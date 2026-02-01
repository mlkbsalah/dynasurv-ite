from pathlib import Path
from typing import Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils.data as TorchData
from sklearn.model_selection import KFold


def stack_by_lines(df: pd.DataFrame, cols: list[str]) -> list[np.ndarray]:
    """Stack variables patient-wise, variable number of lines per patient.   list[np.ndarray(patient_i_lines, len(cols))]
    Args:
        df (pd.DataFrame): DataFrame containing patient data with 'usubjid' and 'lineid' columns.
        cols (list[str]): List of column names to stack.
    Returns:
        list[np.ndarray]: List of numpy arrays, each array corresponds to a patient and has shape (patient_i_lines, len(cols)).
    """

    if not df.set_index(["usubjid", "lineid"]).index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted by ['usubjid', 'lineid'] before stacking by lines."
            "Unsorted DataFrame may lead to incorrect temporal ordering of lines per patient."
        )

    data_numpy = df[cols].to_numpy()
    patient_ids = df["usubjid"].to_numpy()

    change_indices = np.where(np.diff(patient_ids) != 0)[0] + 1

    splits = np.split(data_numpy, change_indices)
    return splits


def pad_sequence_to_length(
    sequences: list[torch.Tensor], target_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of sequences to a target length.

    Args:
        sequences (list[torch.Tensor]): List of tensors of shape (seq_length, *feature_dim). *feature_dim can be any shape.
        target_length (int): Target length to pad sequences to. Must be greater than or equal to the length of the longest sequence.

    Returns:
        -padded_sequences: Padded sequences of shape (batch_size, target_length, *feature_dim).
        -mask: Bool tensor of shape (batch_size, target_length) where True indicates valid (non-padded) time steps and False indicates padding positions.
    """
    first_seq = sequences[0]
    feature_dim = first_seq.shape[1:]
    dtype = first_seq.dtype
    device = first_seq.device

    padded_sequences = torch.zeros(
        (len(sequences), target_length, *feature_dim), dtype=dtype, device=device
    )
    mask = torch.zeros((len(sequences), target_length), dtype=torch.bool, device=device)
    for i, seq in enumerate(sequences):
        seq_length = seq.shape[0]

        if seq_length > target_length:
            raise ValueError(
                "Target_length must be greater than or equal to the length of the longest sequence."
                f"Sequence length {seq_length} exceeds target length {target_length}."
            )

        if seq.shape[1:] != feature_dim:
            raise ValueError(
                "All sequences must have the same dimensionality except for the length dimension."
                f"Expected feature dimension {feature_dim}, but got {seq.shape[1:]}."
            )

        padded_sequences[i, :seq_length] = seq
        mask[i, :seq_length] = True

    return padded_sequences, mask


def transform_time(self, time: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Transform continuous times into discrete interval indices based on provided bounds.
    Args:
        time (torch.Tensor): Tensor of shape (n_patients, n_lines, 1) containing continuous time-to-event values.
        bounds (torch.Tensor): Tensor of shape (n_intervals + 1,) defining the boundaries of time intervals.
    Returns:
        torch.Tensor: Tensor of shape (n_patients, n_lines, 1) containing interval indices.
    """

    n_intervals = len(bounds) - 1
    interval_idx = torch.bucketize(time, bounds) - 1
    interval_idx = torch.clamp(interval_idx, min=0, max=n_intervals - 1)

    return interval_idx


class ESMEOnlineDataset(TorchData.Dataset):
    def __init__(
        self,
        X,
        X_static,
        P,
        treatment_indices,
        P_static,
        d,
        time,
        event,
        interval_idx,
        mask,
        patient_ids,
        n_lines: int,
        interval_bounds: torch.Tensor,
    ):
        """Class constructor for ESME dataset

        Args:
            X (torch.Tensor): Dynamic features of shape (n_patients, n_lines, n_features)
            X_static (torch.Tensor): Static features of shape (n_patients, n_features_static)
            P (torch.Tensor): Treatment assignments of shape (n_patients, n_lines, n_treatments)
            P_static (torch.Tensor): Static treatment features of shape (n_patients, n_treatments_static)
            d (torch.Tensor): Buffer duration between lines of shape (n_patients, n_lines, 1)
            time (torch.Tensor): Time-to-event of shape (n_patients, n_lines, 1)
            event (torch.Tensor): Event indicators of shape (n_patients, n_lines, 1)
            interval_idx (torch.Tensor): Interval indices of shape (n_patients, n_lines, 1)
            patient_ids (np.ndarray): Array of patient identifiers of shape (n_patients,)
            n_lines (int): Number of lines to pad/truncate each patient sample to.
            interval_bounds (torch.Tensor): Tensor of shape (n_intervals + 1,) defining the boundaries of time intervals for survival analysis.

        Remarks:
            Each patient sample contains all their lines, padded to n_lines.
        """
        self.X = X
        self.X_static = X_static
        self.P = P
        self.P_static = P_static
        self.treatment_indices = treatment_indices
        self.d = d
        self.time = time
        self.event = event
        self.interval_idx = interval_idx
        self.mask = mask
        self.patient_ids = patient_ids
        self.interval_bounds = interval_bounds
        self.n_lines = n_lines

        self.n_intervals = len(interval_bounds) - 1

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx):
        x = self.X[idx]  # (n_lines_i, n_features)
        x_static = self.X_static[idx]  # (n_features_static,)
        p = self.P[idx]  # (n_lines_i, n_treatments)
        treatment_idx = self.treatment_indices[idx]  # (n_lines_i,)
        p_static = self.P_static[idx]  # (n_treatments_static,)
        d = self.d[idx]  # (n_lines_i, 1)
        time = self.time[idx]  # (n_lines_i, 1)
        event = self.event[idx]  # (n_lines_i, 1)
        interval_idx = self.interval_idx[idx]  # (n_lines_i, 1)
        patient_id = self.patient_ids[idx]  # scalar
        mask = self.mask[idx]  # (n_lines_i,)

        XPd = torch.cat([x, p, d], dim=-1)
        static = (x_static, p_static)

        return (XPd, static, interval_idx, treatment_idx, time, event, mask, patient_id)


class ESMEOnlineDataModuleCV(L.LightningDataModule):
    VALID_SUBTYPES = ["HR+HER2-", "HER2+", "TN"]

    def __init__(
        self,
        data_dir: str,
        subtype: str,
        n_lines: int,
        n_intervals: int,
        batch_size: int,
        split_seed: int,
        final_training: bool = False,
        fold_idx: int | None = None,
        num_folds: int | None = None,
        holdout_size: float = 0.1,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self._subtype = subtype

        self.n_lines = n_lines
        self.n_intervals = n_intervals
        self.treatment_dict = {}

        self.batch_size = batch_size
        self.fold_idx = fold_idx
        self.num_folds = num_folds
        self._split_seed = split_seed
        self.holdout_size = holdout_size
        self.num_workers = num_workers
        self.final_training = final_training

        self.data = None
        self.ESMEDataset = None
        self.static_data = None
        self.interval_bounds = None

    # ========== Properties ==========
    @property
    def subtype(self) -> str:
        return self._subtype

    @property
    def split_seed(self) -> int:
        return self._split_seed

    @subtype.setter
    def subtype(self, value: str):
        if value not in self.VALID_SUBTYPES:
            raise ValueError(
                f"Invalid subtype: {value}. \n"
                f"subtype must be one of {self.VALID_SUBTYPES}"
            )
        self._subtype = value

    @split_seed.setter
    def split_seed(self, value: int):
        if value is None or not isinstance(value, int):
            raise ValueError("split_seed must be a valid integer.")
        self._split_seed = value

    # ================================

    def prepare_data(self):
        df_dynamic = pd.read_parquet(
            self.data_dir
            / f"model_entry_imputed_data_{self._subtype}_stable_types_categorized.parquet"
        )
        df_static = pd.read_parquet(
            self.data_dir / "model_entry_imputes_data_STATIC_no_staging.parquet"
        )

        df_merge = df_dynamic.merge(df_static, on="usubjid", how="inner")

        df_merge = (
            df_merge.loc[df_merge["lineid"] <= self.n_lines]
            .sort_values(by=["usubjid", "lineid"])
            .reset_index(drop=True)
            .copy()
        )

        self.feature_cols = {
            "x": [
                col
                for col in df_dynamic.columns
                if col.startswith("X_") and col != "X_buffer_time"
            ],
            "x_static": [col for col in df_static.columns if col.startswith("X_")],
            "p": ["T_treatment_category"],
            "p_static": [col for col in df_static.columns if col.startswith("T_")],
            "d": ["X_buffer_time"],
            "time": ["Y_onset_to_death"],
            "event": ["Y_death"],
            "id": ["usubjid", "lineid"],
        }

        p_encoded = pd.get_dummies(
            df_merge[self.feature_cols["p"] + self.feature_cols["id"]],
            prefix="",
            prefix_sep="",
        )
        self.treatment_dict = {
            i: col
            for i, col in enumerate(
                p_encoded.columns.drop(self.feature_cols["id"]).tolist()
            )
        }

        X_list = stack_by_lines(df_merge, self.feature_cols["x"])
        P_list = stack_by_lines(
            p_encoded, p_encoded.columns.drop(self.feature_cols["id"]).tolist()
        )
        d_list = stack_by_lines(df_merge, self.feature_cols["d"])
        time_list = stack_by_lines(df_merge, self.feature_cols["time"])
        event_list = stack_by_lines(df_merge, self.feature_cols["event"])

        X_padded, mask = pad_sequence_to_length(
            [torch.tensor(x, dtype=torch.float32) for x in X_list],
            target_length=self.n_lines,
        )
        P_padded, mask_p = pad_sequence_to_length(
            [torch.tensor(p, dtype=torch.float32) for p in P_list],
            target_length=self.n_lines,
        )
        d_padded, mask_d = pad_sequence_to_length(
            [torch.tensor(d, dtype=torch.float32) for d in d_list],
            target_length=self.n_lines,
        )
        time_padded, mask_time = pad_sequence_to_length(
            [torch.tensor(t, dtype=torch.float32) for t in time_list],
            target_length=self.n_lines,
        )
        event_padded, mask_event = pad_sequence_to_length(
            [torch.tensor(e, dtype=torch.float32) for e in event_list],
            target_length=self.n_lines,
        )

        assert (
            torch.all(mask == mask_p)
            and torch.all(mask == mask_d)
            and torch.all(mask == mask_time)
            and torch.all(mask == mask_event)
        ), "Masks for all sequences must be identical."

        X_static = torch.tensor(
            df_merge.groupby("usubjid")
            .first()[self.feature_cols["x_static"]]
            .to_numpy(),
            dtype=torch.float32,
        )
        P_static = torch.tensor(
            df_merge.groupby("usubjid")
            .first()[self.feature_cols["p_static"]]
            .to_numpy(),
            dtype=torch.float32,
        )

        patient_ids = df_merge["usubjid"].unique()

        treatment_indices = torch.argmax(P_padded, dim=-1)

        self.interval_bounds = torch.linspace(
            0, df_merge["Y_onset_to_death"].max(), self.n_intervals + 1
        )
        event_in_bounds = torch.where(
            time_padded <= self.interval_bounds[-1], event_padded, 0
        )
        time_in_bounds = torch.where(
            time_padded <= self.interval_bounds[-1],
            time_padded,
            self.interval_bounds[-1],
        )
        interval_idx = transform_time(time_padded, event_padded, self.interval_bounds)

        self.ESMEDataset = ESMEOnlineDataset(
            X=X_padded,
            X_static=X_static,
            P=P_padded,
            P_static=P_static,
            treatment_indices=treatment_indices,
            d=d_padded,
            time=time_in_bounds,
            event=event_in_bounds,
            interval_idx=interval_idx,
            patient_ids=patient_ids,
            n_lines=self.n_lines,
            interval_bounds=self.interval_bounds,
            mask=mask,
        )

    def setup(self, stage: str | None = None):
        if self.ESMEDataset is None:
            self.prepare_data()
            assert self.ESMEDataset is not None

        dataset_length = len(self.ESMEDataset)
        holdout_length = int(self.holdout_size * dataset_length)

        generator = torch.Generator().manual_seed(self.split_seed)
        self.cv_dataset, self.holdout_dataset = TorchData.random_split(
            self.ESMEDataset,
            [dataset_length - holdout_length, holdout_length],
            generator=generator,
        )

        if self.final_training:
            if stage == "fit" or stage is None:
                self.train_dataset = self.cv_dataset
                self.val_dataset = self.holdout_dataset
            if stage == "test" or stage is None:
                self.test_dataset = self.holdout_dataset
        else:
            if stage == "fit" or stage is None:
                kfold = KFold(
                    n_splits=self.num_folds or 5,
                    shuffle=True,
                    random_state=self.split_seed,
                )
                all_splits = [k for k in kfold.split(range(len(self.ESMEDataset)))]  # type: ignore
                train_idx, val_idx = all_splits[self.fold_idx]  # type: ignore
                train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

                self.train_dataset = TorchData.Subset(self.ESMEDataset, train_idx)
                self.val_dataset = TorchData.Subset(self.ESMEDataset, val_idx)

            if stage == "test" or stage is None:
                self.test_dataset = self.holdout_dataset

    def train_dataloader(self):
        return TorchData.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return TorchData.DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return TorchData.DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False,
            num_workers=1,
            persistent_workers=False,
        )

    def get_data_dimensions(self):
        self.prepare_data()
        assert self.ESMEDataset is not None, (
            "ESMEDataset must be initialized to get data dimensions."
        )

        feature_cols = self.feature_cols
        x_dim = len(feature_cols["x"])
        p_dim = len(self.treatment_dict)
        x_static_dim = len(feature_cols["x_static"])
        p_static_dim = len(feature_cols["p_static"])

        return {
            "x_input_dim": x_dim,
            "p_input_dim": p_dim,
            "p_static_dim": p_static_dim,
            "x_static_dim": x_static_dim,
            "output_dim": self.n_intervals,
            "time_bins": self.interval_bounds,
        }


if __name__ == "__main__":
    import time

    data_module = ESMEOnlineDataModuleCV(
        data_dir="../../../data",
        subtype="HR+HER2-",
        n_lines=2,
        n_intervals=10,
        batch_size=32,
        fold_idx=3,
        num_folds=10,
        split_seed=12345,
        holdout_size=0.1,
        num_workers=4,
    )
    print("DataModule initialized.")
    start = time.time()
    data_module.prepare_data()
    print(f"Data prepared in {time.time() - start:.2f} seconds.")

    start = time.time()
    data_module.setup()
    print(f"DataModule setup complete in {time.time() - start:.2f} seconds.")

    start = time.time()
    train_loader = data_module.train_dataloader()
    print(f"Train DataLoader created in {time.time() - start:.2f} seconds.")

    start = time.time()
    val_loader = data_module.val_dataloader()
    print(f"Validation DataLoader created in {time.time() - start:.2f} seconds.")

    start = time.time()
    test_loader = data_module.test_dataloader()
    print(f"Test DataLoader created in {time.time() - start:.2f} seconds.")

    start = time.time()
    batch = next(iter(train_loader))
    print(f"First batch retrieved in {time.time() - start:.2f} seconds.")
    print("Batch contents:")
    (
        XPd,
        (x_static, p_static),
        interval_idx,
        treatment_indices,
        time,
        event,
        mask,
        patient_id,
    ) = batch
    print("Batch XPd shape:", XPd.shape, "dtype:", XPd.dtype)
    print("Batch x_static shape:", x_static.shape, "dtype:", x_static.dtype)
    print("Batch p_static shape:", p_static.shape, "dtype:", p_static.dtype)
    print("Batch interval_idx shape:", interval_idx.shape, "dtype:", interval_idx.dtype)
    print(
        "Batch treatment_indices shape:",
        treatment_indices.shape,
        "dtype:",
        treatment_indices.dtype,
    )
    print("Batch time shape:", time.shape, "dtype:", time.dtype)
    print("Batch event shape:", event.shape, "dtype:", event.dtype)
    print("Batch mask shape:", mask.shape, "dtype:", mask.dtype)
    print("Batch patient_id shape:", patient_id.shape, "dtype:", patient_id.dtype)
