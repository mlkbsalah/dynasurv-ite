from pathlib import Path
from typing import Dict, Tuple

import lightning as L
import pandas as pd
import torch
import torch.utils.data as TorchData
from sklearn.model_selection import KFold

from .dataset import ESMEOnlineDataset
from .utils import stack_and_pad, transform_time

FULL_ESME_COLUMN_SCHEME = {
    "x_prefix": "X_",
    "x_static_prefix": "X_",
    "p_cols": ["T_treatment_category"],
    "p_static_prefix": "T_",
    "d_cols": ["X_buffer_time"],
    "time_col": "Y_onset_to_death",
    "event_col": "Y_death",
    "pat_id": ["usubjid"],
    "lineid": ["lineid"],
}


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
        columns_scheme: Dict | None = None,
        final_training: bool = False,
        fold_idx: int | None = None,
        num_folds: int | None = None,
        holdout_size: float = 0.1,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.column_scheme = columns_scheme or FULL_ESME_COLUMN_SCHEME
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
    def subtype(self, value: str) -> None:
        if value not in self.VALID_SUBTYPES:
            raise ValueError(
                f"Invalid subtype: {value}. \n"
                f"subtype must be one of {self.VALID_SUBTYPES}"
            )
        self._subtype = value

    @split_seed.setter
    def split_seed(self, value: int) -> None:
        if value is None or not isinstance(value, int):
            raise ValueError("split_seed must be a valid integer.")
        self._split_seed = value

    # ========= Data Preparation ==========

    def _resolve_columns(self, df: pd.DataFrame, spec: list[str] | str) -> list[str]:
        """Resolve column names base on specification.
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            spec (list[str] | str): Column specification. If list, treated as explicit column names.
                                    If str, treated as prefix to match column names.
        Returns:
            list[str]: Resolved column names.
        Raises:
            ValueError: If specified columns are not found in the DataFrame."""
        if isinstance(spec, list):
            missing = [col for col in spec if col not in df.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame.")
            return spec

        if isinstance(spec, str):
            cols = [col for col in df.columns if col.startswith(spec)]
            if not cols:
                raise ValueError(f"No columns found with prefix '{spec}'.")
            return cols

    def _build_column_map(
        self, df_dynamic: pd.DataFrame, df_static: pd.DataFrame
    ) -> Dict[str, list[str]]:
        """Build column mapping based on the provided column scheme.
        Args:
            df_dynamic (pd.DataFrame): DataFrame containing dynamic data.
            df_static (pd.DataFrame): DataFrame containing static data.
        Returns:
            Dict[str, list[str]]: Mapping of data components to their respective column names.
        """
        column_map = {
            "x": self._resolve_columns(df_dynamic, self.column_scheme["x_prefix"]),
            "x_static": self._resolve_columns(
                df_static, self.column_scheme["x_static_prefix"]
            ),
            "p": self.column_scheme["p_cols"],
            "p_static": self._resolve_columns(
                df_static, self.column_scheme["p_static_prefix"]
            ),
            "d": self.column_scheme["d_cols"],
            "time": [self.column_scheme["time_col"]],
            "event": [self.column_scheme["event_col"]],
            "pat_id": self.column_scheme["pat_id"],
            "lineid": self.column_scheme["lineid"],
        }
        return column_map

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_dynamic = pd.read_parquet(
            self.data_dir
            / f"model_entry_imputed_data_{self._subtype}_stable_types_categorized.parquet"
        )
        df_static = pd.read_parquet(
            self.data_dir / "model_entry_imputes_data_STATIC_no_staging.parquet"
        )

        return df_dynamic, df_static

    def _merge_and_filter(
        self, df_dynamic: pd.DataFrame, df_static: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dynamic and static dataframes on patient ID and filter by number of lines."""

        assert self.column_map is not None, (
            "Column map must be initialized before merging data."
        )
        df_merge = df_dynamic.merge(
            df_static, on=self.column_map["pat_id"][0], how="inner"
        )
        df_merge = (
            df_merge.loc[df_merge[self.column_map["lineid"][0]] <= self.n_lines]
            .sort_values(by=self.column_map["pat_id"] + self.column_map["lineid"])
            .reset_index(drop=True)
            .copy()
        )
        return df_merge

    def _transform_to_tensor(self, df_merge: pd.DataFrame):
        p_encoded = pd.get_dummies(
            df_merge[
                self.column_map["p"]
                + self.column_map["pat_id"]
                + self.column_map["lineid"]
            ],
            prefix="",
            prefix_sep="",
        )
        self.treatment_dict = {
            i: col
            for i, col in enumerate(
                p_encoded.columns.drop(
                    self.column_map["pat_id"] + self.column_map["lineid"]
                ).tolist()
            )
        }

        X_padded, mask = stack_and_pad(df_merge, self.column_map["x"], self.n_lines)
        P_padded, _ = stack_and_pad(
            p_encoded,
            p_encoded.columns.drop(
                self.column_map["pat_id"] + self.column_map["lineid"]
            ).tolist(),
            self.n_lines,
        )
        d_padded, _ = stack_and_pad(df_merge, self.column_map["d"], self.n_lines)
        time_padded, _ = stack_and_pad(df_merge, self.column_map["time"], self.n_lines)
        event_padded, _ = stack_and_pad(
            df_merge, self.column_map["event"], self.n_lines
        )

        X_static = torch.tensor(
            df_merge.groupby(self.column_map["pat_id"][0])
            .first()[self.column_map["x_static"]]
            .to_numpy(),
            dtype=torch.float32,
        )
        P_static = torch.tensor(
            df_merge.groupby(self.column_map["pat_id"][0])
            .first()[self.column_map["p_static"]]
            .to_numpy(),
            dtype=torch.float32,
        )

        patient_ids = (
            df_merge[self.column_map["pat_id"][0]].drop_duplicates().to_numpy()
        )

        treatment_indices = torch.argmax(P_padded, dim=-1)

        interval_bounds = torch.linspace(
            0, df_merge[self.column_map["time"][0]].max(), self.n_intervals + 1
        )
        event_in_bounds = torch.where(
            time_padded <= interval_bounds[-1], event_padded, 0
        )
        time_in_bounds = torch.where(
            time_padded <= interval_bounds[-1],
            time_padded,
            interval_bounds[-1],
        )
        interval_idx = transform_time(time_padded, interval_bounds)

        return (
            {
                "X": X_padded,
                "X_static": X_static,
                "P": P_padded,
                "P_static": P_static,
                "treatment_indices": treatment_indices,
                "d": d_padded,
                "time": time_in_bounds,
                "event": event_in_bounds,
                "interval_idx": interval_idx,
                "patient_ids": patient_ids,
                "mask": mask,
            },
            interval_bounds,
        )

    def prepare_data(
        self,
    ) -> None:
        df_dynamic, df_static = self._load_data()

        self.column_map = self._build_column_map(df_dynamic, df_static)
        df_merge = self._merge_and_filter(df_dynamic, df_static)

        padded_tensor_data, self.interval_bounds = self._transform_to_tensor(df_merge)

        self.ESMEDataset = ESMEOnlineDataset(
            **padded_tensor_data,
        )

    # ========= Data splitting ==========

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

    # ========= DataLoaders ==========

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
        column_map = self._build_column_map(*self._load_data())

        x_dim = len(column_map["x"])
        p_dim = len(self.treatment_dict)
        x_static_dim = len(column_map["x_static"])
        p_static_dim = len(column_map["p_static"])

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

    print("Data dimensions:", data_module.get_data_dimensions())
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
