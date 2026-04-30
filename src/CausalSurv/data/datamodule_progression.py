from pathlib import Path
from typing import Dict, Tuple

import lightning as L
import pandas as pd
import torch
import torch.utils.data as TorchData
from sklearn.model_selection import KFold, train_test_split

from .dataset_progression import ESMEProgressionOnlineDataset
from .utils import pad_sequence_to_length, split_dataframe, transform_time

FULL_ESME_COLUMN_SCHEME = {
    "x_prefix": "X_",
    "x_static_prefix": "X_",
    "p_cols": ["T_treatment_category"],
    "p_static_prefix": "T_",
    "d_cols": ["X_buffer_time"],
    "time_col": "Y_onset_to_death",
    "event_col": "Y_global_death_status",
    "progression_time_col": ["Y_onset_to_progression"],
    "progression_event_col": ["Y_progression"],
    "pat_id": ["usubjid"],
    "lineid": ["lineid"],
}


class ESMEProgressionOnlineDataModuleCV(L.LightningDataModule):
    VALID_SUBTYPES = ["HR+HER2-", "HER2+", "TN"]

    def __init__(
        self,
        data_dir: str,
        subtype: str,
        n_lines: int,
        n_intervals: int,
        batch_size: int,
        split_seed: int,
        min_samples_per_treatment: int = 200,
        columns_scheme: Dict = FULL_ESME_COLUMN_SCHEME,
        final_training: bool = False,
        num_folds: int | None = None,
        fold_idx: int | None = None,
        holdout_size: float = 0.2,
        num_workers: int = 4,
        bound_split: str = "uniform",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.column_scheme = columns_scheme
        self._subtype = subtype

        self.n_lines = n_lines
        self.n_intervals = n_intervals
        self.treatment_dict = {}

        self.batch_size = batch_size
        self.fold_idx = fold_idx
        self.num_folds = num_folds
        self.bound_split = bound_split
        self._split_seed = split_seed
        self.holdout_size = holdout_size
        self.num_workers = num_workers
        self.final_training = final_training

        self.min_samples_per_treatment = min_samples_per_treatment

        self.ESMEDataset = None
        self.death_bounds = None
        self.progression_bounds = None
        self._raw_tensors = None

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
            "progression_time": self.column_scheme["progression_time_col"],
            "progression_event": self.column_scheme["progression_event_col"],
            "pat_id": self.column_scheme["pat_id"],
            "lineid": self.column_scheme["lineid"],
        }
        return column_map

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_dynamic = pd.read_parquet(
            self.data_dir
            / f"model_entry_imputed_data_{self._subtype}_stable_types_categorized_V2.parquet"
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

        self.raw_df = df_merge.copy()
        return df_merge

    def _split_and_pad(
        self, df: pd.DataFrame, cols: list[str], target_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrapper function to split the dataframe and pad sequences to a target length.
        Args:
            df (pd.DataFrame): DataFrame containing patient data with 'usubjid' and 'lineid' columns.
            cols (list[str]): List of column names to split and pad.
            target_length (int): Target length to pad sequences to. Must be greater than or equal to the length of the longest sequence.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Padded sequences and mask tensors.
        """
        sequences = split_dataframe(
            df, cols, self.column_map["pat_id"][0], self.column_map["lineid"][0]
        )
        tensor_sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        padded_sequences, mask = pad_sequence_to_length(tensor_sequences, target_length)
        return padded_sequences, mask

    def _compute_valid_treatments_per_line(
        self, treatment_indices: torch.Tensor, mask: torch.Tensor, min_samples: int = 32
    ) -> dict[int, list[int]]:
        valid_treatments = {}
        n_treatments = len(self.treatment_dict)

        for line in range(treatment_indices.shape[1]):
            valid_mask = mask[:, line].bool()
            if not valid_mask.any():
                continue
            t_line = treatment_indices[valid_mask, line]

            valid = [
                k
                for k in range(n_treatments)
                if (t_line == k).sum().item() >= min_samples
            ]
            valid_treatments[line] = valid

        return valid_treatments

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

        X_padded, mask = self._split_and_pad(
            df_merge, self.column_map["x"], self.n_lines
        )
        P_padded, _ = self._split_and_pad(
            p_encoded,
            p_encoded.columns.drop(
                self.column_map["pat_id"] + self.column_map["lineid"]
            ).tolist(),
            self.n_lines,
        )
        d_padded, _ = self._split_and_pad(df_merge, self.column_map["d"], self.n_lines)
        time_padded, _ = self._split_and_pad(
            df_merge, self.column_map["time"], self.n_lines
        )
        event_padded, _ = self._split_and_pad(
            df_merge, self.column_map["event"], self.n_lines
        )

        progression_time_padded, _ = self._split_and_pad(
            df_merge, self.column_map["progression_time"], self.n_lines
        )
        progression_event_padded, _ = self._split_and_pad(
            df_merge, self.column_map["progression_event"], self.n_lines
        )

        X_static = torch.tensor(
            df_merge.groupby(self.column_map["pat_id"][0])
            .first()[
                self.column_map["x_static"]
            ]  # it's static and repetitive so taking the first is enough
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

        return {
            "X": X_padded,
            "X_static": X_static,
            "P": P_padded,
            "P_static": P_static,
            "treatment_indices": treatment_indices,
            "d": d_padded,
            "time_padded": time_padded,
            "event_padded": event_padded,
            "progression_time_padded": progression_time_padded,
            "progression_event_padded": progression_event_padded,
            "patient_ids": patient_ids,
            "mask": mask,
        }

    def _build_bounds(
        self,
        time_padded: torch.Tensor,
        event_padded: torch.Tensor,
        progression_time_padded: torch.Tensor,
        progression_event_padded: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-line interval bounds from a (training) subset of patients."""
        death_bounds_list = []
        progression_bounds_list = []
        q = torch.linspace(0, 1, self.n_intervals + 1)

        for line in range(self.n_lines):
            valid = mask[:, line].bool()
            t_death = time_padded[valid, line, 0]
            t_prog = progression_time_padded[valid, line, 0]

            if self.bound_split == "quantile":
                ev_death = event_padded[valid, line, 0].bool()
                ev_prog = progression_event_padded[valid, line, 0].bool()
                line_death_bounds = (
                    torch.quantile(t_death[ev_death].float(), q)
                    if ev_death.any()
                    else torch.linspace(0, t_death.max().item(), self.n_intervals + 1)
                )
                line_prog_bounds = (
                    torch.quantile(t_prog[ev_prog].float(), q)
                    if ev_prog.any()
                    else torch.linspace(0, t_prog.max().item(), self.n_intervals + 1)
                )
            elif self.bound_split == "uniform":
                line_death_bounds = torch.linspace(
                    0, t_death.max().item(), self.n_intervals + 1
                )
                line_prog_bounds = torch.linspace(
                    0, t_prog.max().item(), self.n_intervals + 1
                )
            else:
                raise ValueError(f"Unknown bound_split: {self.bound_split!r}")

            death_bounds_list.append(line_death_bounds)
            progression_bounds_list.append(line_prog_bounds)

        death_bounds = torch.stack(death_bounds_list, dim=0)  # (n_lines, n_intervals+1)
        progression_bounds = torch.stack(
            progression_bounds_list, dim=0
        )  # (n_lines, n_intervals+1)
        return death_bounds, progression_bounds

    def _apply_bounds(
        self,
        raw_tensors: dict,
        death_bounds: torch.Tensor,
        progression_bounds: torch.Tensor,
    ) -> dict:
        """Clip times/events to bound range and discretise into intervals."""
        time_padded = raw_tensors["time_padded"]
        event_padded = raw_tensors["event_padded"]
        progression_time_padded = raw_tensors["progression_time_padded"]
        progression_event_padded = raw_tensors["progression_event_padded"]

        death_max_bc = death_bounds[:, -1].view(1, -1, 1)  # (1, n_lines, 1)
        prog_max_bc = progression_bounds[:, -1].view(1, -1, 1)  # (1, n_lines, 1)

        death_event_in_bounds = torch.where(
            time_padded <= death_max_bc, event_padded, torch.zeros_like(event_padded)
        )
        death_time_in_bounds = torch.min(time_padded, death_max_bc)

        progression_event_in_bounds = torch.where(
            progression_time_padded <= prog_max_bc,
            progression_event_padded,
            torch.zeros_like(progression_event_padded),
        )
        progression_time_in_bounds = torch.min(progression_time_padded, prog_max_bc)

        death_interval = torch.stack(
            [
                transform_time(death_time_in_bounds[:, line, :], death_bounds[line])
                for line in range(self.n_lines)
            ],
            dim=1,
        )
        progression_interval = torch.stack(
            [
                transform_time(
                    progression_time_in_bounds[:, line, :], progression_bounds[line]
                )
                for line in range(self.n_lines)
            ],
            dim=1,
        )

        return {
            "X": raw_tensors["X"],
            "X_static": raw_tensors["X_static"],
            "P": raw_tensors["P"],
            "P_static": raw_tensors["P_static"],
            "treatment_indices": raw_tensors["treatment_indices"],
            "d": raw_tensors["d"],
            "death_time": death_time_in_bounds,
            "death_event": death_event_in_bounds,
            "death_interval": death_interval,
            "progression_time": progression_time_in_bounds,
            "progression_event": progression_event_in_bounds,
            "progression_interval": progression_interval,
            "patient_ids": raw_tensors["patient_ids"],
            "mask": raw_tensors["mask"],
        }

    def prepare_data(self) -> None:
        self.df_dynamic, self.df_static = self._load_data()
        self.column_map = self._build_column_map(self.df_dynamic, self.df_static)
        df_merge = self._merge_and_filter(self.df_dynamic, self.df_static)
        self._raw_tensors = self._transform_to_tensor(df_merge)

    # ========= Data splitting ==========

    def setup(self, stage: str | None = None):
        if self._raw_tensors is None:
            self.prepare_data()
        assert self._raw_tensors is not None

        n_patients = len(self._raw_tensors["patient_ids"])
        holdout_length = int(self.holdout_size * n_patients)

        generator = torch.Generator().manual_seed(self.split_seed)
        perm = torch.randperm(n_patients, generator=generator).tolist()
        cv_idx = perm[: n_patients - holdout_length]
        holdout_idx = perm[n_patients - holdout_length :]

        if self.final_training:
            train_idx = cv_idx
        else:
            kfold = KFold(
                n_splits=self.num_folds or 5,
                shuffle=True,
                random_state=self.split_seed,
            )
            all_splits = list(kfold.split(cv_idx))
            train_fold_idx, val_fold_idx = all_splits[self.fold_idx]  # type: ignore
            train_idx = [cv_idx[i] for i in train_fold_idx]
            val_idx = [cv_idx[i] for i in val_fold_idx]
            train_idx, early_stop_idx = train_test_split(
                train_idx, test_size=0.1, shuffle=True, random_state=self.split_seed
            )

        # Build bounds from training patients only
        train_sel = torch.tensor(train_idx)
        raw = self._raw_tensors
        self.death_bounds, self.progression_bounds = self._build_bounds(
            raw["time_padded"][train_sel],
            raw["event_padded"][train_sel],
            raw["progression_time_padded"][train_sel],
            raw["progression_event_padded"][train_sel],
            raw["mask"][train_sel],
        )

        # Apply bounds to all patients, then build the full dataset
        final_tensors = self._apply_bounds(
            raw, self.death_bounds, self.progression_bounds
        )
        self.valid_treatments_per_line = self._compute_valid_treatments_per_line(
            final_tensors["treatment_indices"][train_sel],
            final_tensors["mask"][train_sel],
            self.min_samples_per_treatment,
        )
        self.ESMEDataset = ESMEProgressionOnlineDataset(**final_tensors)

        if self.final_training:
            if stage == "fit" or stage is None:
                self.train_dataset = TorchData.Subset(self.ESMEDataset, cv_idx)
                self.val_dataset = TorchData.Subset(self.ESMEDataset, holdout_idx)
            if stage == "test" or stage is None:
                self.test_dataset = TorchData.Subset(self.ESMEDataset, holdout_idx)
        else:
            if stage == "fit" or stage is None:
                self.train_dataset = TorchData.Subset(self.ESMEDataset, train_idx)
                self.val_dataset = TorchData.Subset(self.ESMEDataset, val_idx)
                self.es_dataset = TorchData.Subset(self.ESMEDataset, early_stop_idx)
            if stage == "test" or stage is None:
                self.test_dataset = TorchData.Subset(self.ESMEDataset, holdout_idx)

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
        dataloader_kwargs = {
            "batch_size": len(self.val_dataset),
            "shuffle": False,
            "num_workers": 1,
            "persistent_workers": True,
        }
        if not self.final_training:
            return [
                TorchData.DataLoader(self.val_dataset, **dataloader_kwargs),
                TorchData.DataLoader(self.es_dataset, **dataloader_kwargs),
            ]
        else:
            return TorchData.DataLoader(self.val_dataset, **dataloader_kwargs)

    def test_dataloader(self):
        return TorchData.DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
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
            "death_time_bins": self.death_bounds,
            "progression_time_bins": self.progression_bounds,
        }
