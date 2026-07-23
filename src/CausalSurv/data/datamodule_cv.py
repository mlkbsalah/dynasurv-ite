from pathlib import Path
from typing import Dict, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils.data as TorchData
from sklearn.model_selection import KFold, train_test_split

from .dataset import ESMEOnlineDataset
from .utils import pad_sequence_to_length, split_dataframe, transform_time

FULL_ESME_COLUMN_SCHEME = {
    "x_prefix": "X_",
    "x_static_prefix": "X_",
    "p_cols": ["T_treatment_category"],
    "p_static_prefix": "T_",
    "d_cols": ["X_time_between_onsets"],
    "time_col": "Y_onset_to_death",
    "event_col": "Y_global_death_status",
    "pat_id": ["usubjid"],
    "lineid": ["lineid"],
    "line_start_col": "line_start_date",
}

# Arms excluded from the *recommendable* action set (they remain in the data so
# that patient histories stay intact and still inform the encoder).
#
#   NO TREATMENT  - counts across lines 1-4 are 207/0/0/1: a line-1 coding
#                   artefact, not a therapeutic option. Structural
#                   non-positivity at every later line.
#   OTHER         - 736 distinct drug-flag combinations, modal share 0.16.
#                   "Set A_k = OTHER" is not a well-defined intervention
#                   (multiple versions of treatment).
#   ET+TT         - 344 distinct combinations, modal share 0.16; same problem.
#
# See improvements.md sections 3, 6 and 7 for the supporting counts.
DEFAULT_EXCLUDED_ARMS = ["NO TREATMENT", "OTHER", "ET+TT"]


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
        min_samples_per_treatment: int = 200,
        columns_scheme: Dict = FULL_ESME_COLUMN_SCHEME,
        final_training: bool = False,
        num_folds: int | None = None,
        fold_idx: int | None = None,
        holdout_size: float = 0.2,
        num_workers: int = 4,
        standardize_continuous: bool = True,
        binary_threshold: int = 2,
        cohort_start_year: int | None = None,
        temporal_split_year: int | None = None,
        add_calendar_feature: bool = False,
        excluded_treatment_arms: list[str] | None = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.column_scheme = columns_scheme
        self._subtype = subtype

        # --- Identifiability controls (see improvements.md) ---
        # cohort_start_year: keep only patients whose FIRST line starts in or
        #   after this year, so every arm in the action set was available for
        #   the whole window (positivity by restriction, section 1).
        # temporal_split_year: patients entering in or after this year form the
        #   holdout, so validation measures generalisation across policy eras
        #   rather than across a random shuffle (section 5).
        # add_calendar_feature: expose calendar time to the encoder as a
        #   covariate, which is safe once availability is stable (section 2).
        self.cohort_start_year = cohort_start_year
        self.temporal_split_year = temporal_split_year
        self.add_calendar_feature = add_calendar_feature
        self.excluded_treatment_arms = (
            DEFAULT_EXCLUDED_ARMS
            if excluded_treatment_arms is None
            else excluded_treatment_arms
        )
        self.patient_entry_years: pd.Series | None = None
        self.recommendable_treatments_per_line: dict[int, list[int]] = {}
        self.cohort_summary: Dict[str, object] = {}

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

        self.min_samples_per_treatment = min_samples_per_treatment

        self.standardize_continuous = standardize_continuous
        self.binary_threshold = binary_threshold
        self.continuous_cols: list[str] = []
        self.scaler: Dict[str, pd.Series] | None = None

        self.ESMEDataset = None
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

        # ic(column_map)

        return column_map

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_dynamic = pd.read_parquet(
            self.data_dir
            / f"model_entry_imputed_data_{self._subtype}_stable_types_categorized_V2.parquet"
        )
        df_static = pd.read_parquet(
            self.data_dir / "model_entry_imputes_data_STATIC_no_staging.parquet"
        )

        # The cohort restriction and the calendar covariate are applied here
        # rather than in prepare_data() because get_data_dimensions() re-loads
        # the frames independently; doing it later would report a feature count
        # that disagrees with the tensors actually built.
        df_dynamic = self._restrict_to_cohort(df_dynamic)
        df_dynamic = self._add_calendar_feature(df_dynamic)

        return df_dynamic, df_static

    def _entry_years(self, df_dynamic: pd.DataFrame) -> pd.Series:
        """Calendar year of each patient's FIRST treatment line."""
        pat_id = self.column_scheme["pat_id"][0]
        start_col = self.column_scheme["line_start_col"]
        return df_dynamic.groupby(pat_id)[start_col].min().dt.year.rename("entry_year")

    def _restrict_to_cohort(self, df_dynamic: pd.DataFrame) -> pd.DataFrame:
        """Keep patients whose first line starts at/after `cohort_start_year`.

        The filter is applied at the PATIENT level, not the record level. A
        record-level cut would keep a patient's line 3 while dropping lines 1-2,
        handing the sequence encoder a history that starts mid-trajectory; on
        this cohort that severs 31% of retained patients. Entry-based selection
        keeps every trajectory whole, which is also what a target trial
        emulation enrolling patients at metastatic diagnosis would do.
        """
        if self.cohort_start_year is None:
            return df_dynamic

        pat_id = self.column_scheme["pat_id"][0]
        entry = self._entry_years(df_dynamic)
        keep = entry[entry >= self.cohort_start_year].index
        restricted = df_dynamic[df_dynamic[pat_id].isin(keep)].copy()

        self.cohort_summary = {
            "cohort_start_year": self.cohort_start_year,
            "patients_before": int(df_dynamic[pat_id].nunique()),
            "patients_after": int(restricted[pat_id].nunique()),
            "records_before": int(len(df_dynamic)),
            "records_after": int(len(restricted)),
        }
        return restricted

    def _add_calendar_feature(self, df_dynamic: pd.DataFrame) -> pd.DataFrame:
        """Add months-since-cohort-start as a dynamic covariate.

        Safe only inside an availability-stable window: there calendar time is
        an ordinary confounder (secular drift in supportive care and in the
        treatment policy) rather than a determinant of which arms exist, so
        conditioning on it removes bias instead of licensing extrapolation.
        """
        if not self.add_calendar_feature:
            return df_dynamic

        start_col = self.column_scheme["line_start_col"]
        df_dynamic = df_dynamic.copy()
        origin_year = self.cohort_start_year or int(df_dynamic[start_col].dt.year.min())
        origin = pd.Timestamp(year=origin_year, month=1, day=1)
        df_dynamic["X_calendar_months"] = (
            df_dynamic[start_col] - origin
        ).dt.days / 30.44
        return df_dynamic

    def _detect_continuous_columns(
        self, df: pd.DataFrame, candidate_cols: list[str]
    ) -> list[str]:
        """Pick columns with more than `binary_threshold` unique values.

        Columns at or below the threshold (typically binary indicators) are
        treated as categorical/binary and left unscaled — z-scoring rare flags
        creates asymmetric spikes that hurt training (see CLAUDE.md notes).
        """
        return [
            col
            for col in candidate_cols
            if df[col].nunique(dropna=True) > self.binary_threshold
        ]

    def _fit_standardizer(
        self, df: pd.DataFrame, cols: list[str]
    ) -> Dict[str, pd.Series]:
        """Compute per-column mean/std on df[cols]. Constant columns get std=1."""
        means = df[cols].mean()
        stds = df[cols].std().replace(0.0, 1.0)
        return {"mean": means, "std": stds}

    def _apply_standardizer(
        self, df: pd.DataFrame, cols: list[str], scaler: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        df = df.copy()
        df[cols] = (df[cols] - scaler["mean"]) / scaler["std"]
        return df

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

    def _compute_recommendable_treatments_per_line(
        self, valid_treatments_per_line: dict[int, list[int]]
    ) -> dict[int, list[int]]:
        """Arms the model is allowed to *recommend*, per line.

        Two filters compose here:

        1. Empirical support -- an arm must clear `min_samples_per_treatment` at
           that line. Below that the counterfactual is extrapolation rather than
           identification, and the viable set genuinely differs by line (at
           line 4 endocrine therapy alone all but disappears while mono-chemo
           dominates), so a single global action space is wrong in both
           directions.
        2. Well-definedness -- arms in `excluded_treatment_arms` are dropped
           regardless of support, either because they are not an intervention
           at all (NO TREATMENT) or because the label pools too many distinct
           regimens to be a single intervention (OTHER, ET+TT).

        Records for excluded arms are deliberately NOT removed from the
        dataset: they are real treatment events that inform the patient history
        and the shared encoder. Only their eligibility to be *recommended* is
        withdrawn.

        Caveat: support is counted on the full cohort, holdout included, so the
        holdout's arm composition informs which arms are eligible. This is an
        eligibility decision rather than a fitted parameter -- at deployment you
        would likewise use all history to decide what may be offered -- but if
        you need a strictly clean split, recompute this from the train indices
        inside setup() and re-read it in the model's on_fit_start.
        """
        excluded_idx = {
            idx
            for idx, name in self.treatment_dict.items()
            if name in self.excluded_treatment_arms
        }
        return {
            line: [k for k in valid if k not in excluded_idx]
            for line, valid in valid_treatments_per_line.items()
        }

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

        valid_treatments_per_line = self._compute_valid_treatments_per_line(
            treatment_indices, mask, self.min_samples_per_treatment
        )
        self.recommendable_treatments_per_line = (
            self._compute_recommendable_treatments_per_line(valid_treatments_per_line)
        )

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
            valid_treatments_per_line,
        )

    def prepare_data(
        self,
    ) -> None:
        self.df_dynamic, self.df_static = self._load_data()

        self.column_map = self._build_column_map(self.df_dynamic, self.df_static)
        df_merge = self._merge_and_filter(self.df_dynamic, self.df_static)

        if self.standardize_continuous:
            # Scaling is restricted to X (dynamic) and X_static. We skip:
            #   - P / P_static: treatment one-hots and history flags (binary by construction)
            #   - d_cols: time-decay input is fed into 1/log(e + d); z-scoring can make
            #     it negative and produce NaNs
            #   - time/event/lineid/pat_id: survival targets and identifiers
            # Caveat: scaler is fit on the full merged frame (including the 20% holdout).
            # Z-score means/stds are robust to that, but if you ever need strict no-leak,
            # move the fit into setup() once train indices are known.
            scaling_candidates = list(
                dict.fromkeys(self.column_map["x"] + self.column_map["x_static"])
            )
            self.continuous_cols = self._detect_continuous_columns(
                df_merge, scaling_candidates
            )
            self.scaler = self._fit_standardizer(df_merge, self.continuous_cols)
            df_merge = self._apply_standardizer(
                df_merge, self.continuous_cols, self.scaler
            )

        padded_tensor_data, self.interval_bounds, self.valid_treatments_per_line = (
            self._transform_to_tensor(df_merge)
        )

        # Entry year per patient, ordered to match padded_tensor_data["patient_ids"],
        # so the temporal split in setup() can index the dataset directly.
        entry = self._entry_years(df_merge)
        self.patient_entry_years = entry.reindex(padded_tensor_data["patient_ids"])

        self.ESMEDataset = ESMEOnlineDataset(
            **padded_tensor_data,
        )

    def describe_cohort(self) -> None:
        """Print the cohort restriction, action mask and split composition.

        Every run states the population it was actually fit on, so a checkpoint
        can never be read as covering more patients or more arms than it does.
        """
        print("\n" + "=" * 72)
        print("COHORT / IDENTIFIABILITY SUMMARY")
        print("=" * 72)

        if self.cohort_summary:
            s = self.cohort_summary
            print(
                f"Cohort restriction: first line >= {s['cohort_start_year']}\n"
                f"  patients {s['patients_before']} -> {s['patients_after']}"
                f"   records {s['records_before']} -> {s['records_after']}"
            )
        else:
            print("Cohort restriction: none (full calendar range)")

        print(
            f"Calendar covariate: {'X_calendar_months' if self.add_calendar_feature else 'none'}"
        )

        if self.patient_entry_years is not None:
            counts = self.patient_entry_years.value_counts().sort_index()
            print(f"Entry years: {dict(counts)}")

        if (
            self.temporal_split_year is not None
            and self.patient_entry_years is not None
        ):
            years = self.patient_entry_years.to_numpy()
            n_tr = int((years < self.temporal_split_year).sum())
            n_ho = int((years >= self.temporal_split_year).sum())
            print(
                f"Temporal split at {self.temporal_split_year}: "
                f"train {n_tr} patients / holdout {n_ho} patients "
                f"({n_ho / (n_tr + n_ho):.1%} held out)"
            )
        else:
            print(f"Split: random, holdout_size={self.holdout_size}")

        print(f"Excluded from action set: {self.excluded_treatment_arms}")
        print(
            f"Recommendable arms per line (min {self.min_samples_per_treatment} obs):"
        )
        for line in sorted(self.recommendable_treatments_per_line):
            names = [
                self.treatment_dict[k]
                for k in self.recommendable_treatments_per_line[line]
            ]
            print(f"  line {line + 1}: {len(names)} arms -> {names}")
        print("=" * 72 + "\n")

    # ========= Data splitting ==========

    def _temporal_split(self) -> Tuple[TorchData.Subset, TorchData.Subset]:
        """Split by patient entry year instead of at random.

        A random split lets a 2019 patient sit in train while another 2019
        patient is scored in validation, so the model is told what the 2019
        treatment policy looked like before being asked to predict on it. Since
        the policy is the thing that drifts, that inflates apparent performance
        in a way deployment will not reproduce. Holding out the latest entry
        years instead measures exactly the generalisation the single-line
        estimand assumes: that the future-treatment policy at deployment stays
        close to the one seen in training.
        """
        assert self.patient_entry_years is not None, (
            "prepare_data() must run before a temporal split can be built."
        )
        years = self.patient_entry_years.to_numpy()
        train_idx = np.flatnonzero(years < self.temporal_split_year).tolist()
        holdout_idx = np.flatnonzero(years >= self.temporal_split_year).tolist()

        if not train_idx or not holdout_idx:
            raise ValueError(
                f"temporal_split_year={self.temporal_split_year} leaves an empty split "
                f"(train={len(train_idx)}, holdout={len(holdout_idx)}). "
                "Pick a year inside the cohort's entry range."
            )

        return (
            TorchData.Subset(self.ESMEDataset, train_idx),
            TorchData.Subset(self.ESMEDataset, holdout_idx),
        )

    def setup(self, stage: str | None = None):
        if self.ESMEDataset is None:
            self.prepare_data()
            assert self.ESMEDataset is not None

        dataset_length = len(self.ESMEDataset)
        holdout_length = int(self.holdout_size * dataset_length)

        if self.temporal_split_year is not None:
            self.cv_dataset, self.holdout_dataset = self._temporal_split()
        else:
            # train/finetuning - RealTest (holdout)
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

                train_idx, early_stop_idx = train_test_split(
                    train_idx, test_size=0.1, shuffle=True
                )

                self.train_dataset = TorchData.Subset(self.ESMEDataset, train_idx)
                self.val_dataset = TorchData.Subset(self.ESMEDataset, val_idx)
                self.es_dataset = TorchData.Subset(self.ESMEDataset, early_stop_idx)

            if stage == "test" or stage is None:
                self.test_dataset = self.holdout_dataset

    # ========= DataLoaders ==========

    def train_dataloader(self):
        return TorchData.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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
