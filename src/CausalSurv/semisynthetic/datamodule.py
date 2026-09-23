"""Datamodule for prefix-expanded semi-synthetic data.

Differs from `ESMEOnlineDataModuleCV` in three places, all consequences of prefix
expansion (see `expand.py`):

* the mask keeps only the sample's last line -- earlier rows are real history with
  placeholder outcomes and must reach neither the losses nor the arm-support counts;
* every split that is not temporal groups by `orig_usubjid`, because the samples of one
  patient share their history and would leak across a train / validation boundary
  (a temporal split already groups them: they share the entry year);
* the training loader shuffles, with a seeded generator.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.utils.data as TorchData

from CausalSurv.data.datamodule_cv import ESMEOnlineDataModuleCV
from CausalSurv.semisynthetic.drivers import PAT_ID
from CausalSurv.semisynthetic.expand import ORIG_ID, PREFIX_LINE


class SemiSyntheticDataModule(ESMEOnlineDataModuleCV):
    group_ids: np.ndarray  # (n_samples,) orig_usubjid, aligned with the dataset

    def _transform_to_tensor(self, df_merge):
        tensors, interval_bounds, _ = super()._transform_to_tensor(df_merge)

        by_patient = df_merge.groupby(PAT_ID)
        prefix = by_patient[PREFIX_LINE].first().reindex(tensors["patient_ids"])
        lines = torch.arange(1, self.n_lines + 1)
        mask = (lines[None, :] == torch.as_tensor(prefix.to_numpy())[:, None]).to(
            tensors["mask"].dtype
        )
        if (mask > tensors["mask"]).any():
            raise ValueError("prefix_line points at a line the sample does not have")
        tensors["mask"] = mask

        # Counted on the endpoint rows only; the parent counted every history row.
        valid = self._compute_valid_treatments_per_line(
            tensors["treatment_indices"], mask, self.min_samples_per_treatment
        )
        self.group_ids = (
            by_patient[ORIG_ID].first().reindex(tensors["patient_ids"]).to_numpy()
        )
        return tensors, interval_bounds, valid

    # ---- splits grouped by original patient ----

    def _permuted_groups(self, indices: np.ndarray, seed: int) -> np.ndarray:
        groups = np.unique(self.group_ids[indices])
        return np.random.default_rng(seed).permutation(groups)

    def _group_split(
        self, indices: np.ndarray, fraction: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(rest, chosen): `fraction` of the groups among `indices` go to `chosen`."""
        groups = self._permuted_groups(indices, seed)
        chosen = np.isin(
            self.group_ids[indices], groups[: max(1, round(fraction * len(groups)))]
        )
        return indices[~chosen], indices[chosen]

    def _split_holdout(self):
        if self.temporal_split_year is not None:
            return super()._split_holdout()
        everything = np.arange(len(self.ESMEDataset))
        rest, holdout = self._group_split(
            everything, self.holdout_size, self.split_seed
        )
        return (
            TorchData.Subset(self.ESMEDataset, rest.tolist()),
            TorchData.Subset(self.ESMEDataset, holdout.tolist()),
        )

    def _cv_split(self):
        """Folds over the training partition only, so the holdout never enters CV."""
        pool = np.asarray(self.cv_dataset.indices)
        n_folds = self.num_folds or 5
        groups = self._permuted_groups(pool, self.split_seed)
        fold_of = {g: i % n_folds for i, g in enumerate(groups)}
        folds = np.array([fold_of[g] for g in self.group_ids[pool]])
        fold_idx = self.fold_idx or 0
        val = pool[folds == fold_idx]
        train, early_stop = self._group_split(
            pool[folds != fold_idx], 0.1, self.validation_seed
        )
        return train.tolist(), val.tolist(), early_stop.tolist()

    def train_dataloader(self):
        return TorchData.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.split_seed),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
