import numpy as np
import torch
import torch.utils.data as TorchData


class ESMEProgressionOnlineDataset(TorchData.Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        X_static: torch.Tensor,
        P: torch.Tensor,
        treatment_indices: torch.Tensor,
        P_static: torch.Tensor,
        d: torch.Tensor,
        death_time: torch.Tensor,
        death_event: torch.Tensor,
        death_interval: torch.Tensor,
        progression_time: torch.Tensor,
        progression_event: torch.Tensor,
        progression_interval: torch.Tensor,
        mask: torch.Tensor,
        patient_ids: np.ndarray,
    ) -> None:
        """Class constructor for ESME dataset

        Args:
            X (torch.Tensor): Dynamic features of shape (n_patients, n_lines, n_features)
            X_static (torch.Tensor): Static features of shape (n_patients, n_features_static)
            P (torch.Tensor): Treatment assignments of shape (n_patients, n_lines, n_treatments)
            P_static (torch.Tensor): Static treatment features of shape (n_patients, n_treatments_static)
            d (torch.Tensor): Buffer duration between lines of shape (n_patients, n_lines, 1)
            death_time (torch.Tensor): Time-to-death of shape (n_patients, n_lines, 1)
            death_event (torch.Tensor): Death event indicators of shape (n_patients, n_lines, 1)
            death_interval (torch.Tensor): Death interval indices of shape (n_patients, n_lines, 1)
            progression_time (torch.Tensor): Time-to-progression of shape (n_patients, n_lines, 1)
            progression_event (torch.Tensor): Progression event indicators of shape (n_patients, n_lines, 1)
            progression_interval (torch.Tensor): Progression interval indices of shape (n_patients, n_lines, 1)
            patient_ids (np.ndarray): Array of patient identifiers of shape (n_patients,)

        Remarks:
            Each patient sample contains all their lines, padded to n_lines.
        """
        self.X = X
        self.X_static = X_static
        self.P = P
        self.P_static = P_static
        self.treatment_indices = treatment_indices
        self.d = d
        self.death_time = death_time
        self.death_event = death_event
        self.death_interval = death_interval
        self.progression_time = progression_time
        self.progression_event = progression_event
        self.progression_interval = progression_interval
        self.mask = mask
        self.patient_ids = patient_ids

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx) -> tuple:
        x = self.X[idx]  # (n_lines_i, n_features)
        x_static = self.X_static[idx]  # (n_features_static,)
        p = self.P[idx]  # (n_lines_i, n_treatments)
        treatment_idx = self.treatment_indices[idx]  # (n_lines_i,)
        p_static = self.P_static[idx]  # (n_treatments_static,)
        d = self.d[idx]  # (n_lines_i, 1)
        death_time = self.death_time[idx]  # (n_lines_i, 1)
        death_event = self.death_event[idx]  # (n_lines_i, 1)
        death_interval = self.death_interval[idx]  # (n_lines_i, 1)
        progression_time = self.progression_time[idx]  # (n_lines_i, 1)
        progression_event = self.progression_event[idx]  # (n_lines_i, 1)
        progression_interval = self.progression_interval[idx]  # (n_lines_i, 1)
        patient_id = self.patient_ids[idx]  # scalar
        mask = self.mask[idx]  # (n_lines_i,)

        XPd = torch.cat([x, p, d], dim=-1)
        static = (x_static, p_static)

        return (
            XPd,
            static,
            death_interval,
            progression_interval,
            treatment_idx,
            death_time,
            death_event,
            progression_time,
            progression_event,
            mask,
            patient_id,
        )
