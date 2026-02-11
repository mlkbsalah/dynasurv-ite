import numpy as np
import pytest
import torch

from CausalSurv.data.dataset import ESMEOnlineDataset


class TestCausalSurvDataset:
    @pytest.fixture
    def create_random_data_instance(self):
        n_patients = 10
        n_lines = 5
        n_features = 3
        n_treatments = 2
        n_features_static = 4
        n_treatments_static = 1

        X = torch.randn(n_patients, n_lines, n_features)
        X_static = torch.randn(n_patients, n_features_static)
        P = torch.randint(0, 2, (n_patients, n_lines, n_treatments)).float()
        P_static = torch.randint(0, 2, (n_patients, n_treatments_static)).float()
        treatment_indices = torch.randint(0, n_treatments, (n_patients, n_lines))
        d = torch.rand(n_patients, n_lines, 1)
        time = torch.rand(n_patients, n_lines, 1)
        event = torch.randint(0, 2, (n_patients, n_lines, 1)).float()
        interval_idx = (
            torch.arange(n_lines).unsqueeze(0).repeat(n_patients, 1).unsqueeze(-1)
        )
        mask = torch.ones(n_patients, n_lines)

        patient_ids = np.arange(n_patients)

        return {
            "X": X,
            "X_static": X_static,
            "P": P,
            "treatment_indices": treatment_indices,
            "P_static": P_static,
            "d": d,
            "time": time,
            "event": event,
            "interval_idx": interval_idx,
            "mask": mask,
            "patient_ids": patient_ids,
        }

    def test_len(self, create_random_data_instance):
        data = create_random_data_instance
        dataset = ESMEOnlineDataset(**data)

        assert len(dataset) == 10

    def test_getitem(self, create_random_data_instance):
        data = create_random_data_instance
        dataset = ESMEOnlineDataset(**data)

        XPd, static, treatment_idx, treatment_idx, time, event, mask, patient_id = (
            dataset[0]
        )
