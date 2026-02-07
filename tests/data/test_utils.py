import numpy as np
import pandas as pd
import pytest
import torch

from CausalSurv.data.utils import (
    pad_sequence_to_length,
    split_dataframe,
    transform_time,
)


class TestSplitDataframe:
    def test_split_dataframe(self):
        data = pd.DataFrame(
            {
                "id": [0, 1, 1, 2],
                "visit": [1, 1, 2, 1],
                "feature1": [3, 10, 20, 50],
                "feature2": [33, 100, 200, 500],
            }
        )

        result = split_dataframe(data, ["feature1", "feature2"], "id", "visit")

        assert len(result) == data["id"].nunique()
        assert np.all(result[0] == np.array([[3, 33]]))
        assert np.all(result[1] == np.array([[10, 100], [20, 200]]))
        assert np.all(result[2] == np.array([[50, 500]]))

    def test_split_dataframe_unsorted_id(self):
        data = pd.DataFrame({"id": [102, 101], "visit": [1, 1], "feature": [100, 200]})

        with pytest.raises(ValueError):
            split_dataframe(data, ["feature"], "id", "visit")

    def test_split_dataframe_unsorted_visit(self):
        data = pd.DataFrame(
            {"id": [101, 101, 103], "visit": [2, 1, 1], "feature": [100, 200, 300]}
        )

        with pytest.raises(ValueError):
            split_dataframe(data, ["feature"], "id", "visit")


class TestPadSequenceToLength:
    @pytest.fixture
    def sample_sequences(self):
        return [torch.tensor([[1, 2, 3]]), torch.tensor([[10, 20, 30], [40, 50, 60]])]

    def test_pad_to_max_sequence_length(self, sample_sequences):
        target_length = 2
        result, mask = pad_sequence_to_length(sample_sequences, target_length)

        expected_result = torch.tensor(
            [[[1, 2, 3], [0, 0, 0]], [[10, 20, 30], [40, 50, 60]]]
        )
        expected_mask = torch.tensor([[True, False], [True, True]])

        assert torch.equal(result, expected_result)
        assert torch.equal(mask, expected_mask)

    def test_pad_beyond_max_sequence_length(self, sample_sequences):
        target_length = 3
        result, mask = pad_sequence_to_length(sample_sequences, target_length)

        expected_mask = torch.tensor([[True, False, False], [True, True, False]])

        assert result.shape[1] == 3
        assert torch.equal(mask, expected_mask)

    def test_raises_error_when_target_too_short(self, sample_sequences):
        with pytest.raises(ValueError):
            pad_sequence_to_length(sample_sequences, target_length=1)


class TestTransformTime:
    @pytest.fixture
    def sample_bounds(self):
        return torch.tensor([0.0, 1.0, 2.0])

    def test_transform_time(self, sample_bounds):
        time = torch.tensor([[0.1, 1.1], [0.2, 1.2], [0.2, 1.3]]).unsqueeze(-1)
        interval_idx = transform_time(time, sample_bounds)

        expected_idx = torch.tensor([[0, 1], [0, 1], [0, 1]]).unsqueeze(-1)

        assert torch.equal(interval_idx, expected_idx)

    def tests_transform_time_out_of_bounds(self, sample_bounds):
        time = torch.tensor([[-1.0, 3.0]]).unsqueeze(-1)
        n_intervals = len(sample_bounds) - 1
        interval_idx = transform_time(time, sample_bounds)

        expected_idx = torch.tensor([[0, n_intervals - 1]]).unsqueeze(-1)

        assert torch.equal(interval_idx, expected_idx)

    def test_transform_time_equal_to_bounds(self, sample_bounds):
        time = torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1)
        interval_idx = transform_time(time, sample_bounds)

        expected_idx = torch.tensor([[0, 0, 1]]).unsqueeze(-1)

        assert torch.equal(interval_idx, expected_idx)
