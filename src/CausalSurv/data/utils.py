from typing import Tuple

import numpy as np
import pandas as pd
import torch


def split_dataframe(
    df: pd.DataFrame, feature_cols: list[str], id_col: str, visit_col: str
) -> list[np.ndarray]:
    """Split the DataFrame into a list of length the unique patient ids of numpy arrays, where each array is of shape (n_visit_i, len(feature_cols)) corresponding to the visits of a single patient.
    Args:
        df (pd.DataFrame): DataFrame containing patient data with id_col, visit_col columns and feature_cols
        feature_cols (list[str]): List of column names
        id_col (str): Column name for patient ID
        visit_col (str): Column name for visit number
    Returns:
        list[np.ndarray]: List of numpy arrays, each array corresponds to a patient and has shape (n_visit_i, len(feature_cols)).
    """

    if not df.set_index([id_col, visit_col]).index.is_monotonic_increasing:
        raise ValueError(
            f"DataFrame must be sorted by [{id_col}, {visit_col}] before stacking by lines."
            "Unsorted DataFrame may lead to incorrect temporal ordering of lines per patient."
        )

    data_numpy = df[feature_cols].to_numpy()
    patient_ids = df[id_col].to_numpy()

    change_indices = np.where(np.diff(patient_ids) != 0)[0] + 1

    splits = np.split(data_numpy, change_indices)
    return splits


def pad_sequence_to_length(
    sequence: list[torch.Tensor], target_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a sequence to a target length.

    Args:
        sequence (list[torch.Tensor]): List of tensors of shape (seq_length, *feature_dim). *feature_dim can be any shape.
        target_length (int): Target length to pad sequences to. Must be greater than or equal to the length of the longest sequence.

    Returns:
        -padded_sequences: Padded sequences of shape (batch_size, target_length, *feature_dim).
        -mask: Bool tensor of shape (batch_size, target_length) where True indicates valid (non-padded) time steps and False indicates padding positions.
    """
    first_seq = sequence[0]
    feature_dim = first_seq.shape[1:]
    dtype = first_seq.dtype
    device = first_seq.device

    padded_sequences = torch.zeros(
        (len(sequence), target_length, *feature_dim), dtype=dtype, device=device
    )
    mask = torch.zeros((len(sequence), target_length), dtype=torch.bool, device=device)
    for i, seq in enumerate(sequence):
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


def transform_time(time: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Transform continuous times into discrete interval indices based on provided bounds.
        `bounds[i-1] < time <= bounds[i] => interval index = i-1`
    if time is less than or equal to the first bound, it will be assigned to interval 0. If time is greater than the last bound, it will be assigned to the last interval (n_intervals - 1).
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
