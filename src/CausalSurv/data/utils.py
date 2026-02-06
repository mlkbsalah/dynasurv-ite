from typing import Tuple

import numpy as np
import pandas as pd
import torch


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


def stack_and_pad(
    df: pd.DataFrame, cols: list[str], target_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrapper function to stack by lines and pad sequences to a target length.
    Args:
        df (pd.DataFrame): DataFrame containing patient data with 'usubjid' and 'lineid' columns.
        cols (list[str]): List of column names to stack and pad.
        target_length (int): Target length to pad sequences to. Must be greater than or equal to the length of the longest sequence.
    Returns:
        padded_sequences: Padded sequences of shape (batch_size, target_length, len(cols)).
        mask: Bool tensor of shape (batch_size, target_length) where True indicates valid (non-padded) time steps and False indicates padding positions.
    """
    sequences = stack_by_lines(df, cols)
    tensor_sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    padded_sequences, mask = pad_sequence_to_length(tensor_sequences, target_length)
    return padded_sequences, mask


def transform_time(time: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
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
