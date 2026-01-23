import torch


def move_to_device(
    list_of_tensors: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    """Move a list of tensors to the specified device.

    Args:
        list_of_tensors (list[torch.Tensor]): List of tensors to move
        device (torch.device): Target device

    Returns:
        list[torch.Tensor]: List of tensors on the target device
    """
    return [tensor.to(device) for tensor in list_of_tensors]
