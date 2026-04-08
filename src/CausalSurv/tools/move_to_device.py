import torch


def move_to_device(
    list_of_tensors: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    return [tensor.to(device) for tensor in list_of_tensors]
