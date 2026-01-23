import torch
import torch.nn as nn


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) with ReLU activations."""

    def __init__(
        self, input_dim: int, output_dim: int, n_units: list[int], dropout: float
    ) -> None:
        """Class constructor
        Args:
            input_dim (int): Dimension of the input features
            output_dim (int): Dimension of the output features
            n_layers (int): Number of layers in the MLP
            n_units (list[int]): Number of units in each hidden layer
        """
        super().__init__()
        if len(n_units) == 0:
            print("Warning: MLP with 0 layers, using identity mapping.")
            self.mlp = nn.Identity()
            return

        layers = []
        prev_layer = input_dim
        self.dropout = dropout
        for i in range(len(n_units)):
            layers.append(nn.Linear(prev_layer, n_units[i]))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(n_units[i]))
            prev_layer = n_units[i]

        layers.append(nn.Linear(prev_layer, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        return self.mlp(x)
