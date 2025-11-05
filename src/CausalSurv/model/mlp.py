import torch 
import torch.nn as nn
from CausalSurv.data.config_loader import load_config

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) with ReLU activations."""

    def __init__(self, input_dim: int, output_dim: int, n_layers: int, n_units: list[int]) -> None:
        """Class constructor
        Args:
            input_dim (int): Dimension of the input features
            output_dim (int): Dimension of the output features
            n_layers (int): Number of layers in the MLP
            n_units (list[int]): Number of units in each hidden layer
        """
        super().__init__()
        assert n_layers == len(n_units), "n_layers must match length of n_units list"

        if n_layers == 0:
            print("Warning: MLP with 0 layers, using identity mapping.")
            self.mlp = nn.Identity()
            return

        layers = []
        in_features = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units[i]))
            layers.append(nn.ReLU())
            in_features = n_units[i]
        layers.append(nn.Linear(in_features, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        return self.mlp(x)
    

if __name__ == "__main__":
    config = load_config("../../../configs/mlp.toml")
    batch_size = 4
    input_dim = 3
    output_dim = config['MLP']['output_dim']  # type: ignore

    MLP_test = MLP(input_dim=input_dim, output_dim=output_dim, n_layers=config['MLP']['n_layers'], n_units=config['MLP']['n_units']) # type: ignore
    
    MLP_test.eval()
    x_test = torch.zeros(size=(batch_size, input_dim))
    y_test = MLP_test(x_test)
    print(y_test.shape)  # should be (batch_size, output_dim)

