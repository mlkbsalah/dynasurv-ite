import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    """
    Feature-wise attention mechanism.
    Computes importance weights for each feature in the input and re-weights them.

    Args:
        input_dim (int): Number of input features.
        hidden_dim (int, optional): Size of hidden layer in attention MLP.
                                    If None, defaults to input_dim // 2.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None):
        super().__init__()
        self.input_dim = input_dim
        if hidden_dim is None:
            hidden_dim = max(1, input_dim // 2)

        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, ..., input_dim)

        Returns:
            weighted_x (torch.Tensor): Tensor of shape (batch, ..., input_dim)
            weights (torch.Tensor): Attention weights of shape (batch, ..., input_dim)
        """
        weights = self.attention_net(x)
        weighted_x = x * weights
        return weighted_x, weights
