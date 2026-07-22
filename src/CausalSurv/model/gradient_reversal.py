import torch


class _GradientReversal(torch.autograd.Function):
    """Identity on the forward pass, sign-flipped and scaled on the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Insert a gradient reversal layer (Ganin & Lempitsky, 2015) into the graph.

    Everything upstream of `x` receives `-alpha` times the gradient that flows back,
    while everything downstream trains normally. This is what makes an adversarial
    head work: the head minimizes its own loss, the encoder maximizes it, and both
    happen in a single backward pass over one scalar objective.

    Negating the loss term instead does not work -- that makes the head maximize its
    own loss too, and it will run its logits to infinity.

    Args:
        x: tensor to pass through unchanged.
        alpha: strength of the reversed gradient. At 0 the upstream encoder receives
            nothing from the head, but the head itself still trains.
    """
    return _GradientReversal.apply(x, alpha)
