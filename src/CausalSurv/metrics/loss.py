import torch

def SURVLoss(sa_true: torch.Tensor, sa_pred: torch.Tensor, epsilon: float = 1e-7, reduction: str = "mean") -> torch.Tensor:
    """
    Survival analysis loss function adapted from nnet_survival.py
    
    Args:
        sa_true: Ground truth, shape (batch, 2*n_intervals)
                 - First n_intervals: survival indicators (1 if survived interval, 0 if not)
                 - Second n_intervals: event indicators (1 for interval where event occurred)
        sa_pred: Predicted conditional survival probabilities, shape (batch, n_intervals)
                 - Probability of surviving each interval given survival to start of interval
        epsilon: Small constant to prevent log(0)
    
    Returns:
        loss: Scalar loss value (mean over batch and time)
    """
    n_intervals = sa_pred.size(1)

    survived_intervals = sa_true[:, :n_intervals]    # (batch, n_intervals)
    event_intervals = sa_true[:, n_intervals:]       # (batch, n_intervals)

    cens_uncens = 1.0 + survived_intervals * (sa_pred - 1.0)
    uncens = 1.0 - event_intervals * sa_pred
    
    cens_uncens = torch.clamp(cens_uncens, min=epsilon)
    uncens = torch.clamp(uncens, min=epsilon)

    loss =  -torch.sum(torch.log(cens_uncens) + torch.log(uncens), dim=1)
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)

    return loss


def PROPLoss(propensity: torch.Tensor, treatment_index: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Propensity loss function using Cross Entropy Loss.
    
    Args:
        propensity: Predicted propensity scores, shape (batch, n_treatments)
        treatment_index: Ground truth treatment indices, shape (batch,)
    Returns:
        loss: Scalar loss value (mean over batch and time)
    """

    n_treatments = propensity.size(-1)
    propensity = propensity.view(-1, n_treatments)  
    loss = torch.nn.CrossEntropyLoss(reduction=reduction)(propensity, treatment_index)
    
    return loss