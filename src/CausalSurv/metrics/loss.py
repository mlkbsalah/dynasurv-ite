import torch

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


class NLLogisticHazard():
    """Class for logistic hazard discrete time survival model loss."""
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, hazard_estimate:torch.Tensor, idx_durations:torch.Tensor, events:torch.Tensor) -> torch.Tensor:
        """

        Args:
            hazard_estimate (torch.Tensor): (batch, n_intervals) tensor of hazard estimates at a specific treatment line
            idx_durations (torch.Tensor): (batch,) tensor of indicies of the time intervals corresponding to observed durations
            events (torch.Tensor): (batch,) tensor of event indicators (1 if event occurred, 0 if censored)

        Returns:
            torch.Tensor: computed loss value with specified reduction
        """
        events = events.view(-1, 1).float()
        idx_durations = idx_durations.view(-1, 1)
        y_true = torch.zeros_like(hazard_estimate).scatter(1, idx_durations, events)
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')(hazard_estimate, y_true)
        loss = bce.cumsum(dim=1).gather(1, idx_durations).view(-1)

        loss = loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss
        return loss