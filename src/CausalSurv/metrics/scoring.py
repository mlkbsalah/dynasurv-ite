import torch
import lightning as L
import numpy as np
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.auc import Auc


def concordance_index(sa_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor, time_bins: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute concordance index for each line in the batch.
    Args:
        sa_pred (torch.Tensor):   predicted survival probabilities, shape (batch, n_lines, n_intervals)
        time (torch.Tensor):      event times, shape (batch, n_lines)
        event (torch.Tensor):     event indicators, shape (batch, n_lines)
        time_bins (torch.Tensor): tensor of shape (n_intervals + 1) defining the time intervals
        mask (torch.Tensor):      mask indicating valid lines, shape (batch, n_lines)
    Returns:
        torch.Tensor:             concordance index for each line, shape (n_lines,)
    """
    
    device = sa_pred.device
    batch_size, n_lines = sa_pred.shape[:2]
    c_index = ConcordanceIndex()
    c_index_values = torch.zeros(n_lines, device=device)
    
    bin_indices = torch.searchsorted(time_bins, time) - 1
    bin_indices = torch.clamp(bin_indices, 0, sa_pred.shape[2] - 1)
    
    survival_at_event_times = torch.zeros((batch_size, n_lines), device=device)
    
    for i in range(batch_size):
        for l in range(n_lines):
            if mask[i, l]:
                bin_idx = bin_indices[i, l].item()
                surv_prob = torch.prod(sa_pred[i, l, :bin_idx+1])
                survival_at_event_times[i, l] = surv_prob
    
    for l in range(n_lines):
        masked_indices = mask[:, l]==1
        if masked_indices.sum() > 0: 
            masked_time = time[masked_indices, l]
            masked_event = event[masked_indices, l].bool()
            masked_survival = survival_at_event_times[masked_indices, l]
            
            c_index_value = c_index(estimate=-masked_survival,  
                                   event=masked_event,
                                   time=masked_time)
            c_index_values[l] = c_index_value
    
    return c_index_values


def brier_score(sa_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor, 
                time_bins: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute Brier Score for survival predictions at time bins.
    
    Args:
        sa_pred (torch.Tensor): (batch, n_lines, n_intervals) predicted conditional survival probabilities
        time (torch.Tensor): (batch, n_lines) event/censoring times
        event (torch.Tensor): (batch, n_lines) event indicators
        time_bins (torch.Tensor): (n_intervals+1,) time bin edges (used as evaluation times)
        mask (torch.Tensor): (batch, n_lines) boolean tensor indicating valid entries
    
    Returns:
        torch.Tensor: (n_lines, n_intervals+1) Brier score for each line at each time bin
    """
    from torchsurv.metrics.brier_score import BrierScore
    
    device = sa_pred.device
    batch_size, n_lines, n_intervals = sa_pred.shape
    n_times = len(time_bins)
    brier = BrierScore()
    
    cumulative_survival = torch.cumprod(sa_pred, dim=2)
    
    ones = torch.ones((batch_size, n_lines, 1), device=device)
    survival_at_time_bins = torch.cat([ones, cumulative_survival], dim=2)
    
    brier_scores = torch.zeros((n_lines, n_times), device=device)
    
    for l in range(n_lines):
        masked_indices = mask[:, l]==1
        if masked_indices.sum() > 0:
            masked_time = time[masked_indices, l]
            masked_event = event[masked_indices, l].bool()
            masked_survival = survival_at_time_bins[masked_indices, l, :]
            
            brier_score_value = brier(estimate=masked_survival,
                                     event=masked_event,
                                     time=masked_time,
                                     new_time=time_bins)
            
            brier_scores[l, :] = brier_score_value
    
    return brier_scores


if __name__ == "__main__":
    from CausalSurv.data.online_data_utils import ESMEDataModule
    from CausalSurv.model.DynaSurvCausalOnline import CausalDynaSurv
    from CausalSurv.data.config_loader import load_config


    data_module = ESMEDataModule(data_dir="../../../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
                                 config="../../../configs/data.toml")
    print("DataModule initialized.")
    data_module.prepare_data()
    print("Data prepared.")
    data_module.setup()
    print("DataModule setup complete.")

    input_dims = data_module.get_input_dimensions()
    x_input_dim = input_dims["x_input_dim"]
    p_input_dim = input_dims["p_input_dim"]
    output_sa_length = input_dims["output_sa_length"]

    model_config = load_config("../../../configs/dynasurv.toml")
    model = CausalDynaSurv(
        x_input_dim=x_input_dim,
        p_input_dim=p_input_dim,
        output_sa_length=output_sa_length,
        n_treatments=p_input_dim,
        n_lines=data_module.n_lines,
        config=model_config,
    )

    with torch.no_grad():
        batch = next(iter(data_module.test_dataloader()))
        XPd, sa_true, treatment_index, time, event, mask = batch

        sa_pred, propensity = model.predict_factual_survival(XPd, treatment_index, mask)
        
        print(brier_score(sa_pred, time, event, data_module.time_bins, mask))
        print(concordance_index(sa_pred, time, event, data_module.time_bins, mask))
