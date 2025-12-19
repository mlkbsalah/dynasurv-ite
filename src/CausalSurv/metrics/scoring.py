import torch
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex 
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator

def kaplan_meier_estimate(time: torch.Tensor, event: torch.Tensor) -> tuple:
    """Compute Kaplan-Meier estimate of survival function.
    Args:
        time (torch.Tensor):  event times, shape (n_samples,)
        event (torch.Tensor): event indicators, shape (n_samples,)
    Returns:
        times (torch.Tensor): unique event times, shape (n_times,)
        survival_prob (torch.Tensor): survival probabilities at each time, shape (n_times,)
    """
    device = time.device

    # Sort times and events
    sorted_indices = torch.argsort(time)
    time_sorted = time[sorted_indices]
    event_sorted = event[sorted_indices]
    
    unique_times, inverse_indices = torch.unique_consecutive(time_sorted, return_inverse=True)
    n_times = unique_times.shape[0]
    
    at_risk = torch.zeros(n_times, device=device)
    events = torch.zeros(n_times, device=device)
    
    for i in range(n_times):
        at_risk[i] = (time_sorted >= unique_times[i]).sum()
        events[i] = event_sorted[inverse_indices == i].sum()
    
    survival_prob = torch.ones(n_times, device=device)
    for i in range(n_times):
        if at_risk[i] > 0:
            survival_prob[i] = survival_prob[i-1] * (1 - events[i] / at_risk[i]) if i > 0 else (1 - events[i] / at_risk[i])
        else:
            survival_prob[i] = survival_prob[i-1] if i > 0 else 1.0
    
    return unique_times, survival_prob


def concordance_index(sa_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor, time_bins: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Compute IPCW-corrected concordance index for each line in the batch."""
    
    device = sa_pred.device
    batch_size, n_lines = sa_pred.shape[:2]
    c_index_values = torch.zeros(n_lines, device=device)
    
    bin_indices = torch.searchsorted(time_bins, time) - 1
    bin_indices = torch.clamp(bin_indices, 0, sa_pred.shape[2] - 1)
    
    survival_at_event_times = torch.zeros((batch_size, n_lines), device=device)

    if mask is None:
        mask = torch.ones((batch_size, n_lines), device=device)

    for i in range(batch_size):
        for l in range(n_lines):
            if mask[i, l]:
                bin_idx = bin_indices[i, l].item()
                surv_prob = torch.prod(sa_pred[i, l, :bin_idx+1])
                survival_at_event_times[i, l] = surv_prob
    
    # ipcw_weights = torch.zeros((batch_size, n_lines), device=device)
    
    for l in range(n_lines):
        masked_indices = mask[:, l] == 1
        if masked_indices.sum() > 1:  
            line_times = time[masked_indices, l]
            line_censored = ~event[masked_indices, l].bool()
            
            unique_times, G_km = kaplan_meier_estimate(line_times, line_censored)
            
            # for i in range(batch_size):
            #     if mask[i, l]:
            #         t_i = time[i, l].item()
            #         if t_i <= unique_times[0].item():
            #             G_t = G_km[0].item()
            #         elif t_i >= unique_times[-1].item():
            #             G_t = G_km[-1].item()
            #         else:
            #             idx = torch.searchsorted(unique_times, time[i, l])
            #             t0, t1 = unique_times[idx-1].item(), unique_times[idx].item()
            #             G0, G1 = G_km[idx-1].item(), G_km[idx].item()
            #             G_t = G0 + (G1 - G0) * (t_i - t0) / (t1 - t0)
                    
            #         ipcw_weights[i, l] = 1.0 / (max(G_t, 1e-6) ** 2)
    
    for l in range(n_lines):
        masked_indices = mask[:, l] == 1
        if masked_indices.sum() > 0:
            masked_time = time[masked_indices, l]
            masked_event = event[masked_indices, l].bool()
            masked_survival = survival_at_event_times[masked_indices, l]
            # masked_ipcw = ipcw_weights[masked_indices, l]
            
            c_index = ConcordanceIndex()
            c_index_value = c_index(
                estimate=-torch.log(masked_survival + 1e-8),
                event=masked_event,
                time=masked_time
            )
            c_index_values[l] = c_index_value
    
    return c_index_values


def brier_score(sa_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor, 
                time_bins: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
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
    device = sa_pred.device
    batch_size, n_lines, n_intervals = sa_pred.shape
    n_times = len(time_bins)
    
    cumulative_survival = torch.cumprod(sa_pred, dim=2)
    
    ones = torch.ones((batch_size, n_lines, 1), device=device)
    survival_at_time_bins = torch.cat([ones, cumulative_survival], dim=2)
    
    brier_scores = torch.zeros((n_lines, n_times), device=device)
    integrated_brier_scores = torch.zeros((n_lines,), device=device)

    if mask is None:
        mask = torch.ones((batch_size, n_lines), device=device)
    
    for l in range(n_lines):
        masked_indices = mask[:, l]==1
        if masked_indices.sum() > 0:
            brier = BrierScore()
            masked_time = time[masked_indices, l]
            masked_event = event[masked_indices, l].bool()
            masked_survival = survival_at_time_bins[masked_indices, l, :]
            
            brier_score_value = brier(estimate=masked_survival,
                                     event=masked_event,
                                     time=masked_time,
                                     new_time=time_bins)
            
            brier_scores[l, :] = brier_score_value
            integrated_brier_scores[l] = brier.integral() 

    return brier_scores, integrated_brier_scores # type: ignore




# if __name__ == "__main__":
#     from CausalSurv.data.online_data_utils import ESMEOnlineDataModule
#     from CausalSurv.model.DynaSurvCausalOnline import CausalDynaSurv
#     from CausalSurv.tools import load_config


#     data_module = ESMEDataModule(data_dir="../../../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
#                                  config="../../../configs/data.toml")
#     print("DataModule initialized.")
#     data_module.prepare_data()
#     print("Data prepared.")
#     data_module.setup()
#     print("DataModule setup complete.")

#     input_dims = data_module.get_input_dimensions()
#     x_input_dim = input_dims["x_input_dim"]
#     p_input_dim = input_dims["p_input_dim"]
#     output_sa_length = input_dims["output_sa_length"]

#     model_config = load_config("../../../configs/dynasurv.toml")
#     model = CausalDynaSurv(
#         x_input_dim=x_input_dim,
#         p_input_dim=p_input_dim,
#         output_sa_length=output_sa_length,
#         n_treatments=p_input_dim,
#         n_lines=data_module.n_lines,
#         config=model_config,
#     )

#     with torch.no_grad():
#         batch = next(iter(data_module.val_dataloader()))
#         XPd, sa_true, treatment_index, time, event, mask = batch

#         sa_pred, propensity = model.predict_factual_survival(XPd, treatment_index, mask)
        
#         print(brier_score(sa_pred, time, event, data_module.time_bins, mask))
#         print(concordance_index(sa_pred, time, event, data_module.time_bins, mask))
