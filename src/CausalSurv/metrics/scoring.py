import torch
import lightning as L
import numpy as np
from torchsurv.metrics.cindex import ConcordanceIndex

def conditionalsurvival_to_cumulativesurvival(sa_pred: torch.Tensor) -> torch.Tensor:
    """
    Convert conditional survival probabilities to cumulative survival probabilities.

    Args:
        sa_pred: (batch, n_lines, n_intervals)
                 First half = conditional survival probabilities, second half = optional extra signals.

    Returns:
        surv_probs: (batch, n_lines, n_intervals)
    """
    surv_probs = torch.cumprod(sa_pred, dim=-1)
    return surv_probs

def kaplan_meier(time: torch.Tensor, event: torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
    """
    Kaplan-Meier estimator evaluated at specified time points.

    Args:
        time: (batch, n_lines) observed times (uncensored + censored)
        event: (batch, n_lines) event indicators (1=event, 0=censored)
        time_grid: (n_time_points,) or (n_time_points, n_lines) time points to evaluate survival

    Returns:
        S: (n_time_points, n_lines) survival estimates at each time in time_grid
    """

    n_lines = time.shape[1]

    if time_grid.ndim == 1:
        time_grid = time_grid[:, None].repeat(1, n_lines)

    n_time_points = time_grid.shape[0]
    S = torch.ones(n_time_points, n_lines, device=time.device)

    # Loop over lines
    for l in range(n_lines):
        t_sorted, idx = torch.sort(time[:,l])
        e_sorted = event[:,l][idx]

        surv = 1.0
        for i, t in enumerate(time_grid[:, l]):
            at_risk = (t_sorted >= t).float()
            n_risk = at_risk.sum()
            n_event = ((t_sorted == t) & (e_sorted == 1)).sum()
            frac = 1.0 - (n_event / torch.clamp(n_risk, min=1.0))
            surv *= frac
            S[i, l] = surv

    return S

def brier_score(sa_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor, time_bins: torch.Tensor) -> torch.Tensor:
    """
    Right-censored Brier score with IPCW correction.

    Args:
        sa_pred: (batch, n_lines, n_intervals) predicted conditional survival probabilities
        time: (batch, n_lines) observed times
        event: (batch, n_lines) event indicators (1=event, 0=censored)
        time_bins: (n_intervals + 1,) bin cutpoints

    Returns:
        brier: (n_intervals, n_lines) Brier score at each bin
    """
    device = sa_pred.device

    # (1) Convert conditional → cumulative survival
    surv_probs = torch.cumprod(sa_pred, dim=-1)  # (batch, n_lines, n_intervals)

    n_intervals = len(time_bins) - 1
    n_lines = time.shape[1]
    brier = torch.zeros(n_intervals, n_lines, device=device)

    # (2) Censoring survival G(t)
    G_full = kaplan_meier(time, 1 - event, time_bins)  # (n_intervals+1, n_lines)
    G_full = torch.clamp(G_full, min=1e-6)  # avoid 0

    # (3) Interpolate G at observed times
    # torch.interp wants 1D x, y, xp, so we vectorize manually
    G_at_events = torch.zeros_like(time, device=device)
    for l in range(n_lines):
        G_at_events[:, l] = torch.from_numpy(np.interp(
            time[:, l].cpu().numpy(),
            time_bins.cpu().numpy(),
            G_full[:, l].cpu().numpy()
        ))
    G_at_events = torch.clamp(G_at_events, min=1e-6)

    # (4) Loop over evaluation times
    for j in range(n_intervals):
        t_j = time_bins[j + 1]
        surv_j = surv_probs[:, :, j]  # (batch, n_lines)
        G_tj = G_full[j + 1, :]       # (n_lines,)

        # Case 1: event occurred before t_j
        case1_mask = (time <= t_j) & (event == 1)
        w1 = 1.0 / G_at_events
        case1 = ((surv_j - 0.0) ** 2) * case1_mask * w1

        # Case 2: subject survived beyond t_j
        case2_mask = (time > t_j)
        w2 = 1.0 / G_tj
        case2 = ((surv_j - 1.0) ** 2) * case2_mask * w2

        # Combine and normalize by valid cases
        valid_mask = case1_mask | case2_mask
        weighted_sum = torch.sum(case1 + case2, dim=0)
        denom = torch.clamp(valid_mask.sum(dim=0), min=1)
        brier[j, :] = weighted_sum / denom

    return brier

def integrated_brier_score(sa_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor, time_bins: torch.Tensor) -> torch.Tensor:
    """
    Integrated Brier Score (IBS).

    Args:
        sa_pred: (batch, n_lines, n_treatments, n_intervals) model outputs
        time: (batch, n_lines)
        event: (batch, n_lines)
        time_bins: (n_intervals + 1,)

    Returns:
        torch.Tensor (n_lines, n_treatments): IBS
    """
    brier = brier_score(sa_pred, time, event, time_bins)  # (n_interval, n_lines)
    dt = time_bins[1:] - time_bins[:-1]  # (n_intervals,)
    ibs = torch.zeros(brier.shape[1], device=sa_pred.device)  # (n_lines,)
    for i in range(brier.shape[1]):
        ibs[i] = torch.sum(brier[:, i] * dt, dim=0) / (time_bins[-1] - time_bins[0])
    return ibs

def concordance_index(sa_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor, time_bins: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        sa_pred (torch.Tensor): (batch, n_lines, n_intervals)
        time (torch.Tensor): (batch, n_lines)
        event (torch.Tensor): (batch, n_lines)
        time_bins (torch.Tensor): (n_intervals+1,)
        mask (torch.Tensor): (batch, n_lines) boolean tensor indicating valid entries (padding mask)

    Returns:
        torch.Tensor: (n_lines,)
    """
    device = sa_pred.device
    batch, n_lines = sa_pred.shape[:2]
    c_index = ConcordanceIndex()
    c_index_values = torch.zeros(n_lines, device=device)
    survival_at_event_times = torch.zeros((batch, n_lines), device=device)

    for l in range(n_lines):
        for i in range(batch):
            t_i = time[i, l].item()
            if t_i >= time_bins[0] and t_i <= time_bins[-1]:
                bin_idx = torch.searchsorted(time_bins, t_i, right=True) - 1
                bin_idx = torch.clamp(bin_idx, 0, sa_pred.shape[2]-1)
                surv_prob = torch.prod(sa_pred[i, l, :bin_idx+1])
                survival_at_event_times[i, l] = surv_prob
            else:
                survival_at_event_times[i, l] = 0.0


    for i in range(n_lines):
        c_index_value = c_index(estimate=survival_at_event_times[:, i],
                                event=event[:, i],
                                time=time[:, i]
        )
        c_index_values[i] = c_index_value

    return c_index_values


if __name__ == "__main__":
    from CausalSurv.data.data_utils import ESMEDataModule
    from CausalSurv.model.DynaSurvCausal import CausalDynaSurv
    from CausalSurv.data.config_loader import load_config


    data_module = ESMEDataModule(data_dir="../../../data/model_entry_imputed_data_HER2+_stable_types_categorized.parquet",
                                 config="../../../configs/data.toml")
    print("DataModule initialized.")
    data_module.prepare_data()
    print("Data prepared.")
    data_module.setup()
    print("DataModule setup complete.")
    train_loader = data_module.train_dataloader()

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
        for batch in train_loader:
            XPd, sa_true, treatment_index, time, event = batch
            print(XPd.shape, sa_true.shape, time.shape, event.shape)
            sa_pred, propensity = model(XPd)
            factual_sa_pred = model._select_factual_head(sa_pred, treatment_index)
            print(factual_sa_pred.shape)

            print(brier_score(factual_sa_pred, time, event, data_module.time_bins).shape)
            print(integrated_brier_score(factual_sa_pred, time, event, data_module.time_bins).shape)
            # print(concordance_index(sa_pred, time, event, data_module.time_bins))

            break
