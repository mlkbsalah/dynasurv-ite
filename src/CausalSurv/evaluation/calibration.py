import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator


def compute_calibration(
    pred_surv, obs_times, obs_events, eval_time, n_bins=10, bin_min_samples=10
):
    pred_surv = pred_surv.numpy()

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(pred_surv, bins) - 1  # (batch, n_lines,)

    calib_pred = []
    calib_obs = []
    bin_count = []
    for i in range(n_bins):
        idx = bin_indices == i
        if idx.sum() < bin_min_samples:
            continue
        agg_pred = np.mean(pred_surv[idx])
        time, survival_prob = kaplan_meier_estimator(  # type:ignore
            obs_events.squeeze()[idx].astype(bool),
            obs_times.squeeze()[idx],
        )
        if eval_time <= time[0]:
            est = 1.0
        elif eval_time >= time[-1]:
            est = survival_prob[-1]
        else:
            est = survival_prob[time <= eval_time][-1]

        calib_obs.append(est)
        calib_pred.append(agg_pred)
        bin_count.append(idx.sum())

    return np.array(calib_pred), np.array(calib_obs), np.array(bin_count)
