from typing import List, Union

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable

from CausalSurv.data.datamodule_progression import ESMEProgressionOnlineDataModuleCV
from CausalSurv.model.dynasurv_causal_online_multihead import (
    DynaSurvCausalOnlineMultiheadProgression,
)

from .calibration import compute_calibration

TimeArg = Union[List[float], float, int]


def _to_time_list(t: TimeArg, repeat: int = 1) -> List[float]:
    if isinstance(t, (int, float)):
        return [float(t)] * repeat
    return [float(v) for v in t]


class DynasurvMultiheadEvaluator:
    def __init__(
        self,
        model: Union[str, DynaSurvCausalOnlineMultiheadProgression],
        datamodule: ESMEProgressionOnlineDataModuleCV,
    ) -> None:
        if isinstance(model, str):
            model = DynaSurvCausalOnlineMultiheadProgression.load_from_checkpoint(
                model, map_location=torch.device("cpu")
            )
        self._model = model
        self._datamodule = datamodule
        self._test_dataloader = datamodule.test_dataloader()

        self._model.fit_censoring_estimator(self._datamodule.train_dataloader())

    @property
    def model(self) -> DynaSurvCausalOnlineMultiheadProgression:
        return self._model

    def test_model(self):
        trainer = L.Trainer()
        return trainer.test(self._model, self._datamodule)

    def brier_score(self, brier_score_tmax: TimeArg, plot: bool = False):
        assert self._model.train_events is not None
        assert self._model.train_times is not None
        N_LINES = self._datamodule.n_lines
        tmax_list = _to_time_list(brier_score_tmax, repeat=N_LINES)

        (
            XPd,
            X_static,
            _,
            _,
            treatment_indices,
            death_time,
            death_event,
            _,
            _,
            test_mask,
            _,
        ) = next(iter(self._test_dataloader))

        disc_surv = self._model.predict_discrete_survival(
            XPd, X_static, treatment_idx=treatment_indices, gather=True
        )

        table = PrettyTable()
        table.field_names = ["Line", "IBS", "Max BS", "Min BS"]
        table.float_format = ".3"
        bs_val_dict: dict = {}
        ipcw_weights: dict = {}

        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=N_LINES, figsize=(20, 3))

        for line in range(N_LINES):
            mask_line = test_mask[:, line] == 1

            ibs, bs_val, bs_ipcw = self._model.eval_brier_score_ipcw(
                train_events=torch.tensor(
                    self._model.train_events[line], dtype=torch.bool
                ),
                train_times=torch.tensor(
                    self._model.train_times[line], dtype=torch.float32
                ),
                test_events=death_event[mask_line, line].bool(),
                test_times=death_time[mask_line, line],
                discrete_survival=disc_surv[mask_line, line],
                tmax=tmax_list[line],
                line_idx=line,
            )
            bs_val_dict[line] = bs_val.cpu().numpy()
            ipcw_weights[line] = bs_ipcw

            table.add_row(
                [line + 1, ibs.item(), bs_val.max().item(), bs_val.min().item()]
            )

            if plot:
                xs = torch.linspace(
                    0, tmax_list[line], self._model.brier_integration_step
                )
                ax[line].plot(xs.numpy(), bs_val.cpu().numpy())
                ax[line].set_title(f"Line {line + 1}")
                ax[line].set_xlabel("Time")
                ax[line].set_ylabel("Brier Score")

        print(table)
        if plot:
            plt.tight_layout()
            plt.show()
        return ibs, bs_val_dict, ipcw_weights

    def line_calibration_error(
        self, eval_time: TimeArg, n_bins: int = 20, plot: bool = False
    ):
        N_LINES = self._datamodule.n_lines
        eval_times = _to_time_list(eval_time)
        eval_time_tn = torch.tensor(eval_times, dtype=torch.float32)

        (
            XPd,
            X_static,
            _,
            _,
            treatment_indices,
            death_time,
            death_event,
            _,
            _,
            mask,
            _,
        ) = next(iter(self._test_dataloader))

        disc_surv = self._model.predict_discrete_survival(
            XPd, X_static, treatment_idx=treatment_indices, gather=True
        )

        table = PrettyTable()
        table.field_names = ["Line", *[f"t = {t}" for t in eval_times]]
        table.float_format = ".3"

        result_df = []
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=N_LINES, figsize=(5 * N_LINES, 4))

        for line in range(N_LINES):
            row: dict = {"Line": line + 1}
            valid_mask = mask[:, line] == 1
            if not valid_mask.any():
                continue

            pred_surv = self._model.eval_factual_survival(
                disc_surv[:, line, :], eval_time_tn, line_idx=line
            )

            for i, t_val in enumerate(eval_times):
                calib_pred, calib_obs, bin_count = compute_calibration(
                    pred_surv[valid_mask, i],
                    death_time[valid_mask, line].numpy(),
                    death_event[valid_mask, line].numpy(),
                    t_val,
                    n_bins,
                )
                err = float(
                    np.sum(
                        bin_count / np.sum(bin_count) * np.abs(calib_obs - calib_pred)
                    )
                )
                row[f"t={t_val}"] = err

                if plot:
                    ax[line].plot(calib_pred, calib_obs, ".-", label=f"t={t_val}")

            if plot:
                ax[line].plot([0, 1], [0, 1], "k--")
                ax[line].set_xlabel("Predicted Survival")
                ax[line].set_ylabel("Observed Survival (KM)")
                ax[line].legend()
                ax[line].set_title(f"Calibration Plot - Line {line + 1}")
                ax[line].grid(True, alpha=0.3)

            result_df.append(row)
            table.add_row(list(row.values()))

        result_df_out = pd.DataFrame(result_df)
        mean_ece = result_df_out.set_index("Line").mean(axis=1).values
        sort_idx = np.argsort(mean_ece)
        table.add_column(
            "Mean",
            [
                f"{mean_ece[i]:.3}({'*' * (np.where(sort_idx == i)[0][0] + 1)})"
                for i in range(len(mean_ece))
            ],
        )
        table.float_format = ".3"
        print(table)
        if plot:
            plt.tight_layout()
            plt.show()
        return result_df_out

    def treatment_calibration_error(
        self, eval_time: TimeArg, n_bins: int = 20, plot: bool = False
    ):
        N_LINES = self._datamodule.n_lines
        eval_times = _to_time_list(eval_time)
        eval_time_tn = torch.tensor(eval_times, dtype=torch.float32)

        (
            XPd,
            X_static,
            _,
            _,
            treatment_indices,
            death_time,
            death_event,
            _,
            _,
            mask,
            _,
        ) = next(iter(self._test_dataloader))

        disc_surv = self._model.predict_discrete_survival(
            XPd, X_static, treatment_idx=treatment_indices, gather=True
        )

        if plot:
            fig, ax = plt.subplots(
                nrows=len(eval_times),
                ncols=N_LINES,
                figsize=(7 * N_LINES, 6 * len(eval_times)),
            )

        for i, t_eval in enumerate(eval_times):
            table = PrettyTable()
            table.title = f"T = {t_eval} months"
            table.add_column(
                "Treatment", list(self._datamodule.treatment_dict.values())
            )

            for line in range(N_LINES):
                col: list = []
                valid_mask = mask[:, line] == 1
                if not valid_mask.any():
                    continue

                pred_surv = self._model.eval_factual_survival(
                    disc_surv[:, line, :], eval_time_tn, line_idx=line
                )
                t_line = treatment_indices[valid_mask, line]
                pred_surv_line = pred_surv[valid_mask]
                time_line = death_time[valid_mask, line]
                event_line = death_event[valid_mask, line]

                for treatment_k in self._datamodule.treatment_dict.keys():
                    if treatment_k in self._datamodule.valid_treatments_per_line[line]:
                        treatment_mask = t_line == treatment_k
                        calib_pred, calib_obs, bin_count = compute_calibration(
                            pred_surv_line[treatment_mask, i],
                            time_line[treatment_mask].numpy(),
                            event_line[treatment_mask].numpy(),
                            t_eval,
                            n_bins,
                        )
                        err = float(
                            np.sum(
                                bin_count
                                / np.sum(bin_count)
                                * np.abs(calib_obs - calib_pred)
                            )
                        )
                        col.append(err)

                        if plot:
                            ax[i, line].plot(
                                calib_pred,
                                calib_obs,
                                ".-",
                                label=f"{self._datamodule.treatment_dict[treatment_k]}",
                            )
                    else:
                        col.append("*")

                if plot:
                    ax[i, line].plot([0, 1], [0, 1], "k--")
                    ax[i, line].set_xlabel("Predicted Survival")
                    ax[i, line].set_ylabel("Observed Survival (KM)")
                    ax[i, line].legend()
                    ax[i, line].set_title(f"Calibration Plot - Line {line + 1}")
                    ax[i, line].grid(True, alpha=0.3)

                table.add_column(f"Line {line + 1}", col)

            table.float_format = ".3"
            print(table)

        print(
            f"*: Not valid due to low counts (<{self._datamodule.min_samples_per_treatment})"
        )
        if plot:
            plt.tight_layout()
            plt.show()
