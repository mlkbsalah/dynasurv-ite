from numbers import Number
from typing import List

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
from sksurv.nonparametric import kaplan_meier_estimator

from CausalSurv.data import ESMEOnlineDataModuleCV
from CausalSurv.model import DynaSurvCausalOnline

from .calibration import compute_calibration


class DynasurvEvaluator:
    def __init__(
        self, model: str | DynaSurvCausalOnline, datamodule: ESMEOnlineDataModuleCV
    ) -> None:
        self._model = model
        self._datamodule = datamodule
        self._test_dataloader = datamodule.test_dataloader()

        self._model.fit_censoring_estimator(self._datamodule.train_dataloader())

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val) -> DynaSurvCausalOnline:
        if isinstance(val, str):
            model = DynaSurvCausalOnline.load_from_checkpoint(
                val, map_location=torch.device("cpu")
            )
        elif isinstance(val, DynaSurvCausalOnline):
            model = val
        return model

    def test_model(self):
        assert isinstance(self._model, DynaSurvCausalOnline)

        trainer = L.Trainer()
        result_dict = trainer.test(self._model, self._datamodule)

        return result_dict

    def brier_score(self, brier_score_tmax: List[float] | float | int, plot=False):
        assert isinstance(self._model, DynaSurvCausalOnline)
        assert self._model.train_events is not None
        assert self._model.train_times is not None
        N_LINES = self._datamodule.n_lines

        if isinstance(brier_score_tmax, list):
            print("TODO:make sur the list is exactly the lenght of the number of lines")
        if isinstance(brier_score_tmax, Number):
            brier_score_tmax = [brier_score_tmax] * N_LINES
        assert isinstance(brier_score_tmax, list)

        (
            XPd,
            X_static,
            interval_idx,
            treatment_indices,
            test_time,
            test_event,
            test_mask,
            patient_id,
        ) = next(iter(self._test_dataloader))
        disc_surv = self._model.predict_discrete_survival(
            XPd, X_static, gather=True, factual_idx=treatment_indices
        )

        table = PrettyTable()
        table.field_names = ["Line", "IBS", "Max BS", "Min BS"]
        table.float_format = ".3"
        bs_val_dict = {}
        ipcw_weights = {}
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=N_LINES, figsize=(20, 3))
        for line in range(N_LINES):
            mask_line = test_mask[:, line] == 1
            e_line_test = test_event[mask_line, line]
            t_line_test = test_time[mask_line, line]

            e_line = self._model.train_events[line]
            t_line = self._model.train_times[line]

            ibs, bs_val, bs_ipcw = self._model.eval_brier_score_ipcw(
                train_events=torch.tensor(e_line, dtype=torch.bool),
                train_times=torch.tensor(t_line),
                test_events=e_line_test.bool(),
                test_times=t_line_test,
                discrete_survival=disc_surv[mask_line, line],
                tmax=brier_score_tmax[line],
            )
            bs_val_dict[line] = bs_val.cpu().numpy()
            ipcw_weights[line] = bs_ipcw

            table.add_row([line, ibs.item(), bs_val.max().item(), bs_val.min().item()])

            if plot:
                ax[line].plot(torch.linspace(0, brier_score_tmax[line], 100), bs_val)
                ax[line].set_xticks(torch.arange(0, brier_score_tmax[line], 6))
        print(table)
        return ibs, bs_val_dict, ipcw_weights

    def line_calibration_error(
        self, eval_time: List[float] | float | int, n_bins: int = 20, plot=False
    ):
        assert isinstance(self._model, DynaSurvCausalOnline)
        N_LINES = self._datamodule.n_lines

        if isinstance(eval_time, Number):
            eval_time = [eval_time]

        eval_time_tn = torch.tensor(eval_time)

        (
            XPd,
            X_static,
            interval_idx,
            treatment_indices,
            time,
            event,
            mask,
            patient_id,
        ) = next(iter(self._test_dataloader))
        disc_surv = self._model.predict_discrete_survival(
            XPd, X_static, gather=True, factual_idx=treatment_indices
        )

        table = PrettyTable()
        table.field_names = ["Line", *[f"t = {time}" for time in eval_time]]
        table.float_format = ".3"

        result_df = []
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=N_LINES, figsize=(5 * N_LINES, 4))
        for line in range(N_LINES):
            row = {"Line": line + 1}
            valid_mask = mask[:, line] == 1
            if not valid_mask.any():
                continue
            pred_surv = self._model.eval_factual_survival(
                disc_surv[:, line, :], eval_time_tn
            )

            for i, t_val in enumerate(eval_time):
                calib_pred, calib_obs, bin_count = compute_calibration(
                    pred_surv[valid_mask, i],
                    time[valid_mask, line].numpy(),
                    event[valid_mask, line].numpy(),
                    t_val,
                    n_bins,
                )

                err = np.sum(
                    bin_count / np.sum(bin_count) * np.abs(calib_obs - calib_pred)
                )
                row[f"t={t_val}"] = err

                ax[line].plot(
                    calib_pred,
                    calib_obs,
                    ".-",
                    label=f"t={t_val}",
                )
                if plot:
                    ax[line].set_xlabel("Predicted Survival")
            if plot:
                ax[line].plot([0, 1], [0, 1], "k--")
                ax[line].set_ylabel("Observed Survival (KM)")
                ax[line].legend()
                ax[line].set_title(f"Calibration Plot - Line {line + 1}")
                ax[line].grid(True, alpha=0.3)

            result_df.append(row)
            table.add_row(row.values())

        result_df = pd.DataFrame(result_df)
        mean_ece = result_df.set_index("Line").mean(axis=1).values
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
        return result_df

    def treatment_calibration_error(
        self, eval_time: List[float] | float | int, n_bins: int = 20, plot=False
    ):
        assert isinstance(self._model, DynaSurvCausalOnline)
        N_LINES = self._datamodule.n_lines

        if isinstance(eval_time, Number):
            eval_time = [eval_time]

        eval_time_tn = torch.tensor(eval_time)

        (
            XPd,
            X_static,
            interval_idx,
            treatment_indices,
            time,
            event,
            mask,
            patient_id,
        ) = next(iter(self._test_dataloader))
        disc_surv = self._model.predict_discrete_survival(
            XPd, X_static, gather=True, factual_idx=treatment_indices
        )

        if plot:
            fig, ax = plt.subplots(
                nrows=len(eval_time),
                ncols=N_LINES,
                figsize=(7 * N_LINES, 6 * len(eval_time)),
            )

        for i, t_eval in enumerate(eval_time):
            table = PrettyTable()
            table.title = f"T = {t_eval} months"
            table.add_column(
                "Treatment", list(self._datamodule.treatment_dict.values())
            )

            for line in range(N_LINES):
                col = []
                valid_mask = mask[:, line] == 1
                if not valid_mask.any():
                    continue

                pred_surv = self._model.eval_factual_survival(
                    disc_surv[:, line, :], eval_time_tn
                )
                t_line = treatment_indices[valid_mask, line]
                pred_surv_line = pred_surv[valid_mask,]
                time_line = time[valid_mask, line]
                event_line = event[valid_mask, line]

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

                        err = np.sum(
                            bin_count
                            / np.sum(bin_count)
                            * np.abs(calib_obs - calib_pred)
                        )
                        col.append(err)
                    else:
                        col.append("*")

                    if plot:
                        ax[i, line].plot(
                            calib_pred,
                            calib_obs,
                            ".-",
                            label=f"{self._datamodule.treatment_dict[treatment_k]}",
                        )

                        ax[i, line].set_xlabel("Predicted Survival")
                        ax[i, line].plot([0, 1], [0, 1], "k--")
                        ax[i, line].set_ylabel("Observed Survival (KM)")
                        ax[i, line].legend()
                        ax[i, line].set_title(f"Calibration Plot - Line {line + 1}")
                        ax[i, line].grid(True, alpha=0.3)
                table.add_column(f"Line {line + 1}", col)
            if plot:
                plt.tight_layout()
            table.float_format = ".3"
            print(table)
        print(
            f"***: Not valid due to very counts (<{self._datamodule.min_samples_per_treatment})"
        )
        if plot:
            plt.show()

        return

    def treatment_calibration_KM(self, save_plots=False):
        assert isinstance(self._model, DynaSurvCausalOnline)
        N_LINES = self._datamodule.n_lines

        (
            XPd,
            X_static,
            interval_idx,
            treatment_indices,
            time,
            event,
            mask,
            patient_id,
        ) = next(iter(self._test_dataloader))

        disc_surv = self._model.predict_discrete_survival(
            XPd, X_static, gather=True, factual_idx=treatment_indices
        )

        for line in range(N_LINES):
            fig, axis = plt.subplots(
                nrows=1,
                ncols=(len(self._datamodule.valid_treatments_per_line[line])),
                figsize=(5 * len(self._datamodule.valid_treatments_per_line[line]), 4),
            )
            valid_mask = mask[:, line] == 1
            if not valid_mask.any():
                continue

            line_surv = disc_surv[valid_mask, line, :]
            t_line = treatment_indices[valid_mask, line]
            time_line = time[valid_mask, line]
            event_line = event[valid_mask, line]

            i = 0
            for treatment_k in self._datamodule.treatment_dict.keys():
                if treatment_k in self._datamodule.valid_treatments_per_line[line]:
                    ax = axis[i]
                    treatment_mask = t_line == treatment_k
                    u_times, surv_probs = kaplan_meier_estimator(
                        event_line[treatment_mask].numpy().astype(bool).squeeze(),
                        time_line[treatment_mask].numpy().squeeze(),
                    )

                    avg_pred_surv = line_surv[treatment_mask].mean(dim=0).cpu().numpy()

                    ax.plot(
                        avg_pred_surv,
                        label=f"Predicted - {self._datamodule.treatment_dict[treatment_k]} (N={treatment_mask.sum()})",
                    )
                    ax.step(
                        u_times,
                        surv_probs,
                        where="post",
                        label=f"{self._datamodule.treatment_dict[treatment_k]}",
                    )
                    ax.set_title(
                        f"Line {line + 1} - Treatment: {self._datamodule.treatment_dict[treatment_k]}"
                    )
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Survival Probability")
                    ax.legend()
                    i += 1
            plt.suptitle(
                f"Predicted Survival vs Kaplan-Meier Estimate - Line {line + 1}"
            )
            plt.tight_layout()
            if save_plots:
                plt.savefig(
                    f"predicted_survival_vs_kaplan_meier_line_{line + 1}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.show()

    def average_km(self):
        assert isinstance(self._model, DynaSurvCausalOnline)
        N_LINES = self._datamodule.n_lines

        (
            XPd,
            X_static,
            interval_idx,
            treatment_indices,
            time,
            event,
            mask,
            patient_id,
        ) = next(iter(self._test_dataloader))

        disc_surv = self._model.predict_discrete_survival(
            XPd, X_static, gather=True, factual_idx=treatment_indices
        )

        figure, axis = plt.subplots(nrows=1, ncols=N_LINES, figsize=(5 * N_LINES, 4))
        for line in range(N_LINES):
            ax = axis[line]
            valid_mask = mask[:, line] == 1
            if not valid_mask.any():
                continue

            line_surv = disc_surv[valid_mask, line, :]
            time_line = time[valid_mask, line]
            event_line = event[valid_mask, line]

            u_times, surv_probs = kaplan_meier_estimator(
                event_line.numpy().astype(bool).squeeze(),
                time_line.numpy().squeeze(),
            )

            avg_pred_surv = line_surv.mean(dim=0).cpu().numpy()
            ax.plot(avg_pred_surv, label=f"Predicted - Line {line + 1}")
            ax.step(u_times, surv_probs, where="post", label="Kaplan-Meier")
            ax.set_title(f"Line {line + 1}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Survival Probability")
            ax.legend()
        plt.suptitle("Average Predicted Survival vs Kaplan-Meier Estimate")
        plt.tight_layout()
        plt.show()
