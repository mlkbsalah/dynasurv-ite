from typing import Tuple

import lightning as L
import numpy as np
import torch
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw

from CausalSurv.metrics.loss import NLLogisticHazard
from CausalSurv.model.embedding_C_LSTM_ITE import embed_LSTM_ITE
from CausalSurv.model.mlp import MLP


class DynaSurvCausalOnline(L.LightningModule):
    """Multi-treatment causal survival model with separate heads.

    Args:
        x_input_dim: Number of X features per timestep.
        p_input_dim: Number of P (patient / static / treatment history) features per timestep.
        output_sa_length: Number of conditional survival intervals (n_intervals).
        n_treatments: Number of distinct treatment heads to model.
        config: Dict or TOML path for model configuration.
    """

    def __init__(
        self,
        x_input_dim: int,
        x_static_dim: int,
        p_input_dim: int,
        p_static_dim: int,
        n_treatments: int,
        output_length: int,
        interval_bounds: torch.Tensor,
        lstm_hidden_length: int,
        x_embed_dim: int,
        p_embed_dim: int,
        init_h_hidden: list[int],
        init_p_hidden: list[int],
        mlpx_hidden_units: list[int],
        mlpp_hidden_units: list[int],
        mlpsa_hidden_units: list[int],
        mlpprop_hidden_units: list[int],
        init_h_dropout: float,
        init_p_dropout: float,
        mlpx_dropout: float,
        mlpp_dropout: float,
        mlpsa_dropout: float,
        mlpprop_dropout: float,
        lambda_prop_loss: float,
        lr: float,
        lr_scheduler_stepsize: int,
        lr_scheduler_gamma: float,
        weight_decay: float,
        attention: bool,
        evaluation_horizon_times: list[float] = [100, 75, 50, 30],
        brier_integration_step: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.x_input_dim = x_input_dim
        self.x_static_dim = x_static_dim
        self.p_input_dim = p_input_dim
        self.p_static_dim = p_static_dim
        self.n_treatments = n_treatments
        self.output_length = output_length

        self.register_buffer("interval_bounds", interval_bounds)
        self.interval_bounds: torch.Tensor

        self.evaluation_horizon_times = evaluation_horizon_times
        self.brier_integration_step = brier_integration_step

        self.lstm = embed_LSTM_ITE(
            x_input_dim=self.x_input_dim,
            p_input_dim=self.p_input_dim,
            output_length=self.output_length,
            hidden_length=lstm_hidden_length,
            x_embed_dim=x_embed_dim,
            p_embed_dim=p_embed_dim,
            mlpx_hidden_units=mlpx_hidden_units,
            mlpp_hidden_units=mlpp_hidden_units,
            mlpsa_hidden_units=mlpsa_hidden_units,
            mlpx_dropout=mlpx_dropout,
            mlpp_dropout=mlpp_dropout,
            mlpsa_dropout=mlpsa_dropout,
            attention=attention,
        )

        self.init_h_mlp = MLP(
            input_dim=x_static_dim,
            n_units=init_h_hidden,
            output_dim=self.lstm.hidden_length,
            dropout=init_h_dropout,
        )
        self.init_p_mlp = MLP(
            input_dim=p_static_dim,
            n_units=init_p_hidden,
            output_dim=self.lstm.p_embed_dim,
            dropout=init_p_dropout,
        )

        self.treatment_head = MLP(
            input_dim=self.lstm.hidden_length,
            output_dim=output_length * n_treatments,
            n_units=mlpsa_hidden_units,
            dropout=mlpsa_dropout,
        )

        self.propensityhead = MLP(
            input_dim=self.lstm.hidden_length,
            output_dim=self.n_treatments,
            n_units=mlpprop_hidden_units,
            dropout=mlpprop_dropout,
        )

        self.surv_loss_fn = NLLogisticHazard(reduction="none")
        self.propensity_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        # Optimizer parameters
        self.lambda_prop_loss = lambda_prop_loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_stepsize = lr_scheduler_stepsize
        self.lr_scheduler_gamma = lr_scheduler_gamma

        # IPCW buffer
        self.train_times = None
        self.train_events = None

    # ====================== Core model logic =============================
    def forward(
        self, XPd: torch.Tensor, X_static: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward pass to retreive all hazards and propensity scores for all treatments and all time steps

        Args:
            XPd: (batch, n_lines, features = x_input_dim + p_input_dim + 1)
            X_static: ( (batch, x_static_dim), (batch, p_static_dim) )


        Returns:
            cond_sa: (batch, n_lines, n_treatments, output_sa_length)
        """
        h, c, p = self._init_lstm_states(X_static, XPd.device)
        hazards_logit = []
        propensity = []

        for t in range(XPd.shape[1]):
            all_responses_t, propensity_t, (h, c, p) = self._forward_one_step(
                XPd[:, t, :], (h, c, p)
            )
            hazards_logit.append(
                all_responses_t
            )  # list of (batch, n_treatments, n_intervals)
            propensity.append(propensity_t)  # list of (batch, n_treatments)
        hazards_logit = torch.stack(
            hazards_logit, dim=1
        )  # (batch, n_lines, n_treatments, n_intervals)
        propensity = torch.stack(propensity, dim=1)  # (batch, n_lines, n_treatments)

        return hazards_logit, propensity

    def _init_lstm_states(
        self, X_static, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden, cell, and treatment embedding states."""
        x_static_general, x_static_treatment = X_static
        batch_size = x_static_general.shape[0]
        h0 = self.init_h_mlp(x_static_general)
        c0 = torch.zeros(batch_size, self.lstm.hidden_length, device=device)
        p0 = self.init_p_mlp(x_static_treatment)
        return h0, c0, p0

    def _forward_one_step(
        self, XPd_t, tuple_in
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """Forward pass for one time step."""
        _, h, c, p = self.lstm(XPd_t, tuple_in)
        propensity_t = torch.softmax(
            self.propensityhead(h), dim=-1
        )  # (batch, n_treatments)
        all_response_t = self.treatment_head(h).view(
            -1, self.n_treatments, self.output_length
        )  # (batch, n_treatments, n_intervals)

        return all_response_t, propensity_t, (h, c, p)

    def get_factual_predictions(
        self,
        XPd: torch.Tensor,
        X_static: Tuple[torch.Tensor, torch.Tensor],
        treatment_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the factual hazard predictions and propensity scores based on the treatment actually received.
        Args:
            XPd (torch.Tensor): (batch, n_lines, features)
            X_static (Tuple[torch.Tensor, torch.Tensor]): ( (batch, x_static_dim), (batch, p_static_dim) )
            treatment_idx (torch.Tensor): (batch, n_lines) integer treatment head indices actually received
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (factual_hazards_logit, propensity)
        """

        hazards_logit, propensity = self.forward(
            XPd, X_static
        )  # (batch, n_lines, n_treatments, n_intervals) / (batch, n_lines, n_treatments)
        gather_idx = (
            treatment_idx.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 1, hazards_logit.shape[-1])
        )  # (batch, n_lines, 1, n_intervals)
        factual_hazards_logit = torch.gather(
            hazards_logit, dim=2, index=gather_idx
        ).squeeze(2)  # (batch, n_lines, n_intervals)
        return factual_hazards_logit, propensity

    # ====================== Training and evaluation ======================
    def training_step(self, batch, batch_idx):
        """perform a training step"""

        XPd, X_static, interval_idx, treatment_idx, time, event, mask, patient_id = (
            batch
        )

        if self.current_epoch == 0:
            self._accumulate_data(time, event, mask)

        loss, surv_loss, prop_loss = self._compute_loss(
            XPd, X_static, treatment_idx, interval_idx, event, mask
        )

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train/survival_loss", surv_loss, prog_bar=True, on_step=True, on_epoch=True
        )
        self.log(
            "train/propensity_loss",
            prop_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """perform a validation step"""
        XPd, X_static, interval_idx, treatment_idx, time, event, mask, patient_id = (
            batch
        )
        N_lines = time.shape[1]

        loss, surv_loss, prop_loss = self._compute_loss(
            XPd, X_static, treatment_idx, interval_idx, event, mask
        )

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/survival_loss", surv_loss, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(
            "val/propensity_loss",
            prop_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        if self.trainer.sanity_checking:
            return loss

        discrete_hazards = torch.sigmoid(
            self.get_factual_predictions(XPd, X_static, treatment_idx)[0]
        )  # (batch, n_lines, n_intervals)

        discrete_survival = torch.cumprod(1 - discrete_hazards, dim=2)
        discrete_survival = torch.cat(
            [torch.ones_like(discrete_survival[:, :, :1]), discrete_survival], dim=2
        )  # (batch, n_lines, n_intervals + 1) to account for S(0)=1

        discrete_cumhazards = torch.cumsum(discrete_hazards, dim=2)
        discrete_cumhazards = torch.cat(
            [torch.zeros_like(discrete_cumhazards[:, :, :1]), discrete_cumhazards],
            dim=2,
        )  # (batch, n_lines, n_intervals + 1) to account for H(0)=0

        ci = []
        ibs = []
        for line in range(N_lines):
            valid_mask = mask[:, line].bool()
            if not valid_mask.any():
                continue

            t_line = time[valid_mask, line]  # (valid_batch, )
            e_line = event[valid_mask, line].bool()  # (valid_batch, )
            line_discrete_survival = discrete_survival[
                valid_mask, line, :
            ]  # (valid_batch, n_intervals + 1)
            line_discrete_cumhazards = discrete_cumhazards[
                valid_mask, line, :
            ]  # (valid_batch, n_intervals + 1)

            if self.train_events is None or self.train_times is None:
                raise ValueError(
                    "IPCW weights cannot be computed before training epoch 0 is completed."
                )

            c_index_td, _ = self.eval_cindex_ipcw(
                train_events=torch.tensor(
                    self.train_events[line], dtype=torch.bool, device="cpu"
                ),
                train_times=torch.tensor(
                    self.train_times[line], dtype=torch.float32, device="cpu"
                ),
                test_events=e_line.cpu(),
                test_times=t_line.cpu(),
                discrete_cumhazards=line_discrete_cumhazards.cpu(),
                device="cpu",
            )
            ci.append(c_index_td)

            ibs_line, bs_val_line, bs_ipcw_weights = self.eval_brier_score_ipcw(
                train_events=torch.tensor(
                    self.train_events[line], dtype=torch.bool, device="cpu"
                ),
                train_times=torch.tensor(
                    self.train_times[line], dtype=torch.float32, device="cpu"
                ),
                test_events=e_line.cpu(),
                test_times=t_line.cpu(),
                discrete_survival=line_discrete_survival.cpu(),
                tmax=self.evaluation_horizon_times[line],
                device="cpu",
            )
            ibs.append(ibs_line)

            self.log(
                f"val/ci_time_step_{line + 1}",
                c_index_td,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"val/ibs_time_step_{line + 1}",
                ibs_line,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        self.log(
            "average_ci",
            float(np.mean(ci)),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "average_ibs",
            float(np.mean(ibs)),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss, np.mean(ci), np.mean(ibs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def _compute_loss(
        self, XPd, X_static, treatment_idx, interval_idx, event, mask
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        haz_pred_logit, propensity = self.get_factual_predictions(
            XPd, X_static, treatment_idx
        )  # (batch, n_lines, n_intervals) / (batch, n_lines, n_treatments)

        batch, n_lines, n_intervals = haz_pred_logit.shape
        haz_pred_logit_flat = haz_pred_logit.reshape(-1, n_intervals)
        interval_idx_flat = interval_idx.reshape(-1)
        event_flat = event.reshape(-1)

        surv_loss_flat = self.surv_loss_fn(
            haz_pred_logit_flat, interval_idx_flat, event_flat
        )  # (batch * n_lines, )
        surv_loss = surv_loss_flat.reshape(batch, n_lines) * mask

        prop_loss_flat = self.propensity_loss_fn(
            propensity.reshape(-1, self.n_treatments), treatment_idx.reshape(-1)
        )
        prop_loss = prop_loss_flat.reshape(batch, n_lines) * mask

        prop_loss = prop_loss.sum() / mask.sum()
        surv_loss = surv_loss.sum() / mask.sum()

        loss = surv_loss - self.lambda_prop_loss * prop_loss
        return loss, surv_loss, prop_loss

    def _accumulate_data(self, time, event, mask):
        """Collect censoring information for IPCW estimation during epoch 0."""
        if (self.train_times is None) or (self.train_events is None):
            n_lines = time.shape[1]
            self.train_times = [[] for _ in range(n_lines)]
            self.train_events = [[] for _ in range(n_lines)]

        for line in range(time.shape[1]):
            valid = mask[:, line].bool()
            if not valid.any():
                continue

            t = time[valid, line]
            e = event[valid, line]

            self.train_events[line].extend(e.tolist())
            self.train_times[line].extend(t.tolist())

    def eval_cindex_ipcw(
        self,
        train_events,
        train_times,
        test_events,
        test_times,
        discrete_cumhazards,
        device,
    ):
        cumhazards = self.eval_factual_cumhazard(
            discrete_cumhazards, test_times.squeeze(), device
        )  # (valid_batch, n_intervals + 1)

        ci_ipcw_weights = get_ipcw(
            event=train_events.squeeze(),
            time=train_times.squeeze(),
            new_time=test_times.squeeze(),
        )

        ci_fun = ConcordanceIndex()
        c_index = ci_fun(
            estimate=cumhazards,
            event=test_events,
            time=test_times,
            weight=ci_ipcw_weights,
        )

        return c_index.item(), ci_ipcw_weights

    def eval_brier_score_ipcw(
        self,
        train_events,
        train_times,
        test_events,
        test_times,
        discrete_survival,
        tmax,
        device,
    ):
        bs_eval_times = torch.linspace(
            0,
            tmax,
            steps=self.brier_integration_step,
            dtype=torch.float32,
            device=device,
        )
        survival_prob = self.eval_factual_survival(
            discrete_survival, bs_eval_times, device
        )  # (valid_batch, n_intervals + 1)

        bs_ipcw_weights = get_ipcw(
            event=train_events.squeeze(),
            time=train_times.squeeze(),
            new_time=bs_eval_times,
        )

        bs_fun = BrierScore()
        bs_val = bs_fun(
            estimate=survival_prob,
            event=test_events,
            time=test_times,
            new_time=bs_eval_times,
            weight_new_time=bs_ipcw_weights,
        )
        ibs = bs_fun.integral()

        return ibs, bs_val, bs_ipcw_weights

    # ====================== Inference methods ============================
    def predict(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple of (hazards, survivals) for all treatments and time steps.

        Args:
            XPd: (batch, n_lines, features)
            X_static: ( (batch, x_static_dim), (batch, p_static_dim) )
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                hazards: (batch, n_lines, n_treatments, n_intervals)
                survivals: (batch, n_lines, n_treatments, n_intervals)
        """
        return (torch.zeros(1), torch.zeros(1))  # Placeholder

    def eval_factual_cumhazard(
        self,
        discrete_cumhazards: torch.Tensor,
        eval_time: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        """return cumulative hazard at eval time for all batch and all lines

        Args:
            discrete_cumhazards (torch.Tensor): tensor of shape (batch, n_intervals+1) cumulative hazards
            eval_time (torch.Tensor): tensor of shape (n_eval_points,) evaluation times

        Returns:
            torch.Tensor: (batch, n_eval_points) cumulative hazards at eval_time for all batch and all lines
        """
        interval_bounds = self.interval_bounds.to(device)
        eval_time = eval_time.to(device)
        discrete_cumhazards = discrete_cumhazards.to(device)

        interval_idx = (
            torch.bucketize(eval_time, interval_bounds, right=True) - 1
        )  # (n_eval_points,)
        if torch.any(interval_idx < 0) or torch.any(interval_idx >= self.output_length):
            print(
                "Warning: eval_time is outside the range of interval_bounds. Clamping to valid range."
            )
            interval_idx = torch.clamp(interval_idx, min=0, max=self.output_length - 1)

        batch_size = discrete_cumhazards.shape[0]
        gather_idx = interval_idx.unsqueeze(0).expand(
            batch_size, -1
        )  # (batch, n_eval_points)
        hazards = torch.gather(
            discrete_cumhazards, dim=1, index=gather_idx
        )  # (batch, n_eval_points)
        return hazards

    def eval_factual_survival(
        self,
        discrete_survival: torch.Tensor,
        eval_time: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        """evaluate `S(t|X,P,A)` at eval_time for all batch and all lines

        Args:
            discrete_survival (torch.Tensor): tensor of shape (batch, n_intervals+1) survival probabilities
            eval_time (torch.Tensor): tensor of shape (n_eval_points,) evaluation times

        Returns:
            torch.Tensor: (batch, n_lines, n_eval_points) survival probabilities at eval_time for all batch and all lines
        """
        interval_bounds = self.interval_bounds.to(device)
        discrete_survival = discrete_survival.to(device)
        eval_time = eval_time.to(device)

        interval_idx = (
            torch.bucketize(eval_time, interval_bounds, right=True) - 1
        )  # (n_eval_points,)
        if torch.any(interval_idx < 0) or torch.any(interval_idx >= self.output_length):
            print(
                "Warning: eval_time is outside the range of interval_bounds. Clamping to valid range."
            )
            interval_idx = torch.clamp(interval_idx, min=0, max=self.output_length - 1)

        batch_size = discrete_survival.shape[0]
        gather_idx = interval_idx.unsqueeze(0).expand(
            batch_size, -1
        )  # (batch, n_eval_points)
        survival = torch.gather(
            discrete_survival, dim=1, index=gather_idx
        )  # (batch, n_eval_points)
        return survival

    # ====================== Optimizer configuration ======================
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_scheduler_stepsize,
            gamma=self.lr_scheduler_gamma,
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    batch_size = 99
    times_steps = 4
    x_input_dim = 10
    p_input_dim = 5
    output_sa_length = 77

    x_static_dim = 8
    p_static_dim = 6

    features = x_input_dim + p_input_dim + 1
    sample_XPd = torch.randn(size=(batch_size, times_steps, features))
    sample_X_static_general = torch.randn(size=(batch_size, x_static_dim))
    sample_X_static_treatment = torch.randn(size=(batch_size, p_static_dim))
    sample_X_static = (sample_X_static_general, sample_X_static_treatment)

    treatment_idx = torch.randint(0, p_input_dim, (batch_size, times_steps))

    model = DynaSurvCausalOnline(
        x_input_dim=x_input_dim,
        p_input_dim=p_input_dim,
        n_treatments=p_input_dim,
        x_static_dim=x_static_dim,
        p_static_dim=p_static_dim,
        output_length=output_sa_length,
        interval_bounds=torch.linspace(0, 100, steps=output_sa_length + 1),
        lstm_hidden_length=32,
        x_embed_dim=16,
        p_embed_dim=16,
        init_h_hidden=[32, 16],
        init_p_hidden=[32, 16],
        mlpx_hidden_units=[32, 16],
        mlpp_hidden_units=[32, 16],
        mlpsa_hidden_units=[32, 16],
        mlpprop_hidden_units=[32, 16],
        init_h_dropout=0.1,
        init_p_dropout=0.1,
        mlpx_dropout=0.1,
        mlpp_dropout=0.1,
        mlpsa_dropout=0.1,
        mlpprop_dropout=0.1,
        lambda_prop_loss=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        lr_scheduler_stepsize=50,
        lr_scheduler_gamma=0.1,
        attention=True,
    )
    hazard_pred = torch.sigmoid(model(sample_XPd, sample_X_static)[0])
    factual_hazard_pred = torch.sigmoid(
        model.get_factual_predictions(
            sample_XPd, sample_X_static, treatment_idx=treatment_idx
        )[0]
    )
    print(f"Input shape: {sample_XPd.shape},\nOutput shape: {hazard_pred.shape}")
    print(f"Factual hazard shape: {factual_hazard_pred.shape}")
