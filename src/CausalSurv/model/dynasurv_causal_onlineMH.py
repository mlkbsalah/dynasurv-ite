from collections import defaultdict
from typing import Tuple

import lightning as L
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw

from ..metrics.emd_loss import EMDLoss
from ..metrics.mmd_loss import MMDLoss
from ..metrics.survival_loss import NLLogisticHazard
from ..model.embedding_C_LSTM_ITE import embed_LSTM_ITE
from ..model.mlp import MLP


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
        n_lines: int,
        output_length: int,
        interval_bounds: torch.Tensor,
        lstm_hidden_length: int = 128,
        lstm_num_layers: int = 4,
        x_embed_dim: int = 64,
        p_embed_dim: int = 16,
        init_h_hidden: list[int] = [32],
        init_p_hidden: list[int] = [32],
        mlpx_hidden_units: list[int] = [128],
        mlpp_hidden_units: list[int] = [32],
        mlpsa_hidden_units: list[int] = [64, 64, 64],
        mlpprop_hidden_units: list[int] = [64, 32],
        lr: float = 1e-5,
        lr_scheduler_stepsize: int = 20,
        lr_scheduler_gamma: float = 0.3,
        weight_decay: float = 0.05,
        attention: bool = True,
        init_h_dropout: float = 0,
        init_p_dropout: float = 0,
        mlpx_dropout: float = 0,
        mlpp_dropout: float = 0,
        mlpsa_dropout: float = 0,
        mlpprop_dropout: float = 0,
        lambda_prop_loss: float = 0,
        lambda_ipm_mmd: float = 0,
        lambda_ipm_emd2: float = 0,
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
        self.n_lines = n_lines
        self.output_length = output_length

        self.register_buffer("interval_bounds", interval_bounds)
        self.interval_bounds: torch.Tensor

        self.evaluation_horizon_times = evaluation_horizon_times
        self.brier_integration_step = brier_integration_step

        self.min_mmd_samples = lstm_hidden_length / 2

        self.lstm = embed_LSTM_ITE(
            x_input_dim=self.x_input_dim,
            p_input_dim=self.p_input_dim,
            output_length=self.output_length,
            hidden_length=lstm_hidden_length,
            num_layers=lstm_num_layers,
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

        self.progression_heads = torch.nn.ModuleList(
            [
                MLP(
                    input_dim=self.lstm.hidden_length,
                    output_dim=self.output_length * n_treatments,
                    n_units=mlpsa_hidden_units,
                    dropout=mlpsa_dropout,
                )
                for _ in range(n_lines)
            ]
        )

        self.propensityhead = MLP(
            input_dim=self.lstm.hidden_length,
            output_dim=self.n_treatments,
            n_units=mlpprop_hidden_units,
            dropout=mlpprop_dropout,
        )

        self.surv_loss_fn = NLLogisticHazard(reduction="none")
        self.propensity_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.mmd_loss = MMDLoss()
        self.emd2_loss = EMDLoss()

        # Optimizer parameters
        self.lambda_prop_loss = lambda_prop_loss
        self.lambda_ipm_mmd = lambda_ipm_mmd
        self.lambda_ipm_emd2 = lambda_ipm_emd2
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
            hazard_logit: (batch, n_lines, n_treatments, n_intervals)
            latent_state: (batch, n_lines, latent_dim)
        """
        h, c, p = self._init_lstm_states(X_static, XPd.device)
        hazards_logit = []
        latent_state = []
        x_static, p_static = X_static

        for t in range(XPd.shape[1]):
            # XPd_aug = torch
            logit_t, (h, c, p) = self._step(XPd[:, t, :], (h, c, p), line_idx=t)
            hazards_logit.append(logit_t)
            latent_state.append(h)
        hazards_logit = torch.stack(hazards_logit, dim=1)
        latent_state = torch.stack(latent_state, dim=1)

        return hazards_logit, latent_state

    def _step(
        self, XPd_t, tuple_in, line_idx=0
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass for one time step."""
        _, h, c, p = self.lstm(XPd_t, tuple_in)
        logit_t = self.progression_heads[line_idx](h).view(
            -1, self.n_treatments, self.output_length
        )  # (batch, n_treatments, n_intervals)

        return logit_t, (h, c, p)

    def _init_lstm_states(
        self, X_static, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden, cell, and treatment embedding states."""
        x_static_general, x_static_treatment = X_static
        batch_size = x_static_general.shape[0]
        # h0 = self.init_h_mlp(x_static_general)
        # c0 = torch.zeros(batch_size, self.lstm.hidden_length, device=device)
        # p0 = self.init_p_mlp(x_static_treatment)

        h0 = torch.zeros(batch_size, self.lstm.hidden_length, device=device)
        c0 = torch.zeros(batch_size, self.lstm.hidden_length, device=device)
        p0 = torch.zeros(batch_size, self.lstm.p_embed_dim, device=device)

        return h0, c0, p0

    def forward_factual(
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
            factual_hazards_logit (troch.Tensor): (batch, n_lines, n_intervals)
            latent_state (troch.Tensor): (batch, n_lines, latent_dim)
        """

        hazards_logit, latent_state = self.forward(
            XPd, X_static
        )  # (batch, n_lines, n_treatments, n_intervals) / (batch, n_lines, lstm_hidden_length)
        gather_idx = (
            treatment_idx.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 1, hazards_logit.shape[-1])
        )  # (batch, n_lines, 1, n_intervals)
        factual_hazards_logit = torch.gather(
            hazards_logit, dim=2, index=gather_idx
        ).squeeze(2)  # (batch, n_lines, n_intervals)
        return factual_hazards_logit, latent_state

    # ====================== Training and evaluation ======================
    def training_step(self, batch, batch_idx):
        """perform a training step"""
        XPd, X_static, interval_idx, treatment_idx, time, event, mask, patient_id = (
            batch
        )
        # (
        #     XPd, X_static,
        #     interval_idx, _,
        #     treatment_idx,
        #     time, event,
        #     _, _,
        #     mask, _,
        # ) = batch
        if self.current_epoch == 0:
            self._accumulate_data(time, event, mask)

        hazard_logits, latent_state = self.forward_factual(XPd, X_static, treatment_idx)

        prop_loss = self._compute_propensity_loss(latent_state, treatment_idx, mask)
        surv_loss = self._compute_sruvival_loss(
            hazard_logits, interval_idx, event, mask
        )
        ipm_mmd_reg = self._compute_ipm_mmd(latent_state, treatment_idx, mask)
        loss = (
            surv_loss
            - self.lambda_prop_loss * prop_loss
            + self.lambda_ipm_mmd * ipm_mmd_reg
        )
        # ipm_wass_reg = self.compute_ipm_w2(latent_state, treatment_idx, mask)

        # ========= logging =========
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train/survival_loss",
            surv_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/propensity_loss",
            prop_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "train/ipm_reg",
            ipm_mmd_reg,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """perform a validation step"""
        XPd, X_static, interval_idx, treatment_idx, time, event, mask, patient_id = (
            batch
        )
        # (
        #     XPd, X_static,
        #     interval_idx, _,
        #     treatment_idx,
        #     time, event,
        #     _, _,
        #     mask, _,
        # ) = batch
        N_lines = time.shape[1]
        hazard_logits, latent_state = self.forward_factual(XPd, X_static, treatment_idx)

        prop_loss = self._compute_propensity_loss(latent_state, treatment_idx, mask)
        surv_loss = self._compute_sruvival_loss(
            hazard_logits, interval_idx, event, mask
        )
        ipm_mmd_reg = self._compute_ipm_mmd(latent_state, treatment_idx, mask)
        imp_emd2_reg = self._compute_ipm_emd2(latent_state, treatment_idx, mask)
        loss = (
            surv_loss
            - self.lambda_prop_loss * prop_loss
            + self.lambda_ipm_mmd * ipm_mmd_reg
            + self.lambda_ipm_emd2 * imp_emd2_reg
        )

        if dataloader_idx == 0:
            self.log(
                "val_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        elif dataloader_idx == 1:
            self.log(
                "early_stop_loss",
                loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        self.log(
            "val/survival_loss",
            surv_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/propensity_loss",
            prop_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "val/ipm_mmd",
            ipm_mmd_reg,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "val/ipm_emd2",
            imp_emd2_reg,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        if self.trainer.sanity_checking:
            return loss

        discrete_hazards = torch.sigmoid(hazard_logits)
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

            t_line = time[valid_mask, line]  # (valid_batch, 1)
            e_line = event[valid_mask, line].bool()  # (valid_batch, 1)
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

            # ic(t_line)

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
                device=torch.device("cpu"),
            )
            ibs.append(ibs_line)

            self.log(
                f"val/ci_time_step_{line + 1}",
                c_index_td,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"val/ibs_time_step_{line + 1}",
                ibs_line,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        # ic(ibs, (np.sum(mask.cpu().numpy(), axis = 0) / np.sum(mask.cpu().numpy())) )

        n_t = np.sum(mask.cpu().numpy(), axis=0)
        w = 1 / n_t
        w = w / w.sum()

        weighted_ibs = np.sum(ibs * w)

        if dataloader_idx == 0:
            self.log(
                "average_ci",
                float(np.mean(ci)),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "average_ibs",
                float(weighted_ibs),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        elif dataloader_idx == 1:
            self.log(
                "early_stop_average_ci",
                float(np.mean(ci)),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "early_stop_average_ibs",
                float(weighted_ibs),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        return loss, np.mean(ci), weighted_ibs

    def _compute_propensity_loss(self, latent_state, treatment_idx, mask):
        batch_size, n_lines, _ = latent_state.shape
        treatment_prediction = torch.softmax(
            self.propensityhead(latent_state.view(-1, latent_state.shape[-1])),
            dim=-1,
        )
        prop_loss = self.propensity_loss_fn(
            treatment_prediction, treatment_idx.view(-1)
        ).view(batch_size, n_lines)
        masked_prop_loss = (prop_loss * mask).sum() / mask.sum()

        return masked_prop_loss

    def _compute_sruvival_loss(self, hazard_logits, interval_idx, event, mask):
        batch_size, n_lines, _ = hazard_logits.shape
        surv_loss = self.surv_loss_fn(
            hazard_logits.view(-1, hazard_logits.shape[-1]),
            interval_idx.view(-1),
            event.view(-1),
        ).view(batch_size, n_lines)
        masked_surv_loss = (surv_loss * mask).sum() / mask.sum()
        return masked_surv_loss

    def _compute_ipm_mmd(self, latent_state, treatment_idx, mask):
        pairwise_mmd = torch.tensor(
            0.0, dtype=torch.float32, device=latent_state.device
        )
        n_pairs = 0

        for line in range(latent_state.shape[1]):
            valid_mask = mask[:, line].bool()
            if not valid_mask.any():
                continue

            z_line = latent_state[valid_mask, line, :]
            t_line = treatment_idx[valid_mask, line]

            valid_treatments = self.valid_treatments_per_line[line]
            z_groups = {k: z_line[t_line == k] for k in valid_treatments}

            for i, k_i in enumerate(valid_treatments):
                for k_j in valid_treatments[i:]:
                    n_i = z_groups[k_i].shape[0]
                    n_j = z_groups[k_j].shape[0]

                    if n_i < self.min_mmd_samples or n_j < self.min_mmd_samples:
                        continue

                    pairwise_mmd += self.mmd_loss(z_groups[k_i], z_groups[k_j])
                    n_pairs += 1

        return pairwise_mmd / n_pairs if n_pairs > 0 else pairwise_mmd

    def _compute_ipm_emd2(self, latent_state, treatment_idx, mask):
        pairwise_w2 = torch.tensor(0.0, dtype=torch.float32, device=latent_state.device)
        n_pairs = 0

        for line in range(latent_state.shape[1]):
            valid_mask = mask[:, line].bool()
            if not valid_mask.any():
                continue

            z_line = latent_state[valid_mask, line, :]
            t_line = treatment_idx[valid_mask, line]

            valid_treatments = self.valid_treatments_per_line[line]
            z_groups = {k: z_line[t_line == k] for k in valid_treatments}

            for i, k_i in enumerate(valid_treatments):
                for k_j in valid_treatments[i:]:
                    n_i = z_groups[k_i].shape[0]
                    n_j = z_groups[k_j].shape[0]

                    if n_i < self.min_mmd_samples or n_j < self.min_mmd_samples:
                        continue

                    pairwise_w2 += self.emd2_loss(z_groups[k_i], z_groups[k_j])
                    n_pairs += 1

        return pairwise_w2 / n_pairs if n_pairs > 0 else pairwise_w2

    def fit_censoring_estimator(self, train_loader):
        """
        Extract train times and events from the training loader and store them
        for IPCW computation during validation/test evaluation.
        Call once after training or before test evaluation.
        """
        all_times = defaultdict(list)
        all_events = defaultdict(list)

        for batch in train_loader:
            (
                XPd,
                X_static,
                interval_idx,
                treatment_idx,
                time,
                event,
                mask,
                patient_id,
            ) = batch

            # (
            #     XPd, X_static,
            #     interval_idx, _,
            #     treatment_idx,
            #     time, event,
            #     _, _,
            #     mask, _,
            # ) = batch
            N_lines = time.shape[1]

            for line in range(N_lines):
                valid_mask = mask[:, line].bool()
                if not valid_mask.any():
                    continue
                all_times[line].append(time[valid_mask, line].cpu().numpy())
                all_events[line].append(event[valid_mask, line].cpu().numpy())

        self.train_times = {line: np.concatenate(all_times[line]) for line in all_times}
        self.train_events = {
            line: np.concatenate(all_events[line]).astype(bool) for line in all_events
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def compute_treatment_prediction_auc(self, XPd, X_static, treatment_idx, mask):
        _, latent_state = self.forward_factual(XPd, X_static, treatment_idx)
        aucs = []

        for line in range(XPd.shape[1]):
            mask_line = mask[:, line].bool()
            X = latent_state[mask_line, line].detach().cpu().numpy()
            y = treatment_idx[mask_line, line].cpu().numpy()

            valid_ks = self.valid_treatments_per_line[line]
            valid_mask = np.isin(y, valid_ks)
            X = X[valid_mask]
            y = y[valid_mask]

            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            rfc = RandomForestClassifier(random_state=42)
            rfc.fit(x_train, y_train)

            rfc_probs = rfc.predict_proba(x_test)
            auc = roc_auc_score(
                y_test,
                rfc_probs,
                multi_class="ovr",
                labels=rfc.classes_,
            )
            aucs.append(auc)

        return aucs

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
        # ic(test_times)
        cumhazards = self.eval_factual_cumhazard(
            discrete_cumhazards, test_times.squeeze(), device
        )  # (valid_batch, n_intervals + 1)

        # ic(train_events.shape, train_times.shape, test_times.shape)
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
        device=torch.device("cpu"),
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
            new_time=bs_eval_times.squeeze(),
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
    def predict(
        self, XPd, X_static, gather: bool = False, factual_idx: None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple of (hazards, survivals) for all time steps.

        Args:
            XPd: (batch, n_lines, features)
            X_static: ( (batch, x_static_dim), (batch, p_static_dim) )
            gather: boolean, gather at the factual indexes
            factual_idx: (batch, n_lines)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                hazards: (batch, n_lines, n_treatments, n_intervals)
                survivals: (batch, n_lines, n_treatments, n_intervals)
        """
        kwargs = {"XPd": XPd, "X_static": X_static}
        if gather:
            if factual_idx is None:
                raise RuntimeError(
                    "Setting gather to True requires factual index but None was given"
                )
            kwargs.update({"gather": gather, "factual_idx": factual_idx})
        return (
            self.predict_discrete_hazard(**kwargs),
            self.predict_discrete_survival(**kwargs),
        )

    def predict_discrete_hazard(
        self,
        XPd,
        X_static,
        gather: bool = False,
        factual_idx: None = None,
        cum: bool = False,
    ) -> torch.Tensor:
        if gather:
            if factual_idx is None:
                raise RuntimeError("gather set to true with no treatment idx")

            discrete_hazards = torch.sigmoid(
                self.forward_factual(XPd, X_static, factual_idx)[0]
            )  # (batch, n_lines, n_intervals)

        else:
            discrete_hazards = torch.sigmoid(
                self.forward(XPd, X_static)[0]
            )  # (batch, n_lines, n_treatments, n_intervals)

        discrete_hazards = torch.cat(
            [torch.zeros_like(discrete_hazards[..., :1]), discrete_hazards],
            dim=-1,
        )  # (batch, n_lines, n_intervals + 1)/(batch, n_lines, n_treatments, n_intervals + 1) to account for H(0)=0

        if cum:
            discrete_cumhazards = torch.cumsum(discrete_hazards, dim=-1)

            return discrete_cumhazards
        else:
            return discrete_hazards

    def predict_discrete_survival(
        self, XPd, X_static, gather: bool = False, factual_idx: None = None
    ):
        if gather:
            if factual_idx is None:
                raise RuntimeError("gather set to true with no treatment idx")

            discrete_hazards = torch.sigmoid(
                self.forward_factual(XPd, X_static, factual_idx)[0]
            )  # (batch, n_lines, n_intervals)

        else:
            discrete_hazards = torch.sigmoid(
                self.forward(XPd, X_static)[0]
            )  # (batch, n_lines, n_treatments, n_intervals)

        discrete_survival = torch.cumprod(1 - discrete_hazards, dim=-1)
        discrete_survival = torch.cat(
            [torch.ones_like(discrete_survival[..., :1]), discrete_survival], dim=-1
        )  # (batch, n_lines, n_intervals + 1)/ (batch, n_lines, n_treatments, n_intervals + 1)to account for S(0)=1

        return discrete_survival

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

        # ic(eval_time.shape, discrete_cumhazards.shape)

        interval_idx = (
            torch.bucketize(eval_time, interval_bounds, right=True) - 1
        )  # (n_eval_points, )

        # ic(interval_idx)

        # ic(interval_idx.shape)
        if torch.any(interval_idx < 0) or torch.any(interval_idx >= self.output_length):
            # print(
            #     "Warning: eval_time is outside the range of interval_bounds. Clamping to valid range."
            # )
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
            # print(
            #     "Warning: eval_time is outside the range of interval_bounds. Clamping to valid range."
            # )
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

    # ====================== Lightning hooks ======================
    def _setup_valid_treatments(self):
        """shared setup for fit and test"""
        assert hasattr(self.trainer, "datamodule"), (
            "Trainer does not have a datamodule, make sure to pass it to trainer.fit/test()"
        )
        self.valid_treatments_per_line = (
            self.trainer.datamodule.valid_treatments_per_line
        )

    def on_fit_start(self) -> None:
        self._setup_valid_treatments()

    def on_test_start(self) -> None:
        self._setup_valid_treatments()
