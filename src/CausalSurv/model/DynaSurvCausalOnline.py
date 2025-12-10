from typing import Tuple

import torch
import lightning as L

from CausalSurv.model.embedding_C_LSTM_ITE import embed_LSTM_ITE
from CausalSurv.model.mlp import MLP
from CausalSurv.metrics.loss import NLLogisticHazard
import torch.nn as nn

from sksurv.metrics import concordance_index_censored

class DynaSurvCausalOnline(L.LightningModule):
    """Multi-treatment causal survival model with separate heads.

    Args:
        x_input_dim: Number of X features per timestep.
        p_input_dim: Number of P (patient / static / treatment history) features per timestep.
        output_sa_length: Number of conditional survival intervals (n_intervals).
        n_treatments: Number of distinct treatment heads to model.
        config: Dict or TOML path for model configuration.
    """

    def __init__(self,
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
                 init_h_dropout: float,
                 init_p_dropout: float,
                 mlpx_dropout: float,
                 mlpp_dropout: float,
                 mlpsa_dropout: float,
                 lr: float,
                 lr_scheduler_stepsize: int,
                 lr_scheduler_gamma: float,
                 weight_decay: float,
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

        
        self.lstm = embed_LSTM_ITE(x_input_dim=self.x_input_dim,
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
                               )
        
        self.init_h_mlp = MLP(input_dim = x_static_dim,
                            n_units= init_h_hidden,
                            output_dim= self.lstm.hidden_length,
                            dropout=init_h_dropout,
                            )
        self.init_p_mlp = MLP(input_dim = p_static_dim,
                                  n_units= init_p_hidden,
                                  output_dim= self.lstm.p_embed_dim,
                                  dropout=init_p_dropout,
                                  )

        self.treatment_heads = nn.ModuleList([
                MLP(input_dim=lstm_hidden_length,
                    n_units=mlpsa_hidden_units,
                    output_dim=self.output_length,
                    dropout=mlpsa_dropout,
                )
                for _ in range(self.n_treatments)
        ])

        self.propensityhead = MLP(
                    input_dim=self.lstm.hidden_length,
                    output_dim=self.n_treatments,
                    n_units=mlpp_hidden_units,
                    dropout=mlpp_dropout,
                    )


        self.loss_fn = NLLogisticHazard(reduction="none")
        # Optimizer parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_stepsize = lr_scheduler_stepsize
        self.lr_scheduler_gamma = lr_scheduler_gamma


    def _init_states(self, X_static, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_static_general, x_static_treatment = X_static
        batch_size = x_static_general.shape[0]
        h0 = self.init_h_mlp(x_static_general)
        c0 = torch.zeros(batch_size, self.lstm.hidden_length, device=device)
        p0 = self.init_p_mlp(x_static_treatment)
        return h0, c0, p0


    def forward(self, XPd: torch.Tensor, X_static: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the shared embed_LSTM and produce treatment head outputs.

        Args:
            XPd: (batch, n_lines, features = x_input_dim + p_input_dim + 1)

        Returns:
            cond_sa: (batch, n_lines, n_treatments, output_sa_length)
        """
        h, c, p = self._init_states(X_static, XPd.device)
        hazards_logit = []

        for t in range(XPd.shape[1]):
            all_responses_t, (h, c, p) = self._predict_response_one_line(XPd[:, t, :], (h, c, p))
            hazards_logit.append(all_responses_t) # list of (batch, n_treatments, n_intervals)
        hazards_logit = torch.stack(hazards_logit, dim=1) # (batch, n_lines, n_treatments, n_intervals)

        propensity = torch.tensor(0)
        return hazards_logit, propensity

    def _predict_response_one_line(self, XPd_t, tuple_in) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        _, h, c, p = self.lstm(XPd_t, tuple_in)
        treatment_responses = []
        for treatment_head in self.treatment_heads:
            response_treatment = treatment_head(h) # (batch, n_intervals)
            treatment_responses.append(response_treatment) # list of (batch, n_intervals)
        all_response_t = torch.stack(treatment_responses, dim=1) # (batch, n_treatments, n_intervals)

        return all_response_t, (h, c, p)


    def forward_factual(self, XPd: torch.Tensor, X_static: Tuple[torch.Tensor, torch.Tensor], treatment_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward pass to retreive factual hazards

        Args:
            XPd (torch.Tensor): (batch, n_lines, features)
            X_static (Tuple[torch.Tensor, torch.Tensor]): ( (batch, x_static_dim), (batch, p_static_dim) )
            treatment_idx (torch.Tensor): (batch, n_lines) integer treatment head indices actually received

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (factual_hazards_logit, propensity)
        """
        
        hazards_logit, propensity = self.forward(XPd, X_static)  # (batch, n_lines, n_treatments, n_intervals) / (batch, n_lines, n_treatments)
        gather_idx = treatment_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, hazards_logit.shape[-1])  # (batch, n_lines, 1, n_intervals)
        factual_hazards_logit = torch.gather(hazards_logit, dim=2, index=gather_idx).squeeze(2)  # (batch, n_lines, n_intervals)
        return factual_hazards_logit, propensity

    def _compute_step_loss(self, XPd, X_static, treatment_idx, interval_idx, event, mask) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        haz_pred_logit, propensity = self.forward_factual(XPd, X_static, treatment_idx)  # (batch, n_lines, n_intervals) / (batch, n_lines, n_treatments)
        loss = torch.tensor(0.0, device=XPd.device)
        surv_loss = torch.tensor(0.0, device=XPd.device)
        prop_loss = torch.tensor(0.0, device=XPd.device)
        total_contributing_losses = torch.tensor(0.0, device=XPd.device)

        n_lines = XPd.shape[1]
        for time_step in range(n_lines):
            mask_line = mask[:, time_step]
            interval_idx_line = interval_idx[:, time_step]
            event_line = event[:, time_step]
            haz_pred_line = haz_pred_logit[:, time_step, :]

            surv_loss_line = self.loss_fn(haz_pred_line, interval_idx_line, event_line) # (batch, )
            surv_loss_line = surv_loss_line * mask_line
            surv_loss += torch.sum(surv_loss_line)
            total_contributing_losses += mask_line.sum()

        surv_loss = surv_loss / total_contributing_losses
        loss = surv_loss

        return loss, surv_loss, prop_loss

    def training_step(self, batch, batch_idx):
        """Compute factual survival loss.

        Expected batch format (tuple):
            XPd: (batch, n_lines, features)
            sa_true: (batch, n_lines, 2*n_intervals) survival + event indicators
            treatment_index: (batch, n_lines) integer treatment head indices actually received
            time: (batch,) continuous time-to-event (optional, unused here)
            event: (batch,) event indicator (optional, unused here)
        """
        XPd, X_static, interval_idx, treatment_idx, time, event, mask = batch  
        loss, surv_loss, prop_loss = self._compute_step_loss(XPd, X_static, treatment_idx, interval_idx, event, mask)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # self.log("train/survival_loss", total_survival_loss, prog_bar=True, on_step=True, on_epoch=True)
        # self.log("train/propensity_loss", total_propensity_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def predict_hazard(self, XPd, X_static, eval_time, treatment_idx):
        hazards_on_grid = torch.sigmoid(self.forward_factual(XPd, X_static, treatment_idx)[0])  # (batch, n_lines, n_intervals)
        interval_idx = torch.bucketize(eval_time, self.interval_bounds) - 1  # type: ignore (batch, n_lines)
        interval_idx = torch.clamp(interval_idx, min=0, max=self.output_length - 1)

        hazards = torch.gather(hazards_on_grid, dim=2, index=interval_idx.unsqueeze(-1)).squeeze(-1)  # (batch, n_lines)
        return hazards


    def validation_step(self, batch, batch_idx):
        XPd, X_static, interval_idx, treatment_idx, time, event, mask = batch
        loss, surv_loss, prop_loss = self._compute_step_loss(XPd, X_static, treatment_idx, interval_idx, event, mask)
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        hazards = self.predict_hazard(XPd, X_static, time, treatment_idx)  # (batch, n_lines)
        for time_step in range(hazards.size(1)):
            mask_time_step = mask[:, time_step]  # (batch, )
            bool_mask_time_step = mask_time_step.bool()

            hazard_line = hazards[bool_mask_time_step, time_step]
            time_line = time[bool_mask_time_step, time_step]
            event_line = event[bool_mask_time_step, time_step].bool()

            ci_time_step = concordance_index_censored(event_line.cpu(), time_line.cpu(), hazard_line.cpu())[0]
            self.log(f"val/ci_time_step_{time_step+1}", ci_time_step, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self): 
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_scheduler_stepsize, gamma=self.lr_scheduler_gamma)
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
        init_h_dropout=0.1,
        init_p_dropout=0.1,
        mlpx_dropout=0.1,
        mlpp_dropout=0.1,
        mlpsa_dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        lr_scheduler_stepsize=50,
        lr_scheduler_gamma=0.1,
    )
    hazard_pred = torch.sigmoid(model(sample_XPd, sample_X_static)[0])
    factual_hazard_pred = torch.sigmoid(model.forward_factual(sample_XPd, sample_X_static, treatment_idx=treatment_idx)[0])
    print(f"Input shape: {sample_XPd.shape},\nOutput shape: {hazard_pred.shape}")
    print(f"Factual hazard shape: {factual_hazard_pred.shape}")
    