from typing import Tuple

import lightning as L
import torch
from sksurv.metrics import concordance_index_censored

from ..metrics.loss import NLLogisticHazard
from ..model.embedding_C_LSTM import embed_LSTM
from ..model.mlp import MLP


class DynaSurvOnline(L.LightningModule):
    """DynaSurv LightningModule

    Dimensions (from config):
      - input_X_length: number of features in X per timestep
      - input_P_length: number of features in P per timestep
      - x_length: embedded dimension of X
      - p_length: embedded dimension of P
      - hidden_length: LSTM hidden size
      - output_sa_length: number of survival intervals (n_intervals)
    """

    def __init__(
        self,
        x_input_dim: int,
        x_static_dim: int,
        p_input_dim: int,
        p_static_dim: int,
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
        self.output_length = output_length
        self.register_buffer("interval_bounds", interval_bounds)

        self.lstm = embed_LSTM(
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

        self.loss_fn = NLLogisticHazard(reduction="none")

        # Optimizer params
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_stepsize = lr_scheduler_stepsize
        self.lr_scheduler_gamma = lr_scheduler_gamma

    def _init_states(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states for LSTM.

        Args:
            batch_size (int): Batch size
            device (torch.device): Device to allocate tensors on

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Initialize to 0 Hidden state, cell state, and previous input state
        """
        h0 = torch.zeros(batch_size, self.lstm.hidden_length, device=device)
        c0 = torch.zeros(batch_size, self.lstm.hidden_length, device=device)
        p0 = torch.zeros(batch_size, self.lstm.p_embed_dim, device=device)
        return h0, c0, p0

    def forward(
        self, XPd: torch.Tensor, X_static: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Run the embedded C-LSTM over a full sequence.

        Args:
            XPd: Tensor of shape (batch, time, features) where features = input_X_length + input_P_length + 1 (d)
            X_static: Tensor of shape (batch, features_static)
        Returns:
            haz_seq: Tensor of shape (batch, time, output_sa_length)
        """
        batch_size, time_steps, _ = XPd.shape
        x_static_general, x_static_treatment = X_static

        h = self.init_h_mlp(x_static_general)
        p = self.init_p_mlp(x_static_treatment)
        c = torch.zeros(batch_size, self.lstm.hidden_length, device=XPd.device)
        hazards_out_logit = []

        for t in range(time_steps):
            haz_t_logit, h, c, p = self.lstm(XPd[:, t, :], (h, c, p))
            hazards_out_logit.append(haz_t_logit)
        hazards_seq_logit = torch.stack(
            hazards_out_logit, dim=1
        )  # (batch_size, time_steps, output_sa_length)
        return hazards_seq_logit

    def compute_loss(self, XPd, X_static, interval_idx, event, mask):
        haz_pred_logit = self.forward(XPd, X_static)  # (batch, time, output_sa_length)
        loss = torch.tensor(0.0, device=haz_pred_logit.device)
        total_contributing_losses = torch.tensor(0.0, device=haz_pred_logit.device)

        n_lines = haz_pred_logit.size(1)
        for time_step in range(n_lines):
            mask_line = mask[:, time_step]
            interval_idx_line = interval_idx[:, time_step]
            event_line = event[:, time_step]
            haz_pred_line = haz_pred_logit[:, time_step, :]

            loss_line = self.loss_fn(
                haz_pred_line, interval_idx_line, event_line
            )  # (batch, )
            loss_line = loss_line * mask_line
            loss += torch.sum(loss_line)
            total_contributing_losses += mask_line.sum()

        loss = loss / total_contributing_losses
        return loss

    def training_step(self, batch, batch_idx):
        XPd, X_static, interval_idx, _, _, event, mask = batch
        loss = self.compute_loss(XPd, X_static, interval_idx, event, mask)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        XPd, X_static, interval_idx, _, time, event, mask = batch
        loss = self.compute_loss(XPd, X_static, interval_idx, event, mask)
        self.log("val/loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)

        hazards = self.predict_hazard(XPd, X_static, time)  # (batch, time)

        for time_step in range(hazards.size(1)):
            mask_time_step = mask[:, time_step]  # (batch, )
            bool_mask_time_step = mask_time_step.bool()

            hazard_line = hazards[bool_mask_time_step, time_step]
            time_line = time[bool_mask_time_step, time_step]
            event_line = event[bool_mask_time_step, time_step].bool()

            ci_time_step = concordance_index_censored(
                event_line.cpu(), time_line.cpu(), hazard_line.cpu()
            )[0]
            self.log(
                f"val/ci_time_step_{time_step + 1}",
                ci_time_step,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        return loss

    def predict_hazard(
        self,
        XPd: torch.Tensor,
        X_static: Tuple[torch.Tensor, torch.Tensor],
        eval_time: torch.Tensor,
    ):
        """_summary_

        Args:
            XPd (torch.Tensor): (batch, time, features)
            X_static (Tuple[torch.Tensor, torch.Tensor]): _static features
            eval_time (torch.Tensor): (batch, time)

        Returns:
            torch.Tensor: (batch, time) predicted hazards at eval_time
        """
        hazards_on_grid = torch.sigmoid(
            self.forward(XPd, X_static)
        )  # (batch, time, output_sa_length)
        interval_idx = torch.bucketize(eval_time, self.interval_bounds) - 1  # type: ignore (batch, time)
        interval_idx = interval_idx.clamp(min=0, max=self.output_length - 1)

        hazards = torch.gather(hazards_on_grid, 2, interval_idx.unsqueeze(-1)).squeeze(
            -1
        )  # (batch, time)
        return hazards

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

    model = DynaSurvOnline(
        x_input_dim=x_input_dim,
        p_input_dim=p_input_dim,
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
    hazard_pred = torch.sigmoid(model(sample_XPd, sample_X_static))
    print(f"Input shape: {sample_XPd.shape},\nOutput shape: {hazard_pred.shape}")
