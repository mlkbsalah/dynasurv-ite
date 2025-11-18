from typing import Tuple

import lightning as L
import torch

from CausalSurv.model.embedding_C_LSTM import embed_LSTM
from CausalSurv.metrics.loss import SURVLoss
from CausalSurv.tools import load_config
from CausalSurv.metrics.scoring import concordance_index, brier_score

class DynaSurv(L.LightningModule):
    """DynaSurv LightningModule

    Dimensions (from config):
      - input_X_length: number of features in X per timestep
      - input_P_length: number of features in P per timestep
      - x_length: embedded dimension of X
      - p_length: embedded dimension of P
      - hidden_length: LSTM hidden size
      - output_sa_length: number of survival intervals (n_intervals)
    """

    def __init__(self, x_input_dim: int, p_input_dim: int, output_sa_length: int, config: dict | str, time_bins: torch.Tensor) -> None:                
        super().__init__()
        self.save_hyperparameters()

        self.config = load_config(config)
        self.train_config = self.config['Optimizer']
        self.lstm_config = self.config['LSTM_cell']

        self.x_input_dim = x_input_dim
        self.p_input_dim = p_input_dim
        self.output_sa_length = output_sa_length
        self.time_bins = time_bins



        self.lstm = embed_LSTM(
            x_input_dim=self.x_input_dim,
            p_input_dim=self.p_input_dim,
            output_sa_length=self.output_sa_length,
            cell_config=self.lstm_config
        )
        self.sigmoid = torch.nn.Sigmoid()
        

        # Optimizer params
        self._optim_name = self.train_config['name'].lower()
        self._lr: float = self.train_config['lr']
        self._weight_decay = self.train_config['weight_decay']

    def _init_states(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def forward(self, XPd: torch.Tensor) -> torch.Tensor:
        """Run the embedded C-LSTM over a full sequence.

        Args:
            XPd: Tensor of shape (batch, time, features) where features = input_X_length + input_P_length + 1 (d)

        Returns:
            sa_seq: Tensor of shape (batch, time, output_sa_length)
        """
        print()
        assert XPd.ndim == 3, f"Expected (batch, time, features), got {tuple(XPd.shape)}"
        batch_size, time_steps, _ = XPd.shape
        h, c, p = self._init_states(batch_size, device=XPd.device)
        sa_out = []

        for t in range(time_steps):
            sa_t, h, c, p = self.lstm(XPd[:, t, :], (h, c, p))
            sa_t = self.sigmoid(sa_t)  # Ensure outputs are in (0, 1)
            sa_out.append(sa_t)
        sa_seq = torch.stack(sa_out, dim=1)  # (batch_size, time_steps, output_sa_length)
        return sa_seq

    def training_step(self, batch, batch_idx):
        XPd, sa_true, _, time, event, mask = batch
        sa_pred = self.forward(XPd) # (batch, time, output_sa_length)
        loss = torch.tensor(0.0, device=sa_pred.device)
        total_contributing_losses = torch.tensor(0.0, device=sa_pred.device)

        n_lines = sa_pred.size(1)
        for line in range(n_lines):
            mask_line = mask[:, line]
            if torch.sum(mask_line) == 0:
                continue

            sa_pred_line = sa_pred[:, line, :]  # (batch, output_sa_length)
            sa_true_line = sa_true[:, line, :]  # (batch, 2*output_sa_length)
            loss_line = SURVLoss(sa_true_line, sa_pred_line)

            masked_loss_line = (loss_line * mask_line).sum()
            contributing_losses = mask_line.sum()
        
            loss += masked_loss_line
            total_contributing_losses += contributing_losses
           
        loss = loss / total_contributing_losses

        # c_index = concordance_index(sa_pred, time, event, self.time_bins, mask)

        # self.log("train/c_index_mean", c_index.mean(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        XPd, sa_true, _, _, _, mask = batch
        sa_pred = self.forward(XPd) # (batch, time, output_sa_length)
        loss = torch.tensor(0.0, device=sa_pred.device)
        total_contributing_losses = torch.tensor(0.0, device=sa_pred.device)

        n_lines = sa_pred.size(1)
        for line in range(n_lines):
            mask_line = mask[:, line]
            if torch.sum(mask_line) == 0:
                continue

            sa_pred_line = sa_pred[:, line, :]  # (batch, output_sa_length)
            sa_true_line = sa_true[:, line, :]  # (batch, 2*output_sa_length)
            loss_line = SURVLoss(sa_true_line, sa_pred_line)

            masked_loss_line = (loss_line * mask_line).sum()
            contributing_losses = mask_line.sum()
        
            loss += masked_loss_line
            total_contributing_losses += contributing_losses
           
        loss = loss / total_contributing_losses

        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        if self._optim_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        elif self._optim_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        elif self._optim_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self._lr, weight_decay=self._weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self._optim_name}")
        return optimizer
    
    def predict_survival_at_times(self, XPd: torch.Tensor, eval_times: torch.Tensor) -> torch.Tensor:
        """Predict survival probabilities at specified evaluation times.

        Args:
            XPd: Tensor of shape (batch, n_lines, features)
            eval_times: Tensor of shape (n_eval_times, n_lines)

        Returns:
            surv_probs: Tensor of shape (batch,) 
        """
        sa_pred = self.forward(XPd)  # (batch, time, output_sa_length)
        batch_size, n_lines, _ = sa_pred.shape
        surv_probs = torch.zeros((batch_size, n_lines), device=sa_pred.device)

        bin_indices = torch.searchsorted(self.time_bins, eval_times) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.output_sa_length - 1)

        for i in range(batch_size):
                bin_idx = bin_indices[i].item()
                surv_prob = torch.prod(sa_pred[i, :, :bin_idx+1], dim=-1).mean()
                surv_probs[i] = surv_prob

        return surv_probs
        

if __name__ == "__main__":
    batch_size = 99
    times_steps = 4
    x_input_dim = 10
    p_input_dim = 5
    output_sa_length = 77

    features = x_input_dim + p_input_dim + 1
    sample_XPd = torch.randn(size=(batch_size, times_steps, features))  
    # model = DynaSurv(x_input_dim=x_input_dim, p_input_dim=p_input_dim, output_sa_length=output_sa_length, config="../../../configs/dynasurv.toml")