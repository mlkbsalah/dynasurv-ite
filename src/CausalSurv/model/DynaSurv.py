from typing import Tuple

import lightning as L
import torch
import torch.nn.functional as F
from CausalSurv.model.embedding_C_LSTM import embed_LSTM
from CausalSurv.metrics.loss import SURVLoss, NLLogisticHazard
from CausalSurv.tools import load_config
from sksurv.metrics import concordance_index_censored, integrated_brier_score

class DynaSurv(L.LightningModule):
    def __init__(self,
                 x_input_dim: int,
                 p_input_dim: int,
                 output_length: int,
                 interval_bounds: torch.Tensor,
                 lstm_hidden_length: int,
                 x_embed_dim: int,
                 p_embed_dim: int,
                 mlpx_hidden_units: list[int],
                 mlpp_hidden_units: list[int],
                 mlpsa_hidden_units: list[int],
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
        self.p_input_dim = p_input_dim
        self.output_length = output_length
        self.interval_bounds = interval_bounds

        self.lstm = embed_LSTM(x_input_dim=self.x_input_dim,
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

        self.loss_fn = NLLogisticHazard()
        
        # Optimizer params
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_stepsize = lr_scheduler_stepsize
        self.lr_scheduler_gamma = lr_scheduler_gamma

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
            haz_seq: Tensor of shape (batch, time, output_sa_length)
        """        
        batch_size, time_steps, n_features = XPd.shape
        h, c, p = self._init_states(batch_size, device=XPd.device)
        hazards_out = []

        for t in range(time_steps):
            haz_t, h, c, p = self.lstm(XPd[:, t, :], (h, c, p))
            hazards_out.append(haz_t)
        hazards_seq = torch.stack(hazards_out, dim=1)  # (batch_size, time_steps, output_sa_length)
        return hazards_seq

    def training_step(self, batch, batch_idx):
        XPd, interval_idx, treatment_idx, time, event = batch
        haz_pred = self.forward(XPd)
        loss = torch.tensor(0.0, device=haz_pred.device)

        for line in range(interval_idx.size(1)):
            interval_idx_line = interval_idx[:, line]
            event_line = event[:, line]
            haz_pred_line = haz_pred[:, line, :]

            loss_fn = NLLogisticHazard()
            loss_line = loss_fn(haz_pred_line, interval_idx_line, event_line)
            loss += loss_line

        
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        XPd, interval_idx, treatment_idx, time, event = batch
        haz_pred = self.forward(XPd)
        loss = torch.tensor(0.0, device=haz_pred.device)

        for line in range(interval_idx.size(1)):
            interval_idx_line = interval_idx[:, line]
            event_line = event[:, line]
            haz_pred_line = haz_pred[:, line, :]
        
            
            loss_line = self.loss_fn(haz_pred_line, interval_idx_line, event_line)
            loss += loss_line

            y_hazard = self._predict_hazard_on_line(XPd[:, line, :], time[:, line])
            results = concordance_index_censored(event_line.cpu().numpy().astype(bool), time[:, line].cpu().numpy(), y_hazard.detach().cpu().numpy().flatten())
            
            
            self.log(f'val_cindex_line_{line}', results[0], prog_bar=True)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_scheduler_stepsize, gamma=self.lr_scheduler_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def predict_survival(self, x: torch.Tensor) -> torch.Tensor:
        haz_pred = torch.sigmoid(self.forward(x))
        surv_pred = torch.cumprod(1 - haz_pred, dim=-1)
        return surv_pred        

    def _predict_hazard_on_line(self, XPd_line: torch.Tensor, eval_time: torch.Tensor) -> torch.Tensor:
        eval_time = eval_time.view(-1, 1)
        interval_idx = torch.searchsorted(self.interval_bounds, eval_time) - 1

        hazards_on_grid = torch.sigmoid(self.forward(XPd_line.unsqueeze(1)).squeeze(1))
        hazards = hazards_on_grid.gather(1, interval_idx)

        return hazards
    
    def predict_hazard(self, x: torch.Tensor, eval_time: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, n_features = x.shape
        hazards_out = []

        for t in range(time_steps):
            haz_t = self._predict_hazard_on_line(x[:, t, :], eval_time[:, t])
            hazards_out.append(haz_t)
        hazards_seq = torch.stack(hazards_out, dim=1)  # (batch_size, time_steps, 1)
        return hazards_seq

if __name__ == "__main__":
    batch_size = 99
    times_steps = 4
    x_input_dim = 10
    p_input_dim = 5
    output_sa_length = 77

    features = x_input_dim + p_input_dim + 1
    sample_XPd = torch.randn(size=(batch_size, times_steps, features))  
    model = DynaSurv(
        x_input_dim=x_input_dim,
        p_input_dim=p_input_dim,
        output_length=output_sa_length,
        interval_bounds=torch.linspace(0, 100, steps=output_sa_length + 1),
        lstm_hidden_length=32,
        x_embed_dim=16,
        p_embed_dim=16,
        mlpx_hidden_units=[32, 16],
        mlpp_hidden_units=[32, 16],
        mlpsa_hidden_units=[32, 16],
        mlpx_dropout=0.1,
        mlpp_dropout=0.1,
        mlpsa_dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        lr_scheduler_stepsize=50,
        lr_scheduler_gamma=0.1,
    )
    sa_pred = model(sample_XPd)
    print(f"Input shape: {sample_XPd.shape},\nOutput shape: {sa_pred.shape}")