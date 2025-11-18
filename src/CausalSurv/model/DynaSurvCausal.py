from typing import List, Tuple

import lightning as L
import torch
import torch.nn as nn

from CausalSurv.model.embedding_C_LSTM_ITE import embed_LSTM_ITE
from CausalSurv.model.mlp import MLP
from CausalSurv.metrics.loss import SURVLoss, PROPLoss
from CausalSurv.tools import load_config


class CausalDynaSurv(L.LightningModule):
    """Multi-treatment causal survival model with separate heads.

    Args:
        x_input_dim: Number of X features per timestep.
        p_input_dim: Number of P (patient / static / treatment history) features per timestep.
        output_sa_length: Number of conditional survival intervals (n_intervals).
        n_treatments: Number of distinct treatment heads to model.
        config: Dict or TOML path for model configuration.
    """

    def __init__( self, x_input_dim: int, p_input_dim: int, output_sa_length: int, n_treatments: int, n_lines: int, config: dict | str = "../../configs/dynasurv.toml") -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = load_config(config)
        self.train_config = self.config.get("Optimizer", {})
        self.lstm_config = self.config.get("LSTM_cell", {})

        self.x_input_dim = x_input_dim
        self.p_input_dim = p_input_dim
        self.output_sa_length = output_sa_length
        self.n_treatments = n_treatments
        self.n_lines = n_lines

        # Shared LSTM representation learner
        self.lstm = embed_LSTM_ITE(
            x_input_dim=self.x_input_dim,
            p_input_dim=self.p_input_dim,
            output_sa_length=self.output_sa_length,  # not used for heads, but embed_LSTM returns an sa; we will ignore that and use hidden state
            cell_config=self.lstm_config,
        )

        self.hidden_length = self.lstm.hidden_length

        self.treatment_heads = nn.ModuleList([
            nn.ModuleList([
                MLP(
                    input_dim=self.hidden_length,
                    output_dim=self.output_sa_length,
                    n_layers=self.config.get("MLP", {}).get("n_layers", 2),
                    n_units=self.config.get("MLP", {}).get("n_units", [64, 32])
                )
                for _ in range(n_treatments)
            ])
            for _ in range(self.n_lines)  # or pass time_steps explicitly
        ])
        self.propensityheads = nn.ModuleList([
            nn.Sequential(
                MLP(
                    input_dim=self.hidden_length,
                    output_dim=self.n_treatments,
                    n_layers=self.config.get("MLP", {}).get("n_layers", 2),
                    n_units=self.config.get("MLP", {}).get("n_units", [64, 32])
                )
                , nn.Softmax(dim=-1)
            )
            for _ in range(self.n_lines)
        ])
        self.output_activation = nn.Sigmoid()

        # Optimizer parameters
        self._optim_name = self.train_config.get("name", "adam").lower()
        self._lr: float = float(self.train_config.get("lr", 3e-4))
        self._weight_decay: float = float(self.train_config.get("weight_decay", 0.0))


    def _init_states(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(batch_size, self.hidden_length, device=device)
        c0 = torch.zeros(batch_size, self.hidden_length, device=device)
        p0 = torch.zeros(batch_size, self.lstm.p_embed_dim, device=device)
        return h0, c0, p0


    def forward(self, XPd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the shared embed_LSTM and produce treatment head outputs.

        Args:
            XPd: (batch, time, features = x_input_dim + p_input_dim + 1)

        Returns:
            cond_sa: (batch, time, n_treatments, output_sa_length)
        """
        assert XPd.ndim == 3, f"Expected (batch, time, features), got {tuple(XPd.shape)}"

        batch_size, time_steps, _ = XPd.shape
        h, c, p = self._init_states(batch_size, XPd.device)

        surv_outputs = [] #list of tensors of size (batch, n_treatments, output_sa_length) per time step (line)
        propensity_outputs = []
        for t in range(time_steps):
            _, h, c, p = self.lstm(XPd[:, t, :], (h, c, p))
            head_outputs = []  # list of tensors of size (batch, output_sa_length) with length n_treatments
            for treatment_head_i_t in self.treatment_heads[t]: # type: ignore
                head_out = self.output_activation(treatment_head_i_t(h))
                head_outputs.append(head_out)
            time_stack = torch.stack(head_outputs, dim=1)  # (batch, n_treatments, output_sa_length)
            surv_outputs.append(time_stack)
            propensity_outputs.append(self.propensityheads[t](h))  # (batch, n_treatments)

        cond_sa = torch.stack(surv_outputs, dim=1)  # (batch, time, n_treatments, output_sa_length)
        propensity = torch.stack(propensity_outputs, dim=1)  # (batch, time, n_treatments)
        return cond_sa, propensity


    def treatment_survival(self, XPd: torch.Tensor) -> torch.Tensor:
        """Compute survival functions per treatment head.

        Args:
            XPd: (batch, time, features)
        Returns:
            survival: (batch, time, n_treatments, n_intervals) cumulative product along intervals
        """
        cond_sa, _ = self.forward(XPd)
        survival = torch.cumprod(cond_sa, dim=-1)
        return survival


    def _select_factual_head(self, cond_sa: torch.Tensor, treatment_index: torch.Tensor) -> torch.Tensor:
        """Select factual treatment head outputs.

        Args:
            cond_sa: (batch, time, n_treatments, n_intervals)
            treatment_index:
                - (batch,)   single factual treatment per sequence (new case)
                - (batch,time) per-time factual treatment indices (legacy / optional)
        Returns:
            factual_sa: (batch, time, n_intervals)
        """
        # Handle different treatment_index shapes
        if treatment_index.ndim == 1:  # (batch,) -> expand to (batch, time)
            treatment_index = treatment_index.unsqueeze(1).expand(-1, cond_sa.size(1))
        
        # Use advanced indexing instead of gather to avoid MPS issues
        batch_size, time_steps, n_treatments, n_intervals = cond_sa.shape
        
        # Create batch and time indices
        batch_idx = torch.arange(batch_size, device=cond_sa.device).unsqueeze(1).expand(-1, time_steps)
        time_idx = torch.arange(time_steps, device=cond_sa.device).unsqueeze(0).expand(batch_size, -1)
        
        # Use advanced indexing: cond_sa[batch_idx, time_idx, treatment_index, :]
        factual = cond_sa[batch_idx, time_idx, treatment_index, :]  # (batch, time, n_intervals)
        return factual


    def training_step(self, batch, batch_idx):
        """Compute factual survival loss.

        Expected batch format (tuple):
            XPd: (batch, time, features)
            sa_true: (batch, time, 2*n_intervals) survival + event indicators
            treatment_index: (batch,) integer treatment head indices actually received
            time: (batch,) continuous time-to-event (optional, unused here)
            event: (batch,) event indicator (optional, unused here)
        """
  
        if len(batch) < 3:
            raise ValueError("Batch must contain at least (XPd, sa_true, treatment_index).")

        XPd, sa_true, treatment_index, _, _ = batch
        cond_sa, propensity = self.forward(XPd)  # (batch, time, n_treatments, n_intervals) , (batch, time, n_treatments)
        factual_sa = self._select_factual_head(cond_sa, treatment_index)  # (batch, time, n_intervals)
        survival_loss = SURVLoss(sa_true, factual_sa)
        propensity_loss = PROPLoss(propensity, treatment_index)
        loss = survival_loss - propensity_loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/survival_loss", survival_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/propensity_loss", propensity_loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        if len(batch) < 3:
            raise ValueError("Batch must contain at least (XPd, sa_true, treatment_index).")
        
        XPd, sa_true, treatment_index, _, _ = batch
        cond_sa, propensity = self.forward(XPd)
        factual_sa = self._select_factual_head(cond_sa, treatment_index)
        survival_loss = SURVLoss(sa_true, factual_sa)
        propensity_loss = PROPLoss(propensity, treatment_index)
        self.log("validation/loss", survival_loss - propensity_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("validation/survival_loss", survival_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("validation/propensity_loss", propensity_loss, prog_bar=True, on_step=False, on_epoch=True)
        return survival_loss


    def configure_optimizers(self): 
        if self._optim_name == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        elif self._optim_name == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        elif self._optim_name == "sgd":
            opt = torch.optim.SGD(self.parameters(), lr=self._lr, weight_decay=self._weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self._optim_name}")
        return opt


    def predict_all_treatments(self, XPd: torch.Tensor) -> torch.Tensor:
        """Alias for forward() for clarity in inference pipelines."""
        sa, _ = self.forward(XPd)
        return sa

    def predict_counterfactual_survival(self, XPd: torch.Tensor) -> torch.Tensor:
        """Return survival functions for each treatment simultaneously.

        Args:
            XPd: (batch, time, features)
        Returns:
            survival: (batch, time, n_treatments, n_intervals)
        """
        return self.treatment_survival(XPd)


if __name__ == "__main__":
    # SMOKE TEST
    batch_size = 3
    time_steps = 5
    x_input_dim = 4
    p_input_dim = 3
    n_intervals = 6
    n_treatments = 2


    features = x_input_dim + p_input_dim + 1
    XPd = torch.randn(batch_size, time_steps, features)
    model = CausalDynaSurv(x_input_dim, p_input_dim, n_intervals, n_treatments=n_treatments, n_lines=time_steps, config="../../../configs/dynasurv.toml")
    out, prop = model(XPd)
    print("cond_sa shape (batch, time, n_treatments, n_intervals):", out.shape)
    surv = model.predict_counterfactual_survival(XPd)
    print("survival shape:", surv.shape) # (batch, time, n_treatments, n_intervals)
    print("propensity shape:", prop.shape) # (batch, time, n_treatments)