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
            output_sa_length=self.output_sa_length,  # not used for heads, but embed_LSTM_ITE returns an sa; we will ignore that and use hidden state
            cell_config=self.lstm_config,
        )

        self.hidden_length = self.lstm.hidden_length

        self.treatment_heads = nn.ModuleList([
                MLP(
                    input_dim=self.hidden_length,
                    output_dim=self.output_sa_length,
                    n_layers=self.config.get("MLP", {}).get("n_layers", 2),
                    n_units=self.config.get("MLP", {}).get("n_units", [16, 8])
                )
                for _ in range(n_treatments)
        ])

        self.propensityhead = MLP(
                    input_dim=self.hidden_length,
                    output_dim=self.n_treatments,
                    n_layers=self.config.get("MLP", {}).get("n_layers", 2),
                    n_units=self.config.get("MLP", {}).get("n_units", [64, 32])
                    )
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
            XPd: (batch, n_lines, features = x_input_dim + p_input_dim + 1)

        Returns:
            cond_sa: (batch, n_lines, n_treatments, output_sa_length)
        """
        assert XPd.ndim == 3, f"Expected (batch, n_lines, features), got {tuple(XPd.shape)}"

        batch_size, n_lines, _ = XPd.shape
        h, c, p = self._init_states(batch_size, XPd.device)

        surv_outputs = [] 
        propensity_outputs = []
        for t in range(n_lines):
            _, h, c, p = self.lstm(XPd[:, t, :], (h, c, p))
            head_outputs = []  # list of tensors of size (batch, output_sa_length) with length n_treatments
            for treatment_head_i in self.treatment_heads: # type: ignore
                head_out = self.output_activation(treatment_head_i(h))
                head_outputs.append(head_out)
            time_stack = torch.stack(head_outputs, dim=1)  # (batch, n_treatments, output_sa_length)
            surv_outputs.append(time_stack)
            propensity_outputs.append(self.propensityhead(h))  # (batch, n_treatments)

        cond_sa = torch.stack(surv_outputs, dim=1)  # (batch, n_lines, n_treatments, output_sa_length)
        propensity = torch.stack(propensity_outputs, dim=1)  # (batch, n_lines, n_treatments)
        return cond_sa, propensity

    def predict_factual_survival(self, XPd: torch.Tensor, treatment_index: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict factual survival for given treatment indices.

        Args:
            XPd: (batch, n_lines, features)
            treatment_index: (batch, n_lines) integer treatment head indices actually received

        Returns:
            torch.Tensor: (batch, n_lines, output_sa_length) predicted survival probabilities
        """
        batch_size, n_lines, _ = XPd.shape
        h, c, p = self._init_states(batch_size, XPd.device)
        cond_sa = torch.zeros((batch_size, n_lines, self.output_sa_length), device=XPd.device)
        propensity = torch.zeros((batch_size, n_lines, self.n_treatments), device=XPd.device)

        for line in range(n_lines):
            mask_line = mask[:, line]
            if torch.sum(mask_line) == 0: # avoid burden of computing on only padded lines
                continue
            _, h, c, p = self.lstm(XPd[:, line, :], (h, c, p))

            line_treatment_index = treatment_index[:, line]
            propensity_line = self.propensityhead(h)  # (batch, n_treatments)

            cond_sa_line = torch.stack([self.treatment_heads[patient_factual_treatment](h[i]) # type: ignore
                                   for i, patient_factual_treatment in enumerate(line_treatment_index)
                                   ], dim=0)  # (batch, output_sa_length)
            cond_sa[:, line, :] = self.output_activation(cond_sa_line)

            propensity[:, line, :] = propensity_line
        return cond_sa, propensity


    def _compute_step_loss(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) < 3:
            raise ValueError("Batch must contain at least (XPd, sa_true, treatment_index).")

        XPd, sa_true, treatment_index, _, _, mask = batch
        device = XPd.device

        batch_size, n_lines, _ = XPd.shape
        h, c, p = self._init_states(batch_size, XPd.device)
        total_survival_loss = torch.tensor(0.0, device=device)
        total_propensity_loss = torch.tensor(0.0, device=device)
        total_valid_lines = torch.tensor(0.0, device=device)
        for line in range(n_lines):
            mask_line = mask[:, line]
            if torch.sum(mask_line) == 0: # avoid burden of computing on only padded lines
                continue
            _, h, c, p = self.lstm(XPd[:, line, :], (h, c, p))

            line_treatment_index = treatment_index[:, line]

            propensity = self.propensityhead(h)  # (batch, n_treatments)

            cond_sa = torch.stack([ self.treatment_heads[patient_factual_treatment](h[i]) 
                for i, patient_factual_treatment in enumerate(line_treatment_index)
            ], dim=0)  # (batch, output_sa_length)
            cond_sa = self.output_activation(cond_sa)

            # Compute losses only for valid (non-padded) lines
            survival_loss = SURVLoss(sa_true[:, line, :], cond_sa, reduction='none')  # (batch,)
            propensity_loss = PROPLoss(propensity, line_treatment_index, reduction='none')  # (batch,)
            
            # Apply mask and average
            masked_survival_loss = (survival_loss * mask_line).sum()
            masked_propensity_loss = (propensity_loss * mask_line).sum()
            contributing_lines = torch.sum(mask_line)

            total_survival_loss += masked_survival_loss
            total_propensity_loss += masked_propensity_loss
            total_valid_lines += contributing_lines
        
        total_survival_loss = total_survival_loss / (total_valid_lines + 1e-8)
        total_propensity_loss = total_propensity_loss / (total_valid_lines + 1e-8)


        loss = total_survival_loss


        return loss, total_survival_loss, total_propensity_loss



    def training_step(self, batch, batch_idx):
        """Compute factual survival loss.

        Expected batch format (tuple):
            XPd: (batch, n_lines, features)
            sa_true: (batch, n_lines, 2*n_intervals) survival + event indicators
            treatment_index: (batch, n_lines) integer treatment head indices actually received
            time: (batch,) continuous time-to-event (optional, unused here)
            event: (batch,) event indicator (optional, unused here)
        """
  
        loss, total_survival_loss, total_propensity_loss = self._compute_step_loss(batch)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/survival_loss", total_survival_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/propensity_loss", total_propensity_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        loss, avg_survival_loss, avg_propensity_loss = self._compute_step_loss(batch)
        
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val/survival_loss", avg_survival_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val/propensity_loss", avg_propensity_loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

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




if __name__ == "__main__":
    # SMOKE TEST
    batch_size = 3
    n_lines = 5
    x_input_dim = 4
    p_input_dim = 3
    n_intervals = 6
    n_treatments = 2


    features = x_input_dim + p_input_dim + 1
    XPd = torch.randn(batch_size, n_lines-3, features)
    model = CausalDynaSurv(x_input_dim, p_input_dim, n_intervals, n_treatments=n_treatments, n_lines=n_lines, config="../../../configs/dynasurv.toml")
    out, prop = model(XPd)
    # print("cond_sa shape (batch, time, n_treatments, n_intervals):", out.shape)
    # surv = model.predict_counterfactual_survival(XPd)
    # print("survival shape:", surv.shape) # (batch, time, n_treatments, n_intervals)
    # print("propensity shape:", prop.shape) # (batch, time, n_treatments)

    # # Test training step
    sa_true = torch.randint(0, 2, (batch_size, n_lines, 2 * n_intervals)).float()
    treatment_index = torch.randint(0, n_treatments, (batch_size,n_lines))
    mask = torch.ones(batch_size, n_lines)
    batch = (XPd, sa_true, treatment_index, None, None, mask)
    # loss = model.training_step(batch, 0)
    # print("Training step loss:", loss.item())

    # Test validation step
    val_loss = model.training_step(batch, 0)
    print("Validation step loss:", val_loss)