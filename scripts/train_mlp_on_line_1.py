import torch
import lightning as L
import torch.nn as nn
from CausalSurv.metrics.loss import SURVLoss
from CausalSurv.data.data_utils import ESMEDataModule
from CausalSurv.metrics.metrics_callback import MetricsCallback
from pytorch_lightning.loggers import WandbLogger
import wandb

data_path = "../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet"


class SurvivalMLP(L.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32], lr=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.lr = lr


        layers = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        layers.append(nn.Sigmoid())

        self.output = layers[-2]  # The last linear layer before Sigmoid
        nn.init.xavier_uniform_(self.output.weight, gain=0.1)
        nn.init.constant_(self.output.bias, 2.0)  
        self.model = nn.Sequential(*layers)

    def forward(self, XPd):
        sa_pred = self.model(XPd)
        return sa_pred

    def training_step(self, batch, batch_idx):
        XPd, sa_true, _, _, _ = batch
        sa_pred = self.forward(XPd)
        for line in range(sa_true.size(1)):
            loss = SURVLoss(sa_true[:, line, :], sa_pred[:, line, :])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        XPd, sa_true, _, _, _ = batch
        sa_pred = self.forward(XPd)
        for line in range(sa_true.size(1)):
            loss = SURVLoss(sa_true[:, line, :], sa_pred[:, line, :])
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def predict_factual_survival(self, XPd, *_):
        sa_pred = self.forward(XPd)
        return sa_pred
    

def main():
    data_path = "../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet"

    data_config = {'data':
                {'settings': {
                        'horizon': 5,
                        'n_lines': 1,
                        'n_time_bins': 100},
                    'splits': {
                        'batch_size': 64,
                        'val_split': 0.2,
                        'test_split': 0.1}
                    }
                }

    data_module = ESMEDataModule(data_dir = data_path,
                                config = data_config,)

    logger = WandbLogger(project = "MLP_survival_model",
                        settings = wandb.Settings(
                            _disable_stats=True,
                            _disable_meta=True,
                            ))

    trainer = L.Trainer(max_epochs=1000,
                        accelerator="auto",
                        logger=logger,
                        check_val_every_n_epoch= 10,
                        callbacks=[MetricsCallback(time_bins=data_module.time_bins)],
    )

    input_dims = data_module.get_input_dimensions()
    input_dim = input_dims['x_input_dim'] + input_dims['p_input_dim'] + 1

