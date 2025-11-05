import torch
import lightning as L
from CausalSurv.metrics.scoring import brier_score, integrated_brier_score, concordance_index

class MetricsCallback(L.Callback):
    def __init__(self, time_bins: torch.Tensor):
        super().__init__()
        self.time_bins = time_bins

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        all_factual_preds, all_time, all_event = [], [], []

        val_loader = trainer.val_dataloaders

        with torch.no_grad():
            for batch in val_loader: # type: ignore
                XPd, _, treatment_index, time, event = batch
                XPd = XPd.to(pl_module.device)
                treatment_index = treatment_index.to(pl_module.device)
                sa_pred, _ = pl_module(XPd)
                factual_sa_pred = pl_module._select_factual_head(sa_pred, treatment_index) # type: ignore
                all_factual_preds.append(factual_sa_pred.cpu())
                all_time.append(time.cpu())
                all_event.append(event.cpu())

        # Concatenate all batches
        factual_sa_pred = torch.cat(all_factual_preds, dim=0)  # (batch, n_lines, n_intervals)
        time = torch.cat(all_time, dim=0)                      # (batch, n_lines)
        event = torch.cat(all_event, dim=0)                    # (batch, n_lines)

        # Compute metrics
        ibs = integrated_brier_score(factual_sa_pred, time, event, self.time_bins) # (n_lines,)
        c_index = concordance_index(factual_sa_pred, time, event, self.time_bins)  # (n_lines,) # what passes shouldb e(batch, n_lines, n_intervals)

        # Log metrics
        ibs_log = {f"IBS/line_{i}": ibs[i].item() for i in range(ibs.shape[0])}
        c_index_log = {f"C_Index/line_{i}": c_index[i].item() for i in range(c_index.shape[0])}
        if trainer.logger is not None:
            trainer.logger.log_metrics({**ibs_log,
                                        **c_index_log}, step=trainer.global_step)
