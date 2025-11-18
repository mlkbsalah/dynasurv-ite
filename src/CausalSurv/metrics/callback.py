import torch
import lightning as L
from CausalSurv.metrics.scoring import brier_score, concordance_index

class MetricsCallback(L.Callback):
    def __init__(self, time_bins: torch.Tensor):
        super().__init__()
        self.time_bins = time_bins


    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        val_loader = trainer.val_dataloaders
        c_index_list = []
        ibs_list = []
        
        # Move time_bins to the correct device
        time_bins = self.time_bins.to(pl_module.device)
        
        with torch.no_grad():
            for batch in val_loader: # type: ignore
                XPd, _, treatment_index, time, event, mask = batch
                XPd = XPd.to(pl_module.device)
                
                time = time.to(pl_module.device)
                event = event.to(pl_module.device)
                mask = mask.to(pl_module.device)
                
                sa_pred = pl_module(XPd)
                _, ibs = brier_score(sa_pred, time, event, time_bins, mask)
                c_index = concordance_index(sa_pred, time, event, time_bins, mask)
                
                c_index_list.append(c_index.cpu())
                ibs_list.append(ibs.cpu())

        # Handle batch size of 1
        if len(c_index_list) == 1:
            avg_c_index = c_index_list[0]
            avg_ibs = ibs_list[0]
        else:
            avg_c_index = torch.cat(c_index_list).mean(dim=0)
            avg_ibs = torch.cat(ibs_list).mean(dim=0)

        ibs_log = {f'val/ibs_line_{i}': avg_ibs[i].item() for i in range(avg_ibs.shape[0])}
        cindex_log = {f'val/c_index_line_{i}': avg_c_index[i].item() for i in range(avg_c_index.shape[0])}
        
        if trainer.logger is not None:
            trainer.logger.log_metrics({**ibs_log, **cindex_log}, step=trainer.global_step)
