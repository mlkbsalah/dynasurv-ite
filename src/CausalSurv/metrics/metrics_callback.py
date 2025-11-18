import torch
import lightning as L
from CausalSurv.metrics.scoring import brier_score, concordance_index
from CausalSurv.tools.move_to_device import move_to_device

class MetricsCallback(L.Callback):
    def __init__(self, time_bins: torch.Tensor):
        super().__init__()
        self.time_bins = time_bins
        
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        val_loader = trainer.val_dataloaders
        c_index_list = []
        ibs_list = []
        
        time_bins = self.time_bins.to(pl_module.device)

        if val_loader is None:
            return
        
        with torch.no_grad():
            for batch in val_loader: 
                if len(batch) == 6:
                    XPd, _, treatment_index, time, event, mask = batch
                else:
                    XPd, _, treatment_index, time, event = batch
                    mask = None

                XPd, treatment_index, time, event = move_to_device([XPd, treatment_index, time, event], pl_module.device)
                if mask is not None:
                    mask = mask.to(pl_module.device)
                
                sa_pred = pl_module.predict_factual_survival(XPd, treatment_index, mask) # type: ignore
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
