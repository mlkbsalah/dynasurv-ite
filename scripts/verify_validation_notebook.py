import tomllib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import lightning as L
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv
from torchsurv.metrics.brier_score import BrierScore

from CausalSurv.model.DynaSurvCausalOnline import DynaSurvCausalOnline
from CausalSurv.data.CV_online_data_utils import ESMEOnlineDataModuleCV

# --- Configuration ---
MODEL_PATH = "/Users/malek/TheLAB/DynaSurv/models/HR+HER2-/4lines/seed_1768340925/checkpoints/dynaSurvCausalOnline-epoch=07-average_ci=0.7227.ckpt"
DATA_PATH = "/Users/malek/TheLAB/DynaSurv/data"
SUBTYPE = "HR+HER2-"
N_LINES = 4
BATCH_SIZE = 16 # Small for test

def run_verification():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = DynaSurvCausalOnline.load_from_checkpoint(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
        model.freeze()
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Loading data...")
    data_module = ESMEOnlineDataModuleCV(
        data_dir=DATA_PATH,
        subtype=SUBTYPE,
        n_intervals=len(model.interval_bounds) - 1,
        n_lines=N_LINES,
        batch_size=BATCH_SIZE,
        final_training=True,
        num_workers=4,
        split_seed=1768340925
    )
    data_module.prepare_data()
    data_module.setup()
    
    # KM Estimator Check
    train_loader = data_module.train_dataloader()
    train_times = [[] for _ in range(N_LINES)]
    train_events = [[] for _ in range(N_LINES)]
    
    print("Accumulating training data (first 5 batches only for speed)...")
    for i, batch in enumerate(train_loader):
        if i > 5: break
        XPd, X_static, interval_idx, treatment_indices, time, event, mask, patient_id = batch
        for line in range(N_LINES):
            valid = mask[:, line].bool()
            if valid.any():
                train_times[line].extend(time[valid, line].numpy())
                train_events[line].extend(event[valid, line].numpy())
                
    kmf_list = []
    print("Fitting KM...")
    for line in range(N_LINES):
        if len(train_times[line]) > 0:
            y_surv = Surv.from_arrays(np.array(train_events[line], dtype=bool), np.array(train_times[line]))
            kmf = CensoringDistributionEstimator().fit(y_surv)
            kmf_list.append(kmf)
        else:
            kmf_list.append(None)
            print(f"Warning: No data for line {line}")

    # Inference Check
    test_loader = data_module.test_dataloader()
    test_interval_bounds = model.interval_bounds.to('cpu')
    
    print("Running Inference (first batch only)...")
    with torch.no_grad():
        for batch in test_loader:
            XPd, X_static, interval_idx, treatment_indices, time, event, mask, patient_id = batch
            
            hazards_logit, _ = model.get_factual_predictions(XPd, X_static, treatment_indices)
            hazards_factual = torch.sigmoid(hazards_logit)
            
            hazards_at_event = model.predict_hazard(XPd, X_static, time, treatment_indices)
            survival_grid = torch.cumprod(1 - hazards_factual, dim=2)
            
            # Check Metrics calculation
            for line in range(N_LINES):
                valid_mask = mask[:, line].bool()
                if not valid_mask.any(): continue
                
                t_line = time[valid_mask, line].cpu()
                e_line = event[valid_mask, line].bool().cpu()
                h_line = hazards_at_event[valid_mask, line].cpu()
                s_grid_line = survival_grid[valid_mask, line, :].cpu()
                
                # C-Index
                try:
                    c = concordance_index_censored(e_line.numpy(), t_line.numpy(), h_line.numpy())[0]
                    print(f"Line {line}: C-index = {c}")
                except Exception as e:
                    print(f"Line {line}: C-index error: {e}")

                # Brier
                if kmf_list[line] is not None:
                    try:
                        y_surv_batch = Surv.from_arrays(e_line, t_line)
                        weights = kmf_list[line].predict_ipcw(y_surv_batch)
                        bs = BrierScore()(s_grid_line, e_line, t_line, test_interval_bounds[:-1], weight_new_time=torch.tensor(weights))
                        print(f"Line {line}: Mean Brier = {torch.mean(bs).item()}")
                    except Exception as e:
                        print(f"Line {line}: Brier error: {e}")
            break # Only one batch

    print("Verification script finished successfully.")

if __name__ == "__main__":
    run_verification()
