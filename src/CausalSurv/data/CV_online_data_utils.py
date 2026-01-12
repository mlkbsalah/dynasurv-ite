from typing import Tuple

import torch
import lightning as L
import torch.utils.data as TorchData

import pandas as pd
import numpy as np
from CausalSurv.tools import load_config
from sklearn.model_selection import KFold

def stack_by_lines(df: pd.DataFrame, cols: list[str]) -> list[np.ndarray]:
    """Stack variables patient-wise, variable number of lines per patient.   list[np.ndarray(patient_i_lines, len(cols))]"""
    grouped = df.groupby('usubjid')[cols]
    return [group.to_numpy() for _, group in grouped]

class ESMEOnlineDataset(TorchData.Dataset):
    def __init__(self, X_list, X_static, P_list, P_static, d_list, time_list, event_list, n_lines: int, interval_bounds: torch.Tensor):
        """Class constructor for ESME dataset

        Args:
            X_list (list[np.ndarray]):   one per patient (each of shape [n_lines_i, n_features])
            P_list (list[np.ndarray]):   one per patient (each of shape [n_lines_i, n_treatments])
            d_list (list[np.ndarray]):   one per patient (each of shape [n_lines_i, 1]) buffer time between lines
            time_list (list[np.ndarray]): one per patient (each of shape [n_lines_i, 1]) event times
            event_list (list[np.ndarray]): one per patient (each of shape [n_lines_i, 1]) event indicators
            time_bins (torch.Tensor):    tensor of shape [n_intervals + 1] defining the time intervals
            n_lines (int):             maximum number of lines to pad to
        
        Remarks:
            Each patient sample contains all their lines, padded to n_lines.  # ← CHANGED
        """
        self.X_list = X_list
        self.X_static = X_static
        self.P_list = P_list
        self.P_static = P_static
        self.d_list = d_list
        self.time_list = time_list
        self.event_list = event_list

        self.interval_bounds = interval_bounds
        self.n_lines = n_lines

        self.n_intervals = len(interval_bounds) - 1

    def __len__(self) -> int:
        return len(self.X_list)
    
    def transform_target_time(self, time: torch.Tensor) -> torch.Tensor:
        interval_idx = torch.bucketize(time, self.interval_bounds) - 1  # (n_lines,)
        interval_idx = torch.clamp(interval_idx, 0, self.n_intervals - 1)
        if interval_idx < 0 or interval_idx >= self.n_intervals:
            raise ValueError(f"Time value out of bounds of interval_bounds: time={time}, bounds={self.interval_bounds}")
        return interval_idx
    
    def _pad_sequence(self, seq: np.ndarray, target_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_lentgh, seq_dim = seq.shape
        
        padded_seq = torch.zeros((target_length, seq_dim), dtype=torch.float32)
        mask = torch.zeros((target_length,), dtype=torch.long)

        padded_seq[:seq_lentgh] = torch.tensor(seq, dtype=torch.float32)
        mask[:seq_lentgh] = 1
        
        return padded_seq, mask

    def __getitem__(self, idx):
        x = self.X_list[idx]  # (n_lines_i, n_features)
        x_static = self.X_static[idx]  # (n_features_static,)
        p = self.P_list[idx]  # (n_lines_i, n_treatments)
        p_static = self.P_static[idx]  # (n_treatments_static,)
        d = self.d_list[idx]  # (n_lines_i, 1)
        time = self.time_list[idx]  # (n_lines_i, 1)
        event = self.event_list[idx]  # (n_lines_i, 1)
        
        x_padded, mask = self._pad_sequence(x, self.n_lines)
        p_padded = self._pad_sequence(p, self.n_lines)[0]
        d_padded = self._pad_sequence(d, self.n_lines)[0]
        time_padded = self._pad_sequence(time, self.n_lines)[0]
        event_padded = self._pad_sequence(event, self.n_lines)[0]
        
        treatment_indices = torch.argmax(p_padded, dim=-1)  # (n_lines,)
        
        interval_idx = torch.tensor([self.transform_target_time(t) for t in time_padded])  # (n_lines,)
        
        XPd = torch.cat([x_padded, p_padded, d_padded], dim=-1)

        return XPd, (x_static, p_static), interval_idx, treatment_indices, time_padded.squeeze(), event_padded.squeeze(), mask.squeeze()


    
class ESMEOnlineDataModuleCV(L.LightningDataModule):
    def __init__(self, 
                 data_dir: str, 
                 n_lines: int, 
                 n_intervals: int,
                 batch_size: int,
                 fold_idx: int | None = None,
                 num_folds: int | None = None,
                 split_seed: int | None = None,
                 holdout_size: float = 0.1,
                 num_workers: int = 4
                 ):
        super().__init__()
        self.data_dir = data_dir

        self.n_lines = n_lines
        self.n_intervals = n_intervals
        self.treatment_dict = {} 

        self.batch_size = batch_size
        self.fold_idx = fold_idx
        self.num_folds = num_folds
        self.split_seed = split_seed
        self.holdout_size = holdout_size
        self.num_workers = num_workers

        self.data = None
        self.ESMEDataset = None
        self.static_data = None
        self.interval_bounds = None

    def prepare_data(self):
        esme_data = pd.read_parquet(self.data_dir)
        # static_data = pd.read_parquet("/workdir/bensalama/DynaSurv/data/model_entry_imputes_data_STATIC_no_staging.parquet")
        static_data = pd.read_parquet("/Users/malek/TheLAB/DynaSurv/data/model_entry_imputes_data_STATIC_no_staging.parquet")  # --- IGNORE ---

        merged = esme_data.merge(
            static_data,
            on='usubjid',
            how='inner',
            suffixes=('', '_static')
        )
        self.data = merged[esme_data.columns]
        self.data = self.data.loc[self.data['lineid'] <= self.n_lines].sort_values(by=['usubjid', 'lineid']).reset_index(drop=True).copy()
        self.static_data = merged[static_data.columns].drop_duplicates("usubjid").reset_index(drop=True).copy()
        
        X_cols = [col for col in self.data.columns if col.startswith('X_') and not col.startswith('X_buffer_time')]
        P_cols = ["T_treatment_category"]
        d_cols = ['X_buffer_time']
        time_cols = ['Y_onset_to_death']
        event_cols = ['Y_death']
        X_static_cols = [col for col in self.static_data.columns if col.startswith('X_')]
        P_static_cols = [col for col in self.static_data.columns if col.startswith('T_')]

        P_encoded = pd.get_dummies(self.data[P_cols + ['usubjid']].astype(str), columns=P_cols, prefix="T") 
        self.treatment_dict = {i: col for i, col in enumerate(P_encoded.columns)}


        X_list = stack_by_lines(self.data, X_cols)
        P_list = stack_by_lines(P_encoded, P_encoded.columns.drop('usubjid').tolist())
        d_list = stack_by_lines(self.data, d_cols)
        time_list = stack_by_lines(self.data, time_cols)
        event_list = stack_by_lines(self.data, event_cols)

        X_static = torch.tensor(self.static_data[X_static_cols].values, dtype=torch.float32)
        P_static = torch.tensor(self.static_data[P_static_cols].values, dtype=torch.float32)

        self.interval_bounds = torch.linspace(0, self.data['Y_onset_to_death'].max(), self.n_intervals + 1)


        self.ESMEDataset = ESMEOnlineDataset(X_list, X_static, P_list, P_static, d_list, time_list, event_list, self.n_lines, self.interval_bounds)

        print(f"Loaded {len(self.data)} rows, {self.data['usubjid'].nunique()} patients.")


    def setup(self, stage: str | None = None):
        if self.ESMEDataset is None:
            self.prepare_data()
        assert self.ESMEDataset is not None, "ESMEDataset must be initialized before setup."
        assert self.fold_idx is not None and self.num_folds is not None and self.split_seed is not None, "fold_idx, num_folds, and split_seed must be provided for cross-validation."

        total_size = len(self.ESMEDataset)
        holdout_size = int(self.holdout_size * total_size)
        cv_size = total_size - holdout_size
        generator = torch.Generator().manual_seed(self.split_seed)
        self.cv_dataset, self.holdout_dataset = TorchData.random_split(self.ESMEDataset, [cv_size, holdout_size], generator=generator)
        


        if stage == 'fit' or stage is None:
            kfold = KFold(n_splits=self.num_folds or 5, shuffle=True, random_state=self.split_seed)
            all_splits = [k for k in kfold.split(range(len(self.ESMEDataset)))] # type: ignore
            train_idx, val_idx = all_splits[self.fold_idx]
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
            
            self.train_dataset = TorchData.Subset(self.ESMEDataset, train_idx)
            self.val_dataset = TorchData.Subset(self.ESMEDataset, val_idx)

        if stage == 'test' or stage is None:
            self.test_dataset = self.holdout_dataset


    def train_dataloader(self):
        return TorchData.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return TorchData.DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False, num_workers=1, persistent_workers=True)

    def test_dataloader(self):
        return TorchData.DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=1, persistent_workers=False)

    def get_data_dimensions(self):
        esme_data = pd.read_parquet(self.data_dir)
        # static_data = pd.read_parquet("/workdir/bensalama/DynaSurv/data/model_entry_imputes_data_STATIC_no_staging.parquet")
        static_data = pd.read_parquet("/Users/malek/TheLAB/DynaSurv/data/model_entry_imputes_data_STATIC_no_staging.parquet")  # --- IGNORE ---
        static_data = static_data.loc[static_data['usubjid'].isin(esme_data['usubjid'].unique())].reset_index(drop=True)
        
        x_dim = len([col for col in esme_data.columns if col.startswith('X_') and not col.startswith('X_buffer_time')])
        p_dim = len(pd.get_dummies(esme_data["T_treatment_category"].astype(str)).columns)
        p_static_dim = len([col for col in static_data.columns if col.startswith('T_')])
        x_static_dim = len([col for col in static_data.columns if col.startswith('X_')])
        interval_bounds = torch.linspace(0, esme_data['Y_onset_to_death'].max(), self.n_intervals + 1)


        return {'x_input_dim': x_dim,
                'p_input_dim': p_dim,
                'p_static_dim': p_static_dim,
                'x_static_dim': x_static_dim,
                'output_dim': self.n_intervals,
                'time_bins': interval_bounds}

if __name__ == "__main__":
    import time 
    data_module = ESMEOnlineDataModuleCV(data_dir="../../../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
                                       n_lines=2,
                                       n_intervals=10,
                                       batch_size=32,
                                       fold_idx = 3,
                                       num_folds=10,
                                       split_seed=12345,
                                       holdout_size=0.1,
                                       num_workers=4
                                       )
    print("DataModule initialized.")
    start = time.time()
    data_module.prepare_data()
    print(f"Data prepared in {time.time() - start:.2f} seconds.")
    
    start = time.time()
    data_module.setup()
    print(f"DataModule setup complete in {time.time() - start:.2f} seconds.")
    
    start = time.time()
    train_loader = data_module.train_dataloader()
    print(f"Train DataLoader created in {time.time() - start:.2f} seconds.")

    start = time.time()
    val_loader = data_module.val_dataloader()
    print(f"Validation DataLoader created in {time.time() - start:.2f} seconds.")

    start = time.time()
    test_loader = data_module.test_dataloader()
    print(f"Test DataLoader created in {time.time() - start:.2f} seconds.")
    
    start = time.time()
    batch = next(iter(train_loader))
    print(f"First batch retrieved in {time.time() - start:.2f} seconds.")
    print("Batch contents:")
    XPd, (x_static, p_static), interval_idx, treatment_indices, time, event, mask = batch
    print("Batch XPd shape:", XPd.shape, "dtype:", XPd.dtype)
    print("Batch x_static shape:", x_static.shape, "dtype:", x_static.dtype)
    print("Batch p_static shape:", p_static.shape, "dtype:", p_static.dtype)
    print("Batch interval_idx shape:", interval_idx.shape, "dtype:", interval_idx.dtype)
    print("Batch treatment_indices shape:", treatment_indices.shape, "dtype:", treatment_indices.dtype)
    print("Batch time shape:", time.shape, "dtype:", time.dtype)
    print("Batch event shape:", event.shape, "dtype:", event.dtype)
    print("Batch mask shape:", mask.shape, "dtype:", mask.dtype)
