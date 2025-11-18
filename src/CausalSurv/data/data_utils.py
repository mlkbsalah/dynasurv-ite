import torch
import lightning as L
import torch.utils.data as TorchData
import pandas as pd
import numpy as np

from CausalSurv.tools import load_config


def stack_by_lines(df:pd.DataFrame, cols: list[str], n_lines: int) -> np.ndarray:
    """Stack variables patient-wise.

    Args:
        df (pd.DataFrame): DataFrame containing the data (number of rows should be divisible by n_lines).
        cols (list[str]): List of column names to stack.
        n_lines (int): Number of lines (timesteps) per patient.

    Returns:
        np.ndarray: Stacked array of shape (n_patients, n_lines, n_features).
    """
    
    arr = df[cols].to_numpy()
    n_patients = len(df) // n_lines
    if len(df) % n_lines != 0:
        raise ValueError("Number of rows in DataFrame must be divisible by n_lines.")
    return arr.reshape(n_patients, n_lines, len(cols))

class ESMEDataset(TorchData.Dataset):
    def __init__(self, X, P, d, time, event, n_lines, time_bins):
        """
        Args:
            X: Tensor (n_samples, n_lines, n_features_X) covariates
            P: Tensor (n_samples, n_lines, n_features_P) treatments
            d: Tensor (n_samples, n_lines, 1) buffer time between lines
            time: Tensor (n_samples, n_lines) survival times
            event: Tensor (n_samples, n_lines) event indicators (1=event, 0=censored)
            time_bins: Tensor (n_bins,) cut-points for discretization
        """
        self.X = X
        self.P = P
        self.d = d
        self.time = time
        self.event = event

        self.time_bins = time_bins
        self.n_intervals = len(time_bins) - 1
        self.n_lines = n_lines

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        p = self.P[idx]
        d = self.d[idx]
        time = self.time[idx]
        event = self.event[idx]
        treatment_index = torch.argmax(p, dim=-1) 
        sa_true = torch.zeros((self.n_lines, 2 * self.n_intervals))

        for line in range(self.n_lines):
            t = time[line]
            e = event[line]
            
            survived_to_interval = (t >= self.time_bins[:-1])
            if e == 1: 
                event_in_interval = (t >= self.time_bins[:-1]) & (t < self.time_bins[1:])
                sa_true[line, :self.n_intervals] = survived_to_interval.clone()
                sa_true[line, self.n_intervals:] = event_in_interval.clone()
            else: 
                midpoints = 0.5 * (self.time_bins[:-1] + self.time_bins[1:])
                survived_intervals_censored = (t >= midpoints)
                sa_true[line, :self.n_intervals] = survived_intervals_censored.clone()      
        
        XPd = torch.cat([x, p, d], dim=-1)
        return XPd, sa_true, treatment_index, time, event



class ESMEDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "", config: dict | str = ""):
        super().__init__()
        self.config = load_config(config)
        self.data_config = self.config['data']['settings']
        self.split_config = self.config['data']['splits']

        self.data_dir = data_dir

        self.n_lines = self.data_config['n_lines']
        self.horizon = self.data_config['horizon']
        self.n_time_bins = self.data_config['n_time_bins']
        self.n_treatments = self.data_config.get('n_treatments', None)
        self.treatment_dict = {} # map treatment index to treatment name (filled after data is loaded)

        self.time_bins = torch.linspace(0, self.horizon, self.n_time_bins + 1)

        self.batch_size = self.split_config.get('batch_size', 32)
        self.val_split = self.split_config.get('val_split', 0.2)
        self.test_split = self.split_config.get('test_split', 0.2)

    def prepare_data(self):
        esme_data = pd.read_parquet(self.data_dir)
        self.data = (
            esme_data
            .groupby('usubjid')
            .filter(lambda x: len(x) >= self.n_lines)
            .sort_values(['usubjid', 'lineid'])
            .groupby('usubjid', group_keys=False)
            .head(self.n_lines)                         
            .reset_index(drop=True)
            )

        print(f"Number of patients per n_lines={self.n_lines}: {len(self.data) // self.n_lines}")
        print(f"Total number of treatments: {self.data['T_treatment_category'].nunique()}")
        
    def setup(self, stage: str | None = None):
        if not hasattr(self, 'data'):
            self.data = pd.read_parquet(self.data_dir)
        
        X_cols = [col for col in self.data.columns if col.startswith('X_') and not col.startswith('X_buffer_time')]
        print(X_cols)
        P_cols = ["T_treatment_category"]
        d_cols = [col for col in self.data.columns if col.startswith('X_buffer_time')]
        time_cols = ['Y_onset_to_death']
        event_cols = ['Y_death']

        P_encoded = pd.get_dummies(self.data[P_cols].astype(str), prefix="T")
        self.treatment_dict = {i: col for i, col in enumerate(P_encoded.columns)}


        X = stack_by_lines(self.data, X_cols, self.n_lines) # (n_patients, n_lines, n_features_X)
        P = stack_by_lines(P_encoded, P_encoded.columns.to_list(), self.n_lines)
        d = stack_by_lines(self.data, d_cols, self.n_lines)
        time = stack_by_lines(self.data, time_cols, self.n_lines).squeeze(-1)  # (n_patients, n_lines)
        event = stack_by_lines(self.data, event_cols, self.n_lines).squeeze(-1)  # (n_patients, n_lines)

        X = torch.tensor(X, dtype=torch.float32)
        P = torch.tensor(P, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        time = torch.tensor(time, dtype=torch.float32)
        event = torch.tensor(event, dtype=torch.float32)

        ESME_Dataset = ESMEDataset(X, P, d, time, event, self.n_lines, self.time_bins)
        Train_data, Val_data, Test_data = TorchData.random_split(ESME_Dataset, lengths = [1 - self.val_split - self.test_split, self.val_split, self.test_split])
        
        if stage == 'fit' or stage is None:
            self.train_dataset = Train_data
            self.val_dataset = Val_data
            
        if stage == 'test' or stage is None:
            self.test_dataset = Test_data
    
    def train_dataloader(self):
        return TorchData.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    
    def val_dataloader(self):
        return TorchData.DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False, num_workers=4, persistent_workers=True)
    
    def test_dataloader(self):
        return TorchData.DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=4, persistent_workers=True)
    
    def get_input_dimensions(self):
        esme_data = pd.read_parquet(self.data_dir)
        x_dim = len([col for col in esme_data.columns if col.startswith('X_') and not col.startswith('X_buffer_time')])
        p_dim = len(pd.get_dummies(esme_data["T_treatment_category"].astype(str)).columns)
        n_intervals = self.n_time_bins

        return {'x_input_dim': x_dim, 'p_input_dim': p_dim, 'output_sa_length': n_intervals}
    
if __name__ == "__main__":
    data_module = ESMEDataModule(data_dir="../../../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
                                 config="../../../configs/data.toml")
    print("DataModule initialized.")
    data_module.prepare_data()
    print("Data prepared.")
    data_module.setup()
    print("DataModule setup complete.")
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        XPd, sa_true, treatment_index, time, event = batch
        print(f"XPd shape: {XPd.shape},\nsa_true shape: {sa_true.shape},\ntreatment_index shape: {treatment_index.shape},\ntime shape: {time.shape},\nevent shape: {event.shape}")
        print(f"treatment_dict: {data_module.treatment_dict}")
        break
