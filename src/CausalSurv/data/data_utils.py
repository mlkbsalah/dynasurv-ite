import torch
import lightning as L
import torch.utils.data as TorchData
import pandas as pd
import numpy as np

from CausalSurv.tools import load_config


def stack_by_lines(df:pd.DataFrame, cols: list[str], n_lines: int) -> torch.Tensor:
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
    return torch.tensor(arr.reshape(n_patients, n_lines, len(cols)), dtype=torch.float32)

class ESMEDataset(TorchData.Dataset):
    def __init__(self, X: torch.Tensor, P: torch.Tensor, d: torch.Tensor, time: torch.Tensor, event: torch.Tensor, n_lines: int, interval_bounds: torch.Tensor) -> None:
        """
        Args:
            X (torch.Tensor): Covariate tensor of shape (n_patients, n_lines, n_features_X).
            P (torch.Tensor): Treatment assignment tensor of shape (n_patients, n_lines, n_treatments).
            d (torch.Tensor): Buffer time tensor of shape (n_patients, n_lines, 1).
            time (torch.Tensor): Time-to-event tensor of shape (n_patients, n_lines).
            event (torch.Tensor): Event indicator tensor of shape (n_patients, n_lines).
            n_lines (int): Number of lines (timesteps) per patient.
            interval_bounds (torch.Tensor): 1D tensor defining the bounds of time intervals for discretization.
        """
        super().__init__()
        self.X = X
        self.P = P
        self.d = d
        self.time = time
        self.event = event

        self.interval_bounds = interval_bounds
        self.n_lines = n_lines
        
        self.n_intervals = len(interval_bounds) - 1

    def __len__(self):
        return len(self.X)
    
    def transform_target_time(self, time):
        interval_idx = torch.searchsorted(self.interval_bounds, torch.tensor([time], dtype=torch.float32), right=True) - 1
        interval_idx = torch.clamp(interval_idx, 0, self.n_intervals - 1)
        if interval_idx < 0 or interval_idx >= self.n_intervals:
            raise ValueError(f"Time value out of bounds of interval_bounds: time={time}, bounds={self.interval_bounds}")
        return interval_idx
    
    def __getitem__(self, idx):
        x = self.X[idx]
        p = self.P[idx]
        d = self.d[idx]
        XPd = torch.cat([x, p, d], dim=-1)

        time = self.time[idx]
        event = self.event[idx]

        treatment_index = torch.argmax(p, dim=-1) 
        
        interval_idx = torch.tensor([self.transform_target_time(t.item()) for t in time]) 

        return XPd, interval_idx, treatment_index, time, event

class ESMEDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 n_lines: int,
                 n_intervals: int,
                 train_batch_size: int,
                 val_split : float = 0.2,
                 test_split : float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir

        self.n_lines = n_lines
        self.n_intervals = n_intervals
        self.treatment_dict = {} 

        self.train_batch_size = train_batch_size
        self.val_split = val_split
        self.test_split = test_split

        self.data = None

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
        if self.data is None:
            self.prepare_data()
        assert type(self.data) == pd.DataFrame, "Data not prepared. Please run prepare_data() before setup()."
        
        X_cols = [col for col in self.data.columns if col.startswith('X_') and not col.startswith('X_buffer_time')]
        P_cols = ["T_treatment_category"]
        d_cols = ['X_buffer_time']
        time_cols = ['Y_onset_to_death']
        event_cols = ['Y_death']

        P_encoded = pd.get_dummies(self.data[P_cols].astype(str), prefix="T")
        self.treatment_dict = {i: col for i, col in enumerate(P_encoded.columns)}

        X = stack_by_lines(self.data, X_cols, self.n_lines) 
        P = stack_by_lines(P_encoded, P_encoded.columns.to_list(), self.n_lines)
        d = stack_by_lines(self.data, d_cols, self.n_lines)
        time = stack_by_lines(self.data, time_cols, self.n_lines).squeeze(-1) 
        event = stack_by_lines(self.data, event_cols, self.n_lines).squeeze(-1)

        
        self.interval_bounds = torch.linspace(0, time.max(), self.n_intervals + 1)

        ESME_Dataset = ESMEDataset(X, P, d, time, event, self.n_lines, self.interval_bounds)
        Train_data, Val_data, Test_data = TorchData.random_split(ESME_Dataset, lengths = [1 - self.val_split - self.test_split, self.val_split, self.test_split])
        
        if stage == 'fit' or stage is None:
            self.train_dataset = Train_data
            self.val_dataset = Val_data 
        if stage == 'test' or stage is None:
            self.test_dataset = Test_data

        return ESME_Dataset
 
    def train_dataloader(self):
        return TorchData.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    
    def val_dataloader(self):
        return TorchData.DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False, num_workers=4, persistent_workers=True)
    
    def test_dataloader(self):
        return TorchData.DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=4, persistent_workers=True)
    
    def get_input_dimensions(self):
        esme_data = pd.read_parquet(self.data_dir)
        x_dim = len([col for col in esme_data.columns if col.startswith('X_') and not col.startswith('X_buffer_time')])
        p_dim = len(pd.get_dummies(esme_data["T_treatment_category"].astype(str)).columns)
        time = esme_data['Y_onset_to_death']
        interval_bounds = torch.linspace(0, time.max(), self.n_intervals + 1)


        return {'x_input_dim': x_dim, 'p_input_dim': p_dim, 'output_dim': self.n_intervals, 'time_bins': interval_bounds}
    
if __name__ == "__main__":
    data_module = ESMEDataModule(data_dir="../../../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
                                 n_lines=1,
                                 n_intervals=10,
                                 train_batch_size=32,
                                 val_split=0.2,
                                 test_split=0.1
                                 )
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
