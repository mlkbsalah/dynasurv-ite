import torch
import lightning as L
import torch.utils.data as TorchData
from CausalSurv.data.config_loader import load_config
import pandas as pd

def stack_by_lines(df: pd.DataFrame, cols: list[str]):
    """Stack variables patient-wise, variable number of lines per patient."""
    patients = df['usubjid'].unique()
    stacked = []
    lengths = []
    for pid in patients:
        arr = df.loc[df['usubjid'] == pid, cols].to_numpy()
        stacked.append(arr)
        lengths.append(len(arr))
    return stacked, lengths


class ESMEDataset(TorchData.Dataset):
    def __init__(self, X_list, P_list, d_list, time_list, event_list, time_bins, max_lines):
        """Class constructor for ESME dataset

        Args:
            X_list (list[np.ndarray]):   one per patient (each of shape [n_lines_i, n_features])
            P_list (list[np.ndarray]):   one per patient (each of shape [n_lines_i, n_treatments])
            d_list (list[np.ndarray]):   one per patient (each of shape [n_lines_i, 1]) buffer time between lines
            time_list (list[np.ndarray]): one per patient (each of shape [n_lines_i, 1]) event times
            event_list (list[np.ndarray]): one per patient (each of shape [n_lines_i, 1]) event indicators
            time_bins (torch.Tensor):    tensor of shape [n_intervals + 1] defining the time intervals
            max_lines (int):             maximum number of lines to pad to
        
        Remarks:
            Each patient sample contains all their lines, padded to max_lines.  # ← CHANGED
        """
        self.X_list = X_list
        self.P_list = P_list
        self.d_list = d_list
        self.time_list = time_list
        self.event_list = event_list
        self.time_bins = time_bins
        self.n_intervals = len(time_bins) - 1
        self.max_lines = max_lines

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_list[idx], dtype=torch.float32)  # (n_lines, n_features)
        p = torch.tensor(self.P_list[idx], dtype=torch.float32)  # (n_lines, n_treatments)
        d = torch.tensor(self.d_list[idx], dtype=torch.float32)  # (n_lines, 1)
        time = torch.tensor(self.time_list[idx], dtype=torch.float32)  # (n_lines, 1)
        event = torch.tensor(self.event_list[idx], dtype=torch.float32)  # (n_lines, 1)

        n_lines = len(x)
        
        # Pad sequences to max_lines
        n_features = x.shape[1]
        n_treatments = p.shape[1]
        
        x_padded = torch.zeros((self.max_lines, n_features), dtype=torch.float32)
        p_padded = torch.zeros((self.max_lines, n_treatments), dtype=torch.float32)
        d_padded = torch.zeros((self.max_lines, 1), dtype=torch.float32)
        time_padded = torch.zeros((self.max_lines, 1), dtype=torch.float32)
        event_padded = torch.zeros((self.max_lines, 1), dtype=torch.float32)

        x_padded[:n_lines] = x
        p_padded[:n_lines] = p
        d_padded[:n_lines] = d
        time_padded[:n_lines] = time
        event_padded[:n_lines] = event
        
        # Create mask (1 for real lines, 0 for padding)
        mask = torch.zeros(self.max_lines, dtype=torch.float32)
        mask[:n_lines] = 1.0
        
        # Treatment indices for each line
        treatment_indices = torch.argmax(p_padded, dim=-1)  # (max_lines,)
        
        # Create survival/event targets for each line
        sa_true = torch.zeros((self.max_lines, 2 * self.n_intervals), dtype=torch.float32)
        
        for line in range(n_lines):
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
        
        XPd = torch.cat([x_padded, p_padded, d_padded], dim=-1)
        return XPd, sa_true, treatment_indices, time_padded.squeeze(), event_padded.squeeze(), mask


class ESMEDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "", config: dict | str = ""):
        super().__init__()
        self.config = load_config(config)
        self.data_config = self.config['data']['settings']
        self.split_config = self.config['data']['splits']

        self.data_dir = data_dir
        self.horizon = self.data_config['horizon']
        self.n_time_bins = self.data_config['n_time_bins']
        self.n_lines = self.data_config['n_lines']
        self.time_bins = torch.linspace(0, self.horizon, self.n_time_bins + 1)

        self.batch_size = self.split_config.get('batch_size', 32)
        self.val_split = self.split_config.get('val_split', 0.2)
        self.test_split = self.split_config.get('test_split', 0.2)

        self.treatment_dict = {}

    def prepare_data(self):
        esme_data = pd.read_parquet(self.data_dir)
        self.data = esme_data.loc[esme_data['lineid'] <= self.n_lines].copy()
        print(f"Loaded {len(self.data)} rows, {self.data['usubjid'].nunique()} patients.")

    def setup(self, stage: str | None = None):
        if not hasattr(self, 'data'):
            self.prepare_data()

        # Define feature, treatment, duration, and event columns
        X_cols = [col for col in self.data.columns if col.startswith('X_') and not col.startswith('X_buffer_time')]
        P_cols = ["T_treatment_category"]
        d_cols = [col for col in self.data.columns if col.startswith('X_buffer_time')]
        time_cols = ['Y_onset_to_death']
        event_cols = ['Y_death']

        P_encoded = pd.get_dummies(self.data[P_cols].astype(str), prefix="T")
        self.treatment_dict = {i: col for i, col in enumerate(P_encoded.columns)}

        X_list, _ = stack_by_lines(self.data, X_cols)
        P_list, _ = stack_by_lines(P_encoded.join(self.data['usubjid']), P_encoded.columns.to_list())
        d_list, _ = stack_by_lines(self.data, d_cols)
        time_list, _ = stack_by_lines(self.data, time_cols)
        event_list, _ = stack_by_lines(self.data, event_cols)

        # Split by patient indices
        n_patients = len(X_list)
        patient_indices = torch.randperm(n_patients).tolist()
        n_test = int(n_patients * self.test_split)
        n_val = int(n_patients * self.val_split)

        test_pids = patient_indices[:n_test]
        val_pids = patient_indices[n_test:n_test+n_val]
        train_pids = patient_indices[n_test+n_val:]

        self.train_dataset = ESMEDataset(
            [X_list[i] for i in train_pids],
            [P_list[i] for i in train_pids],
            [d_list[i] for i in train_pids],
            [time_list[i] for i in train_pids],
            [event_list[i] for i in train_pids],
            self.time_bins,
            self.n_lines
        )

        self.val_dataset = ESMEDataset(
            [X_list[i] for i in val_pids],
            [P_list[i] for i in val_pids],
            [d_list[i] for i in val_pids],
            [time_list[i] for i in val_pids],
            [event_list[i] for i in val_pids],
            self.time_bins,
            self.n_lines
        )

        self.test_dataset = ESMEDataset(
            [X_list[i] for i in test_pids],
            [P_list[i] for i in test_pids],
            [d_list[i] for i in test_pids],
            [time_list[i] for i in test_pids],
            [event_list[i] for i in test_pids],
            self.time_bins,
            self.n_lines
        )

        print(f"Train patients: {len(self.train_dataset)},\nVal patients: {len(self.val_dataset)},\nTest patients: {len(self.test_dataset)}")


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
    data_module = ESMEDataModule(
        data_dir="../../../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
        config="../../../configs/data.toml"
    )
    print("DataModule initialized.")
    data_module.prepare_data()
    print("Data prepared.")
    data_module.setup()
    print("DataModule setup complete.")

    train_loader = data_module.train_dataloader()
    for i, batch in enumerate(train_loader):
        XPd, sa_true, treatment_index, time, event, mask = batch
        # print(mask)
        print(f"Batch {i}:")
        print(f"XPd: {XPd.shape}, sa_true: {sa_true.shape}, treatment index: {treatment_index.shape}, time: {time.shape}, event: {event.shape}, mask: {mask.shape}")
        # print(f"treatment_dict: {data_module.treatment_dict}")
        if i ==1 :
            break