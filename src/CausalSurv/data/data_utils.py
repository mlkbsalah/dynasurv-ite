import torch
import lightning as L
import torch.utils.data as TorchData
import pandas as pd


def stack_by_lines(df: pd.DataFrame, cols: list[str], n_lines: int) -> torch.Tensor:
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
    return torch.tensor(
        arr.reshape(n_patients, n_lines, len(cols)), dtype=torch.float32
    )


class ESMEDataset(TorchData.Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        X_static: torch.Tensor,
        P: torch.Tensor,
        P_static: torch.Tensor,
        d: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
        n_lines: int,
        interval_bounds: torch.Tensor,
    ) -> None:
        """
        Args:
            X (torch.Tensor): Covariate tensor of shape (n_patients, n_lines, n_features_X).
            X_static (torch.Tensor): Static covariate tensor of shape (n_patients, n_features_static).
            P (torch.Tensor): Treatment assignment tensor of shape (n_patients, n_lines, n_treatments).
            P_static (torch.Tensor): Static treatment assignment tensor of shape (n_patients, n_features_static_treatment).
            d (torch.Tensor): Buffer time tensor of shape (n_patients, n_lines, 1).
            time (torch.Tensor): Time-to-event tensor of shape (n_patients, n_lines).
            event (torch.Tensor): Event indicator tensor of shape (n_patients, n_lines).
            n_lines (int): Number of lines (timesteps) per patient.
            interval_bounds (torch.Tensor): 1D tensor defining the bounds of time intervals for discretization.
        """
        super().__init__()
        self.X = X
        self.X_static = X_static
        self.P = P
        self.P_static = P_static
        self.d = d
        self.time = time
        self.event = event

        self.interval_bounds = interval_bounds
        self.n_lines = n_lines

        self.n_intervals = len(interval_bounds) - 1

    def __len__(self):
        return len(self.X)

    def transform_target_time(self, time):
        interval_idx = (
            torch.searchsorted(
                self.interval_bounds,
                torch.tensor([time], dtype=torch.float32),
                right=True,
            )
            - 1
        )
        interval_idx = torch.clamp(interval_idx, 0, self.n_intervals - 1)
        if interval_idx < 0 or interval_idx >= self.n_intervals:
            raise ValueError(
                f"Time value out of bounds of interval_bounds: time={time}, bounds={self.interval_bounds}"
            )
        return interval_idx

    def __getitem__(self, idx):
        x = self.X[idx]
        x_static = self.X_static[idx]
        p = self.P[idx]
        p_static = self.P_static[idx]
        d = self.d[idx]
        XPd = torch.cat([x, p, d], dim=-1)

        time = self.time[idx]
        event = self.event[idx]

        treatment_index = torch.argmax(p, dim=-1)
        interval_idx = torch.tensor(
            [self.transform_target_time(t.item()) for t in time]
        )

        return XPd, (x_static, p_static), interval_idx, treatment_index, time, event


class ESMEDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        n_lines: int,
        n_intervals: int,
        train_batch_size: int,
        val_split: float = 0.2,
        test_split: float = 0.1,
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
        static_data = pd.read_parquet(
            "/Users/malek/TheLAB/DynaSurv/data/model_entry_imputes_data_STATIC_no_staging.parquet"
        )
        merged = esme_data.merge(
            static_data, on="usubjid", how="inner", suffixes=("", "_static")
        )

        merged = (
            merged.groupby("usubjid")
            .filter(lambda x: len(x) >= self.n_lines)
            .sort_values(["usubjid", "lineid"])
        )

        merged = (
            merged.groupby("usubjid", group_keys=False)
            .head(self.n_lines)
            .reset_index(drop=True)
        )
        self.data = merged[esme_data.columns]  # dynamic columns only

        static_columns = static_data.columns
        self.static_data = (
            merged[static_columns]
            .drop_duplicates("usubjid")  # one static row per patient
            .reset_index(drop=True)
        )

        patients_dynamic = self.data["usubjid"].unique()
        patients_static = self.static_data["usubjid"].unique()
        assert (patients_dynamic == patients_static).all(), (
            "Dynamic and static patients are not aligned!"
        )

        print(f"Number of aligned patients: {len(patients_dynamic)}")
        print(
            f"Total number of treatments: {self.data['T_treatment_category'].nunique()}"
        )

    def setup(self, stage: str | None = None):
        if self.data is None:
            self.prepare_data()
        assert isinstance(self.data, pd.DataFrame), (
            "Data not prepared. Please run prepare_data() before setup()."
        )

        X_cols = [
            col
            for col in self.data.columns
            if col.startswith("X_") and not col.startswith("X_buffer_time")
        ]
        P_cols = ["T_treatment_category"]
        d_cols = ["X_buffer_time"]
        time_cols = ["Y_onset_to_death"]
        event_cols = ["Y_death"]
        X_static_cols = [
            col for col in self.static_data.columns if col.startswith("X_")
        ]
        P_static_cols = [
            col for col in self.static_data.columns if col.startswith("T_")
        ]

        P_encoded = pd.get_dummies(self.data[P_cols].astype(str), prefix="T")
        self.treatment_dict = {i: col for i, col in enumerate(P_encoded.columns)}

        X = stack_by_lines(self.data, X_cols, self.n_lines)
        P = stack_by_lines(P_encoded, P_encoded.columns.to_list(), self.n_lines)
        d = stack_by_lines(self.data, d_cols, self.n_lines)
        time = stack_by_lines(self.data, time_cols, self.n_lines).squeeze(-1)
        event = stack_by_lines(self.data, event_cols, self.n_lines).squeeze(-1)

        self.interval_bounds = torch.linspace(0, time.max(), self.n_intervals + 1)

        X_static = torch.tensor(
            self.static_data[X_static_cols].values, dtype=torch.float32
        )
        P_static = torch.tensor(
            self.static_data[P_static_cols].values, dtype=torch.float32
        )

        ESME_Dataset = ESMEDataset(
            X, X_static, P, P_static, d, time, event, self.n_lines, self.interval_bounds
        )
        Train_data, Val_data, Test_data = TorchData.random_split(
            ESME_Dataset,
            lengths=[
                1 - self.val_split - self.test_split,
                self.val_split,
                self.test_split,
            ],
        )

        if stage == "fit" or stage is None:
            self.train_dataset = Train_data
            self.val_dataset = Val_data
        if stage == "test" or stage is None:
            self.test_dataset = Test_data

        return ESME_Dataset

    def train_dataloader(self):
        return TorchData.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return TorchData.DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=1,
            persistent_workers=False,
        )

    def test_dataloader(self):
        return TorchData.DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False,
            num_workers=1,
            persistent_workers=False,
        )

    def get_data_dimensions(self):
        esme_data = pd.read_parquet(self.data_dir)
        static_data = pd.read_parquet(
            "/Users/malek/TheLAB/DynaSurv/data/model_entry_imputes_data_STATIC_no_staging.parquet"
        )
        static_data = static_data.loc[
            static_data["usubjid"].isin(esme_data["usubjid"].unique())
        ].reset_index(drop=True)

        x_dim = len(
            [
                col
                for col in esme_data.columns
                if col.startswith("X_") and not col.startswith("X_buffer_time")
            ]
        )
        p_dim = len(
            pd.get_dummies(esme_data["T_treatment_category"].astype(str)).columns
        )
        p_static_dim = len([col for col in static_data.columns if col.startswith("T_")])
        x_static_dim = len([col for col in static_data.columns if col.startswith("X_")])
        time = esme_data["Y_onset_to_death"]
        interval_bounds = torch.linspace(0, time.max(), self.n_intervals + 1)

        return {
            "x_input_dim": x_dim,
            "p_input_dim": p_dim,
            "p_static_dim": p_static_dim,
            "x_static_dim": x_static_dim,
            "output_dim": self.n_intervals,
            "time_bins": interval_bounds,
        }


if __name__ == "__main__":
    import time

    data_module = ESMEDataModule(
        data_dir="../../../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
        n_lines=1,
        n_intervals=10,
        train_batch_size=32,
        val_split=0.2,
        test_split=0.1,
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

    batch = next(iter(train_loader))
    print(f"Batch fetched in {time.time() - start:.2f} seconds.")
    XPd, (x_static, p_static), interval_idx, treatment_index, time, event = batch
    print(
        f"XPd shape: {XPd.shape},\nx_static shape: {x_static.shape},\np_static shape: {p_static.shape},\ntreatment_index shape: {treatment_index.shape},\ntime shape: {time.shape},\nevent shape: {event.shape}"
    )
    print(f"treatment_dict: {data_module.treatment_dict}")
