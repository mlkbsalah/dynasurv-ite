from sklearn.model_selection import train_test_split, RepeatedKFold
from sksurv.util import Surv
import sksurv.metrics as skmetrics
import pandas as pd
import numpy as np
from typing import Tuple

def train_test_split_survival_data(df:pd.DataFrame, test_size:float=0.2, random_state: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    if df_train['Y_onset_to_death'].max() < df_test['Y_onset_to_death'].max():
        max_train_time = df_train['Y_onset_to_death'].max()
        df_test = df_test[df_test['Y_onset_to_death'] <= max_train_time]

    y_train = Surv.from_dataframe('Y_death', 'Y_onset_to_death', df_train)
    y_test = Surv.from_dataframe('Y_death', 'Y_onset_to_death', df_test)
    X_train = df_train[[col for col in df_train.columns if col.startswith('X')]]
    X_test = df_test[[col for col in df_test.columns if col.startswith('X')]]
    
    
    return X_train, X_test, y_train, y_test

def RepeatedKFold_survival(df:pd.DataFrame, n_splits=5, n_repeats=1, random_state: int | None = None):
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for train_index, test_index in kf.split(df):
        df_train, df_test = df.iloc[train_index],  df.iloc[test_index]
        if df_train['Y_onset_to_death'].max() < df_test['Y_onset_to_death'].max():
            max_train_time = df_train['Y_onset_to_death'].max()
            df_test = df_test[df_test['Y_onset_to_death'] <= max_train_time]

        y_train = Surv.from_dataframe('Y_death', 'Y_onset_to_death', df_train)
        y_test = Surv.from_dataframe('Y_death', 'Y_onset_to_death', df_test)
        X_train = df_train[[col for col in df_train.columns if col.startswith('X') and not col.startswith('X_buffer_time')]]
        X_test = df_test[[col for col in df_test.columns if col.startswith('X') and not col.startswith('X_buffer_time')]]

        yield X_train, X_test, y_train, y_test


def intergrated_brier_score(model, y_train, y_test, x_test) -> float:
    lower, upper = np.percentile(y_test["Y_onset_to_death"], [10, 90])
    times = np.linspace(lower, upper, 100)

    preds = np.asarray([[fn(t) for t in times] for fn in model.predict_survival_function(x_test)])
    ibs = skmetrics.integrated_brier_score(y_train, y_test, preds, times)

    return ibs