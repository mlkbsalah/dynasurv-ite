import numpy as np
import pandas as pd
from sksurv.util import Surv
import sksurv.metrics as skmetrics
from CausalSurv.tools.train_test_split_survival import train_test_split_survival_data, RepeatedKFold_survival

from sksurv.ensemble import RandomSurvivalForest
from tqdm import tqdm 

import csv


def main():
    data_path_list = [
        "../data/model_entry_imputed_data_HER2+_stable_types_categorized.parquet",
        "../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
        "../data/model_entry_imputed_data_TN_stable_types_categorized.parquet",]

    n_splits = 100
    n_repeats = 1
    c_indexs = []

    for data_path in data_path_list:
        print("Processing data from:", data_path)
        df_line = pd.read_parquet(data_path)
        df_line1 = df_line.loc[df_line['lineid']==1].reset_index(drop=True)
        pbar = tqdm(RepeatedKFold_survival(df_line1, n_splits=n_splits, n_repeats=n_repeats), total=n_splits*n_repeats)
        for X_train, X_test, y_train, y_test in pbar:
            model = RandomSurvivalForest(n_estimators=1000,
                                        n_jobs=-1,
                                        verbose=1)
            model.fit(X_train, y_train)
            c_index = skmetrics.concordance_index_censored(y_test['Y_death'], y_test['Y_onset_to_death'], model.predict(X_test))[0]
            c_indexs.append(c_index)
            
            pbar.set_postfix({'C-index': f'{np.mean(c_indexs):.4f} +/- {np.std(c_indexs):.4f}'})
            if np.std(c_indexs) < 0.001 and len(c_indexs) > 10:
                pbar.close()
                break

    print("C-index over 100 splits:", sum(c_indexs)/len(c_indexs))

    with open("output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for x in c_indexs:
            writer.writerow([x])

if __name__ == "__main__":
    main()