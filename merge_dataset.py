from pathlib import Path

import pandas as pd

DATA_PATH = Path('./results/')
DATASET_NAMES = ['dataset1_result_imputation.csv', 'dataset4_result_imputation.csv']
RESULT_FNAME = 'merged_dataset.csv'


merged_df = pd.DataFrame()
for dataset_name in DATASET_NAMES:
    dataset = pd.read_csv(DATA_PATH / dataset_name, index_col=0)
    merged_df = pd.concat([merged_df, dataset], ignore_index=True)

print(merged_df)
merged_df.to_csv(DATA_PATH / RESULT_FNAME)
