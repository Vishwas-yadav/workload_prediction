import pandas as pd
import os

folder_path = 'vm_cpu_readings'

csv1 = pd.read_csv('vmtable.csv', header=None)

hash_id_column = csv1.iloc[:, 0]
cpu_cores_column = csv1.iloc[:, 9]

csv1_relevant = pd.DataFrame({'hash_id': hash_id_column, 'cpu_cores': cpu_cores_column})

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        csv2 = pd.read_csv(file_path, header=None)

        merged_df = pd.merge(csv2, csv1_relevant, left_on=0, right_on='hash_id', how='left')

        missing_hash_ids = merged_df.loc[merged_df['cpu_cores'].isnull(), 0]
        for missing_id in missing_hash_ids:
            print(f"Hash ID {missing_id} not found in vmtable.csv. Setting cpu_cores value to blank.")

        merged_df = merged_df.drop(columns=['hash_id'])

        merged_df.columns = ['timestamp', 'hash_id', 'min_cpu', 'max_cpu', 'avg_cpu', 'cpu_cores']

        merged_df.to_csv(file_path, index=False)

        print(f"Processed and saved: {file_path}")
