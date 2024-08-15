import pandas as pd
import numpy as np

df = pd.read_csv('container_usage.csv')

machine_id_counts = df['machine_id'].value_counts()
valid_machine_ids = machine_id_counts[machine_id_counts > 35000].index

if not valid_machine_ids.empty:
    selected_machine_id = np.random.choice(valid_machine_ids)
    filtered_df = df[df['machine_id'] == selected_machine_id]
    filtered_df.to_csv(f'{selected_machine_id}_filtered.csv', index=False)
    print(f"CSV file created for machine_id {selected_machine_id} containing {len(filtered_df)} records.")
else:
    print("No machine_id found with more than 35,000 occurrences.")
