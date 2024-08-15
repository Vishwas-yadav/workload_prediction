import os
import pandas as pd

folder_path = 'Materna-Trace-1'
trace_info_path = 'Trace-info.txt'

with open(trace_info_path, 'r') as file:
    trace_data = file.readlines()

cpu_cores_values = [line.split('CPU Cores: ')[1].strip() for line in trace_data if 'CPU Cores:' in line]

for idx, core_value in enumerate(cpu_cores_values):
    csv_filename = f'{idx + 1}.csv'
    csv_filepath = os.path.join(folder_path, csv_filename)
    
    if os.path.exists(csv_filepath):
        df = pd.read_csv(csv_filepath)
        df['cpu_cores'] = core_value
        df.to_csv(csv_filepath, index=False)
        print(f"Updated {csv_filename} with CPU cores value: {core_value}")
    else:
        print(f"{csv_filename} does not exist in the specified folder.")

print("All files processed.")
