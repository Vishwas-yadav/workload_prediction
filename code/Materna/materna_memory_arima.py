# -*- coding: utf-8 -*-

import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Function to apply Kalman filter
def apply_kalman_filter(data):
    kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(data)
    return filtered_state_means

# Define folder path containing the dataset files
folder_path = '/content/drive/MyDrive/dataset_cloud/Materna_modified_ds/Materena_traces/GWA-T-13_Materna-Workload-Traces/Materna-Trace-3'
# folder_path = '/content/drive/MyDrive/dataset_cloud/Materna_modified_ds/Materena_traces/GWA-T-13_Materna-Workload-Traces/Materna-Trace-2'
# folder_path = '/content/drive/MyDrive/dataset_cloud/Materna_modified_ds/Materena_traces/GWA-T-13_Materna-Workload-Traces/Materna-Trace-1'


# Get the list of files in the folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]  # Limit to the first 10 CSV files

# Initialize lists to store MAE and MSE for each file
mae_list = []
mse_list = []

# Iterate over each file
for file_name in file_list:
    # Read data from the file
    print("File:::::>>>>", file_name)
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, sep=';')

    # Check if 'Memory usage [%]' and 'cpu_core' columns exist
    if 'Memory usage [%]' not in df.columns or 'CPU cores' not in df.columns:
        print("Skipping file", file_name, "as it doesn't contain required columns")
        continue

    # Replace commas with periods in the 'Memory usage [%]' column
    df['Memory usage [%]'] = df['Memory usage [%]'].str.replace(',', '.')

    # Convert the 'Memory usage [%]' column to numerical values
    df['Memory usage [%]'] = pd.to_numeric(df['Memory usage [%]'])

    # Extract Memory usage column
    memory_usage = df['Memory usage [%]'].values.reshape(-1, 1)

    # Apply Kalman filter to Memory usage
    memory_filtered = apply_kalman_filter(memory_usage)

    # Split data into training and test sets
    train_size = int(len(memory_filtered) * 0.8)
    train_data, test_data = memory_filtered[:train_size], memory_filtered[train_size:]

    # Fit ARIMA model
    model = ARIMA(train_data, order=(5,1,0))
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.forecast(steps=len(test_data))

    # Calculate MAE and MSE
    mae = mean_absolute_error(test_data, predictions)
    mse = mean_squared_error(test_data, predictions)

    mae_list.append(mae)
    mse_list.append(mse)

# Calculate final MAE and MSE on test data
final_mae = np.mean(mae_list)
final_mse = np.mean(mse_list)
print("Test MAE:", final_mae)
print("Test MSE:", final_mse)



