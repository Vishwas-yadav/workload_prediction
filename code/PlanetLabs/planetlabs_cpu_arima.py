# -*- coding: utf-8 -*-

import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pykalman import KalmanFilter
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to apply Kalman filter
def apply_kalman_filter(data):
    kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(data)
    return filtered_state_means

# Define folder path containing the dataset files
folder_path = '/content/drive/MyDrive/dataset/planetLabs'

# Get the list of files in the folder
file_list = os.listdir(folder_path)  

# Initialize lists to store MAE and MSE for each file
mae_list = []
mse_list = []

train_ratio = 0.8  # Define train ratio

# Iterate over each file
for file_name in file_list:
    # Read data from the file
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, header=None)  # Assume data doesn't have column names

    # Apply Kalman filter
    df['cpu_filtered'] = apply_kalman_filter(df[0])

    # Split data into train and test sets
    train_size = int(len(df) * train_ratio)
    train_data = df['cpu_filtered'][:train_size]
    test_data = df['cpu_filtered'][train_size:]

    # Fit ARIMA model
    model = ARIMA(train_data, order=(5,1,0))  # Example order, adjust..
    fitted_model = model.fit()

    # Make predictions
    predictions = fitted_model.forecast(steps=len(test_data))

    # Calculate MAE and MSE
    mae = mean_absolute_error(test_data, predictions)
    mse = mean_squared_error(test_data, predictions)

    # Append MAE and MSE to the lists
    mae_list.append(mae)
    mse_list.append(mse)

# Calculate final MAE and MSE on test data
final_mae = np.mean(mae_list)
final_mse = np.mean(mse_list)

print("Test MAE:", final_mae)
print("Test MSE:", final_mse)

