import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define folder path containing the dataset file
file_path = '/content/drive/MyDrive/dataset/output_filtered.csv'

# Read the dataset
df = pd.read_csv(file_path)
df = df[['ts', 'cpu']]

# Apply Kalman filter
def apply_kalman_filter(data):
    kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(data)
    return filtered_state_means

df['cpu_filtered'] = apply_kalman_filter(df['cpu'])


# Create dataset
data = df['cpu_filtered'].values

# Train-test split
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Fit ARIMA model
model = ARIMA(train_data, order=(5, 1, 0))
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test_data))

# Calculate MAE and MSE
mae = mean_absolute_error(test_data, predictions)
mse = mean_squared_error(test_data, predictions)

print("Test MAE:", mae)
print("Test MSE:", mse)
