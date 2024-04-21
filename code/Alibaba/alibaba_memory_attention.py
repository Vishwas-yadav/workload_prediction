# -*- coding: utf-8 -*-

import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pykalman import KalmanFilter
from keras.layers import Dense, LSTM, GRU, Dropout, Input, Flatten, RepeatVector, Permute, multiply, Lambda, Activation, Conv1D, MaxPooling1D, Bidirectional
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_path = '/content/drive/MyDrive/dataset/output_filtered.csv'
df = pd.read_csv(file_path)
df = df[['ts','memory']]  # Selecting 'ts' and 'memory' columns

def apply_kalman_filter(data):
    kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(data)
    return filtered_state_means


def create_attention_model_LSTM(input_shape, units):
    inputs = Input(shape=input_shape)

    # Adding 1D CNN layer
    cnn_out = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    cnn_out = Dropout(0.2)(cnn_out)

    # LSTM layer
    lstm_out = LSTM(units, return_sequences=True)(cnn_out)

    # Attention mechanism
    attention = Dense(1, activation='relu')(lstm_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)  # Adjusted input shape to match the output shape of LSTM
    attention = Permute([2, 1])(attention)
    sent_representation = multiply([lstm_out, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

    # Output layer
    output = Dense(1)(sent_representation)

    model = Model(inputs=inputs, outputs=output)
    return model

# Function to create BiLSTM model with attention mechanism
def create_attention_model_BiLSTM(input_shape, units):
    inputs = Input(shape=input_shape)

    # Adding 1D CNN layer
    cnn_out = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    cnn_out = Dropout(0.2)(cnn_out)

    # BiLSTM layer
    lstm_out = Bidirectional(LSTM(units, return_sequences=True))(cnn_out)

    # Attention mechanism
    attention = Dense(1, activation='relu')(lstm_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(2 * units)(attention)  # Adjusted input shape to match the output shape of BiLSTM
    attention = Permute([2, 1])(attention)
    sent_representation = multiply([lstm_out, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

    # Output layer
    output = Dense(1)(sent_representation)

    model = Model(inputs=inputs, outputs=output)
    return model


def create_attention_model_GRU(input_shape, units):
    inputs = Input(shape=input_shape)

    # Adding 1D CNN layer
    cnn_out = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    cnn_out = Dropout(0.2)(cnn_out)

    # GRU layer
    gru_out = GRU(units, return_sequences=True)(cnn_out)

    # Attention mechanism
    attention = Dense(1, activation='relu')(gru_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)  # Adjusted input shape to match the output shape of GRU
    attention = Permute([2, 1])(attention)
    sent_representation = multiply([gru_out, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

    # Output layer
    output = Dense(1)(sent_representation)

    model = Model(inputs=inputs, outputs=output)
    return model


# Apply Kalman filter
df['memory_filtered'] = apply_kalman_filter(df['memory'])

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
df['memory_scaled'] = scaler.fit_transform(df['memory_filtered'].values.reshape(-1, 1))

# Define window size
window_size = 60

# Create dataset
data = df['memory_scaled'].values
dataX, dataY = [], []
for i in range(len(data)-window_size-1):
    a = data[i:(i+window_size)]
    dataX.append(a)
    dataY.append(data[i + window_size])
X = np.array(dataX)
y = np.array(dataY)

# Split data into train, validation, test sets
train_ratio = 0.4
validation_ratio = 0.2
test_ratio = 0.4

train_size = int(len(X) * train_ratio)
validation_size = int(len(X) * validation_ratio)
test_size = len(X) - train_size - validation_size

X_train, y_train = X[:train_size], y[:train_size]
X_validation, y_validation = X[train_size:train_size+validation_size], y[train_size:train_size+validation_size]
X_test, y_test = X[train_size+validation_size:], y[train_size+validation_size:]

# Build attention model
attention_model = create_attention_model_LSTM(input_shape=(window_size, 1), units=50)

# Compile attention model
attention_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])

# Train attention model
history = attention_model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=100, batch_size=16, verbose=1)

# Evaluate attention model
score = attention_model.evaluate(X_test, y_test, verbose=0)
print("Attention Model:")
print("Test MSE:", score[1])
print("Test MAE:", score[2])

# Save attention model to drive
model_save_path = '/content/drive/MyDrive/trained_models_alibaba_new/alibaba_Memory_LSTM_W60_1min_attention_model.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Create directory if it doesn't exist
attention_model.save(model_save_path)
print("Attention model saved at:", model_save_path)

# Make predictions on the test data
predictions = attention_model.predict(X_test)

# Inverse transform the predicted and actual values
predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(predictions_inv, label='Predicted')
plt.xlabel('Time Stamp')
plt.ylabel('Memory Utilization %')
plt.legend()
plt.show()

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_inv, label='Actual', color='blue')  # Set color of actual line to blue
plt.plot(predictions_inv, label='Predicted', color='pink', alpha=0.7)  # Set color to pink and alpha to 0.5 for transparency
plt.xlabel('Time Stamp')
plt.ylabel('Memory Utilization %')
plt.legend()
plt.show()
