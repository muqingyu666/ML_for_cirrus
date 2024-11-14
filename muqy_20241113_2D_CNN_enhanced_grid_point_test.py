# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-08-20 08:38
# @Last Modified by:   Muqy
# @Last Modified time: 2024-11-14 15:07

import logging
import warnings

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input
from keras.models import Model
from keras.regularizers import l2
from scipy.stats import norm, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Disable all warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging to display info messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

####################################################################################################
# -----------------------------
# Step 1: Load Your Data
# -----------------------------
data = xr.open_dataset(
    "/RAID01/data/PROJECT_CIRRUS_CLASSIFICATION/CloudSat_Cirrus_classification_grid/monthly_means_met_data_5degree.nc"
)

# -----------------------------
# Step 2: Define Variables
# -----------------------------
# Predictor variables (features)
feature_names = [
    # Stability indices
    "Upper_tropopause_stability",
    # "Instability",
    # Temperature and humidity
    "Tropopause_relative_humidity",
    # "Upper_trop_humidity",
    "Tropopause_temp",
    # "Upper_trop_temp",
    "Skin_temperature",
    # Wind
    "Upper_tropopause_wind_shear",
    "Tropopause_u_wind",
    "Tropopause_v_wind",
    # # Height
    "Tropopause_height",
    # # Vertical motion
    # "Vertical_velocity",
]

# Target variables
target_variables = [
    "insitu_mask",
    "anvil_mask",
]

# Extract dimensions
time = data["time"].values
latitudes = data["lat"].values
longitudes = data["lon"].values

n_time = len(time)  # 58 time points
n_lat = len(latitudes)  # 36 latitudes
n_lon = len(longitudes)  # 72 longitudes
n_features = len(feature_names)

# Extract predictor data: Shape (n_time, n_lat, n_lon, n_features)
X_data = np.stack([data[var].values for var in feature_names], axis=-1)

# Extract target data
y_in_situ_data = data[
    "insitu_mask"
].values  # Shape: (n_time, n_lat, n_lon)
y_anvil_data = data[
    "anvil_mask"
].values  # Shape: (n_time, n_lat, n_lon)

# Split index (e.g., 80% train, 20% test)
split_index = int(n_time * 0.9)  # 46 training samples, 12 test samples

# Training data
X_train = X_data[:split_index]  # Shape: (46, n_lat, n_lon, n_features)
y_in_situ_train = y_in_situ_data[
    :split_index
]  # Shape: (46, n_lat, n_lon)
y_anvil_train = y_anvil_data[:split_index]  # Shape: (46, n_lat, n_lon)

# Testing data
X_test = X_data[split_index:]  # Shape: (12, n_lat, n_lon, n_features)
y_in_situ_test = y_in_situ_data[
    split_index:
]  # Shape: (12, n_lat, n_lon)
y_anvil_test = y_anvil_data[split_index:]  # Shape: (12, n_lat, n_lon)

logging.info(
    f"Data dimensions - Time: {n_time}, Latitude: {n_lat}, Longitude: {n_lon}, Features: {n_features}"
)

# Handle NaNs in features by replacing them with mean values
feature_means = np.nanmean(X_train, axis=(0, 1, 2))
X_train = np.where(np.isnan(X_train), feature_means, X_train)
X_test = np.where(np.isnan(X_test), feature_means, X_test)

# -----------------------------
# Step 5: Prepare Data Arrays
# -----------------------------

# Flatten spatial dimensions for scaling
X_train_flat = X_train.reshape(-1, n_features)
X_test_flat = X_test.reshape(-1, n_features)

# Initialize and fit scaler on training data
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)

# Transform test data
X_test_flat = scaler.transform(X_test_flat)

# Reshape back to original dimensions
X_train = X_train_flat.reshape(-1, n_lat, n_lon, n_features)
X_test = X_test_flat.reshape(-1, n_lat, n_lon, n_features)

# -----------------------------
# Step 5: Prepare Target Data
# -----------------------------

# For in-situ cirrus
y_in_situ_train = y_in_situ_train.reshape(
    -1, n_lat * n_lon
)  # Shape: (46, n_lat * n_lon)
y_in_situ_test = y_in_situ_test.reshape(
    -1, n_lat * n_lon
)  # Shape: (12, n_lat * n_lon)

# For anvil cirrus
y_anvil_train = y_anvil_train.reshape(
    -1, n_lat * n_lon
)  # Shape: (46, n_lat * n_lon)
y_anvil_test = y_anvil_test.reshape(
    -1, n_lat * n_lon
)  # Shape: (12, n_lat * n_lon)

# -----------------------------------------------------------------------------------


def build_cnn_model(input_shape, l2_reg=0.001, dropout_rate=0.5):
    inputs = Input(shape=input_shape)
    x = Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(l2_reg),
    )(inputs)
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(l2_reg),
    )(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(l2_reg),
    )(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(n_lat * n_lon, activation="linear")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# For in-situ cirrus
y_in_situ = y_in_situ_data  # Shape: (n_time, n_lat, n_lon)
y_in_situ = y_in_situ.reshape(
    n_time, n_lat * n_lon
)  # Flatten spatial dimensions

# For anvil cirrus
y_anvil = y_anvil_data  # Shape: (n_time, n_lat, n_lon)
y_anvil = y_anvil.reshape(
    n_time, n_lat * n_lon
)  # Flatten spatial dimensions


def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=80,
    batch_size=50,
):
    # Build model
    model = build_cnn_model(input_shape=(n_lat, n_lon, n_features))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )

    # Early stopping to prevent overfitting
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True
        )
    ]

    # If no validation data is provided, use a validation split
    if X_val is None or y_val is None:
        validation_data = None
        validation_split = (
            0.1  # Use 20% of training data for validation
        )
    else:
        validation_data = (X_val, y_val)
        validation_split = 0.0

    # Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


# --------------------------------------------------------------------------------------------------------

# Training model for in-situ cirrus
print("Training model for in-situ cirrus")
model_in_situ, history_in_situ = train_model(X_train, y_in_situ_train)

# Training model for anvil cirrus
print("\nTraining model for anvil cirrus")
model_anvil, history_anvil = train_model(X_train, y_anvil_train)

# --------------------------------------------------------------------------------------------------------


# For in-situ cirrus
y_in_situ_pred = model_in_situ.predict(X_test)
mse_in_situ = mean_squared_error(y_in_situ_test, y_in_situ_pred)
r2_in_situ = r2_score(y_in_situ_test, y_in_situ_pred)
print(f"In-situ Cirrus Test MSE: {mse_in_situ}")
print(f"In-situ Cirrus Test R2 Score: {r2_in_situ}")

# For anvil cirrus
y_anvil_pred = model_anvil.predict(X_test)
mse_anvil = mean_squared_error(y_anvil_test, y_anvil_pred)
r2_anvil = r2_score(y_anvil_test, y_anvil_pred)
print(f"Anvil Cirrus Test MSE: {mse_anvil}")
print(f"Anvil Cirrus Test R2 Score: {r2_anvil}")

# --------------------------------------------------------------------------------------------------------

# Reshape predictions and targets to (n_time, n_lat, n_lon)
y_in_situ_pred = y_in_situ_pred.reshape(-1, n_lat, n_lon)
y_in_situ_test = y_in_situ_test.reshape(-1, n_lat, n_lon)

y_anvil_pred = y_anvil_pred.reshape(-1, n_lat, n_lon)
y_anvil_test = y_anvil_test.reshape(-1, n_lat, n_lon)

# --------------------------------------------------------------------------------------------------------


def plot_history(history, title):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# Plot for in-situ cirrus model
plot_history(history_in_situ, "In-situ Cirrus Model Loss")

# Plot for anvil cirrus model
plot_history(history_anvil, "Anvil Cirrus Model Loss")

# # ####################################################################################################
