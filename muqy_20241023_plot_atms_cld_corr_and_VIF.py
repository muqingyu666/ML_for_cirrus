# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-10-23 16:28
# @Last Modified by:   Muqy
# @Last Modified time: 2024-10-24 16:36


import logging
import pickle
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set matplotlib style
mpl.style.use("ggplot")
mpl.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "Times New Roman"


# -----------------------------------------------------------------------------------
# Extracting Data
# -----------------------------------------------------------------------------------


def load_from_pkl(filename: str):
    """
    Load a pickled object from a file.

    Parameters:
    - filename (str): Path to the pickle file.

    Returns:
    - Any: The unpickled object.
    """
    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        logging.info(f"Loaded object from {filename}")
        return obj
    except Exception as e:
        logging.error(f"Failed to load object from {filename}: {e}")
        raise


def load_from_hdf5(filename="data_stacked.h5"):
    cld_data = {}
    atms_data = {}
    with h5py.File(filename, "r") as h5f:
        cld_group = h5f["cld_data_stacked"]
        for key in cld_group:
            cld_data[key] = cld_group[key][:]

        atms_group = h5f["atms_data_stacked"]
        for key in atms_group:
            atms_data[key] = atms_group[key][:]
    return cld_data, atms_data


data = load_from_pkl(
    # "C:\\Users\Muqy\OneDrive\Project_Python\Main_python\Data_python\Ridge_regression_data\cleaned_data.pkl"
    "/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/cleaned_data_no_zeros.pkl",  # Define your desired save path
)

(
    X_clean_insitu,
    X_clean_anvil,
    y_clean_insitu,
    y_clean_anvil,
) = (
    data["X_clean_insitu"],
    data["X_clean_anvil"],
    data["y_clean_insitu"],
    data["y_clean_anvil"],
)

cld_data, atms_data = load_from_hdf5(filename="data_stacked.h5")

# -----------------------------------------------------------------------------------
# Assessing Multicollinearity
# -----------------------------------------------------------------------------------

feature_names = [
    "Upper_tropopause_stability",
    "Tropopause_temp",
    "Tropopause_u_wind",
    "Tropopause_v_wind",
    "Upper_tropopause_wind_shear",
    "Skin_temperature",
    "Tropopause_relative_humidity",
]

# Extract and convert each predictor to float32
atms_insitu_df = pd.DataFrame(
    {
        feature_name: X_clean_insitu[:, i].astype(np.float32)
        for i, feature_name in enumerate(feature_names)
    }
)

# Extract and convert each predictor to float32
atms_anvil_df = pd.DataFrame(
    {
        feature_name: X_clean_anvil[:, i].astype(np.float32)
        for i, feature_name in enumerate(feature_names)
    }
)

cld_names = [
    "Upper_tropopause_stability",
    "Tropopause_temp",
    "Tropopause_u_wind",
    "Tropopause_v_wind",
    "Upper_tropopause_wind_shear",
    "Skin_temperature",
    "Tropopause_relative_humidity",
]

cld_name = {
    "Anvil fraction": "anvil_fraction_weighted_2D",
    "Insitu fraction": "insitu_fraction_weighted_2D",
}

cld_insitu_df = pd.DataFrame(
    {"Insitu fraction": y_clean_insitu.astype(np.float32)}
)
cld_anvil_df = pd.DataFrame(
    {"Anvil fraction": y_clean_anvil.astype(np.float32)}
)

# Step 1: Handle NaN values by dropping them (you could also use imputation if appropriate)
combined_df_insitu = pd.concat([atms_insitu_df, cld_insitu_df], axis=1)
clean_df_insitu = combined_df_insitu.dropna()

combined_df_anvil = pd.concat([atms_anvil_df, cld_anvil_df], axis=1)
clean_df_anvil = combined_df_anvil.dropna()

# Step 2: Compute the correlation matrix among meteorological variables
corr_atms_insitu = clean_df_insitu[atms_insitu_df.columns].corr()
corr_atms_anvil = clean_df_anvil[atms_anvil_df.columns].corr()

# Step 3: Compute the correlation between meteorological variables and cloud types
corr_clouds_insitu = clean_df_insitu.corr().loc[
    atms_insitu_df.columns, cld_insitu_df.columns
]
# Step 3: Compute the correlation between meteorological variables and cloud types
corr_clouds_anvil = clean_df_anvil.corr().loc[
    atms_anvil_df.columns, cld_anvil_df.columns
]
corr_clouds = pd.concat([corr_clouds_insitu, corr_clouds_anvil], axis=1)


# Step 4: Plot the correlation heatmaps

# Plot 1: Correlation heatmap among meteorological variables
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_atms_anvil,
    annot=True,
    cmap="coolwarm",
    cbar=True,
    fmt=".2f",
    vmin=-1,
    vmax=1,
)
plt.title("Correlation Heatmap of Meteorological Variables")
plt.xticks(rotation=45)
plt.show()

# Plot 2: Correlation of meteorological variables with cirrus cloud types
plt.figure(figsize=(6, 8))
sns.heatmap(
    corr_clouds,
    annot=True,
    cmap="coolwarm",
    cbar=True,
    fmt=".2f",
    vmin=-1,
    vmax=1,
)
plt.title(
    "Correlation of Meteorological Variables with Cirrus Cloud Types"
)
plt.show()

# Data provided by the user
features = [
    "Upper tropopause stability",
    "Tropopause temp",
    "Tropopause uwind",
    "Tropopause vwind",
    "Upper tropopause wind shear",
    "Skin temperature",
    "Tropopause relative humidity",
]
vif_values = [
    21.6431567462479,
    20.978028989816295,
    3.446739167937659,
    1.0556721895845034,
    3.154393475280124,
    1.1892448155490407,
    1.1517573259132967,
]

# Define threshold for significant collinearity (commonly used threshold: VIF > 5 or VIF > 10)
vif_threshold = 10

# Create the bar plot
plt.figure(figsize=(8, 5))
bars = plt.barh(features, vif_values, color="#77CDFF")

# Mark bars with different colors based on the threshold
for bar, vif in zip(bars, vif_values):
    if vif > vif_threshold:
        bar.set_color("#C62E2E")

# Add the threshold line
plt.axvline(
    x=vif_threshold,
    color="black",
    linestyle="--",
    label=f"Threshold: VIF > {vif_threshold}",
)

# Add labels and title
plt.xlabel("Variance Inflation Factor (VIF)")
plt.title("VIF Values for Collinearity Detection")
plt.gca().invert_yaxis()  # Reverse the y-axis for better readability
plt.legend()

# Show plot
plt.show()
