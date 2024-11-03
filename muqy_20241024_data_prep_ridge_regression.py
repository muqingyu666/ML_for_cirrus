# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-08-20 08:38
# @Last Modified by:   Muqy
# @Last Modified time: 2024-10-29 22:48

import gc
import glob
import logging
import pickle
import warnings
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

# Custom utility functions
from muqy_20240710_util_cirrus_class_freq_micro import (
    extract_start_date,
)

# Disable all warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging to display info messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

####################### Functions ###################################################


class CirrusRegressionPreprocessor:
    def __init__(
        self,
        cld_variables=None,
        atms_variables=None,
        cirrus_obj_paths=None,
        num_jobs=None,
    ):
        if cld_variables is None:
            cld_variables = [
                "insitu_fraction_weighted_2D",
                "anvil_fraction_weighted_2D",
            ]
        if atms_variables is None:
            atms_variables = [
                "Upper_tropopause_stability",
                "Instability",
                "Tropopause_relative_humidity",
                "Upper_tropopause_wind_shear",
                "Tropopause_u_wind",
                "Tropopause_v_wind",
                "Skin_temperature",
            ]
        self.cld_variables = cld_variables
        self.atms_variables = atms_variables
        self.cirrus_obj_paths = cirrus_obj_paths
        self.num_jobs = num_jobs

    def extract_variable_from_single_file(self, file_path):
        cld_data = {key: [] for key in self.cld_variables}
        atms_data = {key: [] for key in self.atms_variables}

        with xr.open_dataset(file_path, engine="h5netcdf") as ds:

            # Extract cloud variables
            for cld_var in self.cld_variables:
                variable_data = ds[cld_var].values.astype(np.float32)
                # Remove values less than -1, as they are misclassifications
                variable_data[variable_data < 0] = np.nan
                # Remove values less than 40, as they could be misclassifications
                # variable_data[variable_data <= 35] = 0
                cld_data[cld_var].append(variable_data)

            # Extract atmospheric variables
            for atms_var in self.atms_variables:
                variable_data = ds[atms_var].values.astype(np.float32)
                variable_data[variable_data <= -99] = np.nan
                atms_data[atms_var].append(variable_data)

        return cld_data, atms_data

    def extract_variable_from_files(self):
        results = Parallel(n_jobs=self.num_jobs, backend="loky")(
            delayed(self.extract_variable_from_single_file)(fp)
            for fp in self.cirrus_obj_paths
        )

        logging.info("Finished extracting variables from files")

        # Results contain a tuple of dictionaries for cloud and atmospheric variables
        # Each element in this result list corresponds to a single file
        cld_data = {
            var: np.hstack(
                [
                    result[0][var]
                    for result in results
                    if var in result[0]
                ]
            )
            for var in self.cld_variables
        }
        atms_data = {
            var: np.hstack(
                [
                    result[1][var]
                    for result in results
                    if var in result[1]
                ]
            )
            for var in self.atms_variables
        }

        return cld_data, atms_data

    def main(self):
        cld_data, atms_data = self.extract_variable_from_files()
        return cld_data, atms_data


def save_to_hdf5(cld_data, atms_data, filename="data_stacked.h5"):
    with h5py.File(filename, "w") as h5f:
        # Create separate groups for cloud and atmospheric data
        cld_group = h5f.create_group("cld_data_stacked")
        for key, array in cld_data.items():
            cld_group.create_dataset(
                key,
                data=array.astype(np.float32),
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,  # Adjust compression level (0-9)
                chunks=True,  # Enable chunking
            )

        atms_group = h5f.create_group("atms_data_stacked")
        for key, array in atms_data.items():
            atms_group.create_dataset(
                key,
                data=array.astype(np.float32),
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,  # Adjust compression level (0-9)
                chunks=True,  # Enable chunking
            )

    print(f"Data saved to {filename}")


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


########################################################################################
########################################################################################
# Data Preparation and Cleaning Pipeline #
########################################################################################
########################################################################################


def combine_predictors(atms_data: pd.DataFrame) -> pd.DataFrame:
    """
    Combine selected atmospheric predictors into a single feature matrix.

    Parameters:
    - atms_data: Atmospheric data as a pandas DataFrame.

    Returns:
    - Combined feature matrix as a pandas DataFrame with float32 data type.
    """
    try:
        logging.info("Combining predictors into a feature matrix...")

        # Define the predictors to include
        predictors = {
            "Upper_tropopause_stability": "Upper_tropopause_stability",
            "Tropopause_temp": "Instability",
            "Tropopause_u_wind": "Tropopause_u_wind",
            "Tropopause_v_wind": "Tropopause_v_wind",
            "Upper_tropopause_wind_shear": "Upper_tropopause_wind_shear",
            "Skin_temperature": "Skin_temperature",
            "Tropopause_relative_humidity": "Tropopause_relative_humidity",
        }

        # Extract and convert each predictor to float32
        data = pd.DataFrame(
            {
                feature_name: atms_data[column_name]
                .squeeze()
                .astype(np.float32)
                for feature_name, column_name in predictors.items()
            }
        )

        logging.info("Predictors combined successfully.")
        return data
    except KeyError as e:
        logging.error(
            f"Missing expected column in atmospheric data: {e}"
        )
        raise
    except Exception as e:
        logging.error(f"Error while combining predictors: {e}")
        raise


def extract_targets(
    cld_data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract target variables for in-situ and anvil cirrus.

    Parameters:
    - cld_data: Cloud data as a pandas DataFrame.

    Returns:
    - Tuple containing y_insitu and y_anvil as numpy arrays with float32 data type.
    """
    try:
        logging.info("Extracting target variables...")
        y_insitu = (
            cld_data["insitu_fraction_weighted_2D"]
            .squeeze()
            .astype(np.float32)
        )
        y_anvil = (
            cld_data["anvil_fraction_weighted_2D"]
            .squeeze()
            .astype(np.float32)
        )
        logging.info("Target variables extracted successfully.")
        return y_insitu, y_anvil
    except KeyError as e:
        logging.error(
            f"Missing expected target column in cloud data: {e}"
        )
        raise
    except Exception as e:
        logging.error(f"Error while extracting targets: {e}")
        raise


def clean_data(
    X: np.ndarray, y_insitu: np.ndarray, y_anvil: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove samples with NaN values from the feature matrix and target vectors.

    Parameters:
    - X: Feature matrix as a numpy array.
    - y_insitu: Target vector for in-situ cirrus.
    - y_anvil: Target vector for anvil cirrus.

    Returns:
    - Tuple containing cleaned X, y_insitu, and y_anvil as numpy arrays.
    """
    logging.info("Cleaning data by removing samples with NaN values...")
    # Create a boolean mask where all features and targets are not NaN
    mask = (
        (~np.isnan(X).any(axis=1))
        & (~np.isnan(y_insitu))
        & (~np.isnan(y_anvil))
    )

    # Apply the mask to filter out invalid samples
    X_clean = X[mask]
    y_clean_insitu = y_insitu[mask]
    y_clean_anvil = y_anvil[mask]

    logging.info(
        f"Number of atms samples after removing NaNs: {X_clean.shape[0]}"
    )
    logging.info(
        f"Number of samples after removing NaNs: {y_clean_insitu.shape[0]}"
    )
    X_clean_insitu = X_clean
    X_clean_anvil = X_clean

    del X_clean
    gc.collect()

    return X_clean_insitu, X_clean_anvil, y_clean_insitu, y_clean_anvil


def clean_data_no_zeros(
    X: np.ndarray, y_insitu: np.ndarray, y_anvil: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove samples with NaN values from the feature matrix and target vectors.

    Parameters:
    - X: Feature matrix as a numpy array.
    - y_insitu: Target vector for in-situ cirrus.
    - y_anvil: Target vector for anvil cirrus.

    Returns:
    - Tuple containing cleaned X, y_insitu, and y_anvil as numpy arrays.
    """
    logging.info("Cleaning data by removing samples with NaN values...")
    # Create a boolean mask where all features and targets are not NaN
    mask = (
        (~np.isnan(X).any(axis=1))
        & (~np.isnan(y_insitu))
        & (~np.isnan(y_anvil))
    )

    # Apply the mask to filter out invalid samples
    X_clean = X[mask]
    y_clean_insitu = y_insitu[mask]
    y_clean_anvil = y_anvil[mask]

    # Apply the mask to filter out 0 data
    mask_insitu = y_clean_insitu != 0
    X_clean_insitu = X_clean[mask_insitu]
    y_clean_insitu = y_clean_insitu[mask_insitu]

    mask_anvil = y_clean_anvil != 0
    X_clean_anvil = X_clean[mask_anvil]
    y_clean_anvil = y_clean_anvil[mask_anvil]

    logging.info(
        f"Number of samples after cleaning insitu data: {X_clean_insitu.shape[0]}"
    )
    logging.info(
        f"Number of samples after cleaning anvil data: {X_clean_anvil.shape[0]}"
    )

    return (
        X_clean_insitu,
        X_clean_anvil,
        y_clean_insitu,
        y_clean_anvil,
    )


def clean_data_balanced_zeros(
    X: np.ndarray, y_insitu: np.ndarray, y_anvil: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove samples with NaN values and balance the dataset by keeping the number of
    zero samples at half the number of non-zero samples.

    Parameters:
    - X: Feature matrix as a numpy array.
    - y_insitu: Target vector for in-situ cirrus.
    - y_anvil: Target vector for anvil cirrus.

    Returns:
    - Tuple containing cleaned and balanced X_insitu, X_anvil, y_insitu, and y_anvil as numpy arrays.
    """
    logging.info("Cleaning data by removing samples with NaN values...")
    # Create a boolean mask where all features and targets are not NaN
    mask = (
        (~np.isnan(X).any(axis=1))
        & (~np.isnan(y_insitu))
        & (~np.isnan(y_anvil))
    )

    # Apply the mask to filter out invalid samples
    X_clean = X[mask]
    y_clean_insitu = y_insitu[mask]
    y_clean_anvil = y_anvil[mask]

    # Process insitu data
    mask_insitu_nonzero = y_clean_insitu != 0
    mask_insitu_zero = y_clean_insitu == 0

    X_insitu_nonzero = X_clean[mask_insitu_nonzero]
    y_insitu_nonzero = y_clean_insitu[mask_insitu_nonzero]
    X_insitu_zero = X_clean[mask_insitu_zero]
    y_insitu_zero = y_clean_insitu[mask_insitu_zero]

    # Calculate desired number of zero samples (half of non-zero samples)
    n_nonzero_insitu = len(y_insitu_nonzero)
    n_desired_zero_insitu = n_nonzero_insitu // 5

    # Randomly select subset of zero samples
    if len(y_insitu_zero) > n_desired_zero_insitu:
        zero_indices = np.random.choice(
            len(y_insitu_zero), n_desired_zero_insitu, replace=False
        )
        X_insitu_zero = X_insitu_zero[zero_indices]
        y_insitu_zero = y_insitu_zero[zero_indices]

    # Process anvil data
    mask_anvil_nonzero = y_clean_anvil != 0
    mask_anvil_zero = y_clean_anvil == 0

    X_anvil_nonzero = X_clean[mask_anvil_nonzero]
    y_anvil_nonzero = y_clean_anvil[mask_anvil_nonzero]
    X_anvil_zero = X_clean[mask_anvil_zero]
    y_anvil_zero = y_clean_anvil[mask_anvil_zero]

    # Calculate desired number of zero samples (half of non-zero samples)
    n_nonzero_anvil = len(y_anvil_nonzero)
    n_desired_zero_anvil = n_nonzero_anvil // 5

    # Randomly select subset of zero samples
    if len(y_anvil_zero) > n_desired_zero_anvil:
        zero_indices = np.random.choice(
            len(y_anvil_zero), n_desired_zero_anvil, replace=False
        )
        X_anvil_zero = X_anvil_zero[zero_indices]
        y_anvil_zero = y_anvil_zero[zero_indices]

    # Combine zero and non-zero samples
    X_clean_insitu = np.vstack((X_insitu_nonzero, X_insitu_zero))
    y_clean_insitu = np.concatenate((y_insitu_nonzero, y_insitu_zero))

    X_clean_anvil = np.vstack((X_anvil_nonzero, X_anvil_zero))
    y_clean_anvil = np.concatenate((y_anvil_nonzero, y_anvil_zero))

    logging.info(
        f"Number of non-zero samples in insitu data: {n_nonzero_insitu}"
    )
    logging.info(
        f"Number of zero samples in insitu data: {len(y_insitu_zero)}"
    )
    logging.info(
        f"Number of non-zero samples in anvil data: {n_nonzero_anvil}"
    )
    logging.info(
        f"Number of zero samples in anvil data: {len(y_anvil_zero)}"
    )

    return (
        X_clean_insitu,
        X_clean_anvil,
        y_clean_insitu,
        y_clean_anvil,
    )


def save_clean_data(
    X_insitu: np.ndarray,
    X_anvil: np.ndarray,
    y_insitu: np.ndarray,
    y_anvil: np.ndarray,
    filepath: str,
) -> None:
    try:
        logging.info(f"Saving cleaned data to {filepath}...")
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "X_clean_insitu": X_insitu,
                    "X_clean_anvil": X_anvil,
                    "y_clean_insitu": y_insitu,
                    "y_clean_anvil": y_anvil,
                },
                f,
            )
        logging.info("Cleaned data saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save cleaned data to {filepath}: {e}")
        raise


def verify_data():
    # Log the number of zero and non-zero values in the cleaned target vectors
    zero_insitu = np.sum(y_clean_insitu == 0)
    non_zero_insitu = np.sum(y_clean_insitu != 0)

    zero_anvil = np.sum(y_clean_anvil == 0)
    non_zero_anvil = np.sum(y_clean_anvil != 0)

    logging.info(
        f"y_clean_insitu: {non_zero_insitu} non-zero values, {zero_insitu} zero values."
    )
    logging.info(
        f"y_clean_anvil: {non_zero_anvil} non-zero values, {zero_anvil} zero values."
    )

    plt.figure(figsize=(6, 5), dpi=200)
    plt.hist(y_clean_insitu, bins=200, label="In-Situ Cirrus")
    plt.title("In-Situ Cirrus Target Distribution")
    plt.xlim(0, np.max(y_clean_insitu) * 0.6)
    plt.tight_layout()
    plt.savefig("in_situ_cirrus_target_distribution.png")

    plt.hist(y_clean_anvil, bins=200, label="Anvil Cirrus")
    plt.title("Anvil Cirrus Target Distribution")
    plt.xlim(0, np.max(y_clean_anvil) * 0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("anvil_cirrus_target_distribution.png")


if __name__ == "__main__":
    # --------------------------------------------------
    # Main Execution Flow
    # --------------------------------------------------

    # Sort file paths by start date to ensure chronological processing
    cirrus_obj_paths = sorted(
        glob.glob(
            "/RAID01/data/PROJECT_CIRRUS_CLASSIFICATION/Height_Weighted_Cirrus_classification_233K_2500m_overlay_aerosol_0085/*.nc"
            # "../../Data_python/CloudSat_data/Height_Weighted_Cirrus_classification_233K_1440m_overlay_aerosol_0085/*.nc"
        ),
        key=extract_start_date,
    )

    # Set the variables to extract
    cld_variables = [
        "insitu_fraction_weighted_2D",
        "anvil_fraction_weighted_2D",
    ]
    atms_variables = [
        "Upper_tropopause_stability",
        "Instability",
        "Tropopause_relative_humidity",
        "Upper_tropopause_wind_shear",
        "Tropopause_u_wind",
        "Tropopause_v_wind",
        "Skin_temperature",
    ]

    # Instantiate the preprocessor with all available cores
    preprocessor = CirrusRegressionPreprocessor(
        cld_variables=cld_variables,
        atms_variables=atms_variables,
        cirrus_obj_paths=cirrus_obj_paths,
        num_jobs=-1,  # Use all available cores
    )

    cld_data_stacked, atms_data_stacked = preprocessor.main()

    # Usage
    save_to_hdf5(
        cld_data_stacked,
        atms_data_stacked,
        filename="/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Data/data_stacked.h5",
    )

    # Step 1: Load Data
    cld_data, atms_data = load_from_hdf5(
        filename="/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Data/data_stacked.h5"
    )

    # Combine predictors into a single feature matrix
    data = combine_predictors(atms_data)

    # Convert the DataFrame to a NumPy array with float32 data type
    X = data.values.astype(np.float32)

    # Step 3: Extract Target Variables
    y_insitu, y_anvil = extract_targets(cld_data)

    # Step 4: Clean Data by Removing Samples with NaNs
    X_clean_insitu, X_clean_anvil, y_clean_insitu, y_clean_anvil = (
        clean_data(X, y_insitu, y_anvil)
    )

    # Step 4: Clean Data by Removing Samples with NaNs
    X_clean_insitu, X_clean_anvil, y_clean_insitu, y_clean_anvil = (
        clean_data_no_zeros(X, y_insitu, y_anvil)
    )

    # # Step 4: Clean Data by Removing Samples with NaNs
    # X_clean_insitu, X_clean_anvil, y_clean_insitu, y_clean_anvil = (
    #     clean_data_balanced_zeros
    # (X, y_insitu, y_anvil)
    # )

    # Delete the original data to free up memory
    del cld_data, atms_data, data
    gc.collect()

    # Verify the cleaned data
    verify_data()

    # -----------------------------------------------------------------------------------
    # PKL data test
    data = load_from_pkl(
        # "C:\\Users\Muqy\OneDrive\Project_Python\Main_python\Data_python\Ridge_regression_data\cleaned_data.pkl"
        "/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Data/cleaned_data.pkl",  # Define your desired save path
        # "/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Data/cleaned_data_5_times_less_zeros.pkl"
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

    # Verify the cleaned data
    verify_data()

    # -----------------------------------------------------------------------------------
    # Optional Step: Save Cleaned Data for Future Use
    save_clean_data(
        X_clean_insitu,
        X_clean_anvil,
        y_clean_insitu,
        y_clean_anvil,
        # "/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Data/cleaned_data_5_times_less_zeros.pkl",  # Define your desired save path
        "/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Data/cleaned_data_no_zeros.pkl",  # Define your desired save path
        # "/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Data/cleaned_data_og.pkl",  # Define your desired save path
    )

    logging.info(
        "Data preparation and cleaning completed successfully."
    )

    # -----------------------------------------------------------------------------------
