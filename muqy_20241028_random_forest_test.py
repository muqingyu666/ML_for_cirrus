# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-08-20 08:38
# @Last Modified by:   Muqy
# @Last Modified time: 2024-11-01 09:31

import gc
from typing import Any, Dict, List
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
import seaborn as sns

# Disable all warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging to display info messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

########################################################################################
########################################################################################
# Ridge Regression Model Training and Evaluation Pipeline #
########################################################################################
########################################################################################


# -----------------------------------------------------------------------------------
# Ridge Regression Model Training and Evaluation Pipeline
# -----------------------------------------------------------------------------------


def load_from_pkl(filepath):
    """
    Load data from a pickle file.
    """
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    return data


def save_pickle(obj: Any, filename: str) -> None:
    """
    Save an object to a pickle file.

    Parameters:
    - obj (Any): The object to be pickled
    - filename (str): The name of the file where the object will be saved
    """
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Saved object to {filename}")
    except Exception as e:
        logging.error(f"Failed to save object to {filename}: {e}")
        raise


def perform_grid_search(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: Any,
    scoring: str,
    n_jobs: int,
    verbose: int,
):
    """
    Perform GridSearchCV with the given parameters.

    Parameters:
    - pipeline (Pipeline): scikit-learn Pipeline object
    - X (np.ndarray): Feature matrix
    - y (np.ndarray): Target vector
    - param_grid (dict): Parameter grid for GridSearch
    - cv (Any): Cross-validation splitting strategy
    - scoring (str): Scoring metric
    - n_jobs (int): Number of parallel jobs
    - verbose (int): Verbosity level

    Returns:
    - GridSearchCV: Fitted GridSearchCV object
    """
    try:
        # logging.info(
        #     f"Starting GridSearchCV with parameters: {param_grid}"
        # )
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,  # Set verbose to 0 to suppress output
        )
        grid_search.fit(X, y)
        logging.info("GridSearchCV completed successfully.")
        return grid_search
    except Exception as e:
        logging.error(f"Error during GridSearchCV: {e}")
        raise


def fit_final_model(
    best_params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
) -> RandomForestRegressor:
    """
    Fit the final Random Forest Regression model with the best hyperparameters.

    Parameters:
    - best_params (dict): Best hyperparameters found from GridSearchCV.
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.

    Returns:
    - RandomForestRegressor: Fitted Random Forest model.
    """
    try:
        logging.info(
            f"Fitting Random Forest model with parameters: {best_params}"
        )
        model = RandomForestRegressor(random_state=42, **best_params)
        model.fit(X, y)
        logging.info("Random Forest model fitted successfully.")
        return model
    except Exception as e:
        logging.error(f"Error during Random Forest model fitting: {e}")
        raise


def plot_feature_importances(
    model: RandomForestRegressor,
    feature_names: list,
    model_type: str,
    title: str = "Feature Importances",
) -> None:
    """
    Plots the feature importances from the Random Forest model.

    Parameters:
        model (RandomForestRegressor): Fitted Random Forest model.
        feature_names (list): List of feature names.
        model_type (str): Type of model for labeling purposes.
        title (str): Title of the plot.
    """
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(10, 8))
        plt.title(f"{model_type}: {title}")
        plt.barh(
            range(len(indices)),
            importances[indices],
            color="b",
            align="center",
        )
        plt.yticks(
            range(len(indices)), [feature_names[i] for i in indices]
        )
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(
            f"/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Figs/{model_type}_feature_importances.png"
        )
        logging.info(
            f"Feature importances plot generated for {model_type}."
        )
    except Exception as e:
        logging.error(
            f"Error during feature importances plotting for {model_type}: {e}"
        )
        raise


def plot_predicted_vs_actual(
    model: RandomForestRegressor,
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "Model",
    title: str = "Predicted vs Actual",
    figsize: tuple = (8, 6),
) -> None:
    """
    Plots Predicted vs Actual values.

    Parameters:
        model (RandomForestRegressor): Fitted Random Forest model.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Actual target values.
        model_type (str): Type of model for labeling purposes.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure.
    """
    try:
        predictions = model.predict(X)
        plt.figure(figsize=figsize)
        sns.scatterplot(x=y, y=predictions, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{model_type}: {title}")
        plt.grid(True, ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(
            f"/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Figs/{model_type}_predicted_vs_actual.png"
        )
        logging.info(
            f"Predicted vs Actual plot generated for {model_type}."
        )
    except Exception as e:
        logging.error(
            f"Error during Predicted vs Actual plotting for {model_type}: {e}"
        )
        raise


def plot_residuals(
    model: RandomForestRegressor,
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "Model",
    title: str = "Residuals vs Predicted",
    figsize: tuple = (8, 6),
) -> None:
    """
    Plots Residuals vs Predicted values.

    Parameters:
        model (RandomForestRegressor): Fitted Random Forest model.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Actual target values.
        model_type (str): Type of model for labeling purposes.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure.
    """
    try:
        predictions = model.predict(X)
        residuals = y - predictions
        plt.figure(figsize=figsize)
        sns.scatterplot(x=predictions, y=residuals, alpha=0.5)
        plt.axhline(0, color="r", linestyle="--", linewidth=2)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f"{model_type}: {title}")
        plt.grid(True, ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(
            f"/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Figs/{model_type}_residuals.png"
        )
        logging.info(
            f"Residuals vs Predicted plot generated for {model_type}."
        )
    except Exception as e:
        logging.error(
            f"Error during Residuals vs Predicted plotting for {model_type}: {e}"
        )
        raise


def calculate_additional_metrics(
    model: RandomForestRegressor,
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "Model",
) -> None:
    """
    Calculates and logs additional performance metrics.

    Parameters:
        model (Ridge): Fitted Ridge Regression model.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Actual target values.
        model_type (str): Type of model for labeling purposes.
    """
    try:
        # Generate predictions
        predictions = model.predict(X)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y, predictions)

        # Calculate Root Mean Squared Error (RMSE)
        rmse = root_mean_squared_error(y, predictions)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y, predictions)

        # Calculate R² Score
        r2 = r2_score(y, predictions)

        # Calculate Pearson Correlation Coefficient
        # Using numpy's corrcoef which returns a matrix; [0,1] element is the correlation
        correlation_matrix = np.corrcoef(y, predictions)
        pearson_corr = correlation_matrix[0, 1]

        # Alternatively, using scipy's pearsonr which returns correlation and p-value
        # from scipy.stats import pearsonr
        # pearson_corr, p_value = pearsonr(y, predictions)

        # Log the metrics
        logging.info(f"{model_type} Performance Metrics:")
        logging.info(f"Mean Squared Error (MSE): {mse:.4f}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        logging.info(f"R² Score: {r2:.4f}")
        logging.info(
            f"Pearson Correlation Coefficient: {pearson_corr:.4f}"
        )

    except Exception as e:
        logging.error(
            f"Error during additional metrics calculation for {model_type}: {e}"
        )
        raise


# -----------------------------------------------------------------------------------
# Main Execution Flow for Verification and Visualization
# -----------------------------------------------------------------------------------


def verification_and_visualization(
    grid_search_insitu: GridSearchCV,
    grid_search_anvil: GridSearchCV,
    final_model_insitu: RandomForestRegressor,
    final_model_anvil: RandomForestRegressor,
    X_clean_insitu: np.ndarray,
    X_clean_anvil: np.ndarray,
    y_clean_insitu: np.ndarray,
    y_clean_anvil: np.ndarray,
    feature_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:

    # Split the data into training and testing sets
    logging.info("Splitting data into training and testing sets...")
    X_train_insitu, X_test_insitu, y_train_insitu, y_test_insitu = (
        train_test_split(
            X_clean_insitu,
            y_clean_insitu,
            test_size=test_size,
            random_state=random_state,
        )
    )
    X_train_anvil, X_test_anvil, y_train_anvil, y_test_anvil = (
        train_test_split(
            X_clean_anvil,
            y_clean_anvil,
            test_size=test_size,
            random_state=random_state,
        )
    )

    logging.info("Data splitting completed.")

    # Define model configurations for iteration
    models = [
        {
            "grid_search": grid_search_insitu,
            "final_model": final_model_insitu,
            "X_test": X_test_insitu,
            "y_test": y_test_insitu,
            "model_type": "In-Situ Cirrus Random Forest",
        },
        {
            "grid_search": grid_search_anvil,
            "final_model": final_model_anvil,
            "X_test": X_test_anvil,
            "y_test": y_test_anvil,
            "model_type": "Anvil Cirrus Random Forest",
        },
    ]

    for model in models:

        # Extract the best parameters for the Random Forest estimator
        model_type = model["model_type"]

        # Calculate additional metrics first
        calculate_additional_metrics(
            model=model["final_model"],
            X=model["X_test"],
            y=model["y_test"],
            model_type=model_type,
        )

        # Evaluate Model Performance on Test Set
        logging.info(f"Evaluating {model_type} on the test set...")
        plot_predicted_vs_actual(
            model=model["final_model"],
            X=model["X_test"],
            y=model["y_test"],
            model_type=model_type,
            title=f"{model_type}: Predicted vs Actual",
            figsize=(8, 6),
        )

        # Plot Feature Importances
        plot_feature_importances(
            model=model["final_model"],
            feature_names=feature_names,
            model_type=model_type,
            title=f"{model_type}: Feature Importances",
        )

        plot_residuals(
            model=model["final_model"],
            X=model["X_test"],
            y=model["y_test"],
            model_type=model_type,
            title=f"{model_type}: Residuals vs Predicted",
            figsize=(8, 6),
        )

    logging.info(
        "Verification and visualization processes completed successfully."
    )


# -----------------------------------------------------------------------------------
# Example Usage within Main Execution Flow
# -----------------------------------------------------------------------------------


def main():
    """
    Main function to execute data loading, preprocessing, modeling, verification, and visualization.
    """
    try:

        # -----------------------------
        # Load or Define Your Data Here
        # -----------------------------
        # Example:
        data = load_from_pkl(
            # "C:\\Users\Muqy\OneDrive\Project_Python\Main_python\Data_python\Ridge_regression_data\cleaned_data.pkl"
            "/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Data/cleaned_data_no_zeros_grid_point_4degree.pkl",  # Define your desired save path
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

        feature_names = [
            "Upper_tropopause_stability",
            "Tropopause_relative_humidity",
            "Tropopause_temp",
            "Skin_temperature",
            "Upper_tropopause_wind_shear",
            "Tropopause_u_wind",
            "Tropopause_v_wind",
        ]

        # -----------------------------------------------------------------------------------
        # Initialize Parameters
        # -----------------------------------------------------------------------------------

        # Define cross-validation strategy
        k_folds = 5  # Number of folds for K-Fold CV
        cv_strategy = KFold(
            n_splits=k_folds, shuffle=True, random_state=42
        )

        # Define scoring metric
        scoring_metric = "r2"

        # Define number of parallel jobs
        parallel_jobs = 10  # Utilize all available cores

        # Define verbosity level
        verbosity = 1

        # -----------------------------------------------------------------------------------
        # Create Ridge Regression Pipeline
        # -----------------------------------------------------------------------------------
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Standardizes the data
                (
                    "random_forest",
                    RandomForestRegressor(random_state=42, n_jobs=16),
                ),  # Random Forest Regression
            ]
        )

        # -----------------------------------------------------------------------------------
        # Grid Search for In-Situ Cirrus
        # -----------------------------------------------------------------------------------
        logging.info("Starting GridSearchCV for In-Situ Cirrus ...")

        # Define parameter grid for coarse search
        param_grid = {
            # The number of trees in the forest. More trees can improve performance but increase computation time.
            "random_forest__n_estimators": [
                1000,
                1200,
                1500,
            ],
            # The maximum depth of the trees. Limiting depth can prevent overfitting.
            "random_forest__max_depth": [10, 20, 30, 40],
            # The minimum number of samples required to split an internal node.
            "random_forest__min_samples_split": [5, 10, 20],
            # The minimum number of samples required to be at a leaf node.
            "random_forest__min_samples_leaf": [2, 4, 6],
            # The number of features to consider when looking for the best split. Options:
            # "auto": Uses all features.
            # "sqrt": Uses the square root of the number of features.
            # "log2": Uses the logarithm base 2 of the number of features.
            "random_forest__max_features": ["auto", "sqrt", "log2"],
        }

        # Perform coarse grid search for In-Situ Cirrus
        grid_search_insitu = perform_grid_search(
            pipeline=pipeline,
            X=X_clean_insitu,
            y=y_clean_insitu,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring_metric,
            n_jobs=parallel_jobs,
            verbose=verbosity,
        )

        best_params_insitu = grid_search_insitu.best_params_
        logging.info(
            f"Best parameters for In-Situ Cirrus: {best_params_insitu}"
        )

        # -----------------------------------------------------------------------------------
        # Grid Search for Anvil Cirrus
        # -----------------------------------------------------------------------------------
        logging.info("Starting GridSearchCV for Anvil Cirrus ...")

        # Perform coarse grid search for Anvil Cirrus
        grid_search_anvil = perform_grid_search(
            pipeline=pipeline,
            X=X_clean_anvil,
            y=y_clean_anvil,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring_metric,
            n_jobs=parallel_jobs,
            verbose=verbosity,
        )

        best_params_anvil = grid_search_anvil.best_params_
        logging.info(
            f"Best parameters for Anvil Cirrus: {best_params_anvil}"
        )

        # -----------------------------------------------------------------------------------
        # Fit Final Models with Best Alphas
        # -----------------------------------------------------------------------------------
        # Fit Final Models with Best Parameters
        logging.info(
            "Fitting the final Random Forest model for In-Situ Cirrus..."
        )
        # Extract the best parameters for the Random Forest estimator
        best_rf_params_insitu = {
            key.replace("random_forest__", ""): value
            for key, value in best_params_insitu.items()
        }
        final_model_insitu = fit_final_model(
            best_params=best_rf_params_insitu,
            X=X_clean_insitu,
            y=y_clean_insitu,
        )
        logging.info(
            "Final Random Forest model for In-Situ Cirrus fitted."
        )

        logging.info(
            "Fitting the final Random Forest model for Anvil Cirrus..."
        )
        best_rf_params_anvil = {
            key.replace("random_forest__", ""): value
            for key, value in best_params_anvil.items()
        }
        final_model_anvil = fit_final_model(
            best_params=best_rf_params_anvil,
            X=X_clean_anvil,
            y=y_clean_anvil,
        )
        logging.info(
            "Final Random Forest model for Anvil Cirrus fitted."
        )

        # -----------------------------------------------------------------------------------
        # Verification and Visualization
        # -----------------------------------------------------------------------------------
        verification_and_visualization(
            grid_search_insitu=grid_search_insitu,
            grid_search_anvil=grid_search_anvil,
            final_model_insitu=final_model_insitu,
            final_model_anvil=final_model_anvil,
            X_clean_insitu=X_clean_insitu,
            X_clean_anvil=X_clean_anvil,
            y_clean_insitu=y_clean_insitu,
            y_clean_anvil=y_clean_anvil,
            feature_names=feature_names,
            test_size=0.1,  # 20% of data used for testing
            random_state=42,
        )

    except Exception as e:
        logging.error(
            f"An error occurred in the main execution flow: {e}"
        )
        raise


def main_without_gridsearch():
    """
    Main function to execute data loading, preprocessing, modeling, verification, and visualization.
    """
    try:
        # --------------------------------
        # Load Your Data
        # --------------------------------
        data = load_from_pkl(
            # "C:\\Users\Muqy\OneDrive\Project_Python\Main_python\Data_python\Ridge_regression_data\cleaned_data.pkl"
            "/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Data/cleaned_data_no_zeros_grid_point.pkl",  # Define your desired save path
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

        feature_names = [
            "Upper_tropopause_stability",
            "Tropopause_temp",
            "Tropopause_u_wind",
            "Tropopause_v_wind",
            "Upper_tropopause_wind_shear",
            "Skin_temperature",
            "Tropopause_relative_humidity",
        ]

        del data
        gc.collect()

        # --------------------------------
        # Data Preprocessing
        # --------------------------------

        # --------------------------------
        # Split the Data into Training and Testing Sets
        # --------------------------------
        test_size = 0.1
        random_state = 42
        X_train_insitu, X_test_insitu, y_train_insitu, y_test_insitu = (
            train_test_split(
                X_clean_insitu,
                y_clean_insitu,
                test_size=test_size,
                random_state=random_state,
            )
        )
        X_train_anvil, X_test_anvil, y_train_anvil, y_test_anvil = (
            train_test_split(
                X_clean_anvil,
                y_clean_anvil,
                test_size=test_size,
                random_state=random_state,
            )
        )

        # --------------------------------
        # Create and Fit the Pipeline
        # --------------------------------
        # Define the pipeline with StandardScaler and RandomForestRegressor
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "random_forest",
                    RandomForestRegressor(
                        random_state=random_state, n_jobs=-1
                    ),
                ),
            ]
        )

        # Fit the model for In-Situ Cirrus
        logging.info(
            "Fitting Random Forest model for In-Situ Cirrus..."
        )
        pipeline.fit(X_train_insitu, y_train_insitu)
        logging.info("Random Forest model for In-Situ Cirrus fitted.")
        final_model_insitu = pipeline.named_steps["random_forest"]

        # Fit the model for Anvil Cirrus
        logging.info("Fitting Random Forest model for Anvil Cirrus...")
        pipeline.fit(X_train_anvil, y_train_anvil)
        logging.info("Random Forest model for Anvil Cirrus fitted.")
        final_model_anvil = pipeline.named_steps["random_forest"]

        # --------------------------------
        # Evaluation and Visualization
        # --------------------------------
        # Define model configurations for iteration
        models = [
            {
                "final_model": final_model_insitu,
                "X_test": X_test_insitu,
                "y_test": y_test_insitu,
                "model_type": "In-Situ Cirrus Random Forest",
            },
            {
                "final_model": final_model_anvil,
                "X_test": X_test_anvil,
                "y_test": y_test_anvil,
                "model_type": "Anvil Cirrus Random Forest",
            },
        ]

        for model in models:
            # Extract the best parameters for the Random Forest estimator
            model_type = model["model_type"]

            calculate_additional_metrics(
                model=model["final_model"],
                X=model["X_test"],
                y=model["y_test"],
                model_type=model_type,
            )

            # Plot Feature Importances
            plot_feature_importances(
                model=model["final_model"],
                feature_names=feature_names,
                model_type=model_type,
                title=f"{model_type}: Feature Importances",
            )

            # Evaluate Model Performance on Test Set
            logging.info(f"Evaluating {model_type} on the test set...")
            plot_predicted_vs_actual(
                model=model["final_model"],
                X=model["X_test"],
                y=model["y_test"],
                model_type=model_type,
                title=f"{model_type}: Predicted vs Actual",
                figsize=(8, 6),
            )

            plot_residuals(
                model=model["final_model"],
                X=model["X_test"],
                y=model["y_test"],
                model_type=model_type,
                title=f"{model_type}: Residuals vs Predicted",
                figsize=(8, 6),
            )

        logging.info(
            "Testing of Random Forest models completed successfully."
        )

    except Exception as e:
        logging.error(
            f"An error occurred in the main execution flow: {e}"
        )
        raise


if __name__ == "__main__":

    main()
