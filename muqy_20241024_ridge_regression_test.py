# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2024-08-20 08:38
# @Last Modified by:   Muqy
# @Last Modified time: 2024-10-31 15:18

from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing
import pickle
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    train_test_split,
)
from scipy.stats import pearsonr

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

########################################################################################
########################################################################################
# Ridge Regression Model Training and Evaluation Pipeline #
########################################################################################
########################################################################################


# -----------------------------------------------------------------------------------
# Ridge Regression Model Training and Evaluation Pipeline
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
) -> GridSearchCV:
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
    alpha: float,
    X: np.ndarray,
    y: np.ndarray,
    solver: str = "saga",
    max_iter: int = 5000,
    tol: float = 1e-3,
) -> Ridge:
    """
    Fit the final Ridge Regression model with the best alpha.

    Parameters:
    - alpha (float): Regularization strength
    - X (np.ndarray): Feature matrix
    - y (np.ndarray): Target vector
    - solver (str): Solver to use in Ridge Regression
    - max_iter (int): Maximum number of iterations
    - tol (float): Tolerance for optimization

    Returns:
    - Ridge: Fitted Ridge model
    """
    try:
        logging.info(
            f"Fitting Ridge Regression model with alpha={alpha}"
        )
        model = Ridge(
            alpha=alpha, solver=solver, max_iter=max_iter, tol=tol
        )
        model.fit(X, y)
        logging.info("Ridge Regression model fitted successfully.")
        return model
    except Exception as e:
        logging.error(f"Error during Ridge model fitting: {e}")
        raise


def plot_performance_vs_alpha(
    grid_search: GridSearchCV, title: str, model_type: str
) -> None:
    """
    Plots the negative mean squared error vs alpha.

    Parameters:
        grid_search (GridSearchCV): Fitted GridSearchCV object.
        title (str): Title of the plot.
    """
    try:
        # Extract alphas and mean_test_score from cv_results_
        alphas = grid_search.cv_results_["param_ridge__alpha"].data
        mean_test_scores = grid_search.cv_results_["mean_test_score"]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(
            alphas,
            mean_test_scores,
            marker="o",
            linestyle="-",
            color="b",
        )
        plt.xscale("log")
        plt.xlabel("Alpha")
        plt.ylabel("Negative Mean Squared Error")
        plt.title(title)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.tight_layout()
        plt.savefig(
            f"/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Figs/{model_type}_performance_vs_alpha.png"
        )
        logging.info(f"Performance vs Alpha plot generated: {title}")
    except Exception as e:
        logging.error(
            f"Error during performance vs alpha plotting: {e}"
        )
        raise


def _fit_single_alpha(args):
    """
    Helper function to fit Ridge model for a single alpha value.
    """
    alpha, X, y = args
    try:
        model = Ridge(
            alpha=alpha, solver="saga", max_iter=5000, tol=1e-3
        )
        model.fit(X, y)
        return model.coef_
    except Exception as e:
        logging.error(f"Error fitting model for alpha={alpha}: {e}")
        raise


def collect_coefficients(
    alphas: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    variables: List[str],
    model_type: str = "Model",
) -> np.ndarray:
    """
    Parallel implementation of coefficient collection using ProcessPoolExecutor.
    """
    try:
        num_cores = max(
            1, multiprocessing.cpu_count() - 1
        )  # Leave one core free
        logging.info(
            f"Collecting coefficients for {model_type} across {len(alphas)} alphas using {num_cores} cores..."
        )

        # Prepare arguments for parallel processing
        args_list = [(alpha, X, y) for alpha in alphas]

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            coefficients = list(
                executor.map(_fit_single_alpha, args_list)
            )

        coefficients_array = np.array(coefficients)
        logging.info(
            f"Coefficient collection for {model_type} complete."
        )
        return coefficients_array
    except Exception as e:
        logging.error(
            f"Error during coefficient collection for {model_type}: {e}"
        )
        raise


def plot_coefficient_paths(
    alphas: np.ndarray,
    coefficients: np.ndarray,
    variables: List[str],
    title: str,
    model_type: str,
) -> None:
    """
    Plots the coefficients of each feature as a function of alpha.

    Parameters:
        alphas (np.ndarray): Array of alpha values.
        coefficients (np.ndarray): Array of coefficients for each alpha.
        variables (List[str]): List of feature names.
        title (str): Title of the plot.
    """
    try:
        plt.figure(figsize=(12, 8))
        for i, var in enumerate(variables):
            plt.plot(alphas, coefficients[:, i], label=var)
        plt.xscale("log")
        plt.xlabel("Alpha")
        plt.ylabel("Coefficient Value")
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(
            f"/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Figs/{model_type} coefficient_paths.png"
        )
        logging.info(f"Coefficient paths plot generated: {title}")
    except Exception as e:
        logging.error(f"Error during coefficient paths plotting: {e}")
        raise


def plot_coefficient_contributions(
    coefficients: np.ndarray,
    variables: List[str],
    best_alpha: float,
    model_type: str = "Model",
) -> None:
    """
    Plots the contribution of each feature in the Ridge Regression model at the best alpha.

    Parameters:
        coefficients (np.ndarray): Array of coefficients at the best alpha.
        variables (List[str]): List of feature names.
        best_alpha (float): The alpha value corresponding to these coefficients.
        model_type (str): Type of model for labeling purposes.
    """
    try:
        # Create a DataFrame for easier plotting
        coef_df = pd.DataFrame(
            {"Feature": variables, "Coefficient": coefficients}
        )

        # Calculate absolute coefficients for contribution
        coef_df["Absolute_Coefficient"] = coef_df["Coefficient"].abs()

        # Sort by absolute coefficient
        coef_df_sorted = coef_df.sort_values(
            by="Absolute_Coefficient", ascending=True
        )

        plt.figure(figsize=(10, 8))
        plt.barh(
            coef_df_sorted["Feature"],
            coef_df_sorted["Coefficient"],
            color="skyblue",
        )
        plt.xlabel("Coefficient Value")
        plt.title(
            f"{model_type}: Feature Contributions at Alpha = {best_alpha:.2e}"
        )
        plt.grid(True, axis="x", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(
            f"/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Figs/{model_type} coefficient_contributions.png"
        )
        logging.info(
            f"Coefficient contributions plot generated for {model_type} at alpha={best_alpha:.2e}"
        )
    except Exception as e:
        logging.error(
            f"Error during coefficient contributions plotting for {model_type}: {e}"
        )
        raise


def plot_predicted_vs_actual(
    model: Ridge,
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "Model",
    title: str = "Predicted vs Actual",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plots Predicted vs Actual values.

    Parameters:
        model (Ridge): Fitted Ridge Regression model.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Actual target values.
        model_type (str): Type of model for labeling purposes.
        title (str): Title of the plot.
        figsize (Tuple[int, int]): Size of the figure.
    """
    try:
        predictions = model.predict(X)
        plt.figure(figsize=figsize)
        sns.scatterplot(x=y, y=predictions, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{model_type}: Predicted vs Actual")
        plt.grid(True, ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(
            f"/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Figs/{model_type} predicted_vs_actual.png"
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
    model: Ridge,
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "Model",
    title: str = "Residuals vs Predicted",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plots Residuals vs Predicted values.

    Parameters:
        model (Ridge): Fitted Ridge Regression model.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Actual target values.
        model_type (str): Type of model for labeling purposes.
        title (str): Title of the plot.
        figsize (Tuple[int, int]): Size of the figure.
    """
    try:
        predictions = model.predict(X)
        residuals = y - predictions
        plt.figure(figsize=figsize)
        sns.scatterplot(x=predictions, y=residuals, alpha=0.5)
        plt.axhline(0, color="r", linestyle="--", linewidth=2)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f"{model_type}: Residuals vs Predicted")
        plt.grid(True, ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(
            f"/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Figs/{model_type} residuals_vs_predicted.png"
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
    model: Ridge,
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
        # correlation_matrix = np.corrcoef(y, predictions)
        # pearson_corr = correlation_matrix[0, 1]

        # Alternatively, using scipy's pearsonr which returns correlation and p-value
        pearson_corr, p_value = pearsonr(y, predictions)

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
    final_model_insitu: Ridge,
    final_model_anvil: Ridge,
    X_clean_insitu: np.ndarray,
    X_clean_anvil: np.ndarray,
    y_clean_insitu: np.ndarray,
    y_clean_anvil: np.ndarray,
    feature_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Perform verification and visualization tasks including multicollinearity assessment
    and plotting of model performance and coefficients.

    Parameters:
        grid_search_insitu_coarse (GridSearchCV): Grid search object for in-situ cirrus (coarse search).
        grid_search_insitu_fine (GridSearchCV): Grid search object for in-situ cirrus (fine search).
        grid_search_anvil_coarse (GridSearchCV): Grid search object for anvil cirrus (coarse search).
        grid_search_anvil_fine (GridSearchCV): Grid search object for anvil cirrus (fine search).
        final_model_insitu (Ridge): Final fitted Ridge model for in-situ cirrus.
        final_model_anvil (Ridge): Final fitted Ridge model for anvil cirrus.
        X_clean (np.ndarray): Cleaned feature matrix.
        y_clean_insitu (np.ndarray): Cleaned target vector for in-situ cirrus.
        y_clean_anvil (np.ndarray): Cleaned target vector for anvil cirrus.
        feature_names (List[str]): List of feature names.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.
        save_vif (bool): Whether to save VIF results to CSV.
        vif_filename_insitu (str): Filename for saving in-situ VIF data.
        vif_filename_anvil (str): Filename for saving anvil VIF data.
    """
    # -----------------------------------------------------------------------------------
    # Split the Data into Training and Testing Sets
    # -----------------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------------
    # 2. Visualization of Model Performance and Coefficients
    # -----------------------------------------------------------------------------------

    # Define model configurations for iteration
    models = [
        {
            "grid_search_coarse": grid_search_insitu,
            "best_alpha": grid_search_insitu.best_params_[
                "ridge__alpha"
            ],
            "model_type": "In-Situ Cirrus Ridge",
            "final_model": final_model_insitu,
            "X_train": X_train_insitu,
            "y_train": y_train_insitu,
            "X_test": X_test_insitu,
            "y_test": y_test_insitu,
        },
        {
            "grid_search_coarse": grid_search_anvil,
            "best_alpha": grid_search_insitu.best_params_[
                "ridge__alpha"
            ],
            "model_type": "Anvil Cirrus Ridge",
            "final_model": final_model_anvil,
            "X_train": X_train_anvil,
            "y_train": y_train_anvil,
            "X_test": X_test_anvil,
            "y_test": y_test_anvil,
        },
    ]

    for model in models:
        model_type = model["model_type"]

        calculate_additional_metrics(
            model=model["final_model"],
            X=model["X_test"],
            y=model["y_test"],
            model_type=model_type,
        )

        # a. Plot Performance Metrics vs Alpha (Coarse Search)
        plot_performance_vs_alpha(
            grid_search=model["grid_search_coarse"],
            title=f"{model_type}: Performance vs Alpha (Coarse Search)",
            model_type=model_type,
        )

        # c. Collect and Plot Coefficient Paths
        best_alphas = model["grid_search_coarse"].param_grid[
            "ridge__alpha"
        ]
        coefficients = collect_coefficients(
            alphas=best_alphas,
            X=model["X_train"],
            y=model["y_train"],
            variables=feature_names,
            model_type=model_type,
        )

        plot_coefficient_paths(
            alphas=best_alphas,
            coefficients=coefficients,
            variables=feature_names,
            title=f"{model_type}: Coefficient Paths vs Alpha",
            model_type=model_type,
        )

        # d. Plot Coefficient Contributions at Best Alpha
        # Find the index of the best alpha
        alpha_index = np.argmin(
            np.abs(
                np.log10(best_alphas) - np.log10(model["best_alpha"])
            )
        )
        best_coefficients = coefficients[alpha_index]

        plot_coefficient_contributions(
            coefficients=best_coefficients,
            variables=feature_names,
            best_alpha=model["best_alpha"],
            model_type=model_type,
        )

        # e. Evaluate Model Performance on Test Set
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
            "/data02/CAR_D1/PYTHON_CODE/CloudSat/filtered_data_generator/Data/cleaned_data_no_zeros_grid_point_scaled_5degree.pkl",  # Define your desired save path
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

        # Define logarithmically spaced alphas for coarse search
        applicant_alphas = np.logspace(-10, 10, 1000)

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
        verbosity = 0

        # -----------------------------------------------------------------------------------
        # Create Ridge Regression Pipeline
        # -----------------------------------------------------------------------------------
        pipeline = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),  # Standardizes the data
                (
                    "ridge",
                    Ridge(solver="saga", max_iter=5000, tol=1e-3),
                ),  # Ridge Regression
            ]
        )

        # -----------------------------------------------------------------------------------
        # Grid Search for In-Situ Cirrus (Coarse Search)
        # -----------------------------------------------------------------------------------
        logging.info("Starting GridSearchCV for In-Situ Cirrus ...")

        # Define parameter grid for coarse search
        param_grid = {"ridge__alpha": applicant_alphas}

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

        best_alpha_insitu = grid_search_insitu.best_params_[
            "ridge__alpha"
        ]
        logging.info(
            f"Best alpha for In-Situ Cirrus: {best_alpha_insitu}"
        )

        # -----------------------------------------------------------------------------------
        # Grid Search for Anvil Cirrus (Coarse Search)
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

        best_alpha_anvil = grid_search_anvil.best_params_[
            "ridge__alpha"
        ]
        logging.info(f"Best alpha for Anvil Cirrus: {best_alpha_anvil}")

        # -----------------------------------------------------------------------------------
        # Fit Final Models with Best Alphas
        # -----------------------------------------------------------------------------------
        logging.info(
            "Fitting the final Ridge model for In-Situ Cirrus..."
        )
        final_model_insitu = fit_final_model(
            alpha=best_alpha_insitu,
            X=X_clean_insitu,
            y=y_clean_insitu,
        )
        logging.info("Final Ridge model for In-Situ Cirrus fitted.")

        logging.info(
            "Fitting the final Ridge model for Anvil Cirrus..."
        )
        final_model_anvil = fit_final_model(
            alpha=best_alpha_anvil,
            X=X_clean_anvil,
            y=y_clean_anvil,
        )
        logging.info("Final Ridge model for Anvil Cirrus fitted.")

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
            test_size=0.2,  # 10% of data used for testing
            random_state=42,
        )

    except Exception as e:
        logging.error(
            f"An error occurred in the main execution flow: {e}"
        )
        raise


if __name__ == "__main__":

    # Set the variables to extract
    cld_variables = [
        "insitu_fraction_weighted_2D",
        "anvil_fraction_weighted_2D",
    ]
    atms_variables = [
        "Upper_tropopause_stability",
        "Tropopause_relative_humidity",
        "Tropopause_temp",
        "Skin_temperature",
        "Upper_tropopause_wind_shear",
        "Tropopause_u_wind",
        "Tropopause_v_wind",
    ]

    main()
