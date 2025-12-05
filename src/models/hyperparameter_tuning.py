"""
Hyperparameter Tuning Module

This module provides functions for hyperparameter tuning using various optimization techniques,
including Optuna and RandomizedSearchCV.
"""

import argparse
import logging
import os
import sys

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class HyperparameterTuner:
    """
    A class for performing hyperparameter tuning using various optimization techniques.
    """

    def __init__(
        self, model_type="xgboost", n_trials=50, cv_folds=5, scoring="roc_auc"
    ):
        """
        Initialize the HyperparameterTuner.

        Args:
            model_type: Type of model to tune ('xgboost', 'logistic_regression', or 'random_forest')
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric to optimize
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.logger = logging.getLogger(__name__)

    def get_param_space(self, model_type):
        """
        Define the hyperparameter search space for different model types.

        Args:
            model_type: Type of model ('xgboost', 'logistic_regression', or 'random_forest')

        Returns:
            Dictionary or function for the respective model type
        """
        if model_type == "xgboost":

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                    "random_state": 42,
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                }
                return xgb.XGBClassifier(**params)

        elif model_type == "logistic_regression":

            def objective(trial):
                params = {
                    "C": trial.suggest_float("C", 0.01, 100, log=True),
                    "penalty": trial.suggest_categorical(
                        "penalty", ["l1", "l2", "elasticnet"]
                    ),
                    "solver": "saga",  # Supports all penalties
                    "max_iter": 1000,
                    "random_state": 42,
                    "l1_ratio": (
                        trial.suggest_float("l1_ratio", 0, 1)
                        if trial.suggest_categorical(
                            "penalty", ["l1", "l2", "elasticnet"]
                        )
                        == "elasticnet"
                        else None
                    ),
                }
                # Remove l1_ratio if not using elasticnet
                if params["penalty"] != "elasticnet":
                    params.pop("l1_ratio", None)

                return LogisticRegression(**params)

        elif model_type == "random_forest":

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical(
                        "max_features", ["sqrt", "log2", None]
                    ),
                    "random_state": 42,
                }
                return RandomForestClassifier(**params)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return objective

    def tune_with_optuna(self, X, y, timeout=300):
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            X: Training features
            y: Training target
            timeout: Timeout in seconds for the optimization

        Returns:
            Best model parameters and best score
        """
        self.logger.info(
            f"Starting hyperparameter tuning with Optuna for {self.model_type}"
        )

        # Get the objective function for the model type
        objective_func = self.get_param_space(self.model_type)

        def objective(trial):
            model = objective_func(trial)

            # Handle class imbalance by calculating scale_pos_weight for XGBoost
            if self.model_type == "xgboost":
                scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
                model.set_params(scale_pos_weight=scale_pos_weight)

            # Handle missing values
            X_filled = X.fillna(X.median(numeric_only=True))

            # Perform cross-validation
            scores = cross_val_score(
                model,
                X_filled,
                y,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=-1,  # Use all available cores
            )

            # Log intermediate results to MLflow if in a run
            try:
                trial_number = trial.number
                mlflow.log_metric(f"trial_{trial_number}_score", scores.mean())
            except Exception as e:
                self.logger.debug(f"MLflow trial logging failed, continuing: {str(e)}")
                pass  # If not in MLflow run, continue without logging

            return scores.mean()

        # Create study and optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, timeout=timeout)

        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best cross-validation score: {best_score:.4f}")

        # Update best params with fixed values that are always needed
        if self.model_type == "xgboost":
            best_params.update(
                {
                    "random_state": 42,
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                }
            )
        elif self.model_type == "logistic_regression":
            best_params.update(
                {
                    "random_state": 42,
                    "max_iter": 1000,
                    "solver": "saga",
                }
            )
        elif self.model_type == "random_forest":
            best_params.update(
                {
                    "random_state": 42,
                }
            )

        return best_params, best_score

    def tune_with_random_search(self, X, y):
        """
        Perform hyperparameter tuning using RandomizedSearchCV.

        Args:
            X: Training features
            y: Training target

        Returns:
            Best model parameters and best score
        """
        self.logger.info(
            f"Starting hyperparameter tuning with RandomizedSearchCV for {self.model_type}"
        )

        if self.model_type == "xgboost":
            model = xgb.XGBClassifier(
                random_state=42, objective="binary:logistic", eval_metric="auc"
            )

            param_distributions = {
                "n_estimators": randint(50, 300),
                "max_depth": randint(3, 10),
                "learning_rate": uniform(0.01, 0.3),
                "subsample": uniform(0.5, 0.5),
                "colsample_bytree": uniform(0.5, 0.5),
                "min_child_weight": randint(1, 10),
                "reg_alpha": uniform(0, 10),
                "reg_lambda": uniform(0, 10),
            }

        elif self.model_type == "logistic_regression":
            model = LogisticRegression(random_state=42, max_iter=1000, solver="saga")

            param_distributions = {
                "C": uniform(0.01, 100),
                "penalty": ["l1", "l2", "elasticnet"],
                "l1_ratio": uniform(0, 1),
            }

        elif self.model_type == "random_forest":
            model = RandomForestClassifier(random_state=42)

            param_distributions = {
                "n_estimators": randint(50, 300),
                "max_depth": randint(3, 20),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", None],
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Handle class imbalance for XGBoost
        if self.model_type == "xgboost":
            scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
            model.set_params(scale_pos_weight=scale_pos_weight)

        # Handle missing values
        X_filled = X.fillna(X.median(numeric_only=True))

        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=self.n_trials,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        random_search.fit(X_filled, y)

        best_params = random_search.best_params_
        best_score = random_search.best_score_

        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best cross-validation score: {best_score:.4f}")

        return best_params, best_score


def load_data(train_path: str):
    """
    Load and prepare training data.

    Args:
        train_path: Path to the training data CSV file

    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Loading training data from {train_path}")
    df = pd.read_csv(train_path)

    logger.info(f"Loaded data with shape: {df.shape}")

    # Separate features and target
    if "default" in df.columns:
        y = df["default"]
        X = df.drop("default", axis=1)
    else:
        raise ValueError("Target column 'default' not found in the dataset")

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    return X, y


def main(
    train_data_path: str,
    model_type: str = "xgboost",
    tuning_method: str = "optuna",  # 'optuna' or 'random_search'
    n_trials: int = 50,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    timeout: int = 300,  # 5 minutes timeout for optuna
    mlflow_tracking_uri: str = "http://localhost:5000",
):
    """
    Main function to perform hyperparameter tuning.

    Args:
        train_data_path: Path to training data
        model_type: Type of model to tune ('xgboost', 'logistic_regression', or 'random_forest')
        tuning_method: Method to use ('optuna' or 'random_search')
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric to optimize
        timeout: Timeout for optuna (in seconds)
        mlflow_tracking_uri: URI for MLflow tracking server
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting hyperparameter tuning...")

    # Set up MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Create or get the experiment
    experiment_name = f"Hyperparameter_Tuning_{model_type}"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception as e:
        # If experiment already exists, get its ID
        logger.debug(f"Experiment creation failed (likely already exists): {str(e)}")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id):
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("tuning_method", tuning_method)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("scoring", scoring)

        # Load data
        X, y = load_data(train_data_path)

        # Create tuner
        tuner = HyperparameterTuner(
            model_type=model_type, n_trials=n_trials, cv_folds=cv_folds, scoring=scoring
        )

        # Perform tuning based on method
        if tuning_method == "optuna":
            best_params, best_score = tuner.tune_with_optuna(X, y, timeout=timeout)
        elif tuning_method == "random_search":
            best_params, best_score = tuner.tune_with_random_search(X, y)
        else:
            raise ValueError(f"Unsupported tuning method: {tuning_method}")

        # Log best parameters and score to MLflow
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        mlflow.log_metric("best_cv_score", best_score)

        logger.info(f"Hyperparameter tuning completed. Best score: {best_score:.4f}")
        logger.info(f"Run ID: {mlflow.active_run().info.run_id}")

        return best_params, best_score


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("hyperparameter_tuning.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Perform hyperparameter tuning for credit scoring models."
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/processed/train.csv",
        help="Path to the training data file (default: data/processed/train.csv)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["logistic_regression", "xgboost", "random_forest"],
        default="xgboost",
        help="Type of model to tune (default: xgboost)",
    )
    parser.add_argument(
        "--tuning_method",
        type=str,
        choices=["optuna", "random_search"],
        default="optuna",
        help="Method to use for hyperparameter tuning (default: optuna)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="roc_auc",
        help="Scoring metric to optimize (default: roc_auc)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,  # 5 minutes
        help="Timeout for optuna in seconds (default: 300)",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI (default: http://localhost:5000)",
    )

    args = parser.parse_args()

    try:
        best_params, best_score = main(
            train_data_path=args.train_data_path,
            model_type=args.model_type,
            tuning_method=args.tuning_method,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
            scoring=args.scoring,
            timeout=args.timeout,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )

        logger.info("Hyperparameter tuning completed successfully!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best cross-validation score: {best_score:.4f}")

    except Exception as e:
        logger.error(f"Hyperparameter tuning failed with error: {str(e)}")
        sys.exit(1)
