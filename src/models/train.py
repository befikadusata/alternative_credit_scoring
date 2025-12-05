"""
Model Training Script

This script trains a baseline model for credit scoring using MLflow for experiment tracking.
It includes parameter tuning, model training, evaluation, and logging of results.
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.cleaning import load_and_clean_data

# Import our custom modules
from src.data.features import apply_feature_engineering


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("model_training.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


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

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")

    return X, y


def train_logistic_regression(X_train, y_train, X_val=None, y_val=None, **params):
    """
    Train a Logistic Regression model with optional validation.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        **params: Additional parameters for LogisticRegression

    Returns:
        Trained model
    """
    logger = logging.getLogger(__name__)

    # Set default parameters if not provided
    default_params = {
        "random_state": 42,
        "max_iter": 1000,
        "class_weight": "balanced",  # Handle imbalanced classes
    }

    # Override defaults with provided params
    default_params.update(params)

    logger.info(f"Training Logistic Regression with params: {default_params}")

    model = LogisticRegression(**default_params)
    model.fit(X_train, y_train)

    # Log parameters
    for param_name, param_value in default_params.items():
        mlflow.log_param(f"logistic_regression_{param_name}", param_value)

    logger.info("Logistic Regression training completed")
    return model


def train_xgboost(X_train, y_train, X_val=None, y_val=None, **params):
    """
    Train an XGBoost model with optional validation.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        **params: Additional parameters for XGBoost

    Returns:
        Trained model
    """
    logger = logging.getLogger(__name__)

    # Set default parameters if not provided
    default_params = {
        "random_state": 42,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "scale_pos_weight": len(y_train[y_train == 0])
        / len(y_train[y_train == 1]),  # Handle class imbalance
    }

    # Override defaults with provided params
    default_params.update(params)

    logger.info(f"Training XGBoost with params: {default_params}")

    # Handle any missing values that might still exist
    X_train = X_train.fillna(X_train.median(numeric_only=True))
    if X_val is not None:
        X_val = X_val.fillna(X_val.median(numeric_only=True))

    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)] if X_val is not None and y_val is not None else None,
        verbose=False,
    )

    # Log parameters
    for param_name, param_value in default_params.items():
        mlflow.log_param(f"xgboost_{param_name}", param_value)

    logger.info("XGBoost training completed")
    return model


def train_random_forest(X_train, y_train, X_val=None, y_val=None, **params):
    """
    Train a Random Forest model with optional validation.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        **params: Additional parameters for RandomForestClassifier

    Returns:
        Trained model
    """
    logger = logging.getLogger(__name__)

    # Set default parameters if not provided
    default_params = {
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",  # Handle imbalanced classes
    }

    # Override defaults with provided params
    default_params.update(params)

    logger.info(f"Training Random Forest with params: {default_params}")

    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)

    # Log parameters
    for param_name, param_value in default_params.items():
        mlflow.log_param(f"random_forest_{param_name}", param_value)

    logger.info("Random Forest training completed")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and log metrics to MLflow.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging purposes

    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger(__name__)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Calculate AUC if probabilities are available
    if y_pred_proba is not None:
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = -1  # Placeholder if AUC cannot be calculated

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    if auc != -1:
        mlflow.log_metric("auc", auc)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring="roc_auc")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    mlflow.log_metric("cv_auc_mean", cv_mean)
    mlflow.log_metric("cv_auc_std", cv_std)

    logger.info(
        f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"
    )
    logger.info(f"{model_name} - CV AUC: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")

    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Log detailed metrics for each class
    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict):
            if "precision" in metrics:
                mlflow.log_metric(f"precision_class_{class_name}", metrics["precision"])
            if "recall" in metrics:
                mlflow.log_metric(f"recall_class_{class_name}", metrics["recall"])
            if "f1-score" in metrics:
                mlflow.log_metric(f"f1_class_{class_name}", metrics["f1-score"])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    mlflow.log_metric("tn", cm[0, 0])
    mlflow.log_metric("fp", cm[0, 1])
    mlflow.log_metric("fn", cm[1, 0])
    mlflow.log_metric("tp", cm[1, 1])

    # Log metrics dict to return
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        "classification_report": class_report,
        "confusion_matrix": cm,
    }

    return metrics


def main(
    train_data_path: str,
    model_type: str = "xgboost",
    test_size: float = 0.2,
    random_state: int = 42,
    mlflow_tracking_uri: str = "http://localhost:5000",
):
    """
    Main function to train a baseline model.

    Args:
        train_data_path: Path to training data
        model_type: Type of model to train ('logistic_regression', 'xgboost', or 'random_forest')
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        mlflow_tracking_uri: URI for MLflow tracking server
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting baseline model training...")

    # Set up MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Create or get the experiment
    experiment_name = "Credit_Scoring_Baseline_Models"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        # If experiment already exists, get its ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id):
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Load data
        X, y = load_data(train_data_path)

        # Split data
        logger.info(
            f"Splitting data with test_size={test_size}, random_state={random_state}"
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

        # Handle missing values by filling with median for numeric columns
        numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
        X_train[numeric_cols] = X_train[numeric_cols].fillna(
            X_train[numeric_cols].median()
        )
        X_val[numeric_cols] = X_val[numeric_cols].fillna(
            X_train[numeric_cols].median()
        )  # Use training median

        # Initialize and train model based on type
        if model_type == "logistic_regression":
            model = train_logistic_regression(X_train, y_train, X_val, y_val)
        elif model_type == "xgboost":
            model = train_xgboost(X_train, y_train, X_val, y_val)
        elif model_type == "random_forest":
            model = train_random_forest(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Evaluate model
        logger.info("Evaluating model performance...")
        metrics = evaluate_model(model, X_val, y_val, model_type)

        # Log the model to MLflow
        if model_type in ["logistic_regression", "random_forest"]:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                conda_env=(
                    "environment.yml" if os.path.exists("environment.yml") else None
                ),
                registered_model_name=f"credit_scoring_{model_type}",
            )
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                conda_env=(
                    "environment.yml" if os.path.exists("environment.yml") else None
                ),
                registered_model_name=f"credit_scoring_{model_type}",
            )

        logger.info(
            f"Model training completed. Run ID: {mlflow.active_run().info.run_id}"
        )
        logger.info(f"Model registered as: credit_scoring_{model_type}")

    return model, metrics


if __name__ == "__main__":
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Train a baseline model for credit scoring."
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
        help="Type of model to train (default: xgboost)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for validation (default: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI (default: http://localhost:5000)",
    )

    args = parser.parse_args()

    try:
        model, metrics = main(
            train_data_path=args.train_data_path,
            model_type=args.model_type,
            test_size=args.test_size,
            random_state=args.random_state,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )

        logger.info("Baseline model training completed successfully!")
    except Exception as e:
        logger.error(f"Model training failed with error: {str(e)}")
        sys.exit(1)
