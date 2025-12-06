"""
Model Training Script

This script trains a baseline model for credit scoring using MLflow for experiment tracking.
It includes parameter tuning, model training, evaluation, and logging of results.
"""

import argparse
import logging
import os
import sys

import joblib
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

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.cleaning import DataCleaner


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
    # for param_name, param_value in default_params.items():
    #     mlflow.log_param(f"logistic_regression_{param_name}", param_value)

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
    }

    num_negative_samples = len(y_train[y_train == 0])
    num_positive_samples = len(y_train[y_train == 1])

    if num_positive_samples == 0:
        logger.warning(
            "No positive samples in training data. Setting scale_pos_weight to 1."
        )
        default_params["scale_pos_weight"] = 1
    elif num_negative_samples == 0:
        logger.warning(
            "No negative samples in training data. Setting scale_pos_weight to 1."
        )
        default_params["scale_pos_weight"] = 1
    else:
        default_params["scale_pos_weight"] = num_negative_samples / num_positive_samples

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
    # for param_name, param_value in default_params.items():
    #     mlflow.log_param(f"xgboost_{param_name}", param_value)

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
    # for param_name, param_value in default_params.items():
    #     mlflow.log_param(f"random_forest_{param_name}", param_value)

    logger.info("Random Forest training completed")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model.

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

    logger.info(
        f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"
    )

    # Cross-validation score
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring="roc_auc")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    logger.info(f"{model_name} - CV AUC: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")

    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

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
):
    """
    Main function to train a baseline model.

    Args:
        train_data_path: Path to training data
        model_type: Type of model to train ('logistic_regression', 'xgboost', or 'random_forest')
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting baseline model training...")

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

    # Initialize and use DataCleaner
    cleaner = DataCleaner()
    X_train = cleaner.clean_loan_data(X_train, exclude_columns=["default"])
    X_val = cleaner.clean_loan_data(X_val, exclude_columns=["default"])

    X_train = cleaner.encode_categorical_features(X_train, fit=True)
    X_val = cleaner.encode_categorical_features(X_val, fit=False)

    X_train = cleaner.scale_numerical_features(X_train, fit=True)
    X_val = cleaner.scale_numerical_features(X_val, fit=False)

    # Save the cleaner
    cleaner_path = "data_cleaner.joblib"
    joblib.dump(cleaner, cleaner_path)
    # mlflow.log_artifact(cleaner_path, "preprocessor")


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

    args = parser.parse_args()

    try:
        model, metrics = main(
            train_data_path=args.train_data_path,
            model_type=args.model_type,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        logger.info("Baseline model training completed successfully!")
    except Exception as e:
        logger.error(f"Model training failed with error: {str(e)}")
        sys.exit(1)
