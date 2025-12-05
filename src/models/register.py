"""
Model Registration Script

This script registers the final, validated model in the MLflow Model Registry
after successful evaluation and fairness analysis.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("model_registration.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def validate_model_performance(
    run_id: str,
    min_auc_threshold: float = 0.7,
    mlflow_tracking_uri: str = "http://localhost:5000",
):
    """
    Validate model performance before registration.

    Args:
        run_id: MLflow run ID of the model to validate
        min_auc_threshold: Minimum AUC threshold for model registration
        mlflow_tracking_uri: MLflow tracking URI

    Returns:
        True if model passes validation, False otherwise
    """
    logger = logging.getLogger(__name__)

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Get the run information
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    # Check performance metrics
    metrics = run.data.metrics

    logger.info(f"Validating model from run {run_id}")
    logger.info(f"Model metrics: {metrics}")

    # Check if AUC is above threshold
    if "auc" in metrics:
        if metrics["auc"] < min_auc_threshold:
            logger.error(
                f"AUC {metrics['auc']} is below minimum threshold {min_auc_threshold}"
            )
            return False
        else:
            logger.info(
                f"AUC {metrics['auc']} passes minimum threshold {min_auc_threshold}"
            )
    else:
        logger.warning("AUC metric not found, proceeding with other validations")

    # Check other important metrics
    required_metrics = ["accuracy", "precision", "recall", "f1_score"]
    missing_metrics = [metric for metric in required_metrics if metric not in metrics]

    if missing_metrics:
        logger.warning(f"Missing metrics: {missing_metrics}")
    else:
        logger.info("All required metrics are present")

    return True


def validate_fairness_report(
    fairness_report_path: str, max_disparity_threshold: float = 0.1
):
    """
    Validate fairness report before registration.

    Args:
        fairness_report_path: Path to the fairness report JSON file
        max_disparity_threshold: Maximum allowed disparity threshold

    Returns:
        True if model passes fairness validation, False otherwise
    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(fairness_report_path):
        logger.error(f"Fairness report not found at {fairness_report_path}")
        return False

    # Load fairness report
    with open(fairness_report_path, "r") as f:
        fairness_report = json.load(f)

    logger.info(f"Validating fairness report from {fairness_report_path}")

    # Check for bias indicators
    bias_indicators = fairness_report.get("bias_indicators", {})

    for feature, bias_info in bias_indicators.items():
        has_bias = bias_info.get("concern") == "Potential bias detected"

        if has_bias:
            logger.error(f"Potential bias detected for feature {feature}")

            # Check specific disparities
            disparities = bias_info.get("disparities", {})
            for metric, value in disparities.items():
                if value > max_disparity_threshold:
                    logger.error(
                        f"Disparity for {metric} ({value}) exceeds threshold ({max_disparity_threshold})"
                    )
                    return False

    logger.info("Fairness validation passed")
    return True


def register_model(
    run_id: str,
    model_name: str,
    model_alias: str = "champion",
    stage: str = "staging",  # "staging" or "production"
    min_auc_threshold: float = 0.7,
    max_disparity_threshold: float = 0.1,
    fairness_report_path: str = "fairness_report.json",
    mlflow_tracking_uri: str = "http://localhost:5000",
):
    """
    Register the model in MLflow Model Registry after validation.

    Args:
        run_id: MLflow run ID containing the model
        model_name: Name for the registered model
        model_alias: Alias for the model version (default: "champion")
        stage: Stage to transition the model to ("staging" or "production")
        min_auc_threshold: Minimum AUC threshold
        max_disparity_threshold: Maximum allowed disparity
        fairness_report_path: Path to fairness report
        mlflow_tracking_uri: MLflow tracking URI

    Returns:
        Version of the registered model
    """
    logger = logging.getLogger(__name__)

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Validate model performance
    if not validate_model_performance(run_id, min_auc_threshold, mlflow_tracking_uri):
        logger.error("Model performance validation failed")
        return None

    # Validate fairness report if provided
    if fairness_report_path and not validate_fairness_report(
        fairness_report_path, max_disparity_threshold
    ):
        logger.error("Fairness validation failed")
        return None

    # Register the model
    try:
        logger.info(f"Registering model {model_name} from run {run_id}")

        # Register the model artifact named 'model'
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)

        logger.info(f"Model registered with version {result.version}")

        # Create a model registry client
        client = mlflow.tracking.MlflowClient()

        # Add model alias
        try:
            client.set_registered_model_alias(model_name, model_alias, result.version)
            logger.info(f"Set alias '{model_alias}' for version {result.version}")
        except Exception as e:
            logger.warning(f"Could not set alias '{model_alias}': {str(e)}")

        # Transition model to specified stage
        if stage.lower() in ["staging", "production"]:
            client.transition_model_version_stage(
                name=model_name, version=result.version, stage=stage.capitalize()
            )
            logger.info(
                f"Model version {result.version} transitioned to {stage.capitalize()}"
            )

        # Set model description
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=f"Credit scoring model registered on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        logger.info(
            f"Model {model_name} version {result.version} successfully registered and validated"
        )

        # Log registration info to MLflow if we're in a run
        try:
            mlflow.log_param("registered_model_name", model_name)
            mlflow.log_param("registered_model_version", result.version)
            mlflow.log_param("registration_stage", stage)
        except:
            # If not in MLflow run, continue without logging
            pass

        return result.version

    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        return None


def main(
    run_id: str,
    model_name: str,
    model_alias: str = "champion",
    stage: str = "staging",
    min_auc_threshold: float = 0.7,
    max_disparity_threshold: float = 0.1,
    fairness_report_path: str = "fairness_report.json",
    mlflow_tracking_uri: str = "http://localhost:5000",
):
    """
    Main function to register the final, validated model.

    Args:
        run_id: MLflow run ID containing the model to register
        model_name: Name for the registered model
        model_alias: Alias for the model version
        stage: Stage to transition the model to ("staging" or "production")
        min_auc_threshold: Minimum AUC threshold
        max_disparity_threshold: Maximum allowed disparity
        fairness_report_path: Path to fairness report
        mlflow_tracking_uri: MLflow tracking URI
    """
    logger = setup_logging()
    logger.info("Starting model registration process...")

    # Set up MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Create or get the experiment
    experiment_name = "Model_Registration"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        # If experiment already exists, get its ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # Start MLflow run for registration process
    with mlflow.start_run(experiment_id=experiment_id):
        # Log parameters
        mlflow.log_param("source_run_id", run_id)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_alias", model_alias)
        mlflow.log_param("stage", stage)
        mlflow.log_param("min_auc_threshold", min_auc_threshold)
        mlflow.log_param("max_disparity_threshold", max_disparity_threshold)
        mlflow.log_param("fairness_report_path", fairness_report_path)

        # Register the model
        version = register_model(
            run_id=run_id,
            model_name=model_name,
            model_alias=model_alias,
            stage=stage,
            min_auc_threshold=min_auc_threshold,
            max_disparity_threshold=max_disparity_threshold,
            fairness_report_path=fairness_report_path,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

        if version:
            logger.info(
                f"Model registration completed successfully. Registered model: {model_name}, version: {version}"
            )

            # Log success
            mlflow.log_metric("registration_success", 1)
            mlflow.log_param("registered_version", version)

            return version
        else:
            logger.error("Model registration failed")

            # Log failure
            mlflow.log_metric("registration_success", 0)

            return None


if __name__ == "__main__":
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Register the final, validated model in MLflow Model Registry."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="MLflow run ID containing the model to register",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name for the registered model"
    )
    parser.add_argument(
        "--model_alias",
        type=str,
        default="champion",
        help="Alias for the model version (default: 'champion')",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["staging", "production"],
        default="staging",
        help="Stage to transition the model to (default: 'staging')",
    )
    parser.add_argument(
        "--min_auc_threshold",
        type=float,
        default=0.7,
        help="Minimum AUC threshold for model registration (default: 0.7)",
    )
    parser.add_argument(
        "--max_disparity_threshold",
        type=float,
        default=0.1,
        help="Maximum allowed disparity for fairness (default: 0.1)",
    )
    parser.add_argument(
        "--fairness_report_path",
        type=str,
        default="fairness_report.json",
        help="Path to fairness report JSON file (default: 'fairness_report.json')",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI (default: http://localhost:5000)",
    )

    args = parser.parse_args()

    try:
        version = main(
            run_id=args.run_id,
            model_name=args.model_name,
            model_alias=args.model_alias,
            stage=args.stage,
            min_auc_threshold=args.min_auc_threshold,
            max_disparity_threshold=args.max_disparity_threshold,
            fairness_report_path=args.fairness_report_path,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )

        if version:
            logger.info(
                f"Model registration completed successfully! Model: {args.model_name}, Version: {version}"
            )
        else:
            logger.error("Model registration failed.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Model registration failed with error: {str(e)}")
        sys.exit(1)
