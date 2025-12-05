"""
Model Evaluation Framework

This module provides comprehensive evaluation functions for credit scoring models,
including AUC, precision-recall analysis, confusion matrix, and fairness metrics.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluator:
    """
    A class for comprehensive model evaluation with multiple metrics and visualizations.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Set up matplotlib for saving plots
        plt.style.use("default")

    def evaluate_model(
        self, model, X_test, y_test, model_name="Model", save_plots=True
    ):
        """
        Comprehensive model evaluation with multiple metrics and visualizations.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for logging
            save_plots: Whether to save plots to MLflow

        Returns:
            Dictionary of evaluation results
        """
        self.logger.info(f"Starting evaluation for {model_name}")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Calculate standard metrics
        results = {}

        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["precision"] = precision_score(y_test, y_pred, zero_division=0)
        results["recall"] = recall_score(y_test, y_pred, zero_division=0)
        results["f1_score"] = f1_score(y_test, y_pred, zero_division=0)

        if y_pred_proba is not None:
            results["auc"] = roc_auc_score(y_test, y_pred_proba)
            results["avg_precision"] = average_precision_score(y_test, y_pred_proba)
        else:
            results["auc"] = -1  # Placeholder if AUC cannot be calculated
            results["avg_precision"] = -1

        # Detailed classification report
        results["classification_report"] = classification_report(
            y_test, y_pred, output_dict=True
        )

        # Confusion matrix
        results["confusion_matrix"] = confusion_matrix(y_test, y_pred)

        self.logger.info(
            f"{model_name} - Accuracy: {results['accuracy']:.4f}, "
            f"Precision: {results['precision']:.4f}, "
            f"Recall: {results['recall']:.4f}, "
            f"F1: {results['f1_score']:.4f}, "
            f"AUC: {results['auc']:.4f}"
        )

        # Log metrics to MLflow if in a run
        try:
            for metric_name, metric_value in results.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)

                # Log detailed classification report metrics
                if metric_name == "classification_report":
                    for class_name, metrics in metric_value.items():
                        if isinstance(metrics, dict):
                            for sub_metric, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    mlflow.log_metric(
                                        f"{sub_metric}_{class_name}", value
                                    )

            # Log confusion matrix elements
            if "confusion_matrix" in results:
                cm = results["confusion_matrix"]
                mlflow.log_metric("tn", cm[0, 0])
                mlflow.log_metric("fp", cm[0, 1])
                mlflow.log_metric("fn", cm[1, 0])
                mlflow.log_metric("tp", cm[1, 1])
        except:
            # If not in MLflow run, continue without logging
            pass

        # Generate visualizations if requested
        if save_plots and y_pred_proba is not None:
            self._create_visualizations(
                X_test, y_test, y_pred, y_pred_proba, model_name
            )

        return results

    def _create_visualizations(self, X_test, y_test, y_pred, y_pred_proba, model_name):
        """
        Create and save evaluation visualizations to MLflow.
        """
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"{model_name} - Model Evaluation", fontsize=16)

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        axes[0, 0].plot(
            fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})", color="blue"
        )
        axes[0, 0].plot([0, 1], [0, 1], "k--", label="Random Classifier")
        axes[0, 0].set_xlabel("False Positive Rate")
        axes[0, 0].set_ylabel("True Positive Rate")
        axes[0, 0].set_title("ROC Curve")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        axes[0, 1].plot(
            recall_vals,
            precision_vals,
            label=f"PR Curve (AP = {avg_precision:.3f})",
            color="red",
        )
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].set_title("Precision-Recall Curve")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0])
        axes[1, 0].set_title("Confusion Matrix")
        axes[1, 0].set_xlabel("Predicted")
        axes[1, 0].set_ylabel("Actual")

        # 4. Prediction Distribution
        axes[1, 1].hist(
            y_pred_proba[y_test == 0],
            bins=50,
            alpha=0.5,
            label="Non-Default",
            density=True,
        )
        axes[1, 1].hist(
            y_pred_proba[y_test == 1], bins=50, alpha=0.5, label="Default", density=True
        )
        axes[1, 1].set_xlabel("Predicted Probability")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Prediction Distribution by Class")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Log the figure to MLflow
        try:
            mlflow.log_figure(fig, f"{model_name}_evaluation_plots.png")
        except:
            # If not in MLflow run, save locally
            fig.savefig(f"{model_name}_evaluation_plots.png")

        plt.close(fig)

    def create_shap_explanation(self, model, X_test, model_name="Model"):
        """
        Create SHAP explanations for model predictions.

        Args:
            model: Trained model
            X_test: Test features
            model_name: Name of the model

        Returns:
            SHAP explainer object
        """
        try:
            # Create SHAP explainer based on model type
            if hasattr(model, "booster"):
                # XGBoost model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
            elif hasattr(model, "feature_importances_"):
                # Tree-based model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
            else:
                # Use KernelExplainer for other model types
                explainer = shap.LinearExplainer(
                    model, X_test[:100]
                )  # Use subset for efficiency
                shap_values = explainer.shap_values(X_test)

            # Create SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, show=False)
            plt.title(f"{model_name} - SHAP Feature Importance")

            # Log to MLflow
            try:
                mlflow.log_figure(fig, f"{model_name}_shap_summary.png")
            except:
                # If not in MLflow run, save locally
                fig.savefig(f"{model_name}_shap_summary.png")

            plt.close(fig)

            return explainer
        except Exception as e:
            self.logger.warning(f"Could not create SHAP explanations: {str(e)}")
            return None

    def evaluate_model_by_group(
        self, model, X_test, y_test, sensitive_features, model_name="Model"
    ):
        """
        Evaluate model performance across different groups based on sensitive features.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            sensitive_features: DataFrame with sensitive features (e.g., gender, race)
            model_name: Name of the model

        Returns:
            Dictionary of performance metrics by group
        """
        results_by_group = {}

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Evaluate for each sensitive feature
        for col in sensitive_features.columns:
            self.logger.info(f"Evaluating model performance by {col}")

            # Get unique values for the sensitive feature
            unique_values = sensitive_features[col].unique()

            # Calculate metrics for each group
            group_metrics = {}
            for value in unique_values:
                mask = sensitive_features[col] == value

                if mask.sum() == 0:  # Skip if no samples for this group
                    continue

                y_test_group = y_test[mask]
                y_pred_group = y_pred[mask]

                if y_pred_proba is not None:
                    y_pred_proba_group = y_pred_proba[mask]

                # Calculate metrics for this group
                group_metrics[value] = {
                    "count": len(y_test_group),
                    "accuracy": (
                        accuracy_score(y_test_group, y_pred_group)
                        if len(y_test_group) > 0
                        else 0
                    ),
                    "precision": (
                        precision_score(y_test_group, y_pred_group, zero_division=0)
                        if len(y_test_group) > 0
                        else 0
                    ),
                    "recall": (
                        recall_score(y_test_group, y_pred_group, zero_division=0)
                        if len(y_test_group) > 0
                        else 0
                    ),
                    "f1_score": (
                        f1_score(y_test_group, y_pred_group, zero_division=0)
                        if len(y_test_group) > 0
                        else 0
                    ),
                }

                if y_pred_proba is not None and len(y_test_group) > 0:
                    group_metrics[value]["auc"] = roc_auc_score(
                        y_test_group, y_pred_proba_group
                    )
                else:
                    group_metrics[value]["auc"] = -1  # Placeholder

            results_by_group[col] = group_metrics

        # Log fairness metrics to MLflow if in run
        try:
            for feature, metrics in results_by_group.items():
                for group, group_metrics in metrics.items():
                    for metric_name, metric_value in group_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(
                                f"{feature}_{group}_{metric_name}", metric_value
                            )
        except:
            # If not in MLflow run, continue without logging
            pass

        return results_by_group


def load_model_from_mlflow(run_id, model_name=None):
    """
    Load a model from MLflow.

    Args:
        run_id: MLflow run ID
        model_name: Optional registered model name

    Returns:
        Loaded model
    """
    logger = logging.getLogger(__name__)

    try:
        if model_name:
            # Load from model registry
            model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
        else:
            # Load from run artifacts
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

        logger.info(f"Model loaded successfully from run {run_id}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def load_test_data(test_path: str):
    """
    Load and prepare test data.

    Args:
        test_path: Path to the test data CSV file

    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Loading test data from {test_path}")
    df = pd.read_csv(test_path)

    logger.info(f"Loaded test data with shape: {df.shape}")

    # Separate features and target
    if "default" in df.columns:
        y = df["default"]
        X = df.drop("default", axis=1)
    else:
        raise ValueError("Target column 'default' not found in the dataset")

    logger.info(f"Test features shape: {X.shape}, Test target shape: {y.shape}")

    return X, y


def main(
    test_data_path: str,
    run_id: str = None,
    model_name: str = None,
    model_type: str = "xgboost",
    mlflow_tracking_uri: str = "http://localhost:5000",
):
    """
    Main function to evaluate a trained model.

    Args:
        test_data_path: Path to test data
        run_id: MLflow run ID (optional if using model_name)
        model_name: Registered model name (optional if using run_id)
        model_type: Type of model for evaluation
        mlflow_tracking_uri: URI for MLflow tracking server
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")

    # Set up MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Create or get the experiment
    experiment_name = "Model_Evaluation"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        # If experiment already exists, get its ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id):
        # Log parameters
        mlflow.log_param("run_id", run_id or "N/A")
        mlflow.log_param("model_name", model_name or "N/A")
        mlflow.log_param("model_type", model_type)

        # Load model
        if run_id:
            model = load_model_from_mlflow(run_id=run_id)
        elif model_name:
            model = load_model_from_mlflow(model_name=model_name)
        else:
            raise ValueError("Either run_id or model_name must be provided")

        # Load test data
        X_test, y_test = load_test_data(test_path=test_data_path)

        # Initialize evaluator
        evaluator = ModelEvaluator()

        # Perform evaluation
        model_name_tag = f"{model_type}_model"
        results = evaluator.evaluate_model(model, X_test, y_test, model_name_tag)

        # Create SHAP explanations if possible
        try:
            evaluator.create_shap_explanation(model, X_test, model_name_tag)
        except Exception as e:
            logger.warning(f"Could not create SHAP explanations: {str(e)}")

        logger.info(
            f"Model evaluation completed. Run ID: {mlflow.active_run().info.run_id}"
        )

        return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("model_evaluation.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Evaluate a trained credit scoring model."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="data/processed/test.csv",
        help="Path to the test data file (default: data/processed/test.csv)",
    )
    parser.add_argument(
        "--run_id", type=str, help="MLflow run ID (optional if using --model_name)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Registered model name (optional if using --run_id)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="xgboost",
        help="Type of model for evaluation (default: xgboost)",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI (default: http://localhost:5000)",
    )

    args = parser.parse_args()

    if not args.run_id and not args.model_name:
        print("Error: Either --run_id or --model_name must be provided")
        sys.exit(1)

    try:
        results = main(
            test_data_path=args.test_data_path,
            run_id=args.run_id,
            model_name=args.model_name,
            model_type=args.model_type,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )

        logger.info("Model evaluation completed successfully!")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"Model evaluation failed with error: {str(e)}")
        sys.exit(1)
