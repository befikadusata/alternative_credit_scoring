"""
Fairness and Bias Analysis Module

This module provides functions for analyzing model fairness and bias across
different demographic groups, including statistical parity, equal opportunity,
and demographic parity metrics.
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


class FairnessAnalyzer:
    """
    A class for performing fairness and bias analysis on ML models.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_fairness_metrics(self, y_true, y_pred, sensitive_features):
        """
        Calculate fairness metrics across different groups.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: DataFrame with sensitive attributes (e.g., gender, race)

        Returns:
            Dictionary of fairness metrics
        """
        results = {}

        # Convert to pandas Series if they aren't already
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred)

        # Combine for easier grouping
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

        # Calculate metrics for each sensitive feature
        for col in sensitive_features.columns:
            self.logger.info(f"Analyzing fairness for sensitive feature: {col}")

            # Get unique values for the sensitive feature
            unique_values = sensitive_features[col].unique()

            # Calculate metrics for each group
            group_metrics = {}
            for value in unique_values:
                mask = sensitive_features[col] == value
                subset_df = df[mask]

                if len(subset_df) == 0:  # Skip if no samples for this group
                    continue

                y_true_group = subset_df["y_true"]
                y_pred_group = subset_df["y_pred"]

                # Calculate various fairness metrics for this group
                group_metrics[value] = {
                    "count": len(y_true_group),
                    "positive_rate": y_pred_group.mean(),  # Statistical parity difference
                    "true_positive_rate": (
                        ((y_pred_group == 1) & (y_true_group == 1)).sum()
                        / (y_true_group == 1).sum()
                        if (y_true_group == 1).sum() > 0
                        else 0
                    ),  # Equal opportunity
                    "false_positive_rate": (
                        ((y_pred_group == 1) & (y_true_group == 0)).sum()
                        / (y_true_group == 0).sum()
                        if (y_true_group == 0).sum() > 0
                        else 0
                    ),
                    "accuracy": accuracy_score(y_true_group, y_pred_group),
                    "precision": precision_score(
                        y_true_group, y_pred_group, zero_division=0
                    ),
                    "recall": recall_score(y_true_group, y_pred_group, zero_division=0),
                    "f1_score": f1_score(y_true_group, y_pred_group, zero_division=0),
                }

                # Calculate AUC if possible
                if len(np.unique(y_true_group)) > 1:
                    try:
                        group_metrics[value]["auc"] = roc_auc_score(
                            y_true_group, y_pred_group
                        )
                    except:
                        group_metrics[value]["auc"] = -1  # Placeholder
                else:
                    group_metrics[value][
                        "auc"
                    ] = -1  # Placeholder for groups with only one class

            # Calculate fairness disparities
            if len(group_metrics) > 1:
                disparities = self._calculate_disparities(group_metrics)
                group_metrics["disparities"] = disparities

            results[col] = group_metrics

        return results

    def _calculate_disparities(self, group_metrics):
        """
        Calculate fairness disparities between groups.

        Args:
            group_metrics: Metrics calculated for each group

        Returns:
            Dictionary of disparity metrics
        """
        # Extract metric values for different groups
        metrics_of_interest = [
            "positive_rate",
            "true_positive_rate",
            "false_positive_rate",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc",
        ]

        disparities = {}

        for metric in metrics_of_interest:
            values = [
                gm[metric]
                for gm in group_metrics.values()
                if metric in gm and gm[metric] != -1
            ]
            if len(values) > 1:
                # Calculate max difference (a simple measure of disparity)
                disparities[f"max_{metric}_difference"] = max(values) - min(values)
                # Calculate standard deviation (another measure of disparity)
                disparities[f"{metric}_std"] = np.std(values)
                # Calculate coefficient of variation
                disparities[f"{metric}_cv"] = (
                    np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                )

        return disparities

    def detect_bias(self, fairness_results, threshold=0.1):
        """
        Detect potential bias based on fairness metrics.

        Args:
            fairness_results: Results from calculate_fairness_metrics
            threshold: Threshold for considering a disparity significant

        Returns:
            Dictionary indicating which groups show potential bias
        """
        bias_indicators = {}

        for feature, groups in fairness_results.items():
            if "disparities" in groups:
                disparities = groups["disparities"]
                significant_disparities = {
                    metric: value
                    for metric, value in disparities.items()
                    if value > threshold
                }

                if significant_disparities:
                    bias_indicators[feature] = {
                        "disparities": significant_disparities,
                        "concern": (
                            "Potential bias detected"
                            if significant_disparities
                            else "No significant bias detected"
                        ),
                    }
                else:
                    bias_indicators[feature] = {
                        "disparities": {},
                        "concern": "No significant bias detected",
                    }

        return bias_indicators

    def create_fairness_report(self, fairness_results, bias_indicators):
        """
        Create a comprehensive fairness report.

        Args:
            fairness_results: Results from calculate_fairness_metrics
            bias_indicators: Results from detect_bias

        Returns:
            Dictionary with detailed fairness report
        """
        report = {
            "summary": {},
            "detailed_results": fairness_results,
            "bias_indicators": bias_indicators,
        }

        for feature, groups in fairness_results.items():
            # Calculate summary statistics for the feature
            summary = {}

            # Get all metric names
            if len(groups) > 0 and "disparities" in groups:
                # Exclude the disparities key when looking for metric names
                sample_group = {
                    k: v
                    for k, v in next(iter(groups.values())).items()
                    if k != "disparities" and not k.startswith("count")
                }

                for metric in sample_group.keys():
                    values = [
                        g[metric]
                        for g in groups.values()
                        if metric in g and g[metric] != -1 and "disparities" not in g
                    ]  # Exclude disparities and groups with placeholder values
                    if values:
                        summary[metric] = {
                            "min": min(values),
                            "max": max(values),
                            "mean": np.mean(values),
                            "std": np.std(values),
                        }

            report["summary"][feature] = summary

        return report

    def create_fairness_visualizations(
        self, fairness_results, model_name="Model", save_plots=True
    ):
        """
        Create visualizations for fairness analysis.

        Args:
            fairness_results: Results from calculate_fairness_metrics
            model_name: Name of the model for visualization titles
            save_plots: Whether to save plots to MLflow
        """
        for feature, groups in fairness_results.items():
            # Prepare data for visualization
            if "disparities" in groups:
                groups_no_disparities = {
                    k: v for k, v in groups.items() if k != "disparities"
                }
            else:
                groups_no_disparities = groups

            if not groups_no_disparities:
                continue

            # Create a figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"{model_name} - Fairness Analysis by {feature}", fontsize=16)

            # Get metrics that are most relevant for fairness
            metrics_to_plot = [
                "positive_rate",
                "true_positive_rate",
                "false_positive_rate",
                "accuracy",
            ]

            # Remove metrics not present in the data
            available_metrics = [
                m
                for m in metrics_to_plot
                if any(m in g for g in groups_no_disparities.values())
            ]

            # Plot up to 4 metrics
            for i, metric in enumerate(available_metrics[:4]):
                ax = axes[i // 2, i % 2] if len(available_metrics) > 1 else axes[0, 0]

                values = []
                labels = []
                for group_name, group_metrics in groups_no_disparities.items():
                    if metric in group_metrics and group_metrics[metric] != -1:
                        values.append(group_metrics[metric])
                        labels.append(str(group_name))

                ax.bar(labels, values)
                ax.set_title(f'{metric.replace("_", " ").title()} by {feature}')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.tick_params(axis="x", rotation=45)

                # Add value labels on bars
                for j, v in enumerate(values):
                    ax.text(j, v, f"{v:.3f}", ha="center", va="bottom")

            # If we have fewer than 4 metrics, hide empty subplots
            for i in range(len(available_metrics), 4):
                if i > 0:  # Only hide if we have other plots
                    axes[i // 2, i % 2].set_visible(False)

            plt.tight_layout()

            # Save to MLflow if requested
            if save_plots:
                try:
                    mlflow.log_figure(fig, f"{model_name}_fairness_{feature}.png")
                except:
                    # If not in MLflow run, save locally
                    fig.savefig(f"{model_name}_fairness_{feature}.png")

            plt.close(fig)


def analyze_model_fairness(
    model, X_test, y_test, sensitive_features, model_name="Model"
):
    """
    Perform complete fairness analysis on a model.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        sensitive_features: DataFrame with sensitive features
        model_name: Name of the model for logging

    Returns:
        Fairness analysis report
    """
    logger = logging.getLogger(__name__)

    # Make predictions
    y_pred = model.predict(X_test)

    # Initialize fairness analyzer
    analyzer = FairnessAnalyzer()

    # Calculate fairness metrics
    logger.info("Calculating fairness metrics...")
    fairness_results = analyzer.calculate_fairness_metrics(
        y_test, y_pred, sensitive_features
    )

    # Detect bias
    logger.info("Detecting potential bias...")
    bias_indicators = analyzer.detect_bias(fairness_results, threshold=0.1)

    # Create comprehensive report
    logger.info("Creating fairness report...")
    report = analyzer.create_fairness_report(fairness_results, bias_indicators)

    # Create visualizations
    logger.info("Creating fairness visualizations...")
    analyzer.create_fairness_visualizations(fairness_results, model_name)

    # Log results to MLflow if in a run
    try:
        # Log summary metrics
        for feature, summary in report["summary"].items():
            for metric, stats in summary.items():
                mlflow.log_metric(f"{feature}_{metric}_min", stats["min"])
                mlflow.log_metric(f"{feature}_{metric}_max", stats["max"])
                mlflow.log_metric(f"{feature}_{metric}_mean", stats["mean"])
                mlflow.log_metric(f"{feature}_{metric}_std", stats["std"])

        # Log whether bias was detected for each feature
        for feature, bias_info in report["bias_indicators"].items():
            has_bias = 1 if bias_info["concern"] == "Potential bias detected" else 0
            mlflow.log_metric(f"{feature}_has_bias", has_bias)

            # Log the maximum disparity for the feature
            if "disparities" in bias_info:
                max_disparity = max(
                    [v for v in bias_info["disparities"].values()], default=0
                )
                mlflow.log_metric(f"{feature}_max_disparity", max_disparity)
    except:
        # If not in MLflow run, continue without logging
        pass

    logger.info(f"Fairness analysis completed for {model_name}")

    return report


def load_sensitive_features(data_path: str, sensitive_feature_cols: list):
    """
    Load sensitive features from data.

    Args:
        data_path: Path to the data file
        sensitive_feature_cols: List of column names for sensitive features

    Returns:
        DataFrame with sensitive features
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Loading sensitive features {sensitive_feature_cols} from {data_path}")
    df = pd.read_csv(data_path)

    # Select only the sensitive feature columns
    sensitive_df = df[sensitive_feature_cols].copy()

    logger.info(f"Loaded sensitive features with shape: {sensitive_df.shape}")

    return sensitive_df


def main(
    test_data_path: str,
    sensitive_feature_cols: list,
    run_id: str = None,
    model_name: str = None,
    model_type: str = "xgboost",
    threshold: float = 0.1,
    mlflow_tracking_uri: str = "http://localhost:5000",
):
    """
    Main function to perform fairness analysis on a trained model.

    Args:
        test_data_path: Path to test data
        sensitive_feature_cols: List of sensitive feature column names
        run_id: MLflow run ID (optional if using model_name)
        model_name: Registered model name (optional if using run_id)
        model_type: Type of model for evaluation
        threshold: Threshold for considering a disparity significant
        mlflow_tracking_uri: URI for MLflow tracking server
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting fairness analysis...")

    # Set up MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Create or get the experiment
    experiment_name = "Fairness_Analysis"
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
        mlflow.log_param("sensitive_features", ", ".join(sensitive_feature_cols))
        mlflow.log_param("threshold", threshold)

        # Load model
        from src.models.evaluate import load_model_from_mlflow

        if run_id:
            model = load_model_from_mlflow(run_id=run_id)
        elif model_name:
            model = load_model_from_mlflow(model_name=model_name)
        else:
            raise ValueError("Either run_id or model_name must be provided")

        # Load test data
        from src.models.evaluate import load_test_data

        X_test, y_test = load_test_data(test_path=test_data_path)

        # Load sensitive features
        sensitive_features = load_sensitive_features(
            test_data_path, sensitive_feature_cols
        )

        # Perform fairness analysis
        fairness_report = analyze_model_fairness(
            model=model,
            X_test=X_test,
            y_test=y_test,
            sensitive_features=sensitive_features,
            model_name=f"{model_type}_model",
        )

        # Save the fairness report as an artifact
        with open("fairness_report.json", "w") as f:
            json.dump(fairness_report, f, indent=2, default=str)

        try:
            mlflow.log_artifact("fairness_report.json")
        except:
            # If not in MLflow run, save locally
            pass

        # Clean up temporary file
        if os.path.exists("fairness_report.json"):
            os.remove("fairness_report.json")

        logger.info(
            f"Fairness analysis completed. Run ID: {mlflow.active_run().info.run_id}"
        )

        return fairness_report


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("fairness_analysis.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Perform fairness analysis on a trained credit scoring model."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="data/processed/test.csv",
        help="Path to the test data file (default: data/processed/test.csv)",
    )
    parser.add_argument(
        "--sensitive_feature_cols",
        nargs="+",
        default=[
            "home_ownership",
            "purpose",
        ],  # Common sensitive features in credit data
        help="List of sensitive feature column names (default: ['home_ownership', 'purpose'])",
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
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for considering a disparity significant (default: 0.1)",
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
        report = main(
            test_data_path=args.test_data_path,
            sensitive_feature_cols=args.sensitive_feature_cols,
            run_id=args.run_id,
            model_name=args.model_name,
            model_type=args.model_type,
            threshold=args.threshold,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )

        logger.info("Fairness analysis completed successfully!")
        logger.info(f"Summary of fairness by feature: {report['summary']}")

    except Exception as e:
        logger.error(f"Fairness analysis failed with error: {str(e)}")
        sys.exit(1)
