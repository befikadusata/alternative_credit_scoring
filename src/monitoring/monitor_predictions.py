"""
Script for active model monitoring using Evidently AI.

This script compares live prediction data against a reference dataset to detect
data drift and prediction drift. It generates Evidently AI reports and
can optionally log key metrics to MLflow.
"""

import argparse
import logging
import os
import sys
import pandas as pd
import json

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, PredictionDriftPreset, ClassificationPreset
from evidently.options.color_scheme import ColorOptions
from evidently.utils.data_drift_detection import EvidentlyColumnMapping # Add this import

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)

def main(
    reference_data_path: str,
    prediction_logs_path: str,
    output_report_path: str = None,
    output_json_metrics_path: str = None,
    mlflow_tracking_uri: str = None,
    mlflow_experiment_name: str = "Model Monitoring",
    monitored_model_name: str = None,
    monitored_model_version: str = None,
    target_column: str = "default", # Assuming logs might contain ground truth
    prediction_column: str = "prediction", # Assuming logs contain model prediction
    probability_column: str = "probability_default" # Assuming logs contain probability
):
    """
    Main function to perform model monitoring using Evidently AI.

    Args:
        reference_data_path: Path to the reference dataset.
        prediction_logs_path: Path to the current prediction logs dataset.
        output_report_path: Optional path to save the HTML Evidently report.
        output_json_metrics_path: Optional path to save the JSON Evidently metrics.
        mlflow_tracking_uri: MLflow Tracking URI.
        mlflow_experiment_name: MLflow Experiment name for logging monitoring results.
        monitored_model_name: Name of the model being monitored in MLflow.
        monitored_model_version: Version of the model being monitored.
        target_column: Name of the target column in the prediction logs (if ground truth is available).
        prediction_column: Name of the prediction column in the prediction logs.
        probability_column: Name of the probability column for the positive class.
    """
    logger = setup_logging()
    logger.info("Starting model monitoring with Evidently AI...")

    # Load data
    try:
        reference_data = pd.read_csv(reference_data_path)
        prediction_logs = pd.read_csv(prediction_logs_path)
        logger.info(f"Loaded reference data shape: {reference_data.shape}")
        logger.info(f"Loaded prediction logs shape: {prediction_logs.shape}")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    # Filter out target column from reference data if it exists
    # The reference data used for Evidently should contain features only
    reference_data_for_evidently = reference_data.drop(columns=[target_column], errors='ignore')
    
    # The current_data for Evidently should contain features, prediction, and optionally target
    current_data_for_evidently = prediction_logs.copy()
    
    # Identify numerical and categorical features from the reference_data_for_evidently
    numerical_features = [col for col in reference_data_for_evidently.columns if pd.api.types.is_numeric_dtype(reference_data_for_evidently[col])]
    categorical_features = [col for col in reference_data_for_evidently.columns if not pd.api.types.is_numeric_dtype(reference_data_for_evidently[col])]

    # Define ColumnMapping for Evidently AI
    column_mapping = EvidentlyColumnMapping(
        target=target_column,
        prediction=prediction_column,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        datetime_features=[]
    )
    
    # Check if target is actually in prediction_logs for performance metrics
    has_ground_truth = target_column in prediction_logs.columns

    # Evidently AI Report Generation
    monitoring_report = Report(metrics=[
        DataDriftPreset(stattest="ks", num_stattest="ks", cat_stattest="chi2"), # Kolmogorov-Smirnov for num, Chi2 for cat
        PredictionDriftPreset(stattest="ks"), # Data and Prediction drift
        ClassificationPreset(target_name=target_column, prediction_name=prediction_column, probrability_columns=[probability_column]) if has_ground_truth else None # Model performance if ground truth is available
    ], options=[ColorOptions(primary_color="#735cfa", success_color="#32a852", warning_color="#ffc107", danger_color="#dc3545")])
    
    # Filter out None if ClassificationPreset was not added
    monitoring_report.metrics = [m for m in monitoring_report.metrics if m is not None]

    monitoring_report.run(current_data=current_data_for_evidently, reference_data=reference_data_for_evidently, column_mapping=column_mapping)
    logger.info("Evidently AI Report run completed.")

    # Save reports
    if output_report_path:
        os.makedirs(os.path.dirname(output_report_path) or '.', exist_ok=True)
        monitoring_report.save_html(output_report_path)
        logger.info(f"Evidently HTML report saved to {output_report_path}")
    
    if output_json_metrics_path:
        os.makedirs(os.path.dirname(output_json_metrics_path) or '.', exist_ok=True)
        monitoring_report.save_json(output_json_metrics_path)
        logger.info(f"Evidently JSON metrics saved to {output_json_metrics_path}")

    # Log metrics to MLflow
    if mlflow_tracking_uri and mlflow_experiment_name:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        with mlflow.start_run(run_name="model_monitoring_run", nested=True, experiment_id=mlflow.get_experiment_by_name(mlflow_experiment_name).experiment_id if mlflow.get_experiment_by_name(mlflow_experiment_name) else None) as run:
            if not mlflow.get_experiment_by_name(mlflow_experiment_name):
                mlflow.create_experiment(mlflow_experiment_name)
                mlflow.set_experiment(mlflow_experiment_name)
            else:
                mlflow.set_experiment(mlflow_experiment_name)

            logger.info(f"MLflow Run ID for monitoring: {run.info.run_id}")
            
            # Log parameters
            mlflow.log_param("monitored_model_name", monitored_model_name)
            mlflow.log_param("monitored_model_version", monitored_model_version)
            mlflow.log_param("reference_data_path", reference_data_path)
            mlflow.log_param("prediction_logs_path", prediction_logs_path)
            
            # Extract and log key metrics from the Evidently report
            metrics_json = json.loads(monitoring_report.json())
            
            # Data Drift metrics
            # Assuming DataDriftPreset is always the first metric in the list
            data_drift_metric_result = metrics_json.get('metrics', [{}])[0].get('result', {})
            data_drift_summary = data_drift_metric_result.get('drift_by_columns', {})

            for column_name, drift_info in data_drift_summary.items():
                mlflow.log_metric(f"data_drift_{column_name}_drift_score", drift_info.get('drift_score', 0))
                mlflow.log_metric(f"data_drift_{column_name}_drift_detected", 1 if drift_info.get('drift_detected', False) else 0)

            mlflow.log_metric("dataset_drift_score", data_drift_metric_result.get('dataset_drift', 0))
            mlflow.log_metric("number_of_drifted_columns", data_drift_metric_result.get('number_of_drifted_columns', 0))
            mlflow.log_metric("share_of_drifted_columns", data_drift_metric_result.get('share_of_drifted_columns', 0))

            # Prediction Drift metrics
            # Assuming PredictionDriftPreset is always the second metric
            prediction_drift_metric_result = metrics_json.get('metrics', [{}, {}])[1].get('result', {})
            mlflow.log_metric(f"prediction_drift_{prediction_column}_drift_score", prediction_drift_metric_result.get('prediction_drift', 0))

            # If ClassificationPreset was used
            if has_ground_truth:
                classification_metric_result = metrics_json.get('metrics', [{}, {}, {}])[2].get('result', {})
                mlflow.log_metric("model_performance_accuracy", classification_metric_result.get('accuracy', 0))
                mlflow.log_metric("model_performance_f1", classification_metric_result.get('f1', 0))
                mlflow.log_metric("model_performance_precision", classification_metric_result.get('precision', 0))
                mlflow.log_metric("model_performance_recall", classification_metric_result.get('recall', 0))
                mlflow.log_metric("model_performance_roc_auc", classification_metric_result.get('roc_auc', 0))
                # Log other relevant classification metrics as needed

            if output_report_path:
                mlflow.log_artifact(output_report_path)
            if output_json_metrics_path:
                mlflow.log_artifact(output_json_metrics_path)

            logger.info("Evidently AI monitoring metrics logged to MLflow.")

    logger.info("Model monitoring script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform model monitoring using Evidently AI Reports."
    )
    parser.add_argument(
        "--reference-data-path",
        type=str,
        required=True,
        help="Path to the reference dataset for comparison.",
    )
    parser.add_argument(
        "--prediction-logs-path",
        type=str,
        required=True,
        help="Path to the current prediction logs dataset.",
    )
    parser.add_argument(
        "--output-report-path",
        type=str,
        default=None,
        help="Optional: Path to save the Evidently HTML report.",
    )
    parser.add_argument(
        "--output-json-metrics-path",
        type=str,
        default=None,
        help="Optional: Path to save the Evidently JSON metrics.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow Tracking URI.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default="Model Monitoring",
        help="MLflow Experiment name for logging monitoring results.",
    )
    parser.add_argument(
        "--monitored-model-name",
        type=str,
        default="credit_scoring_model",
        help="Name of the model being monitored in MLflow.",
    )
    parser.add_argument(
        "--monitored-model-version",
        type=str,
        default="production",
        help="Version or stage of the model being monitored.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="default",
        help="Name of the target column in the prediction logs (if ground truth is available).",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="prediction",
        help="Name of the prediction column in the prediction logs.",
    )
    parser.add_argument(
        "--probability-column",
        type=str,
        default="probability_default",
        help="Name of the probability column for the positive class in the prediction logs.",
    )

    args = parser.parse_args()
    main(
        reference_data_path=args.reference_data_path,
        prediction_logs_path=args.prediction_logs_path,
        output_report_path=args.output_report_path,
        output_json_metrics_path=args.output_json_metrics_path,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        monitored_model_name=args.monitored_model_name,
        monitored_model_version=args.monitored_model_version,
        target_column=args.target_column,
        prediction_column=args.prediction_column,
        probability_column=args.probability_column,
    )
