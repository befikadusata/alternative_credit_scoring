from __future__ import annotations

import pendulum
import os
import json

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# MLflow
import mlflow
from mlflow.tracking import MlflowClient

# Import custom modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.evaluate import main as evaluate_model_script_main # Assuming evaluate.py has a main function

# --- Configuration ---
# Ensure these match the values in .env or docker-compose for the API service
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL_NAME = os.getenv("CHAMPION_MODEL_NAME", "credit_scoring_model")
DEFAULT_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/lending_club_loan.csv") # Placeholder, adjust as needed

# --- Helper Functions for Python Operators ---
def _get_latest_run_metrics(**kwargs):
    """
    Queries MLflow to get the metrics from the latest run for a given model.
    Pushes the run_id and metrics to XCom.
    """
    ti = kwargs['ti']
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    runs = client.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name(REGISTERED_MODEL_NAME).experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError(f"No MLflow runs found for experiment '{REGISTERED_MODEL_NAME}'.")

    latest_run = runs[0]
    latest_metrics = latest_run.data.metrics
    run_id = latest_run.info.run_id

    ti.xcom_push(key="latest_run_id", value=run_id)
    ti.xcom_push(key="latest_run_metrics", value=latest_metrics)
    kwargs['ti'].log.info(f"Latest MLflow run_id: {run_id}, Metrics: {latest_metrics}")

def _performance_validation_gate(**kwargs):
    """
    Checks if the latest model's performance meets the defined threshold.
    """
    ti = kwargs['ti']
    latest_metrics = ti.xcom_pull(key="latest_run_metrics", task_ids="get_latest_run_metrics")

    # Define performance threshold (e.g., from docs/model/evaluation.md)
    # This should ideally be loaded from a config file
    REQUIRED_AUC_THRESHOLD = float(os.getenv("REQUIRED_AUC_THRESHOLD", "0.75"))

    current_auc = latest_metrics.get("auc")
    if current_auc is None:
        raise ValueError("AUC metric not found in latest MLflow run.")

    kwargs['ti'].log.info(f"Latest model AUC: {current_auc}, Required AUC: {REQUIRED_AUC_THRESHOLD}")

    if current_auc >= REQUIRED_AUC_THRESHOLD:
        kwargs['ti'].log.info("Model performance meets the required threshold. Proceeding to registration.")
        return "pass"
    else:
        kwargs['ti'].log.error("Model performance DOES NOT meet the required threshold. Aborting registration.")
        raise ValueError("Model performance did not meet the required threshold.")

def _model_registration_task(**kwargs):
    """
    Transitions the latest successful model to the Staging stage in MLflow Model Registry.
    """
    ti = kwargs['ti']
    run_id = ti.xcom_pull(key="latest_run_id", task_ids="get_latest_run_metrics")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Get the latest model version for the given run_id
    filter_string = f"run_id='{run_id}'"
    model_versions = client.search_model_versions(filter_string=filter_string)

    if not model_versions:
        raise ValueError(f"No model version found for run_id: {run_id}")

    # Assuming train.py logs only one model, pick the first one
    latest_model_version = model_versions[0]
    version = latest_model_version.version
    
    kwargs['ti'].log.info(f"Registering model version {version} for run {run_id} to '{REGISTERED_MODEL_NAME}' and setting stage to 'Staging'.")
    
    # Transition to Staging
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Staging"
    )
    kwargs['ti'].log.info(f"Model '{REGISTERED_MODEL_NAME}' version {version} transitioned to 'Staging'.")


def _send_notification(context):
    """Placeholder for sending notifications on DAG failure or success."""
    dag_id = context['dag'].dag_id
    run_id = context['run_id']
    status = "failed" if context['exception'] else "succeeded"
    kwargs['ti'].log.info(f"DAG {dag_id} run {run_id} {status}. Details: {context}")
    # Here you would integrate with Slack, email, etc.
    pass

# --- DAG Definition ---
with DAG(
    dag_id="training_pipeline_dag",
    start_date=days_ago(1), # pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule=None,  # Set to None for manual triggering, or a cron expression like '0 0 * * 0' for weekly
    catchup=False,
    tags=["mlops", "training", "credit_scoring"],
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "email_on_failure": True, # Changed to True for production-like behavior
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": pendulum.duration(minutes=5),
        "on_failure_callback": _send_notification,
        "on_success_callback": _send_notification,
    },
) as dag:
    # Task 0: Ensure MLflow Tracking URI is set
    set_mlflow_uri = BashOperator(
        task_id="set_mlflow_tracking_uri",
        bash_command=f"export MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}",
        do_xcom_push=False,
    )

    # Task 1: Data Ingestion
    # Combines downloading raw data, making processed data, and creating reference dataset
    # The --output-path for make_dataset.py should match train_data_path in train.py
    data_ingestion = BashOperator(
        task_id="data_ingestion",
        bash_command=f"""
            echo "Starting data ingestion tasks..."
            python scripts/download_dataset.py --output-path {DEFAULT_DATA_PATH}
            python scripts/make_dataset.py --raw-data-path {DEFAULT_DATA_PATH} --output-path data/processed/train.csv
            python scripts/create_reference_dataset.py --input-path data/processed/train.csv --output-path data/reference/reference.csv
            echo "Data ingestion tasks completed."
        """,
    )

    # Task 2: Data Validation (Gate)
    # Assumes validate_data.py exits with non-zero on failure
    data_validation_gate = BashOperator(
        task_id="data_validation_gate",
        bash_command="python src/monitoring/validate_data.py --input-path data/processed/train.csv --reference-path data/reference/reference.csv",
    )

    # Task 3: Model Training
    # train.py logs the model to MLflow automatically
    model_training = BashOperator(
        task_id="model_training",
        bash_command=f"python src/models/train.py --train_data_path data/processed/train.csv --registered_model_name {REGISTERED_MODEL_NAME}",
        # XCom push model_run_id if train.py were to return it, for now we search for it later
    )
    
    # Task to get the latest MLflow run ID and metrics from the model training task
    get_latest_run_metrics = PythonOperator(
        task_id="get_latest_run_metrics",
        python_callable=_get_latest_run_metrics,
    )

    # Task 4: Model Evaluation (on a separate test set if available, for now using validation set)
    # This assumes evaluate.py takes a model_name, version, and test_data_path
    model_evaluation = BashOperator(
        task_id="model_evaluation",
        bash_command=f"python src/models/evaluate.py --model_name {REGISTERED_MODEL_NAME} --model_version latest --test_data_path data/processed/test.csv",
    )

    # Task 5: Performance Validation (Gate)
    performance_validation_gate = PythonOperator(
        task_id="performance_validation_gate",
        python_callable=_performance_validation_gate,
        provide_context=True,
    )

    # Task 6: Model Registration (Transition to Staging)
    model_registration = PythonOperator(
        task_id="model_registration",
        python_callable=_model_registration_task,
        provide_context=True,
    )

    # Define task dependencies
    data_ingestion >> data_validation_gate >> model_training
    model_training >> get_latest_run_metrics >> performance_validation_gate
    performance_validation_gate >> model_registration
