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

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REFERENCE_DATA_PATH = "data/reference/reference.csv"
# Placeholder for where prediction logs are aggregated.
# In a real system, this would fetch from a database, data lake, or stream.
PREDICTION_LOGS_PATH = "data/prediction_logs/current_hour.csv" 
MONITORING_REPORT_HTML_PATH = "/tmp/evidently_monitoring_report.html"
MONITORING_METRICS_JSON_PATH = "/tmp/evidently_monitoring_metrics.json"

# --- Helper Functions for Python Operators ---
def _send_notification(context):
    """Placeholder for sending notifications on DAG failure or success."""
    dag_id = context['dag'].dag_id
    run_id = context['run_id']
    status = "failed" if context['exception'] else "succeeded"
    print(f"DAG {dag_id} run {run_id} {status}. Details: {context}")
    # Here you would integrate with Slack, email, etc.
    pass

# --- DAG Definition ---
with DAG(
    dag_id="model_monitoring_dag",
    start_date=days_ago(1),
    schedule_interval="@hourly", # Run hourly
    catchup=False,
    tags=["mlops", "monitoring", "credit_scoring"],
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "email_on_failure": True,
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

    # Task 1: Prepare Prediction Logs
    # In a real system, this would query a database, S3, or a streaming platform
    # to aggregate prediction logs for the last hour.
    # For now, this is a placeholder. You would replace 'cp' with actual aggregation logic.
    prepare_prediction_logs = BashOperator(
        task_id="prepare_prediction_logs",
        bash_command=f"""
            echo "Aggregating prediction logs for the last hour..."
            # Placeholder: Create a dummy prediction logs file if it doesn't exist
            if [ ! -f {PREDICTION_LOGS_PATH} ]; then
                mkdir -p $(dirname {PREDICTION_LOGS_PATH})
                echo "loan_id,feature1,feature2,prediction,probability_default,default" > {PREDICTION_LOGS_PATH}
                echo "2,0.5,10,0,0.1,0" >> {PREDICTION_LOGS_PATH}
                echo "3,0.8,12,1,0.8,1" >> {PREDICTION_LOGS_PATH}
                echo "4,0.2,5,0,0.2,0" >> {PREDICTION_LOGS_PATH}
            fi
            echo "Prediction logs prepared at {PREDICTION_LOGS_PATH}"
        """,
        # You might need to adjust default_args to handle missing directories etc.
    )

    # Task 2: Run Evidently AI Monitoring Script
    run_monitoring_script = BashOperator(
        task_id="run_evidently_monitoring",
        bash_command=f"""
            python src/monitoring/monitor_predictions.py \
                --reference-data-path {REFERENCE_DATA_PATH} \
                --prediction-logs-path {PREDICTION_LOGS_PATH} \
                --output-report-path {MONITORING_REPORT_HTML_PATH} \
                --output-json-metrics-path {MONITORING_METRICS_JSON_PATH} \
                --mlflow-tracking-uri {MLFLOW_TRACKING_URI} \
                --mlflow-experiment-name "Model Monitoring"
        """,
    )

    # Task Dependencies
    set_mlflow_uri >> prepare_prediction_logs >> run_monitoring_script
