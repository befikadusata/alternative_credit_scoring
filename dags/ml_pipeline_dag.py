from __future__ import annotations

import pendulum
import os
import json
import pandas as pd
import redis

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# MLflow
import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL_NAME = os.getenv("CHAMPION_MODEL_NAME", "credit_scoring_model")
RAW_DATA_PATH = "data/raw/LC_loans_granting_model_dataset.csv"
PROCESSED_DATA_PATH = "data/processed/train.csv"
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# --- Python Functions for Operators ---

def _materialize_features_to_redis():
    """
    Reads processed data and loads it into Redis to simulate a feature store.
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    for _, row in df.iterrows():
        loan_id = row['loan_id']
        feature_dict = row.to_dict()
        r.set(f"loan_features:{loan_id}", json.dumps(feature_dict))
    
    print(f"Materialized {len(df)} feature sets to Redis.")

def _get_latest_run_id(**kwargs):
    """
    Gets the run_id from the latest MLflow run of the training task.
    """
    ti = kwargs['ti']
    run_id = ti.xcom_pull(task_ids='model_training', key='run_id')
    if not run_id:
        raise ValueError("Could not find run_id in XComs from model_training task.")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    run = client.get_run(run_id)
    ti.xcom_push(key="latest_run_metrics", value=run.data.metrics)
    print(f"Latest MLflow run_id: {run_id}, Metrics: {run.data.metrics}")

def _register_model(**kwargs):
    """
    Registers the model from the latest run in the MLflow Model Registry.
    """
    ti = kwargs['ti']
    run_id = ti.xcom_pull(task_ids='model_training', key='run_id')
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    
    print(f"Registered model '{REGISTERED_MODEL_NAME}' with version '{registered_model.version}'.")
    
    # Transition to Staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        stage="Staging"
    )
    print(f"Model version '{registered_model.version}' transitioned to 'Staging'.")


# --- DAG Definition ---
with DAG(
    dag_id="ml_pipeline_dag",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["mlops", "training", "credit_scoring", "redis", "xai"],
) as dag:

    data_ingestion = BashOperator(
        task_id="data_ingestion",
        bash_command=f"python scripts/download_dataset.py",
    )

    data_preparation = BashOperator(
        task_id="data_preparation",
        bash_command=f"python scripts/make_dataset.py --raw-data-path {RAW_DATA_PATH} --output-path {PROCESSED_DATA_PATH}",
    )

    materialize_features = PythonOperator(
        task_id="materialize_features_to_redis",
        python_callable=_materialize_features_to_redis,
    )

    model_training = BashOperator(
        task_id="model_training",
        bash_command=(
            f"python src/models/train.py --train_data_path {PROCESSED_DATA_PATH} "
            f"--registered_model_name {REGISTERED_MODEL_NAME} "
            f"&& mkdir -p /airflow/xcom "
            f"&& echo '{{\"run_id\": \"'$(cat /airflow/xcom/run_id)'\"}}' > /airflow/xcom/return.json"
        ),
        do_xcom_push=True,
    )

    model_evaluation = BashOperator(
        task_id="model_evaluation",
        bash_command=(
            "python src/models/evaluate.py --model_name {{ task_instance.xcom_pull(task_ids='model_training', key='run_id') }} "
            "--test_data_path data/processed/test.csv"
        ),
    )

    get_run_id = PythonOperator(
        task_id='get_latest_run_id',
        python_callable=_get_latest_run_id,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
    )

    # Define task dependencies
    data_ingestion >> data_preparation >> materialize_features
    materialize_features >> model_training
    model_training >> model_evaluation
    model_evaluation >> get_run_id
    get_run_id >> register_model
