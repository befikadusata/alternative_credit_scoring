import os
import requests
import time
import pytest
import mlflow
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# API and MLflow URLs
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
TEST_MODEL_NAME = "integration_test_model"

@pytest.fixture(scope="module")
def wait_for_api():
    """Fixture to wait for the API to be healthy before running tests."""
    start_time = time.time()
    while time.time() - start_time < 60:
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                print("API is healthy!")
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    pytest.fail("API did not become healthy within 60 seconds.")

@pytest.fixture(scope="module")
def registered_dummy_model():
    """
    Trains, logs, and registers a dummy model to the live MLflow instance.
    Yields the model name and version.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Use a unique experiment name for testing
    experiment_name = f"integration-tests-{int(time.time())}"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        # Create a simple dummy model
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model = LogisticRegression()
        model.fit(X, y)
        
        # Log the model to MLflow, which also registers it
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=TEST_MODEL_NAME,
        )
        model_version = model_info.registered_model_version
        print(f"Registered dummy model '{TEST_MODEL_NAME}' version '{model_version}'")
        
    yield TEST_MODEL_NAME, model_version

    # Cleanup: Delete the registered model after tests are done
    try:
        client = mlflow.tracking.MlflowClient()
        client.delete_registered_model(name=TEST_MODEL_NAME)
        print(f"Cleaned up registered model '{TEST_MODEL_NAME}'")
    except Exception as e:
        print(f"Error cleaning up model '{TEST_MODEL_NAME}': {e}")


def test_health_check(wait_for_api):
    """Test the /health endpoint."""
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "champion" in data
    assert "redis" in data

def test_root_endpoint(wait_for_api):
    """Test the root / endpoint."""
    response = requests.get(f"{API_BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Credit Scoring API"
    assert data["status"] == "running"

def test_prediction_endpoint_with_loaded_model(wait_for_api, registered_dummy_model):
    """
    Test the full prediction flow: load a model, then make a prediction.
    """
    model_name, model_version = registered_dummy_model

    # Step 1: Tell the API to load the model as the champion
    load_payload = {
        "model_name": model_name,
        "model_version": model_version,
        "model_type": "champion"
    }
    load_response = requests.put(f"{API_BASE_URL}/model/load", json=load_payload)
    assert load_response.status_code == 200, f"Failed to load model: {load_response.text}"
    assert f"loaded successfully as champion" in load_response.json()["message"]

    # Step 2: Make a prediction
    # This payload is simplified because the dummy model doesn't care about features
    predict_payload = {
        "input": {
            "loan_amnt": 10000, "term": "36 months", "int_rate": 10.0,
            "installment": 300.0, "grade": "B", "sub_grade": "B2", "emp_length": 5.0,
            "home_ownership": "MORTGAGE", "annual_inc": 75000.0, 
            "verification_status": "Verified", "purpose": "debt_consolidation", 
            "dti": 15.0, "delinq_2yrs": 0, "inq_last_6mths": 1, "open_acc": 8, 
            "pub_rec": 0, "revol_bal": 12000, "revol_util": 50.0, "total_acc": 20,
            "initial_list_status": "f", "total_pymnt": 1000.0, "total_pymnt_inv": 1000.0,
            "total_rec_prncp": 800.0, "total_rec_int": 200.0, "total_rec_late_fee": 0.0,
            "recoveries": 0.0, "collection_recovery_fee": 0.0, "last_pymnt_amnt": 300.0
        }
    }
    pred_response = requests.post(f"{API_BASE_URL}/predict", json=predict_payload)
    assert pred_response.status_code == 200
    data = pred_response.json()
    assert "prediction" in data
    assert "probability_default" in data
    assert data["model_name"] == model_name
    assert data["model_version"] == model_version
