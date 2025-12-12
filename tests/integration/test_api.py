"""
Integration tests for the Credit Scoring API.

These tests run against a live instance of the API service and its dependencies,
ensuring that all components work together correctly.
"""

import os
import requests
import time
import pytest

# API base URL - assuming it's running locally on port 8000
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

@pytest.fixture(scope="module")
def wait_for_api():
    """Fixture to wait for the API to be healthy before running tests."""
    start_time = time.time()
    while time.time() - start_time < 60:  # Wait for up to 60 seconds
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                print("API is healthy!")
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    pytest.fail("API did not become healthy within 60 seconds.")

def test_health_check(wait_for_api):
    """Test the /health endpoint."""
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"] # It will be degraded if no model is loaded yet
    assert "champion" in data
    assert "redis" in data

def test_root_endpoint(wait_for_api):
    """Test the root / endpoint."""
    response = requests.get(f"{API_BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Credit Scoring API"
    assert data["status"] == "running"

def test_prediction_endpoint_no_model(wait_for_api):
    """
    Test the /predict endpoint when no model is loaded.
    It should return a 503 Service Unavailable error.
    """
    sample_payload = {
        "loan_amnt": 10000,
        "int_rate": 12.5,
        "installment": 333.33,
        "emp_length": 7,
        "annual_inc": 60000,
        "dti": 20.0,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "open_acc": 10,
        "pub_rec": 0,
        "revol_bal": 5000,
        "revol_util": 35.0,
        "total_acc": 25,
        "term": " 36 months",
        "grade": "B",
        "sub_grade": "B2",
        "home_ownership": "MORTGAGE",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "initial_list_status": "f"
    }
    response = requests.post(f"{API_BASE_URL}/predict", json=sample_payload)
    # The initial state is no model loaded, so API returns 503
    assert response.status_code == 503 
    assert "Champion model not loaded" in response.json()["detail"]

# To run a test with a loaded model, you would first need a step in your CI
# to run the training script and register a model in MLflow.
# Then, you would need to load it via the API's /model/load endpoint
# before running a test like the one below.

# @pytest.mark.skip(reason="Requires a trained and registered model in MLflow")
# def test_prediction_endpoint_with_loaded_model(wait_for_api):
#     """
#     Test the /predict endpoint after loading a model.
#     This is a more complex test that requires a full MLOps environment.
#     """
#     # Step 1: Ensure a model is trained and registered in MLflow (done outside this test)
#
#     # Step 2: Tell the API to load the model
#     # In a real CI, you'd get the version from the training run
#     load_payload = {
#         "model_name": "credit_scoring_model",
#         "model_version": 1 
#     }
#     load_response = requests.put(f"{API_BASE_URL}/model/load", json=load_payload)
#     assert load_response.status_code == 200, f"Failed to load model: {load_response.text}"
#
#     # Step 3: Make a prediction
#     sample_payload = {
#         "loan_amnt": 10000,
#         # ... (add other features)
#     }
#     pred_response = requests.post(f"{API_BASE_URL}/predict", json=sample_payload)
#     assert pred_response.status_code == 200
#     data = pred_response.json()
#     assert "prediction" in data
#     assert "probability_default" in data
#
