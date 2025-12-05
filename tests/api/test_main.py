import os

# This is a bit of a hack to make sure the app can be imported
# In a real project, you'd have a better packaging structure
import sys
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.api.main import app
from src.api.models import PredictionInput


@pytest.fixture(scope="module")
def client():
    """Pytest fixture for the FastAPI TestClient."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def dummy_model_path():
    """Trains and saves a dummy LogisticRegression model for testing."""
    # Generate a dummy dataset with feature names matching PredictionInput
    feature_names_excluding_loan_id = [
        field for field in PredictionInput.__fields__ if field != "loan_id"
    ]
    n_features_needed = len(feature_names_excluding_loan_id)
    X, y = make_classification(
        n_samples=100,
        n_features=n_features_needed,
        n_informative=5,
        n_redundant=0,
        random_state=42,
    )

    # Create a DataFrame with correct column names
    X_df = pd.DataFrame(X, columns=feature_names_excluding_loan_id)

    model = LogisticRegression(
        random_state=42, solver="liblinear"
    )  # Use liblinear for smaller datasets
    model.fit(X_df, y)

    path = "tests/test_model.joblib"
    joblib.dump(model, path)
    yield path
    os.remove(path)


@pytest.fixture(scope="function")
def mock_models(dummy_model_path):
    """Mocks the champion and challenger models with a real dummy model."""
    # Load the dummy model
    real_dummy_model = joblib.load(dummy_model_path)

    # Create a mock for challenger, as it's not the focus here
    mock_challenger_model = MagicMock()
    # Ensure it returns something sensible if called
    mock_challenger_model.predict_proba.return_value = np.array([[0.7, 0.3]])
    mock_challenger_model.predict.return_value = np.array(
        [0]
    )  # Prediction: 0 (No Default)

    with (
        patch("src.api.main.champion_model", real_dummy_model),
        patch("src.api.main.challenger_model", mock_challenger_model),
    ):
        yield real_dummy_model  # Yield the champion model for specific assertions


@pytest.fixture(scope="function")
def mock_redis():
    """Mocks the Redis client."""
    mock_redis_client = MagicMock()

    with patch("src.api.main.redis_client", mock_redis_client):
        yield mock_redis_client


def test_health_check(client):
    """Tests the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] in ["healthy", "degraded"]
    assert "champion" in json_response
    assert "challenger" in json_response
    assert "redis" in json_response


def test_predict_endpoint_no_redis(client, mock_models, mock_redis):
    """Tests the /predict endpoint with no loan_id, so Redis is not used."""
    mock_redis.get_features.return_value = None

    # Sample valid input payload
    payload = {
        "input": {
            "loan_amnt": 12000.0,
            "term": "36 months",
            "int_rate": 10.0,
            "installment": 300.0,
            "grade": "B",
            "sub_grade": "B2",
            "emp_length": 5.0,
            "home_ownership": "MORTGAGE",
            "annual_inc": 75000.0,
            "verification_status": "Verified",
            "purpose": "debt_consolidation",
            "dti": 15.0,
            "delinq_2yrs": 0,
            "inq_last_6mths": 1,
            "open_acc": 8,
            "pub_rec": 0,
            "revol_bal": 12000,
            "revol_util": 50.0,
            "total_acc": 20,
            "initial_list_status": "f",
            "total_pymnt": 1000.0,
            "total_pymnt_inv": 1000.0,
            "total_rec_prncp": 800.0,
            "total_rec_int": 200.0,
            "total_rec_late_fee": 0.0,
            "recoveries": 0.0,
            "collection_recovery_fee": 0.0,
            "last_pymnt_amnt": 300.0,
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    json_response = response.json()
    # With a real model, we cannot assert specific fixed values, but we can assert the structure and type
    assert "prediction" in json_response
    assert "probability_default" in json_response
    assert "risk_level" in json_response

    # Verify that redis was not called to get or set features since no loan_id was provided
    mock_redis.get_features.assert_not_called()
    mock_redis.set_features.assert_not_called()


def test_predict_endpoint_with_redis_miss(client, mock_models, mock_redis):
    """Tests the /predict endpoint with a loan_id, but a Redis cache miss."""
    loan_id = "test-loan-123"
    mock_redis.get_features.return_value = None

    payload = {
        "input": {
            "loan_id": loan_id,
            "loan_amnt": 12000.0,
            "term": "36 months",
            "int_rate": 10.0,
            "installment": 300.0,
            "grade": "B",
            "sub_grade": "B2",
            "emp_length": 5.0,
            "home_ownership": "MORTGAGE",
            "annual_inc": 75000.0,
            "verification_status": "Verified",
            "purpose": "debt_consolidation",
            "dti": 15.0,
            "delinq_2yrs": 0,
            "inq_last_6mths": 1,
            "open_acc": 8,
            "pub_rec": 0,
            "revol_bal": 12000,
            "revol_util": 50.0,
            "total_acc": 20,
            "initial_list_status": "f",
            "total_pymnt": 1000.0,
            "total_pymnt_inv": 1000.0,
            "total_rec_prncp": 800.0,
            "total_rec_int": 200.0,
            "total_rec_late_fee": 0.0,
            "recoveries": 0.0,
            "collection_recovery_fee": 0.0,
            "last_pymnt_amnt": 300.0,
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    mock_redis.get_features.assert_called_once_with(loan_id)
    # Verify that features are written to cache after a miss
    mock_redis.set_features.assert_called_once()

    # Check the actual call arguments to set_features
    args, kwargs = mock_redis.set_features.call_args
    assert args[0] == loan_id
    assert isinstance(args[1], PredictionInput)
    assert args[1].loan_id == loan_id


def test_predict_endpoint_with_redis_hit(client, mock_models, mock_redis):
    """Tests the /predict endpoint with a Redis cache hit."""
    loan_id = "test-loan-456"

    # The features that are "cached" in Redis
    cached_features = PredictionInput(
        **{
            "loan_id": loan_id,
            "loan_amnt": 9999.0,
            "term": "60 months",
            "int_rate": 20.0,
            "installment": 400.0,
            "grade": "E",
            "sub_grade": "E1",
            "emp_length": 1.0,
            "home_ownership": "RENT",
            "annual_inc": 40000.0,
            "verification_status": "Not Verified",
            "purpose": "credit_card",
            "dti": 35.0,
            "delinq_2yrs": 2,
            "inq_last_6mths": 4,
            "open_acc": 15,
            "pub_rec": 1,
            "revol_bal": 20000,
            "revol_util": 80.0,
            "total_acc": 25,
            "initial_list_status": "w",
            "total_pymnt": 500.0,
            "total_pymnt_inv": 500.0,
            "total_rec_prncp": 100.0,
            "total_rec_int": 400.0,
            "total_rec_late_fee": 20.0,
            "recoveries": 0.0,
            "collection_recovery_fee": 0.0,
            "last_pymnt_amnt": 400.0,
        }
    )
    mock_redis.get_features.return_value = cached_features

    # The payload sent by the user might be different, but since loan_id matches,
    # the cached features should be used.
    payload = {
        "input": {
            "loan_id": loan_id,
            "loan_amnt": 1000.0,
            # ... other fields can be different or missing
            "term": "36 months",
            "int_rate": 10.0,
            "installment": 300.0,
            "grade": "B",
            "sub_grade": "B2",
            "emp_length": 5.0,
            "home_ownership": "MORTGAGE",
            "annual_inc": 75000.0,
            "verification_status": "Verified",
            "purpose": "debt_consolidation",
            "dti": 15.0,
            "delinq_2yrs": 0,
            "inq_last_6mths": 1,
            "open_acc": 8,
            "pub_rec": 0,
            "revol_bal": 12000,
            "revol_util": 50.0,
            "total_acc": 20,
            "initial_list_status": "f",
            "total_pymnt": 1000.0,
            "total_pymnt_inv": 1000.0,
            "total_rec_prncp": 800.0,
            "total_rec_int": 200.0,
            "total_rec_late_fee": 0.0,
            "recoveries": 0.0,
            "collection_recovery_fee": 0.0,
            "last_pymnt_amnt": 300.0,
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    mock_redis.get_features.assert_called_once_with(loan_id)

    # Check that the model was called with the cached data, not the payload data
    # The mock model's `predict_proba` is called with a DataFrame
    call_args, _ = mock_models.predict_proba.call_args
    input_df = call_args[0]
    assert isinstance(input_df, pd.DataFrame)

    # We can assert that the DataFrame passed to the model matches the cached features
    # Excluding loan_id which is not part of the model input
    expected_input_df = pd.DataFrame([cached_features.dict(exclude={"loan_id"})])
    pd.testing.assert_frame_equal(input_df, expected_input_df)

    # Also check that the write-through cache is still active
    mock_redis.set_features.assert_called_once()
