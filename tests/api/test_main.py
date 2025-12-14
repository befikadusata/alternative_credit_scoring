import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.api.main import app
from src.api.models import PredictionInput

# Removed DataCleaner and apply_feature_engineering imports as they are not needed for this simplified dummy model
# from src.data.cleaning import DataCleaner
# from src.data.features import apply_feature_engineering


@pytest.fixture(scope="module")
def client():
    """Pytest fixture for the FastAPI TestClient."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def dummy_model_path():
    """Trains and saves a dummy LogisticRegression model for testing."""
    # Generate a dummy dataset with feature names matching PredictionInput, excluding loan_id
    # and creating realistic values for categorical features.
    raw_input_data = {
        "loan_amnt": [12000.0] * 100,
        "term": ["36 months"] * 50 + ["60 months"] * 50,  # 100 elements
        "int_rate": [10.0] * 100,
        "installment": [300.0] * 100,
        "grade": (
            ["A"] * 15
            + ["B"] * 15
            + ["C"] * 15
            + ["D"] * 15
            + ["E"] * 15
            + ["F"] * 15
            + ["G"] * 10
        ),  # 100 elements
        "sub_grade": (
            ["A1"] * 5
            + ["A2"] * 5
            + ["A3"] * 5
            + ["A4"] * 5
            + ["A5"] * 5
            + ["B1"] * 5
            + ["B2"] * 5
            + ["B3"] * 5
            + ["B4"] * 5
            + ["B5"] * 5
            + ["C1"] * 5
            + ["C2"] * 5
            + ["C3"] * 5
            + ["C4"] * 5
            + ["C5"] * 5
            + ["D1"] * 5
            + ["D2"] * 5
            + ["D3"] * 5
            + ["D4"] * 5
            + ["D5"] * 5
        ),  # 100 elements (20 categories * 5 repetitions)
        "emp_length": [0.0, 1.0, 5.0, 10.0] * 25,  # 100 elements
        "home_ownership": ["MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY"] * 16
        + ["MORTGAGE", "RENT", "OWN", "OTHER"],  # 100 elements
        "annual_inc": [75000.0] * 100,
        "verification_status": ["Verified", "Not Verified", "Source Verified"] * 33
        + ["Verified"],  # 100 elements
        "purpose": [
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "other",
            "major_purchase",
            "medical",
            "car",
            "small_business",
            "wedding",
            "house",
            "moving",
            "vacation",
        ]
        * 8
        + [
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "other",
        ],  # 100 elements
        "dti": [15.0] * 100,
        "delinq_2yrs": [0] * 100,
        "inq_last_6mths": [1] * 100,
        "open_acc": [8] * 100,
        "pub_rec": [0] * 100,
        "revol_bal": [12000] * 100,
        "revol_util": [50.0] * 100,
        "total_acc": [20] * 100,
        "initial_list_status": ["f", "w"] * 50,  # 100 elements
        "total_pymnt": [1000.0] * 100,
        "total_pymnt_inv": [1000.0] * 100,
        "total_rec_prncp": [800.0] * 100,
        "total_rec_int": [200.0] * 100,
        "total_rec_late_fee": [0.0] * 100,
        "recoveries": [0.0] * 100,
        "collection_recovery_fee": [0.0] * 100,
        "last_pymnt_amnt": [300.0] * 100,
    }
    X_raw = pd.DataFrame(raw_input_data)
    y = pd.Series(np.random.randint(0, 2, 100), name="default")  # Dummy target

    # Apply the same one-hot encoding logic as in src/api/main.py
    X_encoded = X_raw.copy()
    for col in X_encoded.select_dtypes(include=["object"]).columns:
        if col in X_encoded.columns:
            dummies = pd.get_dummies(X_encoded[col], prefix=col)
            X_encoded = pd.concat([X_encoded.drop(columns=[col]), dummies], axis=1)

    # Ensure all columns are float types
    X_final = X_encoded.astype(float)

    # Train a simple Logistic Regression model
    model = LogisticRegression(
        random_state=42, solver="liblinear", max_iter=1000, class_weight="balanced"
    )
    model.fit(X_final, y)

    path = "tests/test_model.joblib"
    joblib.dump(model, path)

    # Save feature names for later use by the API
    feature_names_path = "tests/test_model_features.joblib"
    joblib.dump(X_final.columns.tolist(), feature_names_path)

    yield path
    os.remove(path)
    os.remove(feature_names_path)


@pytest.fixture(scope="module")
def data_cleaner():
    """Loads the DataCleaner from the saved joblib file."""
    return joblib.load("data_cleaner.joblib")


@pytest.fixture(scope="function")
def mock_models(dummy_model_path, data_cleaner):
    """Mocks the champion and challenger models with a real dummy model."""
    # Load the dummy model and feature names
    import joblib

    real_dummy_model = joblib.load(dummy_model_path)

    # Load the feature names that were saved with the dummy model
    feature_names_path = "tests/test_model_features.joblib"
    feature_names = (
        joblib.load(feature_names_path) if os.path.exists(feature_names_path) else []
    )

    # Create a mock for challenger, as it's not the focus here
    mock_challenger_model = MagicMock()
    # Ensure it returns something sensible if called
    mock_challenger_model.predict_proba.return_value = np.array([[0.7, 0.3]])
    mock_challenger_model.predict.return_value = np.array(
        [0]
    )  # Prediction: 0 (No Default)

    # Create model info that includes the feature names
    model_info = {
        "model_name": "credit_scoring_model",
        "model_version": "champion",
        "load_time": datetime.now(timezone.utc),
        "last_prediction_time": None,
        "feature_names": feature_names,
    }

    # Create a mock wrapper that calls the real model but tracks calls
    from unittest.mock import Mock

    mock_champion_model = Mock(wraps=real_dummy_model)

    with (
        patch("src.api.main.champion_model", mock_champion_model),
        patch("src.api.main.challenger_model", mock_challenger_model),
        patch("src.api.main.champion_cleaner", data_cleaner),  # Mock the cleaner
        patch(
            "src.api.main.challenger_cleaner", MagicMock()
        ),  # Mock challenger cleaner
        patch("src.api.main.champion_model_info", model_info),
        patch("src.api.main.challenger_model_info", {"model_name": "challenger_model", "model_version": "challenger", "feature_names": feature_names}),
    ):
        yield mock_champion_model, mock_challenger_model



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
    mock_champion_model, _ = mock_models  # Unpack the tuple

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
    assert "prediction" in json_response
    assert "probability_default" in json_response
    assert "risk_level" in json_response

    # Verify that redis was not called
    mock_redis.get_features.assert_not_called()
    mock_redis.set_features.assert_not_called()
    # Verify champion model was called
    mock_champion_model.predict_proba.assert_called_once()


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
    mock_redis.set_features.assert_called_once()
    args, kwargs = mock_redis.set_features.call_args
    assert args[0] == loan_id
    assert isinstance(args[1], PredictionInput)
    assert args[1].loan_id == loan_id


def test_predict_endpoint_with_redis_hit(client, mock_models, mock_redis):
    """Tests the /predict endpoint with a Redis cache hit."""
    loan_id = "test-loan-456"
    mock_champion_model, _ = mock_models

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

    payload = {
        "input": {
            "loan_id": loan_id,
            "loan_amnt": 1000.0,
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

    call_args, _ = mock_champion_model.predict_proba.call_args
    input_df = call_args[0]
    assert isinstance(input_df, pd.DataFrame)

    expected_input_df = pd.DataFrame([cached_features.model_dump(exclude={"loan_id"})])

    for col in expected_input_df.select_dtypes(include=["object"]).columns:
        if col in expected_input_df.columns:
            dummies = pd.get_dummies(expected_input_df[col], prefix=col)
            expected_input_df = pd.concat(
                [expected_input_df.drop(columns=[col]), dummies], axis=1
            )

    expected_input_df = expected_input_df.astype(float)

    feature_names_path = "tests/test_model_features.joblib"
    if os.path.exists(feature_names_path):
        model_feature_names = joblib.load(feature_names_path)
        expected_input_df = expected_input_df.reindex(
            columns=model_feature_names, fill_value=0
        )

    pd.testing.assert_frame_equal(input_df, expected_input_df)
    mock_redis.set_features.assert_called_once()


def test_predict_endpoint_invalid_input(client, mock_models):
    """Tests the /predict endpoint with a missing required field."""
    payload = {
        "input": {
            "term": "36 months",
            "int_rate": 10.0,
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    json_response = response.json()
    assert "detail" in json_response
    assert any(
        "loan_amnt" in error["loc"] and "Field required" in error["msg"]
        for error in json_response["detail"]
    )


def test_predict_endpoint_challenger_model(client, mock_models, mock_redis):
    """Tests that the challenger model is called when random() is low."""
    mock_champion_model, mock_challenger_model = mock_models

    with patch("src.api.main.challenger_traffic_percentage", 50.0):
        with patch("src.api.main.random.random", return_value=0.0):
            payload = {
                "input": {
                    "loan_amnt": 15000.0,
                    "term": "60 months",
                    "int_rate": 15.0,
                    "installment": 400.0,
                    "grade": "C",
                    "sub_grade": "C3",
                    "emp_length": 2.0,
                    "home_ownership": "RENT",
                    "annual_inc": 55000.0,
                    "verification_status": "Source Verified",
                    "purpose": "other",
                    "dti": 25.0,
                    "delinq_2yrs": 1,
                    "inq_last_6mths": 0,
                    "open_acc": 12,
                    "pub_rec": 0,
                    "revol_bal": 15000,
                    "revol_util": 70.0,
                    "total_acc": 30,
                    "initial_list_status": "f",
                    "total_pymnt": 2000.0,
                    "total_pymnt_inv": 2000.0,
                    "total_rec_prncp": 1500.0,
                    "total_rec_int": 500.0,
                    "total_rec_late_fee": 0.0,
                    "recoveries": 0.0,
                    "collection_recovery_fee": 0.0,
                    "last_pymnt_amnt": 400.0,
                }
            }

            response = client.post("/predict", json=payload)
            assert response.status_code == 200

            mock_challenger_model.predict_proba.assert_called_once()
            mock_champion_model.predict_proba.assert_not_called()
