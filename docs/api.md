# API Specification

This document details the endpoints for the Alternative Credit Scoring API. The API is built with FastAPI, which provides automatic, interactive documentation.

**For a complete, interactive experience, run the service locally and visit [http://localhost:8000/docs](http://localhost:8000/docs).**

---

## Authentication

The API is currently configured for demonstration purposes and does not require an API key.

---

## Endpoints

### `GET /`

The root endpoint provides basic information about the API's status.

-   **Description:** Confirms the API is running and checks if the primary model is loaded.
-   **`curl` Example:**
    ```bash
    curl http://localhost:8000/
    ```
-   **Example Response:**
    ```json
    {
      "message": "Credit Scoring API",
      "version": "1.1.0",
      "status": "running",
      "model_loaded": true,
      "model_name": "credit_scoring_model",
      "model_version": "1"
    }
    ```

### `GET /health`

The health endpoint provides a detailed status of the API and its dependencies.

-   **Description:** Checks the status of the loaded champion and challenger models, as well as the connection to the Redis feature cache.
-   **`curl` Example:**
    ```bash
    curl http://localhost:8000/health
    ```
-   **Example Response:**
    ```json
    {
      "status": "healthy",
      "timestamp": "2025-12-18T20:00:00.000Z",
      "champion": {
        "status": "healthy",
        "model_name": "credit_scoring_model",
        "model_version": "1",
        "load_time": "2025-12-18T19:59:00.000Z"
      },
      "challenger": {
        "status": "not_loaded"
      },
      "redis": {
        "status": "healthy"
      },
      "challenger_traffic_percentage": 10.0
    }
    ```

### `POST /predict`

The primary endpoint for generating a single credit risk prediction.

-   **Description:** Accepts a JSON object with applicant features and returns a detailed prediction. This endpoint routes a percentage of traffic to a "challenger" model if one is configured.
-   **Request Body:** A JSON object containing an `input` field with all required features.
-   **`curl` Example:**
    ```bash
    curl -X 'POST' \
      'http://localhost:8000/predict' \
      -H 'Content-Type: application/json' \
      -d '{
        "input": {
            "loan_amnt": 10000.0, "term": "36 months", "int_rate": 12.12,
            "installment": 335.23, "grade": "B", "sub_grade": "B1",
            "emp_length": 5.0, "home_ownership": "RENT", "annual_inc": 65000.0,
            "verification_status": "Verified", "purpose": "debt_consolidation", "dti": 18.5,
            "delinq_2yrs": 0, "inq_last_6mths": 1, "open_acc": 10, "pub_rec": 0,
            "revol_bal": 15000, "revol_util": 65.0, "total_acc": 25,
            "initial_list_status": "f", "total_pymnt": 4022.76, "total_pymnt_inv": 4022.76,
            "total_rec_prncp": 870.85, "total_rec_int": 3151.91,
            "total_rec_late_fee": 0.0, "recoveries": 0.0, "collection_recovery_fee": 0.0,
            "last_pymnt_amnt": 335.23
        }
    }'
    ```
-   **Response Body:**
    ```json
    {
      "prediction": 0,
      "probability_default": 0.2345,
      "probability_repayment": 0.7655,
      "risk_level": "low",
      "prediction_time_seconds": 0.0123,
      "model_name": "credit_scoring_model",
      "model_version": "1",
      "timestamp": "2025-12-18T20:01:00.000Z"
    }
    ```

### `POST /predict/batch`

An endpoint for making multiple predictions in a single request.

-   **Description:** Accepts a list of up to 1000 applicant feature sets.
-   **Request Body:** A JSON object containing an `inputs` field with a list of feature objects.
-   **Response Body:** A list of prediction results.

### `PUT /model/load`

An administrative endpoint to dynamically load a new model from the MLflow Model Registry.

-   **Description:** Allows you to load a registered model as either the `champion` or `challenger`.
-   **Request Body:**
    ```json
    {
      "model_name": "credit_scoring_model_v2",
      "model_version": "staging",
      "model_type": "challenger"
    }
    ```
-   **Response Body:**
    ```json
    {
      "message": "Model credit_scoring_model_v2 version staging loaded successfully as challenger",
      "model_name": "credit_scoring_model_v2",
      "model_version": "staging",
      "timestamp": "2025-12-18T20:02:00.000Z"
    }
    ```
