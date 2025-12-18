# 30-Second Demo Guide

This guide provides the fastest path to see the Alternative Credit Scoring platform in action. In under a minute, you will start the entire platform and get a real-time credit risk prediction from the API.

## Prerequisites

You must have **Docker** and **Docker Compose** installed on your machine.

- [Install Docker](https://docs.docker.com/get-docker/)
- [Install Docker Compose](https://docs.docker.com/compose/install/)

---

### Step 1: Start the Platform

First, you need to create a `.env` file from the template. This file holds configuration variables for the Docker services.

```bash
cp .env.template .env
```

Now, launch all services using Docker Compose. This command will pull the necessary images and start all containers in the background, including the API, database, MLflow server, and more.

```bash
docker-compose up -d
```

It may take a minute for all services to become healthy. You can check the status by running `docker-compose ps`.

---

### Step 2: Check the API Health

Once the services are running, you can check the API's health status. Open a terminal and use `curl` to hit the `/health` endpoint.

```bash
curl http://localhost:8000/health
```

You should receive a response indicating a `"healthy"` status, which confirms the API is running and has successfully loaded the default credit scoring model.

```json
{
  "status": "healthy",
  "timestamp": "...",
  "model_loaded": true,
  "model_name": "credit_scoring_model",
  "model_version": "1",
  ...
}
```

---

### Step 3: Get a Live Credit Prediction

Now you can send a request to the `/predict` endpoint with sample applicant data. The `curl` command below sends a JSON payload with the required features for a prediction.

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
        "loan_amnt": 10000.0,
        "term": "36 months",
        "int_rate": 12.12,
        "installment": 335.23,
        "grade": "B",
        "sub_grade": "B1",
        "emp_length": 5.0,
        "home_ownership": "RENT",
        "annual_inc": 65000.0,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": 18.5,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "open_acc": 10,
        "pub_rec": 0,
        "revol_bal": 15000,
        "revol_util": 65.0,
        "total_acc": 25,
        "initial_list_status": "f",
        "total_pymnt": 4022.76,
        "total_pymnt_inv": 4022.76,
        "total_rec_prncp": 870.85,
        "total_rec_int": 3151.91,
        "total_rec_late_fee": 0.0,
        "recoveries": 0.0,
        "collection_recovery_fee": 0.0,
        "last_pymnt_amnt": 335.23
    }
}'
```

The API will instantly return a JSON response with the credit risk assessment:

```json
{
  "prediction": 0,
  "probability_default": 0.2345,
  "risk_level": "low",
  "model_name": "credit_scoring_model",
  "model_version": "1",
  ...
}
```

- **`prediction`**: `0` for low risk (likely to repay) and `1` for high risk (likely to default).
- **`probability_default`**: The model's confidence in the prediction.
- **`risk_level`**: A human-friendly interpretation of the score.

---

## Explore the Platform

The demo doesn't stop here. The `docker-compose` command launched a full MLOps platform that you can explore:

-   **MLflow UI**: `http://localhost:5000` (Track experiments and browse the model registry)
-   **MinIO Console**: `http://localhost:9001` (Browse the artifact store for models)
-   **API Docs**: `http://localhost:8000/docs` (Interactive Swagger UI for the API)
-   **Airflow UI**: `http://localhost:8080` (Orchestrate and monitor ML pipelines)
