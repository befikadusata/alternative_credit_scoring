# Architectural Components

This document describes the key components that make up the credit scoring system.

## 1. API Layer (FastAPI)

The API layer is the front door to the system, providing a robust interface for clients.

- **Responsibilities:**
    - HTTP request handling and routing for all incoming traffic.
    - Input validation using Pydantic schemas to ensure data integrity.
    - Authentication, authorization, and rate limiting to protect the service.
    - Centralized request/response logging and standardized error handling.
    - Tracking requests with unique IDs for distributed tracing.

- **Key Features:**
    - **Auto-generated Documentation:** FastAPI provides OpenAPI (Swagger) and ReDoc documentation automatically, simplifying integration.
    - **Asynchronous Support:** Async request handling allows for high concurrency and efficient I/O operations.
    - **Technology:** Built with Python 3.10+, FastAPI, and the Uvicorn ASGI server.

## 2. Prediction Service Layer

This is the core logic engine of the application. It orchestrates the process of generating a credit score.

- **Responsibilities:**
    - Executing the feature engineering pipeline.
    - Loading and caching machine learning models.
    - Orchestrating the inference process.
    - Post-processing model results to generate final scores and recommendations.
    - Generating explanations for each prediction using SHAP.

- **Design Patterns:**
    - **Singleton:** A single instance of the model loader is used per model version to conserve memory.
    - **Strategy:** The Champion-Challenger routing logic is implemented using a strategy pattern to easily switch between routing schemes.
    - **Pipeline:** Feature transformations are applied as a sequential pipeline.
    - **Factory:** Models are instantiated using a factory pattern based on metadata from the model registry.

## 3. MLflow Infrastructure

MLflow is the backbone of the MLOps lifecycle, managing experiments, models, and artifacts.

- **Components:**
    - **Tracking Server:** Logs experiments, parameters, metrics, and artifacts.
    - **Model Registry:** Manages the lifecycle of models, including versioning and stage transitions (Staging, Production, Archived).
    - **Backend Store:** A PostgreSQL database that stores all metadata for experiments and the model registry.
    - **Artifact Store:** A MinIO (S3-compatible) object storage service that stores model files, plots, and other large artifacts.

- **Model Registry Schema Example:**
  The registry stores rich metadata for each model version, providing a full audit trail.
  ```json
  {
    "model_name": "credit_scoring_xgboost",
    "version": "2",
    "stage": "Production",
    "creation_timestamp": "2024-12-01T10:30:00Z",
    "tags": {
      "algorithm": "XGBoost",
      "dataset_version": "v2.1",
      "auc_roc": "0.78",
      "training_duration_min": "45"
    },
    "run_id": "abc123...",
    "description": "Retrained with Q4 2024 data"
  }
  ```

## 4. Champion-Challenger Framework

This framework enables the safe deployment and testing of new models.

- **Purpose:** To test a new "Challenger" model against the current "Champion" (production) model in a live environment before a full rollout.

- **Traffic Routing:**
  - A configurable portion of production traffic (e.g., 10%) is routed to the Challenger model. The remaining 90% goes to the Champion.
  - For every request, both models may score the data to allow for direct performance comparison. However, the user only receives the prediction from the selected model (either Champion or Challenger).
  - The predictions from both models are logged for later analysis.

- **Promotion Logic:** The Challenger is promoted to Champion only if it demonstrates a statistically significant performance improvement over a defined period.

## 5. Feature Store (Simplified)

The feature store provides a centralized location for pre-computed features, ensuring consistency between model training and serving.

- **Purpose:**
    - To avoid feature skew between the training and inference environments.
    - To reduce prediction latency by serving pre-computed features.

- **Implementation:**
    - For this project, a simplified feature store is implemented using a **Redis-based caching layer**.
    - Features are computed and stored with a unique key (e.g., `user_id` and `version`).
    - A Time-to-Live (TTL) is set on each feature set (e.g., 1 hour) to ensure data freshness.
    - If a feature set is not found in the cache (a cache miss), the system computes the features on-the-fly and then caches them (a write-through cache strategy).

- **Future Enhancements:** For a full production environment, this would be replaced by a dedicated feature store solution like Feast or AWS Feature Store.
