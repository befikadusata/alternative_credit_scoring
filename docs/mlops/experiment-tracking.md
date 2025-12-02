# Experiment Tracking & Model Registry

This document describes the MLflow setup used for experiment tracking and managing the model lifecycle with the Model Registry.

## 1. MLflow Architecture

MLflow serves as the central hub for managing the entire machine learning lifecycle. The architecture consists of several key components orchestrated via Docker Compose for local development.

-   **MLflow Tracking Server:** A central server that provides a REST API and a UI for logging and viewing experiments.
-   **Backend Store (PostgreSQL):** A PostgreSQL database that stores all MLflow metadata, including experiment runs, parameters, metrics, and model registry information. Using a robust database ensures data integrity.
-   **Artifact Store (MinIO):** A MinIO server, which provides an S3-compatible object storage solution. It is used to store large artifacts like model files, plots, and datasets.

### Docker Compose Configuration

The following `docker-compose.yml` defines the multi-container setup for the MLflow stack.

```yaml
services:
  # PostgreSQL for MLflow backend
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "mlflow"]
      interval: 10s
      timeout: 5s
      retries: 5

  # MinIO for artifact storage (S3-compatible)
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  # MLflow Tracking Server
  mlflow:
    build:
      context: ./docker
      dockerfile: Dockerfile.mlflow
    ports:
      - "5000:5000"
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:${DB_PASSWORD}@postgres:5432/mlflow
      MLFLOW_ARTIFACT_ROOT: s3://mlflow-artifacts
      AWS_ACCESS_KEY_ID: ${MINIO_USER}
      AWS_SECRET_ACCESS_KEY: ${MINIO_PASSWORD}
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:${DB_PASSWORD}@postgres:5432/mlflow
      --default-artifact-root s3://mlflow-artifacts
      --host 0.0.0.0
      --port 5000

volumes:
  postgres_data:
  minio_data:
```

## 2. Model Registry and Versioning

The MLflow Model Registry is used to manage the lifecycle of models, providing a centralized repository for versioning, stage management, and lineage tracking.

### Model Stages

Each model version in the registry is assigned a stage, which defines its status in the deployment lifecycle.

-   **None:** A newly registered model version that has not yet been reviewed.
-   **Staging:** The model version is a candidate for production and is undergoing validation and testing.
-   **Production:** The model version has been fully validated and is actively serving live traffic.
-   **Archived:** The model version is deprecated and no longer in use, but is kept for historical tracking.

### Model Promotion Workflow

The process of moving a model from development to production follows a clear, auditable workflow:

1.  **Registration:** After a successful training run, the model is registered with the MLflow Model Registry. This creates a new version (e.g., `version 5`) and assigns it the `None` stage. A `validation_status` tag is set to `pending`.

2.  **Transition to Staging:** The model version is manually or automatically transitioned to the `Staging` stage. At this point, it becomes a "Challenger" model and begins receiving a small portion of production traffic for live testing.

3.  **Validation:** The Challenger model's performance is compared against the current Production "Champion" model based on key business and performance metrics. It also undergoes a final round of validation checks (see [Model Evaluation](./evaluation.md)).

4.  **Promotion to Production:**
    -   If the `Staging` model meets or exceeds the validation criteria and outperforms the current `Production` model, it is promoted.
    -   The script first transitions the current `Production` model to the `Archived` stage.
    -   It then transitions the `Staging` model to the `Production` stage.
    -   This atomic switch ensures that only one model is in the `Production` stage at any given time.

5.  **Archival:** Older models are kept in the `Archived` stage to maintain a complete history for auditing and rollback purposes.
