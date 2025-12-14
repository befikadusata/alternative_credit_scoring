# Alternative Credit Scoring Platform

This project demonstrates a production-ready Machine Learning system designed for alternative credit scoring. It showcases comprehensive MLOps practices, including real-time prediction, model versioning, drift detection, and explainable AI.

## Project Overview

The core purpose is to provide a robust, scalable, and observable platform for assessing creditworthiness using non-traditional data sources, thereby fostering financial inclusion for underserved populations.

## Documentation

All detailed project documentation can be found in the [`docs/`](./docs/index.md) directory.

*   **[Repository Structure](./docs/repository-structure.md)**: Explains the layout of the project repository.
*   **[API Specification](./docs/api/spec.md)**: Details the available API endpoints and their usage.
*   **[MLOps Overview](./docs/mlops/index.md)**: Describes the MLOps principles and tools used.

For a high-level plan of upcoming work, please refer to the [PROJECT_ROADMAP.md](./PROJECT_ROADMAP.md).

## Getting Started

Detailed instructions on how to set up the local development environment and run the project are provided in [GETTING_STARTED.md](./docs/GETTING_STARTED.md).

## Key Features

*   **Real-time Prediction:** Low-latency API for instant credit decisions, with streaming support via Kafka.
*   **ML Pipeline Orchestration:** Automated training, evaluation, and deployment pipeline using Apache Airflow.
*   **MLflow Integration:** Comprehensive experiment tracking, model registry, and lifecycle management.
*   **Champion-Challenger Deployment:** Safe model deployments with A/B testing capabilities.
*   **Drift Detection:** Automated monitoring for data and model drift using Evidently AI.
*   **Explainable AI:** SHAP-based explanations for every prediction.
*   **Observability:** Prometheus and Grafana for infrastructure and model monitoring.

## Technology Stack

*   **ML Framework:** XGBoost, Scikit-learn
*   **Orchestration:** Apache Airflow
*   **Streaming:** Apache Kafka
*   **MLOps:** MLflow, Evidently AI
*   **API:** FastAPI, Uvicorn
*   **Data Storage:** PostgreSQL (MLflow & Airflow metadata), MinIO (Artifacts), Redis (Feature Store)
*   **Monitoring:** Prometheus, Grafana
*   **Containerization:** Docker, Docker Compose
*   **CI/CD:** GitHub Actions (planned)

## Running the ML Pipeline

The entire ML pipeline can be executed using Apache Airflow.

1.  **Start all services:**
    ```bash
    docker-compose up -d
    ```
2.  **Access the Airflow UI:**
    Open your browser and navigate to `http://localhost:8080`.
    - **Username:** `airflow`
    - **Password:** `airflow`

3.  **Trigger the ML Pipeline DAG:**
    - In the Airflow UI, find the `ml_pipeline_dag`.
    - Unpause the DAG by clicking the toggle on the left.
    - Manually trigger the DAG by clicking the "play" button on the right.
