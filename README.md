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

*   **Real-time Prediction:** Low-latency API for instant credit decisions.
*   **MLflow Integration:** Comprehensive experiment tracking, model registry, and lifecycle management.
*   **Champion-Challenger Deployment:** Safe model deployments with A/B testing capabilities.
*   **Drift Detection:** Automated monitoring for data and model drift using Evidently AI.
*   **Explainable AI:** SHAP-based explanations for every prediction.
*   **Observability:** Prometheus and Grafana for infrastructure and model monitoring.

## Technology Stack

*   **ML Framework:** XGBoost, Scikit-learn
*   **MLOps:** MLflow, Evidently AI
*   **API:** FastAPI, Uvicorn
*   **Data Storage:** PostgreSQL (MLflow metadata), MinIO (Artifacts), Redis (Feature Store)
*   **Monitoring:** Prometheus, Grafana
*   **Containerization:** Docker, Docker Compose
*   **CI/CD:** GitHub Actions (planned)
