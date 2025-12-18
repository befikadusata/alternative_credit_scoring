# Alternative Credit Scoring Platform

This project is a complete, production-ready Machine Learning platform that assesses creditworthiness using alternative data sources. It is designed to serve as a comprehensive template for building, deploying, and monitoring robust MLOps systems.

---

## Key Features

-   **Real-time Prediction API:** A low-latency FastAPI server for instant credit risk assessments.
-   **End-to-End MLOps:** Full lifecycle management using MLflow for experiment tracking and model registry.
-   **Champion-Challenger Deployments:** Safely test new models in production by routing a portion of live traffic.
-   **Automated Model Retraining:** Orchestrated via Apache Airflow to keep models fresh.
-   **Comprehensive Monitoring:** Includes system monitoring with Prometheus/Grafana and ML-specific drift detection with Evidently AI.
-   **Declarative Infrastructure:** The entire stack is containerized with Docker and can be stood up with a single command.

---

## Live Demo in 30 Seconds

To see the platform in action, follow the quick-start guide to launch all services and make a live prediction against the API.

➡️ **[View the Demo Guide](./docs/demo-guide.md)**

---

## Project Documentation

This repository contains extensive documentation covering the system architecture, technical decisions, API specifications, and model performance.

-   **[Portfolio Pitch](./docs/portfolio-pitch.md):** A high-level overview for investors and hiring managers.
-   **[System Architecture](./docs/architecture.md):** A deep dive into the components, data flows, and infrastructure.
-   **[Technology Decisions](./docs/tech-decisions.md):** The rationale behind the chosen technology stack.
-   **[API Specification](./docs/api.md):** Detailed documentation for all API endpoints.
-   **[Model Evaluation](./docs/evaluation.md):** Performance metrics and fairness analysis for the credit model.

---

## Core Technology Stack

-   **Backend & Serving:** FastAPI, Docker
-   **ML & Data Science:** XGBoost, Scikit-learn, Pandas
-   **MLOps & Orchestration:** MLflow, Apache Airflow, Evidently AI
-   **Databases & Storage:** PostgreSQL, Redis, MinIO
-   **Monitoring:** Prometheus, Grafana

For a detailed breakdown, see the [Technology Decisions](./docs/tech-decisions.md) document.
