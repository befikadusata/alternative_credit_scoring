# Gemini Project Overview: Alternative Credit Scoring Platform

This document provides a comprehensive overview of the Alternative Credit Scoring Platform project, designed to facilitate understanding and interaction with the codebase for Gemini.

## 1. Project Purpose and Core Functionality

This is a production-ready Machine Learning system for **alternative credit scoring**. Its primary goal is to assess creditworthiness using non-traditional data sources, aiming to improve financial inclusion.

The system is built with a strong focus on **MLOps best practices**, featuring:
- **Real-time Prediction API:** For instant credit decisions.
- **Experiment Tracking:** Using MLflow for logging experiments, parameters, and metrics.
- **Model Registry:** MLflow is used for versioning and managing models.
- **Champion-Challenger Deployments:** The API supports routing traffic between a primary "champion" model and a "challenger" model for live A/B testing.
- **Drift Detection:** The architecture is designed for monitoring data and model drift (likely using a tool like Evidently AI).
- **Infrastructure as Code:** Terraform is used for managing infrastructure.
- **Containerization:** The entire application stack can be run locally using Docker Compose.

## 2. Technology Stack

- **Backend Framework:** FastAPI
- **ML Frameworks:** XGBoost, Scikit-learn
- **MLOps Platform:** MLflow
- **Hyperparameter Tuning:** Optuna
- **Data Handling:** Pandas, NumPy
- **Database (MLflow Backend):** PostgreSQL
- **Artifact Store:** MinIO
- **Feature Caching:** Redis
- **Containerization:** Docker, Docker Compose
- **CI/CD:** GitHub Actions
- **Code Formatting & Linting:** Black, isort, flake8, Ruff

## 3. Project Structure Highlights

The repository is well-organized, separating concerns into distinct directories:

- `src/`: Contains all the core application code.
  - `src/api/`: The FastAPI application, including endpoints for prediction, health checks, and model loading. **`main.py` is the entry point.**
  - `src/models/`: Logic for model training (`train.py`), evaluation (`evaluate.py`), and hyperparameter tuning.
  - `src/data/`: Modules for data cleaning and feature engineering.
- `notebooks/`: For exploratory data analysis (EDA) and model prototyping.
- `configs/`: For storing configuration files (e.g., model parameters).
- `scripts/`: Standalone scripts for tasks like downloading data or running a training pipeline.
- `docker-compose.yml`: Defines the local development environment, including services for the API, MLflow, PostgreSQL, and MinIO.
- `pyproject.toml`: Manages Python dependencies and tool configurations.

## 4. How to Build, Run, and Test

### Running the Local Development Environment

The entire stack can be brought up using Docker Compose.

**Command:**
```bash
docker-compose up -d
```

This will start the following services:
- **`postgres`**: The database for MLflow.
- **`minio`**: The object store for MLflow artifacts.
- **`mlflow`**: The MLflow tracking server (accessible at `http://localhost:5000`).
- **API**: The FastAPI application (accessible at `http://localhost:8000`).

### Running Tests

The project uses `pytest` for testing.

**Command:**
```bash
pytest
```

## 5. Development Conventions

- **Code Style:** The project uses `black` for code formatting and `isort` for import sorting. `flake8` and `ruff` are used for linting.
- **Pre-commit Hooks:** The `.pre-commit-config.yaml` file suggests that pre-commit hooks are used to enforce code style and quality before committing code.
- **Dependency Management:** Dependencies are managed using `poetry`, as indicated by the `pyproject.toml` and `poetry.lock` files.
- **Configuration:** Configuration is separated from code and managed through environment variables and configuration files in the `configs/` directory.
