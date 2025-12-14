# Credit Scoring Platform - Qwen Context File

## Project Overview

The Alternative Credit Scoring Platform is a production-ready Machine Learning system designed for alternative credit scoring, showcasing comprehensive MLOps practices. The project focuses on providing a robust, scalable, and observable platform for assessing creditworthiness using non-traditional data sources, fostering financial inclusion for underserved populations.

## Architecture

The platform follows a microservices architecture with the following key components:

- **API Service**: FastAPI-based RESTful API for real-time predictions
- **Model Registry**: MLflow for experiment tracking, model versioning and lifecycle management
- **Feature Store**: Redis for caching and serving features
- **Data Pipeline**: Apache Airflow for orchestration of data processing workflows
- **Monitoring**: Prometheus and Grafana for infrastructure and model monitoring
- **Drift Detection**: Evidently AI for automated monitoring of data and model drift
- **Explainable AI**: SHAP-based explanations for every prediction

## Technology Stack

- **ML Frameworks**: XGBoost, Scikit-learn
- **MLOps**: MLflow, Evidently AI
- **API**: FastAPI, Uvicorn
- **Data Storage**: PostgreSQL (MLflow metadata), MinIO (Artifacts), Redis (Feature Store)
- **Monitoring**: Prometheus, Grafana, AlertManager
- **Orchestration**: Apache Airflow
- **Containerization**: Docker, Docker Compose
- **Languages**: Python 3.13

## Project Structure

```
├── configs/          # Configuration files
├── data/
│   ├── processed/    # Processed datasets
│   ├── raw/          # Raw datasets
│   └── reference/    # Reference datasets
├── docs/             # Documentation files
├── dags/             # Apache Airflow DAGs for workflow orchestration
├── docker/           # Additional Docker configurations
├── grafana/          # Grafana dashboards and configurations
├── notebooks/        # Jupyter notebooks for experimentation
├── prometheus/       # Prometheus configuration files
├── scripts/          # Scripts for data processing, model training, etc.
├── src/              # Source code
│   ├── api/          # API implementation with Champion-Challenger model support
│   ├── data/         # Data processing modules
│   ├── features/     # Feature engineering modules
│   ├── models/       # Model training and evaluation modules
│   └── monitoring/   # Monitoring and drift detection modules
├── tests/            # Unit and integration tests
├── terraform/        # Infrastructure as Code definitions
├── .env.template     # Environment variables template
├── docker-compose.yml # Docker services configuration
├── Dockerfile        # Container definition
├── pyproject.toml    # Project dependencies and configuration
└── README.md         # Project overview
```

## Building and Running

### Prerequisites
- Docker and Docker Compose
- Python 3.13.x (specifically required for Apache Airflow compatibility)
- Poetry (recommended for dependency management)

### Quick Setup

1. Clone the repository and navigate to the project directory
2. Copy environment template:
   ```bash
   cp .env.template .env
   ```
3. Start services with Docker Compose:
   ```bash
   docker-compose up -d
   ```
4. Install dependencies:
   ```bash
   poetry install
   # or
   pip install -r requirements.txt
   ```

### Starting Services
- **API Server**: `cd src/api && uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- **All services**: `docker-compose up -d`
- **Data processing**: `python scripts/make_dataset.py`
- **Model training**: `python src/models/train.py`

## Development Conventions

- **Code Formatting**: Black and Ruff formatting tools are configured
- **Testing**: Pytest is used for unit and integration tests
- **Dependencies**: Managed via Poetry with pyproject.toml
- **Environment Variables**: Use .env file based on .env.template

## Key Features

- **Real-time Prediction**: Low-latency API for instant credit decisions
- **Champion-Challenger Model Deployment**: Safe model deployments with A/B testing capabilities
- **Model Versioning**: Comprehensive tracking using MLflow
- **Feature Store**: Redis-based caching for efficient feature retrieval
- **Drift Detection**: Automated monitoring using Evidently AI
- **Explainable AI**: SHAP-based explanations for model predictions
- **Comprehensive Monitoring**: Prometheus and Grafana for observability

## API Endpoints

- `/` - Root endpoint with API information
- `/health` - Health check for services and models
- `/predict` - Single prediction endpoint
- `/predict/batch` - Batch prediction endpoint
- `/model/info` - Information about loaded models
- `/model/load` - Endpoint to load/reload models
- `/docs` - Interactive API documentation

## Testing

Run tests with:
```bash
pytest
```

## Data Flow

1. Raw data is processed using scripts in the `scripts/` directory
2. Data is cleaned and features engineered using modules in `src/data/`
3. Models are trained using algorithms like XGBoost in `src/models/`
4. Models are registered and tracked with MLflow
5. Predictions are served through the FastAPI application
6. Results are monitored using Prometheus and Grafana

## Champion-Challenger Architecture

The platform implements a Champion-Challenger model deployment strategy where:
- A **Champion** model serves the majority of prediction requests
- A **Challenger** model receives a configurable percentage of requests (default 10%)
- Performance metrics are compared to determine if the challenger should become the new champion

## MLOps Practices

The platform implements several key MLOps practices:
- Experiment tracking with MLflow
- Model versioning and registry
- Continuous monitoring for data and model drift
- Automated pipeline orchestration with Apache Airflow
- Feature store with Redis for consistent feature serving
- Model explainability with SHAP