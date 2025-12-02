# Repository Structure

This document outlines the structure of the repository, which follows best practices for organizing a production-oriented MLOps project.

```
.
├── .github/
│   └── workflows/      # GitHub Actions CI/CD pipelines
├── configs/            # Configuration files (e.g., model params, feature lists)
├── data/
│   ├── processed/      # Intermediate or processed data
│   ├── raw/            # Original, immutable data
│   └── reference/      # Reference data for monitoring (e.g., training set stats)
├── docs/               # Project documentation (what you are reading now)
├── notebooks/          # Jupyter notebooks for exploration and research
├── scripts/            # Standalone scripts (e.g., run_training.py)
├── src/
│   ├── api/            # Source code for the prediction API (FastAPI)
│   ├── data/           # Data processing and feature engineering modules
│   ├── models/         # Model training, prediction, and evaluation logic
│   └── monitoring/     # Code for drift detection and monitoring hooks
├── tests/              # Unit and integration tests
├── .gitignore          # Files and directories to be ignored by Git
├── docker-compose.yml  # Docker Compose file for local development environment
└── README.md           # Top-level project readme
```

## Component Descriptions

-   **`.github/workflows/`**: Contains YAML files that define the CI/CD pipelines using GitHub Actions. This includes workflows for testing, linting, and building Docker images.

-   **`configs/`**: A centralized directory for all configuration. Separating configuration from code allows for easier management of parameters for different environments (development, production) without changing the source code.

-   **`data/`**: Holds all datasets used in the project.
    -   `raw/`: The original, immutable data dump. Data here should be treated as read-only.
    -   `processed/`: Cleaned, transformed, or feature-engineered data.
    -   `reference/`: A baseline dataset (usually the training set) used by monitoring tools like Evidently AI to compare against live production data for drift detection.

-   **`docs/`**: Contains all project documentation, including architectural diagrams, API specifications, and model evaluation reports.

-   **`notebooks/`**: For interactive, exploratory work. This is where initial data analysis (EDA), model prototyping, and experimentation happen. Code is typically moved from notebooks to the `src` directory once it's ready to be part of the production pipeline.

-   **`scripts/`**: Holds standalone scripts that perform key actions, such as `run_training.py` to execute the model training pipeline or `make_dataset.py` to run data processing steps.

-   **`src/`**: The main source code for the project, structured as a Python package.
    -   `api/`: The FastAPI application code, including endpoints, request/response schemas, and serving logic.
    -   `data/`: Reusable modules for data loading, validation, preprocessing, and feature engineering.
    -   `models/`: Core modeling logic, including scripts for training models (`train.py`), making predictions (`predict.py`), and evaluation.
    -   `monitoring/`: Modules related to model monitoring, such as calculating drift metrics.

-   **`tests/`**: Contains all automated tests. The structure of this directory should mirror the `src` directory to make it easy to find tests for specific modules.

-   **Root Files**:
    -   `docker-compose.yml`: Defines the services, networks, and volumes for the local development environment.
    -   `README.md`: The entry point for the repository, containing a project summary and basic setup instructions.
