# CI/CD Pipeline

This document outlines the Continuous Integration and Continuous Deployment (CI/CD) pipeline implemented using GitHub Actions. The pipeline ensures code quality, automates testing, and facilitates the building of deployment artifacts.

## 1. Pipeline Overview

The CI/CD pipeline is triggered automatically on specific events within the GitHub repository and consists of several distinct stages:

1.  **Linting and Formatting Checks:** Ensures code adheres to established style guides and best practices.
2.  **Unit and Integration Tests:** Validates the functionality of individual components and their interactions.
3.  **Docker Image Build (Future):** Creates and tags Docker images for the application services.

## 2. GitHub Actions Workflow

The pipeline is defined in the `.github/workflows/ci.yml` file in the repository.

### Triggers

The workflow is configured to run on the following events:

*   **Push to `main` branch:** Automatically triggers the full CI pipeline.
*   **Pull Requests to `main` branch:** Runs the full CI pipeline to ensure code quality before merging.

### Stages

The pipeline consists of two main jobs: `lint-and-format` and `test`.

#### 2.1. Lint and Format (`lint-and-format`)

*   **Purpose:** Enforce code style and identify basic syntax errors before running the test suite.
*   **Steps:**
    *   Checks out the repository code.
    *   Sets up a Python 3.13.x environment (due to Apache Airflow compatibility).
    *   Installs [Poetry](https://python-poetry.org/) for dependency management.
    *   Installs project dependencies using `poetry install`.
    *   Runs `poetry run black --check .` to ensure code is formatted correctly.
    *   Runs `poetry run isort --check .` to ensure imports are sorted correctly.
    *   Runs `poetry run flake8 .` to check for linting errors.
    *   The job will fail if any of these checks do not pass.

#### 2.2. Test (`test`)

*   **Purpose:** Verify the correctness of the application logic by running the test suite. This job depends on the successful completion of the `lint-and-format` job.
*   **Steps:**
    *   Checks out the repository code.
    *   Sets up a Python 3.13.x environment (due to Apache Airflow compatibility) and installs Poetry.
    *   Installs project dependencies using `poetry install`.
    *   Runs the test suite using `poetry run pytest`.

#### 2.3. Build Docker Images (Future Implementation)

*   **Purpose:** Create production-ready Docker images for deployment.
*   **Note:** This stage is not yet implemented in the current CI/CD workflow.
*   **Future Steps:**
    *   Log in to a Docker registry (e.g., Docker Hub, AWS ECR).
    *   Build Docker images for each service (e.g., `prediction-service`).
    *   Tag images with the Git commit SHA and/or version number.
    *   Push tagged images to the Docker registry.

## 3. Future Enhancements

*   **Deployment Stage:** Automatically deploy new Docker images to a staging or production environment (e.g., Kubernetes, AWS ECS) upon successful completion of all CI steps.
*   **Model Validation in CI:** Integrate more robust model validation checks within the CI pipeline, beyond just a training run, to ensure model quality before deployment.
*   **Security Scanning:** Add steps for static application security testing (SAST) and dependency vulnerability scanning.
