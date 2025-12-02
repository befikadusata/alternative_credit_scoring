# CI/CD Pipeline

This document outlines the Continuous Integration and Continuous Deployment (CI/CD) pipeline implemented using GitHub Actions. The pipeline ensures code quality, automates testing, and facilitates the building of deployment artifacts.

## 1. Pipeline Overview

The CI/CD pipeline is triggered automatically on specific events within the GitHub repository and consists of several distinct stages:

1.  **Linting and Formatting Checks:** Ensures code adheres to established style guides and best practices.
2.  **Unit and Integration Tests:** Validates the functionality of individual components and their interactions.
3.  **Docker Image Build:** Creates and tags Docker images for the application services.

## 2. GitHub Actions Workflow

The pipeline is defined in a YAML file within the `.github/workflows/` directory of the repository.

### Triggers

The workflow is configured to run on the following events:

*   **Push to `main` branch:** Automatically triggers the full CI/CD pipeline, including tests and Docker image builds.
*   **Pull Requests:** Runs linting and tests to ensure code quality before merging.

### Stages

#### 2.1. Lint and Format

*   **Purpose:** Enforce code style and identify basic syntax errors.
*   **Steps:**
    *   Checkout code.
    *   Set up Python environment.
    *   Install linting tools (e.g., `flake8`, `black`).
    *   Run linters and formatters. Fails if any issues are found.

#### 2.2. Test

*   **Purpose:** Verify the correctness of the application logic.
*   **Steps:**
    *   Checkout code.
    *   Set up Python environment.
    *   Install project dependencies.
    *   Run unit tests (e.g., using `pytest`).
    *   Run integration tests (e.g., testing API endpoints against a mocked or temporary database).
    *   (Optional but recommended) Run a simplified model training pipeline with a small dataset to catch potential regressions in the ML training logic.

#### 2.3. Build Docker Images

*   **Purpose:** Create production-ready Docker images for deployment.
*   **Steps:**
    *   Checkout code.
    *   Log in to a Docker registry (e.g., Docker Hub, AWS ECR).
    *   Build Docker images for each service (e.g., `prediction-service`, `mlflow-server`).
    *   Tag images appropriately (e.g., `latest`, Git commit SHA, version number).
    *   Push tagged images to the Docker registry.

## 3. Future Enhancements

*   **Deployment Stage:** Automatically deploy new Docker images to a staging or production environment (e.g., Kubernetes, AWS ECS) upon successful completion of all CI steps.
*   **Model Validation in CI:** Integrate more robust model validation checks within the CI pipeline, beyond just a training run, to ensure model quality before deployment.
*   **Security Scanning:** Add steps for static application security testing (SAST) and dependency vulnerability scanning.
