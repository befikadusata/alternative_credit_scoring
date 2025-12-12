# Project Roadmap: MLOps Enhancements

This document outlines the key initiatives and actionable tasks required to advance the project from a strong proof-of-concept to a fully automated, production-grade MLOps system.

---

## 0. Foundational Local Environment Setup

**Objective:** Create a "one-command" local development environment that is fully integrated and mirrors the production setup, addressing current gaps in the local developer experience.

-   [x] **Integrate Redis into Docker Compose**
    -   [x] Add a `redis` service to the `docker-compose.yml` file using the official `redis:alpine` image.
    -   [x] Configure a persistent volume for the Redis service to retain cached features across restarts.
    -   [x] Update the `.env.template` to ensure the API's `redis_client.py` is pre-configured to connect to the Dockerized Redis instance.

-   [x] **Integrate the FastAPI Application into Docker Compose**
    -   [x] Add an `api` service to the `docker-compose.yml` that builds from the existing project `Dockerfile`.
    -   [x] Configure the `api` service to depend on the `mlflow`, `postgres`, and `redis` services to manage startup order.
    -   [x] Map port `8000` on the host to the container to make the API accessible.
    -   [x] Set environment variables (e.g., `MLFLOW_TRACKING_URI`, `REDIS_HOST`) directly in the `docker-compose.yml` for the `api` service, so it automatically connects to the other containerized services.

-   [x] **Implement Real Model Loading from MLflow**
    -   [x] Refactor `src/api/model_loader.py` to remove the placeholder `DummyModel`.
    -   [x] Implement the logic to use `mlflow.pyfunc.load_model()` to download the specified model from the MLflow Tracking Server.
    -   [x] Ensure the `tracking_uri` is configured via the environment variable provided by Docker Compose.
    -   [x] Modify the logic to also load the `DataCleaner` preprocessor, which should be logged as an artifact alongside the model in MLflow.

-   [x] **Update Project Documentation**
    -   [x] Update the `README.md` and create a `GETTING_STARTED.md` to reflect the new, simplified "one-command" setup (`docker-compose up --build`).
    -   [x] Remove obsolete instructions related to running the API or its dependencies manually.

---

## 1. Implement Workflow Orchestration

**Objective:** Automate the entire model training and registration pipeline to ensure it is reproducible, schedulable, and monitored.

**Tool:** Apache Airflow

-   [x] **Setup Airflow Environment**
    -   [x] Add `apache-airflow` and required providers (e.g., `apache-airflow-providers-docker`) to `pyproject.toml`.
    -   [x] Create a `dags/` directory in the project root.
    -   [x] Update `docker-compose.yml` to include Airflow services (webserver, scheduler, worker, metadata database).

-   [x] **Define the Training DAG**
    -   [x] Create a new DAG file: `dags/training_pipeline_dag.py`.
    -   [x] Define the DAG to run on a weekly schedule and also be triggerable manually.
    -   [x] **Task 1: Data Ingestion:** Create a task to fetch the latest raw data for training.
    -   [x] **Task 2: Data Validation (Gate):** Implement a task that runs the `validate_data.py` script (from Initiative #2). The pipeline must fail if data quality or drift issues are detected.
    -   [x] **Task 3: Data Processing:** Create a task to run the data cleaning and feature engineering pipeline.
    -   [x] **Task 4: Model Training:** Create a task to execute the `src/models/train.py` script, logging the run to MLflow.
    -   [x] **Task 5: Model Evaluation:** Create a task that runs an evaluation script against the test set and logs metrics to MLflow.
    -   [x] **Task 6: Performance Validation (Gate):** Implement a conditional task that checks if the new model's performance (e.g., AUC > 0.78) meets the criteria defined in `docs/model/evaluation.md`. The pipeline must fail if the model is not performant enough.
    -   [x] **Task 7: Model Registration:** If all gates pass, create a final task that uses the MLflow API to transition the trained model to the "Staging" stage in the Model Registry.
    -   [x] **Task 8: Notifications:** Implement tasks that send success or failure notifications to a designated channel (e.g., Slack or email).

---

## 2. Implement Automated Data and Model Validation

**Objective:** Create an automated validation gate to prevent data quality issues or drift from impacting model training and performance.

**Tool:** Evidently AI

-   [x] **Setup and Tooling**
    -   [x] Add `evidently` to the project dependencies in `pyproject.toml`.
    -   [x] Create a new script `scripts/create_reference_dataset.py` that generates a baseline dataset from the training data and saves it to `data/reference/`.

-   [x] **Create the Validation Script**
    -   [x] Create the new script: `src/monitoring/validate_data.py`.
    -   [x] The script will accept two arguments: a path to `current_data` and a path to the `reference_data`.
    -   [x] Use Evidently AI's `TestSuite` to define a battery of tests.

-   [x] **Define Validation Tests in the Script**
    -   [x] Add `DataDriftTestPreset` to detect statistical drift in features.
    -   [x] Add `DataQualityTestPreset` to check for missing values, column types, and duplicates.
    -   [x] Add column-specific value range tests (e.g., `TestColumnValueMin`, `TestColumnValueMax`) for critical features like `annual_inc`.

-   [x] **Integrate and Automate**
    -   [x] The script must export the `TestSuite` results as a JSON object and exit with a non-zero status code if the test suite fails.
    -   [x] The script should also generate an HTML report for visual inspection, which will be saved as an artifact by the Airflow pipeline.
    -   [x] Integrate this script as the second task in the Airflow training DAG to act as a data quality gate.

---

## 3. Implement Active Model Monitoring and Alerting

**Objective:** Move from passive logging to an active monitoring system that tracks model performance in production and automatically alerts the team when issues are detected.

-   [x] **Create a Monitoring Job**
    -   [x] Develop a new script, `src/monitoring/monitor_predictions.py`, that:
        -   [x] Fetches prediction logs from the production environment from the last hour.
        -   [x] Loads the reference dataset from `data/reference/`.
        -   [x] Uses Evidently AI to generate a `Report` or `TestSuite` comparing the live data to the reference data.
    -   [x] Create a new Airflow DAG (`dags/monitoring_dag.py`) that runs this script on an hourly basis.

-   [x] **Expose and Visualize Metrics**
    -   [x] Modify the monitoring script to calculate key drift metrics (e.g., share of drifting features, prediction drift score).
    -   [x] Use the `prometheus_client` library to expose these metrics on a port.
    -   [x] Update the project's Grafana instance (`grafana/dashboard.json`) to add new panels that visualize these drift metrics over time, alongside existing infrastructure metrics.

-   [x] **Implement Alerting**
    -   [x] Configure Prometheus Alertmanager to handle alerts.
    -   [x] Define alert rules in Prometheus based on the thresholds in `docs/mlops/monitoring.md` (e.g., `ALERT if feature_drift_share > 0.3`).
    -   [x] Configure Alertmanager to route triggered alerts to a team channel (e.g., Slack).

---

## 4. Mature the CI/CD Pipeline for ML

**Objective:** Enhance the existing CI pipeline and build out a CD (Continuous Deployment) workflow to automate testing and deployment of both the API and the models.

-   [x] **Enhance the CI Pipeline (`.github/workflows/ci.yml`)**
    -   [x] **Add Integration Testing:** Create a new job that uses `docker-compose` to spin up the entire application stack.
    -   [x] Develop a test script (`tests/integration/test_api.py`) that sends requests to the running API and asserts that the responses are valid. This ensures all services work together correctly.
    -   [x] **Add Security Scanning:** Integrate a tool like `trivy` or `snyk` to scan the Docker image for vulnerabilities.

-   [x] **Create a CD (Continuous Deployment) Workflow**
    -   [x] Create a new workflow file: `.github/workflows/cd.yml`.
    -   [x] **Trigger:** Configure the workflow to run on pushes to the `main` branch.
    -   [x] **Step 1: Build & Push Image:** Add a step to build the application's Docker image and push it to AWS ECR with a new version tag.
    -   [x] **Step 2: Deploy to Staging:** Add a step that uses Terraform or the AWS CLI to deploy the new image to a dedicated staging environment on AWS ECS.
    -   [x] **Step 3: Run E2E Tests:** Run an end-to-end test suite against the live staging environment to confirm deployment success.
    -   [x] **Step 4: Manual Approval Gate:** Add a `manual approval` step in the GitHub Actions workflow to require a team member to sign off before deploying to production.
    -   [x] **Step 5: Deploy to Production:** Upon approval, a final job will deploy the validated image to the production environment, potentially using a blue/green deployment strategy to ensure zero downtime.
