# Project Roadmap: Alternative Credit Scoring Platform

This document outlines a comprehensive, phased roadmap for the development and deployment of the credit scoring MLOps project. Each phase represents a major milestone.

---

### Phase 1: Project Setup & Foundational Work (Milestone 1)

*This phase focuses on establishing the project's foundation, including the repository structure, development environment, and initial data exploration.*

- [x] **Task 1.1: Initialize Project Structure**
  - [x] Set up the Git repository and define branching strategies.
  - [x] Create the standard repository directory structure (`src`, `data`, `docs`, `configs`, etc.).
  - [x] Add initial documentation (`README.md`, `.gitignore`, `docs/repository-structure.md`).

- [x] **Task 1.2: Data Acquisition & Initial Exploration**
  - [x] Download and document the source of the LendingClub dataset.
  - [x] Perform initial Exploratory Data Analysis (EDA) in a Jupyter Notebook to understand features, distributions, and data quality issues.
  - [x] Document initial findings and hypotheses in the notebook.

- [x] **Task 1.3: Setup Local Development Environment**
  - [x] Create a `docker-compose.yml` file to manage services like MLflow, PostgreSQL, and MinIO.
  - [x] Define project dependencies in `requirements.txt`.
  - [x] Create and document a `.env.template` file for environment variables.
  - [x] Write a `GETTING_STARTED.md` guide for new developers.

---

### Phase 2: Core Model Development (Milestone 2)

*This phase covers the data science lifecycle, from cleaning data and engineering features to training and evaluating the model.*

- [x] **Task 2.1: Data Preprocessing & Feature Engineering Pipeline**
  - [x] Develop reusable data cleaning and imputation logic in `src/data/`.
  - [x] Implement a feature engineering pipeline to create predictive features.
  - [x] Create a master script (`scripts/make_dataset.py`) to run the entire data processing pipeline.
  - [x] Implement a data versioning strategy and store a reference dataset for monitoring.

- [x] **Task 2.2: Baseline Model Training**
  - [x] Develop a model training script (`src/models/train.py`).
  - [x] Integrate MLflow tracking to log all experiments, including parameters, metrics, and model artifacts.
  - [x] Train a baseline model (e.g., Logistic Regression or default XGBoost) to establish a performance benchmark.

- [x] **Task 2.3: Advanced Modeling & Evaluation**
  - [x] Implement a hyperparameter tuning process (e.g., using Optuna or `RandomizedSearchCV`).
  - [x] Develop a robust model evaluation framework (`src/models/evaluate.py`) that includes AUC, precision-recall, and a confusion matrix.
  - [x] Perform bias and fairness analysis on the best model candidate across different demographic groups.
  - [x] Register the final, validated model in the MLflow Model Registry.

---

### Phase 3: API Implementation & MLOps Integration (Milestone 3)

*This phase focuses on operationalizing the model by building a serving API and integrating MLOps tools.*

- [x] **Task 3.1: Develop Prediction API**
  - [x] Set up a FastAPI application in `src/api/` with endpoints for prediction and health checks.
  - [x] Create Pydantic schemas for request and response validation.
  - [x] Implement model loading logic that pulls the specified model version from the MLflow Registry.

- [x] **Task 3.2: Implement Serving Strategies**
  - [x] Develop the Champion-Challenger routing logic to safely test new models.
  - [x] Implement structured logging for all predictions from both models to enable offline analysis.
  - [x] Implement a simple feature store using Redis for low-latency predictions.

- [x] **Task 3.3: Integrate Model Monitoring**
  - [x] Develop a script (`src/monitoring/drift_detection.py`) to run data and prediction drift analysis using Evidently AI.
  - [x] Configure the API to export key metrics (latency, errors) to Prometheus.
  - [x] Create a basic Grafana dashboard to visualize infrastructure and model metrics.

---

### Phase 4: Testing, CI/CD & Deployment (Milestone 4)

*This phase focuses on ensuring reliability through automated testing, building a CI/CD pipeline, and preparing for deployment.*

- [x] **Task 4.1: Implement Comprehensive Testing**
  - [x] Write unit tests for data processing functions and helper utilities.
  - [x] Write integration tests for the FastAPI endpoints to validate the full request/response cycle.
  - [x] Add tests for the model prediction logic to check for consistent outputs.

- [x] **Task 4.2: Build CI/CD Pipeline**
  - [x] Create a GitHub Actions workflow to automate CI checks.
  - [x] Add jobs for linting, formatting, and running the test suite on every pull request.
  - [x] Create a CD workflow that builds and pushes Docker images to a registry on merges to the `main` branch. (Note: CI part is implemented, CD is future work).
  - [x] Document the CI/CD pipeline in `docs/mlops/ci-cd.md`.

- [x] **Task 4.3: Package & Prepare for Deployment**
  - [x] Write Dockerfiles for the prediction service and any other custom components.
  - [x] (Optional) Create deployment manifests (e.g., Kubernetes YAML or Terraform scripts) for a target cloud environment.
  - [x] Document the deployment process and required infrastructure.

---

### Phase 5: Advanced Features & Future Work (Post-Launch)

*This phase includes potential enhancements to mature the platform and extend its capabilities.*

- [ ] **Task 5.1: Enhance Feature Management**
  - [ ] Evolve the Redis-based cache into a full-fledged Feature Store (e.g., by integrating Feast).
  - [ ] Create automated batch pipelines for ingesting new features into the feature store.

- [ ] **Task 5.2: Automate Retraining & Deployment**
  - [ ] Develop a workflow (e.g., using Airflow or Kubeflow Pipelines) to automatically trigger model retraining based on drift alerts or a schedule.
  - [ ] Implement a process for automated promotion of models to production if they pass all validation criteria.

- [ ] **Task 5.3: Build an Interactive UI**
  - [ ] Create a simple Streamlit or Gradio web application for stakeholders to upload data, receive predictions, and visualize SHAP explanations.

- [ ] **Task 5.4: Implement A/B Testing**
  - [ ] Enhance the Champion-Challenger framework to support true A/B testing, allowing for the measurement of business KPIs (e.g., default rates, profitability) for different model versions.
