# Architectural Technology Decisions

This document records the key technology choices and their justifications for this project. The goal is to build a robust, scalable, and maintainable MLOps platform using a curated stack of modern, open-source tools.

### 1. API Framework: FastAPI

**Decision:** Use FastAPI for the prediction and management API.

**Justification:**
-   **Performance:** FastAPI is one of the fastest Python web frameworks available, built on Starlette (for ASGI) and Pydantic (for data validation). Its asynchronous nature allows for high concurrency, making it ideal for low-latency prediction serving.
-   **Developer Experience:** It features intuitive, type-hint-based development that reduces bugs.
-   **Automatic Documentation:** It automatically generates interactive OpenAPI (Swagger) and ReDoc documentation, which is invaluable for API discoverability and testing.
-   **Data Validation:** Its seamless integration with Pydantic enforces strict data validation at the API boundary, ensuring data quality before it enters the core application logic.

### 2. MLOps Lifecycle Management: MLflow

**Decision:** Use MLflow as the central platform for managing the machine learning lifecycle.

**Justification:**
-   **All-in-One Solution:** MLflow provides four tightly integrated components (Tracking, Projects, Models, and Registry) that cover the entire lifecycle, from experiment logging to production deployment. This avoids the complexity of integrating multiple disparate tools.
-   **Framework Agnostic:** It works with any ML library (Scikit-learn, XGBoost, TensorFlow, etc.), making it highly flexible.
-   **Reproducibility:** By tracking code versions, parameters, and artifacts, MLflow ensures that any experiment or model can be perfectly reproduced.
-   **Model Governance:** The Model Registry provides a formal workflow for managing model versions and their lifecycle stages (e.g., Staging, Production), which is critical for compliance and safe deployments.

### 3. Containerization & Orchestration: Docker & Docker Compose

**Decision:** Use Docker to containerize all services and Docker Compose to manage the local development environment.

**Justification:**
-   **Consistency & Portability:** Docker containers encapsulate each service and its dependencies, ensuring that the application runs identically on any machine, from a developer's laptop to a production server. This eliminates "it works on my machine" problems.
-   **Isolation:** Each service runs in its own isolated environment, preventing dependency conflicts and improving security.
-   **Simplified Local Development:** Docker Compose allows the entire multi-container stack (API, databases, MLflow, etc.) to be defined in a single YAML file and launched with a single command (`docker-compose up`), dramatically simplifying the setup process.

### 4. Data & Artifact Storage: PostgreSQL & MinIO

**Decision:** Use PostgreSQL for metadata and MinIO for artifact storage.

**Justification:**
-   **PostgreSQL (Metadata):** A powerful, open-source, and ACID-compliant relational database is required to reliably store the metadata for MLflow experiments and Airflow pipeline runs. PostgreSQL is a proven, production-grade choice.
-   **MinIO (Artifacts):** Machine learning artifacts (like model files, datasets, and plots) are large, unstructured blobs that are best stored in an object store. MinIO provides an S3-compatible API, making it a perfect open-source choice for local development. This also ensures a seamless transition to a cloud provider like AWS S3 for production.

### 5. Monitoring Stack: Prometheus, Grafana, & Evidently AI

**Decision:** Use a three-part stack for comprehensive observability.

**Justification:**
-   **Prometheus & Grafana (System Metrics):** This is the de-facto industry standard for system observability. Prometheus excels at scraping and storing time-series metrics (e.g., API latency, CPU usage), and Grafana is unmatched for creating powerful, intuitive dashboards to visualize them.
-   **Evidently AI (ML-Specific Monitoring):** While Prometheus is great for system health, it is not designed to understand machine learning concepts. Evidently AI specializes in detecting **data drift** (when input data changes over time) and **prediction drift** (when model output distributions change), which are critical for maintaining model performance in production.

### 6. Pipeline Orchestration: Apache Airflow

**Decision:** Use Apache Airflow for orchestrating the batch training pipelines.

**Justification:**
-   **Python-Native:** DAGs (Directed Acyclic Graphs) are defined in Python, which allows for dynamic pipeline generation and access to the full Python ecosystem. This makes it a natural fit for ML engineers.
-   **Scalability & Extensibility:** Airflow has a highly scalable architecture (Scheduler, Worker, Webserver) and a vast library of providers for integrating with nearly any external system.
-   **Mature & Battle-Tested:** As a long-standing and widely adopted project, it is a reliable and feature-rich choice for complex workflow management.
