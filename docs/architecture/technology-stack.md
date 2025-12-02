# Technology Stack

This project uses a curated set of modern, open-source technologies chosen for their performance, reliability, and strong community support.

| Layer | Technology | Justification |
| --- | --- | --- |
| **ML Training** | XGBoost 2.0+ | Industry standard for tabular data, providing excellent performance and speed. |
| **ML Framework** | Scikit-learn 1.3+ | Used for the feature engineering pipeline and model evaluation metrics. |
| **MLOps Platform** | MLflow 2.8+ | Provides a complete solution for the ML lifecycle, from experiment tracking to model deployment. |
| **API Framework** | FastAPI 0.104+ | A high-performance Python web framework for building APIs, with automatic documentation and async support. |
| **Data Validation** | Pydantic v2 | Enables type-safe data validation and settings management, integrated seamlessly with FastAPI. |
| **Model Serving** | MLflow Models | The native MLflow format for packaging models, allowing for consistent deployment. |
| **Feature Store** | Redis 7+ | Provides an in-memory, key-value cache for serving pre-computed features at low latency. |
| **Metadata Database** | PostgreSQL 14+ | A reliable, ACID-compliant relational database used as the backend store for MLflow. |
| **Artifact Storage** | MinIO | An S3-compatible object storage solution for storing large model artifacts. Can be swapped for AWS S3 in production. |
| **ML Monitoring** | Evidently AI 0.4+ | A specialized open-source tool for detecting data drift, prediction drift, and model performance issues. |
| **Infra Monitoring**| Prometheus & Grafana | The industry-standard combination for collecting time-series metrics and creating observability dashboards. |
| **Containerization**| Docker 24+ | Ensures consistent and reproducible development and deployment environments. |
| **Orchestration** | Docker Compose | Used for managing the multi-container local development environment. |
| **CI/CD** | GitHub Actions | Integrated directly with the source code repository for continuous integration and deployment automation. |
| **Cloud (Target)** | AWS (EC2, S3, RDS, EKS) | The architecture is designed to be cloud-agnostic but is primarily targeted for deployment on AWS for production. |
