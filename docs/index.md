# Alternative Credit Scoring MLOps Project

## 1. Project Overview

This project demonstrates a production-ready Machine Learning system for alternative credit scoring. It is designed to assess the creditworthiness of individuals with limited or no formal credit history by using alternative data sources, such as employment records, digital footprint, and payment patterns.

The solution is an end-to-end ML platform featuring real-time credit risk prediction, robust model versioning, automated drift detection, and explainable AI, showcasing enterprise-grade MLOps capabilities.

### 1.1. Key Capabilities

| Capability | Implementation | Business Value |
| --- | --- | --- |
| **Real-time Prediction** | FastAPI-based API with <100ms latency (p95) | Enables instant lending decisions for a better user experience. |
| **Model Versioning** | MLflow Registry for tracking and managing model versions. | Provides a complete audit trail and rollback capabilities. |
| **Champion-Challenger** | A/B traffic routing between production and staging models. | Ensures safe and reliable deployment of new models. |
| **Drift Detection** | Evidently AI for automated data and prediction drift monitoring. | Acts as an early warning system for model performance degradation. |
| **Explainability** | SHAP values generated for each prediction. | Ensures regulatory compliance and builds trust with stakeholders. |
| **Observability** | Prometheus for metrics collection and Grafana for visualization. | Delivers operational excellence and system health monitoring. |

### 1.2. Target Use Case

**Primary Scenario:** The system is designed for small-value loan origination (e.g., $500 - $5,000) for individuals with limited traditional credit history.

**User Journey:**
1. An applicant submits a loan request through a web or mobile application.
2. The system ingests application data along with alternative data sources.
3. A real-time credit score and risk assessment are generated in under two seconds.
4. A loan officer reviews the score and the accompanying explanation.
5. A final decision (approve, reject, or manual review) is made.
6. The outcome is logged to a feedback loop for continuous model improvement.

### 1.3. Stakeholders

| Stakeholder | Needs | Success Criteria |
| --- | --- | --- |
| **Loan Officers** | Fast, reliable scores with clear explanations. | Over 95% of lending decisions are aided by the model's output. |
| **Data Scientists** | Efficient experiment tracking and model comparison. | Easy and rapid retraining and deployment of models. |
| **ML Engineers** | Reliable serving, monitoring, and debugging tools. | >99.5% API uptime with latency under 100ms. |
| **Compliance Team**| Complete audit trail, fairness metrics, and explainability. | Full lineage tracking for models and predictions. |
| **Applicants** | Fair and transparent credit decisions. | A demographic parity gap of less than 10% across groups. |
