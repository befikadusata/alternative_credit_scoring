# MLOps Overview

This section describes the MLOps infrastructure and workflows that support the continuous integration, deployment, and monitoring of the machine learning model.

### Guiding Principles

1.  **Automation:** The entire ML lifecycle, from training to deployment and monitoring, is automated to ensure reliability and speed.
2.  **Reproducibility:** Every model and prediction is versioned and tracked, making the entire system auditable and reproducible.
3.  **Collaboration:** A centralized platform (MLflow) enables seamless collaboration between data scientists, ML engineers, and other stakeholders.
4.  **Continuous Monitoring:** The model's performance is continuously monitored in production to detect issues like data drift and performance degradation proactively.

### Key Documents

*   [**Experiment Tracking & Registry**](./experiment-tracking.md): Details on the MLflow setup for tracking experiments and managing the model lifecycle through the Model Registry.
*   [**Model Serving**](./model-serving.md): The strategy for safely deploying models using a Champion-Challenger framework.
*   [**Monitoring**](./monitoring.md): The three-layer approach to monitoring, covering infrastructure, model performance, and business KPIs.
