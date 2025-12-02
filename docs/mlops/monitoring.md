# Model & System Monitoring

A comprehensive monitoring strategy is in place to ensure the health, performance, and reliability of the entire system. The approach is built on three layers of monitoring.

## 1. The Three Layers of Monitoring

#### Layer 1: Infrastructure Metrics

This layer tracks the health and performance of the serving infrastructure itself.

-   **Tooling:** **Prometheus** for metrics collection and **Grafana** for visualization.
-   **Key Metrics (The RED Method):**
    -   **Rate:** The number of requests per second (RPS) the API is handling.
    -   **Errors:** The rate of server-side errors (e.g., HTTP 5xx).
    -   **Duration:** The latency of requests, typically measured at the 50th, 95th, and 99th percentiles (p50, p95, p99).
-   **Other Metrics:**
    -   System resource usage (CPU, memory, disk).
    -   Database connection pool status.
    -   Model loading times.

#### Layer 2: ML-Specific Metrics

This layer focuses on monitoring the performance of the machine learning model, specifically looking for signs of degradation.

-   **Tooling:** **Evidently AI** for generating detailed drift and performance reports.
-   **Key Metrics:**
    -   **Data Drift:** Detects statistical changes in the distribution of input features between the training data and live production data. For example, has the average income of applicants suddenly changed?
    -   **Prediction Drift:** Detects statistical changes in the distribution of the model's output (the prediction scores). This can be an early indicator of data drift.
    -   **Model Performance:** When ground truth labels become available (i.e., when we know if a loan actually defaulted), the model's performance metrics (e.g., AUC-ROC, precision) are re-calculated and tracked over time.

#### Layer 3: Business Metrics

This layer tracks high-level Key Performance Indicators (KPIs) that are relevant to business stakeholders.

-   **Tooling:** Custom dashboards in **Grafana**.
-   **Key Metrics:**
    -   Approval rate trends over time.
    -   The distribution of loan amounts by the model's risk category.
    -   The rate at which loan officers override the model's recommendations.

## 2. Monitoring Workflow and Alerting

1.  **Logging:** Every prediction request, its input features, and the model's output are logged to a structured format.
2.  **Aggregation:** These logs are collected and processed in near-real-time batches (e.g., hourly).
3.  **Analysis:** The aggregated batches are fed into **Evidently AI** to compare against a baseline reference dataset (e.g., the training data).
4.  **Reporting:** Evidently AI generates drift reports and calculates drift scores.
5.  **Metrics Export:** Key metrics from all three layers (e.g., p95 latency, error rate, feature drift score) are scraped by **Prometheus**.
6.  **Visualization:** **Grafana** pulls data from Prometheus to display on dashboards, providing a single pane of glass for system observability.
7.  **Alerting:** Prometheus is configured with alert rules that trigger notifications (e.g., via Slack or email) if a metric crosses a predefined threshold.

### Example Alerting Thresholds

```python
ALERT_THRESHOLDS = {
    "api_latency_p95_ms": 150,
    "api_error_rate": 0.01,  # 1%
    "data_drift_detected": True,
    "feature_drift_share": 0.30,  # 30% of features have drifted
    "prediction_drift_score": 0.15, # Kolmogorov-Smirnov test p-value
    "model_performance_drop": 0.05  # 5% drop in AUC
}
```
