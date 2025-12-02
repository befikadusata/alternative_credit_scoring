# Model Evaluation, Validation, and Fairness

A rigorous evaluation framework is essential to ensure that the model is not only performant but also reliable, fair, and ready for production.

## 1. Performance Metrics

A combination of metrics is used to get a holistic view of the model's performance.

-   **Primary Metric: AUC-ROC** (Area Under the Receiver Operating Characteristic Curve). This is the key metric for optimization. It is threshold-independent and robust to class imbalance, making it ideal for this use case. A higher AUC-ROC indicates better-ranking ability of the model.
    -   *Target: > 0.75 (good), > 0.80 (excellent)*

-   **Secondary Metrics:**
    -   **AUC-PR:** The area under the Precision-Recall curve is also monitored, as it gives a better sense of performance on highly imbalanced datasets where the positive class is of primary interest.
    -   **Precision:** Of all the applicants the model predicts will default, what percentage actually do?
    -   **Recall:** Of all the applicants who actually defaulted, what percentage did the model correctly identify?
    -   **F1-Score:** The harmonic mean of precision and recall.
    -   **Brier Score:** Measures the accuracy of probabilistic predictions. A lower score indicates better calibration.

### Confusion Matrix Analysis

The confusion matrix is used to analyze the business impact of the model's errors.

| | Predicted: No Default | Predicted: Default |
| :--- | :--- | :--- |
| **Actual: No Default** | True Negative (TN) | False Positive (FP) |
| **Actual: Default** | False Negative (FN) | True Positive (TP) |

-   **Business Impact:**
    -   **False Positives (FP):** Incorrectly rejecting good applicants, leading to lost opportunity cost.
    -   **False Negatives (FN):** Incorrectly approving risky applicants, leading to financial loss.

The model's decision threshold can be tuned to balance the trade-off between FPs and FNs based on business goals.

## 2. Model Validation Checklist

Before a model version can be promoted to production, it must pass a series of automated validation checks.

| Category | Metric | Threshold |
| :--- | :--- | :--- |
| **Performance** | `auc_roc` (on test set) | `min: 0.75` |
| | `precision` | `min: 0.70` |
| | `recall` | `min: 0.60` |
| | `train_val_auc_gap` | `max: 0.05` |
| **Fairness** | `demographic_parity_diff` | `max: 0.10` |
| | `equal_opportunity_diff`| `max: 0.10`|
| **Stability** | `cv_auc_std` (cross-val) | `max: 0.03`|
| **Inference** | `prediction_time_ms` (p95)| `max: 80` |
| | `model_size_mb` | `max: 100` |

## 3. Bias & Fairness Analysis

To ensure the model makes equitable decisions, it is tested for bias across sensitive demographic groups (e.g., age groups, geographic regions).

-   **Protected Attributes:** The model's performance is compared across different subgroups for a given attribute.
-   **Fairness Metrics:**
    -   **Demographic Parity:** Checks if the approval rate (the proportion of positive predictions) is similar across all subgroups. A large difference suggests the model may be systematically favoring one group.
    -   **Equal Opportunity:** Checks if the true positive rate (recall) is similar across all subgroups. A large difference suggests the model is better at identifying positive outcomes for one group than another.

If the difference in these metrics between groups exceeds a predefined threshold (e.g., 10%), the model is flagged for a manual review by the data science team.
