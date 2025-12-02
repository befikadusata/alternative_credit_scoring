# Model Interpretability

To build trust and meet regulatory requirements, it is critical that the model's decisions can be explained. This project uses SHAP (SHapley Additive exPlanations) to provide both global and local interpretability.

## 1. Global Interpretability

Global interpretability helps us understand the model's behavior as a whole across the entire dataset.

-   **Method:** A SHAP Summary Plot is used to visualize the most important features and their impact on the model's output. This plot shows:
    -   **Feature Importance:** Features are ranked by the magnitude of their average SHAP value.
    -   **Impact Direction:** For each feature, it shows whether high or low values tend to increase or decrease the prediction (i.e., the probability of default).

-   **Example Insights:**
    -   "Higher FICO scores consistently decrease the predicted risk."
    -   "A high debt-to-income ratio tends to increase the predicted risk."

This analysis provides a high-level sanity check that the model has learned sensible relationships from the data.

## 2. Local Interpretability

Local interpretability explains an individual prediction, which is crucial for loan officers and for providing feedback to applicants.

-   **Method:** For each prediction, a unique SHAP explanation is generated. This explanation breaks down the prediction, showing how each feature contributed to moving the final score away from the "base value" (the average prediction across the entire dataset).

-   **Process:**
    1.  For a single applicant, calculate the SHAP values for each of their features.
    2.  Identify the top 3-5 features that had the largest positive impact (increased the risk of default) and the largest negative impact (decreased the risk).
    3.  Present these contributions in a human-readable format.

### Example Local Explanation

This is the type of output that a loan officer would see when reviewing an application. It clearly and concisely explains the reasoning behind the model's score.

> **This applicant has a low risk of default (15.3% probability).**
>
> **Key factors increasing risk:**
> *   `delinquencies_2yrs`: 1.0 (impact: +4.2%)
> *   `revolving_utilization`: 0.85 (impact: +3.1%)
> *   `inquiries_6months`: 3.0 (impact: +2.5%)
>
> **Key factors decreasing risk:**
> *   `fico_score`: 720.0 (impact: -8.5%)
> *   `debt_to_income_ratio`: 0.25 (impact: -5.2%)
> *   `employment_length`: 8.0 (impact: -3.8%)

This level of transparency is essential for building trust in the system and ensuring that decisions are made fairly and can be justified.
