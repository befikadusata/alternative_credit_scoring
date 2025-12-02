# Dataset and Target Variable

This document provides details on the dataset used for this project and how the target variable for the classification task is defined.

## 1. Dataset Selection

**Dataset:** LendingClub Loan Data (2007-2018)

This dataset was chosen as it is a well-documented and publicly available dataset that serves as a realistic proxy for a fintech lending scenario.

- **Justification:**
    - **Publicly Available:** No licensing or access issues.
    - **Realistic Data:** Contains over 150,000 loan records with a rich feature set, including demographic, financial, and credit history information.
    - **Well-Suited for Problem:** The data includes a clear binary outcome (loan default) and features that can be used to model an "alternative" credit scoring story.
    - **Widely Used:** It is a standard benchmark dataset in the credit risk modeling community.

- **Characteristics:**
    - **Size:** ~150,000 loans
    - **Features:** 150+ raw columns (reduced to ~35 relevant features).
    - **Target Variable:** `loan_status` is used to derive a binary target.
    - **Class Distribution:** The dataset is imbalanced, with a default rate of approximately 18-20%.

## 2. Target Variable

The goal of the model is to predict the likelihood of a loan defaulting. A binary target variable is engineered from the `loan_status` column.

- **Definition:**
  A loan is considered a **Default (1)** if its status is one of the following:
    - 'Charged Off'
    - 'Default'
    - 'Does not meet the credit policy. Status:Charged Off'
    - 'Late (31-120 days)'
    - 'Late (16-30 days)'

  All other statuses, primarily 'Fully Paid', are considered **No Default (0)**.

- **Class Imbalance Handling:**
  Given the ~80/20 class split, the imbalance is addressed through several strategies:
    - **Stratified Splitting:** Ensuring that the train, validation, and test sets all have the same class distribution as the original dataset.
    - **Evaluation Metrics:** Using metrics that are robust to class imbalance, such as **AUC-ROC** and **AUC-PR (Precision-Recall)**, instead of relying solely on accuracy.
    - **Model Parameterization:** Utilizing the `scale_pos_weight` hyperparameter in XGBoost, which assigns a higher penalty to misclassifying the minority class (defaults).
