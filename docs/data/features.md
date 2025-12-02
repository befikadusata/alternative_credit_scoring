# Feature Engineering and Preprocessing

This document details the process of selecting, creating, and transforming features to prepare them for the model. It also describes the data splitting strategy.

## 1. Feature Selection

A subset of ~35 of the most relevant features were selected from the original 150+ columns in the dataset. They are grouped into the following categories:

#### Applicant Demographics (8 features)
These features describe the applicant's personal and financial situation.
- `emp_length`
- `home_ownership`
- `annual_inc`
- `zip_code`
- `addr_state`
- `verification_status`
- `dti` (Debt-to-Income ratio)
- `purpose`

#### Credit History (12 features)
These features provide insight into the applicant's past credit behavior.
- `fico_range_low`
- `fico_range_high`
- `open_acc`
- `pub_rec`
- `revol_bal`
- `revol_util`
- `total_acc`
- `delinq_2yrs`
- `inq_last_6mths`
- `mths_since_last_delinq`
- `mths_since_last_record`
- `earliest_cr_line`

#### Loan Characteristics (5 features)
These features describe the loan itself.
- `loan_amnt`
- `funded_amnt`
- `term`
- `int_rate`
- `installment`

## 2. Data Preprocessing and Feature Engineering Pipeline

A multi-step pipeline is used to transform the raw data into a format suitable for the XGBoost model. This ensures that the same transformations are applied consistently during both training and inference.

1.  **Data Cleaning:**
    - **Missing Value Imputation:** Missing numerical values are filled using the median, while categorical missing values are filled with the mode.
    - **Outlier Handling:** Extreme values for fields like income and loan amount are capped at a reasonable maximum. The top and bottom 1% of values for some features are winsorized.
    - **Invalid Data Removal:** Records with invalid data (e.g., zero income) are removed.

2.  **Feature Engineering:** New features are created to capture more complex relationships in the data. This includes:
    - **Ratio Features:** Creating new predictors like `loan_to_income_ratio` and `installment_to_income_ratio`.
    - **Time-based Features:** Calculating the length of the applicant's credit history in years from the `earliest_cr_line` feature.
    - **Interaction Features:** Creating boolean flags for risky combinations, such as having both high credit utilization and a high number of recent inquiries.

3.  **Encoding:**
    - **Categorical Features:** Non-numeric features like `home_ownership` and `purpose` are converted into numerical representations using label encoding.

4.  **Scaling:**
    - **Numeric Features:** All numerical features are standardized (scaled to have a mean of 0 and a standard deviation of 1) using a `StandardScaler`. This is important for some models, and while not strictly necessary for XGBoost, it is good practice.

The parameters for these transformations (e.g., medians for imputation, fitted encoders, and scalers) are learned from the **training data only** and then applied to the validation and test sets to prevent data leakage.

## 3. Data Splitting

A robust data splitting strategy is crucial for reliable model evaluation.

- **Method:** The data is split using a **stratified** approach to ensure that the class distribution (Default vs. No Default) is maintained across all sets.
- **Proportions:**
    - **Training Set:** 70% of the data, used to train the model.
    - **Validation Set:** 15% of the data, used for hyperparameter tuning and early stopping.
    - **Test Set:** 15% of the data, held back for the final, unbiased evaluation of the model's performance.

For a true production scenario, a **time-based split** (e.g., training on older data and testing on more recent data) would be preferred to better simulate how the model would perform on future, unseen data.
