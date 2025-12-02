# Model Development and Training

This document describes the XGBoost model configuration, the training pipeline, and the hyperparameter tuning strategy.

## 1. Model Configuration

The XGBoost model is configured to optimize for classification performance while controlling for overfitting.

-   **Objective:** `binary:logistic` for binary classification, returning probabilities.
-   **Evaluation Metric:** `auc` (Area Under the ROC Curve) is used as the primary metric for optimization because it is well-suited for imbalanced datasets.
-   **Regularization:** Both L1 (`alpha`) and L2 (`lambda`) regularization are used to prevent overfitting.
-   **Tree Parameters:** Tree depth and child weight are constrained to create simpler trees that generalize better.
-   **Imbalance Handling:** `scale_pos_weight` is used to give more importance to the minority class (defaults), helping the model learn its characteristics.

#### Example XGBoost Parameters:
```python
XGBOOST_PARAMS = {
    # Core parameters
    "objective": "binary:logistic",
    "eval_metric": "auc",
    # Tree parameters
    "max_depth": 6,
    "min_child_weight": 3,
    "gamma": 0.1,
    # Regularization
    "alpha": 0.1,
    "lambda": 1.0,
    # Learning parameters
    "learning_rate": 0.1,
    "n_estimators": 200,
    # Sampling
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # Imbalanced data handling
    "scale_pos_weight": 4.5, # ~ ratio of negative/positive samples
    # Performance
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1
}
```

## 2. Training Pipeline

The model training process is executed as a scripted pipeline that logs all relevant information to MLflow.

1.  **Data Loading:** The preprocessed training and validation data splits are loaded.
2.  **MLflow Experiment Setup:** An MLflow experiment is initiated. A new run is started within this experiment to track the training process. All hyperparameters are logged to MLflow.
3.  **Model Training:** The XGBoost classifier is trained on the training dataset. The validation dataset is used for **early stopping**, a technique that stops the training process when the model's performance on the validation set no longer improves, which helps prevent overfitting.
4.  **Model Evaluation:** After training, the model's performance is evaluated on the validation set using a comprehensive set of metrics (AUC-ROC, Precision, Recall, etc.). These metrics are logged to MLflow.
5.  **Artifact Logging:** Key assets, known as artifacts, are saved and logged to MLflow. This includes the trained model itself, feature importance plots, and a confusion matrix.
6.  **Model Registration:** The logged model is then registered in the MLflow Model Registry. This creates a new, versioned model entry that can be referenced later for deployment.

## 3. Hyperparameter Tuning

To find the optimal set of hyperparameters for the XGBoost model, a systematic tuning process is employed.

-   **Approach:** **Randomized Search with Cross-Validation** (`RandomizedSearchCV`). This method is more efficient than a grid search, as it samples a fixed number of parameter combinations from a specified distribution.
-   **Process:**
    1.  A search space (a dictionary of hyperparameters and their possible values) is defined.
    2.  `RandomizedSearchCV` runs a series of training jobs with different parameter combinations, using cross-validation to ensure the results are robust.
    3.  Each trial (a single training job with one parameter combination) is logged as a separate run in MLflow.
    4.  The best-performing set of hyperparameters is identified based on the mean cross-validated AUC score.

#### Example Hyperparameter Search Space:
```python
PARAM_GRID = {
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}
```
This tuning process typically improves the model's AUC from a baseline of ~0.75 to a tuned score of ~0.78-0.80.
