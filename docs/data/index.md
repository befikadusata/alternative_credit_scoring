# Data Strategy Overview

A robust data strategy is fundamental to building a reliable machine learning model. This section outlines the approach to dataset selection, feature engineering, data quality, and versioning.

### Guiding Principles

1.  **Relevance:** The chosen dataset must be a realistic proxy for the problem domain of alternative credit scoring.
2.  **Quality:** Data quality is enforced through automated validation checks at every stage of the pipeline.
3.  **Consistency:** Feature engineering logic is consistently applied across both training and inference to prevent skew.
4.  **Reproducibility:** Data and features are versioned to ensure that experiments and models are fully reproducible.

### Key Documents

*   [**Dataset**](./dataset.md): Details on the LendingClub dataset, target variable definition, and class imbalance handling.
*   [**Features**](./features.md): Information on feature selection, the different categories of features used, and the feature engineering pipeline.
*   [**Data Quality & Versioning**](./quality-and-versioning.md): The strategy for ensuring data quality and versioning datasets for reproducibility.
