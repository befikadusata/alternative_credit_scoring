# Data Quality & Versioning

Ensuring data quality and maintaining clear versions of datasets are critical MLOps practices for creating reproducible and reliable models.

## 1. Data Quality Checks

Automated data quality validation is integrated into the data pipeline to catch issues early. The framework for these checks includes:

-   **Schema Validation:**
    -   Verifies that all required columns are present.
    -   Ensures that each column has the expected data type (e.g., `annual_inc` is numeric).

-   **Completeness Checks:**
    -   Asserts that critical columns (`loan_amnt`, `annual_inc`, etc.) have no missing values.
    -   Flags any column where the percentage of missing values exceeds a set threshold (e.g., 5%).

-   **Validity Checks:**
    -   Confirms that values are within a plausible range (e.g., `age` between 18 and 100, `FICO score` between 300 and 850).

-   **Consistency Checks:**
    -   Performs logical checks between columns (e.g., ensures `funded_amnt` is not greater than `loan_amnt`).

-   **Distribution Checks:**
    -   Monitors for severe class imbalance to ensure there is enough signal to train a model.
    -   Checks that numerical features have a minimum level of variance to be useful.

Any dataset that fails these checks is flagged, and the pipeline is stopped to prevent "garbage in, garbage out."

## 2. Data Versioning

To ensure reproducibility, every dataset used for training is versioned. This allows data scientists to link a specific model version back to the exact dataset it was trained on.

-   **Approach:**
    A combination of a structured directory layout and metadata files is used to manage data versions. For a more robust system, a tool like DVC (Data Version Control) would be integrated.

-   **Directory Structure:**
    The `data/` directory is organized to separate raw, processed, and reference data. Version numbers are included in the filenames.
    ```
    data/
    ├── raw/
    │   ├── lending_club_2007_2018.csv
    │   └── data_version.json
    ├── processed/
    │   ├── train_v1.parquet
    │   ├── validation_v1.parquet
    │   └── test_v1.parquet
    ├── reference/
    │   └── baseline_statistics_v1.json
    └── metadata/
        └── data_quality_report_v1.html
    ```

-   **Version Metadata:**
    A JSON file accompanies each version, capturing critical information about the dataset's source, size, and characteristics. This file acts as a "data card" for the version.

    **Example `data_version.json`:**
    ```json
    {
        "version": "1.0",
        "created_at": "2024-12-01T10:00:00Z",
        "source": "kaggle:lending-club",
        "records_total": 145832,
        "features_count": 35,
        "target_distribution": {
            "no_default": 0.82,
            "default": 0.18
        },
        "splits": {
            "train": {"records": 102082, "default_rate": 0.18},
            "validation": {"records": 21875, "default_rate": 0.18},
            "test": {"records": 21875, "default_rate": 0.18}
        },
        "data_quality_score": 0.95,
        "sha256_hash": "abc123..."
    }
    ```
