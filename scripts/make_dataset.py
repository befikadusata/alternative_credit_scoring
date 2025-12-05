#!/usr/bin/env python3
"""
Data Processing Pipeline Script

This script runs the complete data processing pipeline from raw data to
processed features ready for model training. It includes:
1. Data loading
2. Data cleaning and imputation
3. Feature engineering
4. Data splitting
5. Saving processed datasets
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.model_selection import train_test_split

# Import our custom modules
from src.data.cleaning import DataCleaner, load_and_clean_data
from src.data.features import FeatureEngineer, apply_feature_engineering


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("data_processing.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def validate_paths(raw_data_path, processed_data_path):
    """Validate input and output paths."""
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file does not exist: {raw_data_path}")

    # Create processed data directory if it doesn't exist
    Path(processed_data_path).mkdir(parents=True, exist_ok=True)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from file with error handling."""
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def save_data(df: pd.DataFrame, file_path: str):
    """Save data to file with error handling."""
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Saving data to {file_path} with shape {df.shape}")
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        raise


def main(
    raw_data_path: str,
    processed_data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Main function to run the data processing pipeline.

    Args:
        raw_data_path: Path to raw input data
        processed_data_path: Path to save processed output data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data processing pipeline...")

    # Validate paths
    validate_paths(raw_data_path, processed_data_path)

    # Load raw data
    df = load_data(raw_data_path)

    # Initialize processing classes
    cleaner = DataCleaner()
    engineer = FeatureEngineer()

    # Step 1: Clean the data
    logger.info("Starting data cleaning...")
    df_cleaned = cleaner.clean_loan_data(df)
    logger.info(f"Data cleaning completed. Shape after cleaning: {df_cleaned.shape}")

    # Step 2: Engineer features
    logger.info("Starting feature engineering...")
    df_engineered = apply_feature_engineering(df_cleaned)
    logger.info(
        f"Feature engineering completed. Shape after feature engineering: {df_engineered.shape}"
    )

    # Step 3: Create target variable if not already present
    if (
        "default" not in df_engineered.columns
        and "loan_status" in df_engineered.columns
    ):
        df_engineered = engineer.create_target_variable(df_engineered, "default")

    # Step 4: Separate features and target
    if "default" in df_engineered.columns:
        X = df_engineered.drop("default", axis=1)
        y = df_engineered["default"]
        logger.info(
            f"Separated features (X) and target (y). X shape: {X.shape}, y shape: {y.shape}"
        )
    else:
        # If no target column found, use all columns as features
        X = df_engineered
        y = None
        logger.warning(
            "No target column 'default' found in the data. All columns will be treated as features."
        )

    # Step 5: Split the data
    if y is not None:
        logger.info(
            f"Splitting data with test_size={test_size}, random_state={random_state}"
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Train set: X {X_train.shape}, y {y_train.shape}")
        logger.info(f"Test set: X {X_test.shape}, y {y_test.shape}")
    else:
        # If no target, just split the features
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        logger.info(
            f"Feature-only split - Train: {X_train.shape}, Test: {X_test.shape}"
        )
        y_train, y_test = None, None

    # Step 6: Save processed datasets
    train_path = os.path.join(processed_data_path, "train.csv")
    test_path = os.path.join(processed_data_path, "test.csv")

    # Combine features and target for saving if target exists
    if y_train is not None:
        train_df = pd.concat([X_train, y_train], axis=1)
    else:
        train_df = X_train

    if y_test is not None:
        test_df = pd.concat([X_test, y_test], axis=1)
    else:
        test_df = X_test

    save_data(train_df, train_path)
    save_data(test_df, test_path)

    # Also save the full processed dataset
    if y is not None:
        full_processed_df = pd.concat([X, y], axis=1)
    else:
        full_processed_df = X

    full_path = os.path.join(processed_data_path, "full_processed.csv")
    save_data(full_processed_df, full_path)

    # Save feature names for reference
    features_path = os.path.join(processed_data_path, "feature_names.txt")
    with open(features_path, "w") as f:
        for feature in X.columns:
            f.write(f"{feature}\n")

    logger.info("Data processing pipeline completed successfully!")
    logger.info(f"Processed datasets saved to {processed_data_path}")
    logger.info(f"- Training data: {train_path}")
    logger.info(f"- Test data: {test_path}")
    logger.info(f"- Full processed data: {full_path}")
    logger.info(f"- Feature names: {features_path}")


if __name__ == "__main__":
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/raw/credit_data.csv",
        help="Path to the raw input data file (default: data/raw/credit_data.csv)",
    )
    parser.add_argument(
        "--processed_data_path",
        type=str,
        default="data/processed/",
        help="Path to save processed output data (default: data/processed/)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    try:
        main(
            raw_data_path=args.raw_data_path,
            processed_data_path=args.processed_data_path,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        sys.exit(1)
