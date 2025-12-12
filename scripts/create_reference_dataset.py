"""
Script to create a reference dataset for Evidently AI monitoring.

This script loads the processed training data, applies the same cleaning and
preprocessing steps used during model training, and saves the resulting
feature-only DataFrame to a specified output path. This cleaned, feature-only
dataset serves as a baseline for data quality and drift checks.
"""

import argparse
import logging
import os
import sys

import pandas as pd
from src.data.cleaning import DataCleaner # Import DataCleaner from src

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)

def main(input_path: str, output_path: str):
    """
    Main function to create the reference dataset.

    Args:
        input_path: Path to the processed training data (e.g., data/processed/train.csv)
        output_path: Path where the cleaned, feature-only reference dataset will be saved
    """
    logger = setup_logging()
    logger.info(f"Starting creation of reference dataset from {input_path}...")

    # Load data
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Input file not found at {input_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        sys.exit(1)

    # Separate features and target (if 'default' column exists)
    if "default" in df.columns:
        # We only want features for the reference dataset
        X = df.drop("default", axis=1)
        logger.info("Removed 'default' target column for reference dataset.")
    else:
        X = df.copy()
        logger.info("No 'default' column found. Processing all columns as features.")

    # Initialize and use DataCleaner
    cleaner = DataCleaner()
    
    # Apply cleaning steps
    # Note: exclude_columns should not contain "default" here as it's already dropped
    X_cleaned = cleaner.clean_loan_data(X, exclude_columns=[])
    X_encoded = cleaner.encode_categorical_features(X_cleaned, fit=True)
    X_scaled = cleaner.scale_numerical_features(X_encoded, fit=True)

    logger.info(f"Cleaned and processed features shape: {X_scaled.shape}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the processed DataFrame
    X_scaled.to_csv(output_path, index=False)
    logger.info(f"Reference dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a reference dataset for Evidently AI monitoring."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/processed/train.csv",
        help="Path to the processed training data (e.g., data/processed/train.csv).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/reference/reference.csv",
        help="Path where the cleaned, feature-only reference dataset will be saved.",
    )

    args = parser.parse_args()
    main(input_path=args.input_path, output_path=args.output_path)