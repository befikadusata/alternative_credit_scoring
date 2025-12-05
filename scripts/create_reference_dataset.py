#!/usr/bin/env python3
"""
Script to create and register a reference dataset with versioning for monitoring.

This script applies the full data processing pipeline to create a reference
dataset that will be used for model monitoring and data drift detection.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

# Import our custom modules
from src.data.versioning import DataVersioner, create_reference_dataset


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("reference_dataset_creation.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def main(
    raw_data_path: str,
    reference_data_path: str = "data/reference/reference_dataset.csv",
    mlflow_tracking_uri: str = "http://localhost:5000",
):
    """
    Main function to create and register a reference dataset with versioning.

    Args:
        raw_data_path: Path to raw input data
        reference_data_path: Path to save the reference dataset
        mlflow_tracking_uri: URI for MLflow tracking server
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting reference dataset creation with versioning...")

    # Validate raw data path
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file does not exist: {raw_data_path}")

    # Create reference data directory if it doesn't exist
    Path(reference_data_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize the data versioner
    versioner = DataVersioner(mlflow_tracking_uri=mlflow_tracking_uri)

    # Create the reference dataset
    logger.info("Creating reference dataset...")
    reference_df = create_reference_dataset(
        raw_data_path=raw_data_path, output_path=reference_data_path
    )

    # Register the reference dataset with MLflow
    logger.info("Registering reference dataset with versioning...")
    dataset_name = Path(raw_data_path).stem
    run_id = versioner.register_reference_dataset(
        df=reference_df,
        dataset_name=dataset_name,
        description="Reference dataset for credit scoring model monitoring",
    )

    logger.info(f"Reference dataset registered successfully with run ID: {run_id}")
    logger.info(f"Reference dataset saved to: {reference_data_path}")

    # Display basic info about the reference dataset
    logger.info(f"Reference dataset shape: {reference_df.shape}")
    logger.info(f"Columns in reference dataset: {list(reference_df.columns)}")

    # If target column exists, show class distribution
    if "default" in reference_df.columns:
        target_dist = reference_df["default"].value_counts(normalize=True)
        logger.info(f"Target distribution in reference dataset: \n{target_dist}")


if __name__ == "__main__":
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Create and register a reference dataset with versioning for monitoring."
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        required=True,
        help="Path to the raw input data file",
    )
    parser.add_argument(
        "--reference_data_path",
        type=str,
        default="data/reference/reference_dataset.csv",
        help="Path to save the reference dataset (default: data/reference/reference_dataset.csv)",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI (default: http://localhost:5000)",
    )

    args = parser.parse_args()

    try:
        main(
            raw_data_path=args.raw_data_path,
            reference_data_path=args.reference_data_path,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )
    except Exception as e:
        logger.error(f"Reference dataset creation failed with error: {str(e)}")
        sys.exit(1)
