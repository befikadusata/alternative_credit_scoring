"""
Script for automated data validation using Evidently AI.

This script compares a 'current' dataset against a 'reference' dataset
(e.g., training data) to detect data quality issues and data drift.
It uses Evidently AI's TestSuite to perform various checks and
exits with a non-zero code if any critical tests fail.
"""

import argparse
import logging
import os
import sys
import pandas as pd

from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
from evidently.test_suite import TestSuite
from evidently.test_case import TestColumnValueMin, TestColumnValueMax

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

def main(current_data_path: str, reference_data_path: str, output_report_path: str = None, output_json_path: str = None):
    """
    Main function to perform data validation using Evidently AI TestSuite.

    Args:
        current_data_path: Path to the current dataset (e.g., data/processed/train.csv for pipeline data).
        reference_data_path: Path to the reference dataset (e.g., data/reference/reference.csv).
        output_report_path: Optional path to save the HTML Evidently report.
        output_json_path: Optional path to save the JSON Evidently test results.
    """
    logger = setup_logging()
    logger.info("Starting data validation with Evidently AI...")

    # Load data
    try:
        current_data = pd.read_csv(current_data_path)
        reference_data = pd.read_csv(reference_data_path)
        logger.info(f"Loaded current data shape: {current_data.shape}")
        logger.info(f"Loaded reference data shape: {reference_data.shape}")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Ensure columns match between current and reference for consistent testing
    # This might need more sophisticated handling if features can be added/removed over time
    common_columns = list(set(current_data.columns) & set(reference_data.columns))
    current_data = current_data[common_columns]
    reference_data = reference_data[common_columns]
    logger.info(f"Using {len(common_columns)} common columns for validation.")

    # Define the TestSuite
    data_validation_test_suite = TestSuite(tests=[
        DataQualityTestPreset(),
        DataDriftTestPreset(),
        # Example: Add specific tests for critical columns
        # TestColumnValueMin(column_name="annual_inc", gte=0),
        # TestColumnValueMax(column_name="int_rate", lte=30)
    ])

    # Run the test suite
    data_validation_test_suite.run(current_data=current_data, reference_data=reference_data, column_mapping=None)
    logger.info("Evidently AI TestSuite run completed.")

    # Save reports
    if output_report_path:
        os.makedirs(os.path.dirname(output_report_path) or '.', exist_ok=True)
        data_validation_test_suite.save_html(output_report_path)
        logger.info(f"Evidently report saved to {output_report_path}")
    
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
        data_validation_test_suite.save_json(output_json_path)
        logger.info(f"Evidently JSON results saved to {output_json_path}")

    # Check results and exit accordingly
    if not data_validation_test_suite.as_dict().get('summary', {}).get('all_passed', False):
        logger.error("Evidently AI TestSuite detected failures. Data validation failed!")
        sys.exit(1)
    else:
        logger.info("Evidently AI TestSuite passed successfully. Data validation passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform data validation using Evidently AI TestSuite."
    )
    parser.add_argument(
        "--current-data-path",
        type=str,
        required=True,
        help="Path to the current dataset to be validated.",
    )
    parser.add_argument(
        "--reference-data-path",
        type=str,
        default="data/reference/reference.csv",
        help="Path to the reference dataset for comparison (default: data/reference/reference.csv).",
    )
    parser.add_argument(
        "--output-report-path",
        type=str,
        default=None,
        help="Optional: Path to save the Evidently HTML report.",
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        default=None,
        help="Optional: Path to save the Evidently JSON test results.",
    )

    args = parser.parse_args()
    main(
        current_data_path=args.current_data_path,
        reference_data_path=args.reference_data_path,
        output_report_path=args.output_report_path,
        output_json_path=args.output_json_path,
    )
