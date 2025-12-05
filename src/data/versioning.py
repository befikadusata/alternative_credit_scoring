"""
Data versioning module for the credit scoring platform.

This module provides functionality to version datasets using MLflow and track
data changes over time for model monitoring purposes.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.artifacts
import pandas as pd


class DataVersioner:
    """
    A class for managing data versioning using MLflow.
    """

    def __init__(self, mlflow_tracking_uri: Optional[str] = None):
        """
        Initialize the DataVersioner.

        Args:
            mlflow_tracking_uri: URI for MLflow tracking server. If None, uses default.
        """
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

        self.logger = logging.getLogger(__name__)
        self.experiment_name = "Data_Versioning"

        # Create or get the experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except Exception as e:
            # If experiment already exists, get its ID
            self.logger.debug(
                f"Experiment creation failed (likely already exists): {str(e)}"
            )
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id

    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """
        Calculate a hash of the dataframe to uniquely identify its content.

        Args:
            df: Input DataFrame

        Returns:
            SHA256 hash of the dataframe content
        """
        # Convert dataframe to string representation
        df_string = df.to_json(orient="records", sort_keys=True)
        # Calculate SHA256 hash
        data_hash = hashlib.sha256(df_string.encode()).hexdigest()
        return data_hash

    def _get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic statistics of the dataframe for version tracking.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing basic statistics
        """
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        }

        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe().to_dict()
            stats["numeric_summary"] = numeric_stats

        return stats

    def save_dataset_version(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        version_description: str = "",
        artifact_path: str = "datasets",
    ) -> str:
        """
        Save a version of the dataset to MLflow tracking server.

        Args:
            df: DataFrame to save
            dataset_name: Name of the dataset
            version_description: Description of this version
            artifact_path: Path in MLflow artifacts where to save the dataset

        Returns:
            Run ID of the MLflow run that contains the dataset version
        """
        # Calculate data hash and statistics
        data_hash = self._calculate_data_hash(df)
        stats = self._get_data_statistics(df)

        # Start MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            # Log dataset metadata as parameters
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("data_hash", data_hash)
            mlflow.log_param("version_description", version_description)
            mlflow.log_param("created_at", datetime.now().isoformat())

            # Log statistics as parameters (for simple metrics)
            mlflow.log_param("row_count", stats["row_count"])
            mlflow.log_param("column_count", stats["column_count"])

            # Log detailed statistics as artifacts
            stats_path = f"{dataset_name}_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2, default=str)

            mlflow.log_artifact(stats_path)

            # Save the actual dataset
            dataset_filename = f"{dataset_name}_{data_hash[:8]}.csv"
            df.to_csv(dataset_filename, index=False)

            mlflow.log_artifact(dataset_filename, artifact_path)

            # Clean up temporary files
            os.remove(dataset_filename)
            os.remove(stats_path)

        self.logger.info(
            f"Dataset version saved: {dataset_name} with hash {data_hash[:8]} in run {run.info.run_id}"
        )
        return run.info.run_id

    def load_dataset_version(
        self, run_id: str, dataset_name: str, artifact_path: str = "datasets"
    ) -> pd.DataFrame:
        """
        Load a specific version of the dataset from MLflow.

        Args:
            run_id: MLflow run ID that contains the dataset
            dataset_name: Name of the dataset to load
            artifact_path: Path in MLflow artifacts where the dataset is stored

        Returns:
            Loaded DataFrame
        """
        # Download the artifact
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"{artifact_path}/"
        )

        # Find the dataset file
        dataset_files = list(Path(local_path).glob(f"{dataset_name}_*.csv"))

        if not dataset_files:
            raise FileNotFoundError(
                f"No dataset file found for {dataset_name} in run {run_id}"
            )

        # Load and return the dataset
        df = pd.read_csv(dataset_files[0])
        self.logger.info(f"Dataset loaded from run {run_id}: {df.shape}")
        return df

    def register_reference_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        description: str = "Reference dataset for model monitoring",
    ):
        """
        Register a reference dataset for model monitoring.

        Args:
            df: Reference DataFrame
            dataset_name: Name of the reference dataset
            description: Description of the reference dataset
        """
        # Save the reference dataset with a specific naming convention
        ref_run_id = self.save_dataset_version(
            df=df,
            dataset_name=f"reference_{dataset_name}",
            version_description=description,
        )

        self.logger.info(
            f"Reference dataset registered: {dataset_name} in run {ref_run_id}"
        )
        return ref_run_id


def create_reference_dataset(
    raw_data_path: str, output_path: str = "data/reference/reference_dataset.csv"
) -> pd.DataFrame:
    """
    Create a reference dataset from raw data by applying cleaning and feature engineering.

    Args:
        raw_data_path: Path to raw input data
        output_path: Path to save the reference dataset

    Returns:
        Reference dataset as a DataFrame
    """
    logger = logging.getLogger(__name__)

    # Import our custom modules
    from src.data.cleaning import DataCleaner
    from src.data.features import FeatureEngineer

    # Load raw data
    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    # Initialize processing classes
    cleaner = DataCleaner()
    engineer = FeatureEngineer()

    # Clean the data
    logger.info("Cleaning reference data...")
    df_cleaned = cleaner.clean_loan_data(df)

    # Engineer features
    logger.info("Applying feature engineering to reference data...")
    df_engineered = apply_feature_engineering(df_cleaned)

    # Create target variable if needed
    if (
        "default" not in df_engineered.columns
        and "loan_status" in df_engineered.columns
    ):
        df_engineered = engineer.create_target_variable(df_engineered, "default")

    # Ensure data types are consistent
    df_engineered = df_engineered.infer_objects()

    # Save the reference dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_engineered.to_csv(output_path, index=False)

    logger.info(f"Reference dataset created and saved to {output_path}")
    logger.info(f"Reference dataset shape: {df_engineered.shape}")

    return df_engineered


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline to the input data.
    This is a helper function to avoid circular imports.

    Args:
        df: Input DataFrame with raw credit data

    Returns:
        DataFrame with cleaned and engineered features
    """
    from src.data.features import FeatureEngineer

    engineer = FeatureEngineer()
    df = engineer.create_features(df)
    return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Example usage would be:
    # versioner = DataVersioner()
    # df = pd.read_csv('path/to/data.csv')
    # run_id = versioner.save_dataset_version(df, 'credit_data', 'Initial dataset version')
    # ref_run_id = versioner.register_reference_dataset(df, 'credit_data', 'Baseline reference dataset')
