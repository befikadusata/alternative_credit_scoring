"""
Model Loading Module for Credit Scoring API

This module handles loading models from the MLflow Model Registry.
"""

import logging
import os
from typing import List, Optional, Union
import json

import joblib
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyModel:
    """A dummy model for testing purposes.
    This is used as a fallback if a real model cannot be loaded,
    or for component tests that don't need real model logic.
    """

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        # Return dummy predictions (e.g., all 0s)
        return np.zeros(len(data), dtype=int)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        # Return dummy probabilities (e.g., 90% chance of 0, 10% chance of 1)
        return np.array([[0.9, 0.1]] * len(data))


def load_model_from_registry(
    model_name: str,
    model_version: Union[str, int] = "latest",
    tracking_uri: Optional[str] = None,
) -> tuple[object, object, List[str]]:
    """
    Load a model and its corresponding DataCleaner and feature names
    from the MLflow Model Registry.

    Args:
        model_name: Name of the model in the registry
        model_version: Version of the model (can be version number, "latest", or alias like "champion")
        tracking_uri: MLflow tracking URI (if None, uses environment variable or default)

    Returns:
        A tuple containing the loaded PyFuncModel instance, the DataCleaner instance,
        and a list of feature names.
    """
    logger.info(
        f"Attempting to load model '{model_name}' version/alias '{model_version}' from MLflow."
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    else:
        logger.error("MLflow Tracking URI not set. Cannot load model.")
        raise ValueError("MLflow Tracking URI not configured.")

    model = None
    data_cleaner = None
    feature_names = []

    try:
        # Construct the MLflow model URI
        model_uri = f"models:/{model_name}/{model_version}"
        logger.info(f"Loading MLflow model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        # Get MLflow client to fetch model run details for artifacts
        client = MlflowClient()
        model_version_obj = None

        if isinstance(model_version, int):
            model_version_obj = client.get_model_version(model_name, model_version)
        else:  # "latest", "champion", "challenger", or other aliases/stages
            # Use search_model_versions to find by name and filter by stage/alias
            # Note: MLflow's search_model_versions doesn't directly support alias filtering prior to 2.11+
            # This logic provides a workaround by fetching all and filtering.
            all_versions = client.search_model_versions(f"name='{model_name}'")
            if model_version in ["latest", "champion", "challenger", "staging", "production", "archived"]:
                # Try to find by stage
                filtered_versions = [
                    mv for mv in all_versions if mv.current_stage.lower() == model_version.lower()
                ]
                if filtered_versions:
                    # Get the most recent one if multiple are in the same stage (unlikely for "latest" stage)
                    model_version_obj = sorted(filtered_versions, key=lambda mv: mv.creation_timestamp, reverse=True)[0]
                else:
                    logger.warning(
                        f"Could not find model version for alias/stage '{model_version}'. "
                        "Attempting to load the latest 'Production' or 'Staging' version."
                    )
                    # Fallback to get_latest_versions for "Production" or "Staging"
                    latest_alias_versions = client.get_latest_versions(
                        model_name, stages=["Production", "Staging"]
                    )
                    if latest_alias_versions:
                        model_version_obj = latest_alias_versions[0] # Take the first one (usually latest)
            else: # If a specific version string is passed, try to get it directly
                try:
                    model_version_obj = client.get_model_version(model_name, model_version)
                except Exception:
                    logger.warning(f"Model version '{model_version}' not found directly. Falling back to search.")
                    # If direct get fails, it might be an unlisted alias or tag, which is harder to resolve without specific API support.
                    # For now, we'll let the outer try-except handle the final failure if model_version_obj remains None.
                    pass


        if not model_version_obj:
            raise ValueError(f"No model version found for '{model_name}' with version/alias '{model_version}'")
            
        run_id = model_version_obj.run_id
        logger.info(f"Model '{model_name}' version '{model_version}' is from MLflow Run ID: {run_id}")

        # Load the DataCleaner (preprocessor) artifact
        try:
            cleaner_artifact_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/preprocessor"
            )
            data_cleaner = joblib.load(cleaner_artifact_path)
            logger.info("Successfully loaded DataCleaner preprocessor from MLflow artifacts.")
        except Exception as e:
            logger.error(f"Failed to load DataCleaner from MLflow artifacts for run {run_id}: {e}")
            data_cleaner = None

        # Load feature names artifact
        try:
            feature_names_artifact_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/feature_names"
            )
            with open(feature_names_artifact_path, 'r') as f:
                feature_names = json.load(f)
            logger.info("Successfully loaded feature names from MLflow artifacts.")
        except Exception as e:
            logger.error(f"Failed to load feature names from MLflow artifacts for run {run_id}: {e}")
            feature_names = []


        logger.info(
            f"Successfully loaded model '{model_name}' version '{model_version}' and its associated artifacts."
        )

        return model, data_cleaner, feature_names

    except Exception as e:
        logger.error(
            f"Failed to load model or preprocessor for '{model_name}' version '{model_version}': {str(e)}"
        )
        raise


if __name__ == "__main__":
    # Example usage (will need an MLflow server running and a model registered)
    # Ensure MLFLOW_TRACKING_URI is set in your environment or passed explicitly
    # model, cleaner, features = load_model_from_registry("credit_scoring_model", "latest")
    logger.info("Model loader module loaded successfully")