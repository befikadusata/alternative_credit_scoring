"""
Model Loading Module for Credit Scoring API

This module handles loading models from the MLflow Model Registry.
"""

import logging
import os
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd  # Import pandas for DummyModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyModel:
    """A dummy model for testing purposes."""

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
) -> tuple[object, object, List[str]]:  # Changed return type hint
    """
    Load a model and its corresponding DataCleaner from the MLflow Model Registry.

    Args:
        model_name: Name of the model in the registry
        model_version: Version of the model (can be version number, "latest", or alias like "champion")
        tracking_uri: MLflow tracking URI (if None, uses environment variable or default)

    Returns:
        A tuple containing the loaded PyFuncModel instance, the DataCleaner instance, and a list of feature names.
    """
    logger.info(
        f"Attempting to load (dummy) model '{model_name}' version/alias '{model_version}' locally"
    )

    try:
        # Load the DataCleaner from the locally saved file
        cleaner_path = "data_cleaner.joblib"
        if os.path.exists(cleaner_path):
            data_cleaner = joblib.load(cleaner_path)
            logger.info(f"Successfully loaded DataCleaner from {cleaner_path}")
        else:
            logger.warning(
                f"DataCleaner not found at {cleaner_path}. Returning None for cleaner."
            )
            data_cleaner = None

        # Load feature names
        feature_names_path = "tests/test_model_features.joblib"
        if os.path.exists(feature_names_path):
            feature_names = joblib.load(feature_names_path)
            logger.info(f"Successfully loaded feature names from {feature_names_path}")
        else:
            # Generate expected feature names based on the categorical variables that would be one-hot encoded
            logger.warning(
                f"Feature names not found at {feature_names_path}. Generating default feature names."
            )

            # Create a representative sample DataFrame with all possible categorical values
            import pandas as pd

            raw_input_data = {
                "loan_amnt": [12000.0],
                "int_rate": [10.0],
                "installment": [300.0],
                "emp_length": [5.0],
                "annual_inc": [75000.0],
                "dti": [15.0],
                "delinq_2yrs": [0],
                "inq_last_6mths": [1],
                "open_acc": [8],
                "pub_rec": [0],
                "revol_bal": [12000],
                "revol_util": [50.0],
                "total_acc": [20],
                "total_pymnt": [1000.0],
                "total_pymnt_inv": [1000.0],
                "total_rec_prncp": [800.0],
                "total_rec_int": [200.0],
                "total_rec_late_fee": [0.0],
                "recoveries": [0.0],
                "collection_recovery_fee": [0.0],
                "last_pymnt_amnt": [300.0],
                "term": ["36 months"],  # The other possible value is '60 months'
                "grade": ["A"],  # Possible values: A, B, C, D, E, F, G
                "sub_grade": ["A1"],  # Possible values: A1-A5, B1-B5, ..., G1-G5
                "home_ownership": [
                    "MORTGAGE"
                ],  # Possible values: MORTGAGE, RENT, OWN, OTHER, NONE, ANY
                "verification_status": [
                    "Verified"
                ],  # Possible values: Verified, Not Verified, Source Verified
                "purpose": [
                    "debt_consolidation"
                ],  # Possible values: debt_consolidation, credit_card, home_improvement, etc.
                "initial_list_status": ["f"],  # Possible values: 'f', 'w'
            }

            # Add all possible categorical values to ensure all dummies are generated
            all_terms = ["36 months", "60 months"]
            all_grades = ["A", "B", "C", "D", "E", "F", "G"]
            all_sub_grades = [
                f"{g}{i}"
                for g in ["A", "B", "C", "D", "E", "F", "G"]
                for i in range(1, 6)
            ]
            all_home_ownership = ["MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY"]
            all_verification_status = ["Verified", "Not Verified", "Source Verified"]
            all_purposes = [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "other",
                "major_purchase",
                "medical",
                "car",
                "small_business",
                "wedding",
                "house",
                "moving",
                "vacation",
            ]
            all_initial_list_status = ["f", "w"]

            # Create a DataFrame that will include all possible categories when one-hot encoded
            X_raw = pd.DataFrame(
                {
                    "loan_amnt": [12000.0] * len(all_grades),
                    "int_rate": [10.0] * len(all_grades),
                    "installment": [300.0] * len(all_grades),
                    "emp_length": [5.0] * len(all_grades),
                    "annual_inc": [75000.0] * len(all_grades),
                    "dti": [15.0] * len(all_grades),
                    "delinq_2yrs": [0] * len(all_grades),
                    "inq_last_6mths": [1] * len(all_grades),
                    "open_acc": [8] * len(all_grades),
                    "pub_rec": [0] * len(all_grades),
                    "revol_bal": [12000] * len(all_grades),
                    "revol_util": [50.0] * len(all_grades),
                    "total_acc": [20] * len(all_grades),
                    "total_pymnt": [1000.0] * len(all_grades),
                    "total_pymnt_inv": [1000.0] * len(all_grades),
                    "total_rec_prncp": [800.0] * len(all_grades),
                    "total_rec_int": [200.0] * len(all_grades),
                    "total_rec_late_fee": [0.0] * len(all_grades),
                    "recoveries": [0.0] * len(all_grades),
                    "collection_recovery_fee": [0.0] * len(all_grades),
                    "last_pymnt_amnt": [300.0] * len(all_grades),
                    "term": (all_terms * (len(all_grades) // len(all_terms) + 1))[
                        : len(all_grades)
                    ],
                    "grade": all_grades,
                    "sub_grade": all_sub_grades[
                        : len(all_grades)
                    ],  # Use a subset that matches grade length
                    "home_ownership": (
                        all_home_ownership
                        * (len(all_grades) // len(all_home_ownership) + 1)
                    )[: len(all_grades)],
                    "verification_status": (
                        all_verification_status
                        * (len(all_grades) // len(all_verification_status) + 1)
                    )[: len(all_grades)],
                    "purpose": (
                        all_purposes * (len(all_grades) // len(all_purposes) + 1)
                    )[: len(all_grades)],
                    "initial_list_status": (
                        all_initial_list_status
                        * (len(all_grades) // len(all_initial_list_status) + 1)
                    )[: len(all_grades)],
                }
            )

            # Apply the same one-hot encoding logic as in src/api/main.py
            X_encoded = X_raw.copy()
            for col in X_encoded.select_dtypes(include=["object"]).columns:
                if col in X_encoded.columns:
                    dummies = pd.get_dummies(X_encoded[col], prefix=col, dtype=float)
                    X_encoded = pd.concat(
                        [X_encoded.drop(columns=[col]), dummies], axis=1
                    )

            # Convert all columns to float
            X_final = X_encoded.astype(float)

            feature_names = X_final.columns.tolist()
            logger.info(
                f"Generated default feature names with {len(feature_names)} features."
            )

        # Return a dummy model and the loaded cleaner
        model = DummyModel()
        logger.info(
            f"Returning DummyModel for '{model_name}' version '{model_version}'"
        )

    """
    Load a model and its corresponding DataCleaner from the MLflow Model Registry.

    Args:
        model_name: Name of the model in the registry
        model_version: Version of the model (can be version number, "latest", or alias like "champion")
        tracking_uri: MLflow tracking URI (if None, uses environment variable or default)

    Returns:
        A tuple containing the loaded PyFuncModel instance, the DataCleaner instance, and a list of feature names.
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

    try:
        # Construct the MLflow model URI
        model_uri = f"models:/{model_name}/{model_version}"
        logger.info(f"Loading MLflow model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        # Get MLflow client to fetch model run details for artifacts
        client = MlflowClient()
        if isinstance(model_version, int):
            model_version_obj = client.get_model_version(model_name, model_version)
        else: # "latest", "champion", "challenger"
            latest_versions = client.search_model_versions(f"name='{model_name}'")
            # Filter by alias if specified (e.g., "champion", "challenger")
            if model_version in ["champion", "challenger"]:
                # MLflow 2.11+ supports aliases, older versions don't directly expose in search_model_versions
                # For compatibility, we'll try to find by tags or assume 'latest' if alias not found.
                # A more robust solution might involve iterating through aliases if available in API or tags.
                found_version = None
                for mv in latest_versions:
                    if mv.current_stage == model_version.capitalize():
                        found_version = mv
                        break
                if found_version:
                    model_version_obj = found_version
                else:
                    logger.warning(f"Could not find model version with stage/alias '{model_version}'. Attempting to load 'latest'.")
                    model_version_obj = client.get_latest_versions(model_name, stages=["Production", "Staging"])[0] if client.get_latest_versions(model_name, stages=["Production", "Staging"]) else None
            elif model_version == "latest":
                model_version_obj = client.get_latest_versions(model_name, stages=["Production", "Staging"])[0] if client.get_latest_versions(model_name, stages=["Production", "Staging"]) else None
            else:
                raise ValueError(f"Unsupported model version/alias: {model_version}")

        if not model_version_obj:
            raise ValueError(f"No model version found for '{model_name}' with alias/stage '{model_version}'")
            
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
            data_cleaner = None # Set to None if loading fails

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
            feature_names = [] # Set to empty list if loading fails


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
