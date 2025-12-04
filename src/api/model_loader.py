"""
Model Loading Module for Credit Scoring API

This module handles loading models from the MLflow Model Registry.
"""
import logging
import os
from typing import Optional, Union

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_registry(
    model_name: str, 
    model_version: Union[str, int] = "latest",
    tracking_uri: Optional[str] = None
) -> mlflow.pyfunc.PyFuncModel:
    """
    Load a model from the MLflow Model Registry.
    
    Args:
        model_name: Name of the model in the registry
        model_version: Version of the model (can be version number, "latest", or alias like "champion")
        tracking_uri: MLflow tracking URI (if None, uses environment variable or default)
        
    Returns:
        Loaded PyFuncModel instance
    """
    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    logger.info(f"Attempting to load model '{model_name}' version/alias '{model_version}' from registry")
    
    try:
        # Construct the model URI
        if model_version == "latest":
            # Get the latest version
            client = MlflowClient()
            latest_version_info = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])
            
            if not latest_version_info:
                raise ValueError(f"No versions found for model '{model_name}'")
            
            # Prefer Production, then Staging, then any
            latest_version = None
            for version_info in latest_version_info:
                if version_info.current_stage == "Production":
                    latest_version = version_info.version
                    break
                elif version_info.current_stage == "Staging" and latest_version is None:
                    latest_version = version_info.version
                    break
                elif latest_version is None:
                    latest_version = version_info.version
            
            model_uri = f"models:/{model_name}/{latest_version}"
            logger.info(f"Resolved 'latest' to version {latest_version}, URI: {model_uri}")
        else:
            # Check if model_version is an alias by trying to get alias info
            client = MlflowClient()
            try:
                # Try to get model version by alias
                alias_version_info = client.get_model_version_by_alias(model_name, str(model_version))
                model_uri = f"models:/{model_name}/{alias_version_info.version}"
                logger.info(f"Resolved alias '{model_version}' to version {alias_version_info.version}, URI: {model_uri}")
            except:
                # Assume it's a version number
                model_uri = f"models:/{model_name}/{model_version}"
                logger.info(f"Using direct version '{model_version}', URI: {model_uri}")
        
        # Load the model
        model = mlflow.pyfunc.load_model(model_uri)
        
        logger.info(f"Successfully loaded model from {model_uri}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}' version '{model_version}': {str(e)}")
        raise


def get_model_version_info(
    model_name: str,
    model_version: Union[str, int],
    tracking_uri: Optional[str] = None
) -> dict:
    """
    Get detailed information about a specific model version.
    
    Args:
        model_name: Name of the model in the registry
        model_version: Version of the model
        tracking_uri: MLflow tracking URI
        
    Returns:
        Dictionary with model version information
    """
    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    client = MlflowClient()
    
    try:
        if isinstance(model_version, str) and model_version != "latest":
            # Check if it's an alias
            try:
                version_info = client.get_model_version_by_alias(model_name, model_version)
                model_version = version_info.version
            except:
                # Not an alias, assume it's a version number
                pass
        
        if model_version == "latest":
            # Get latest version
            latest_versions = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])
            if not latest_versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            
            # Prefer Production, then Staging, then any
            version_info = None
            for v_info in latest_versions:
                if v_info.current_stage == "Production":
                    version_info = v_info
                    break
                elif v_info.current_stage == "Staging" and version_info is None:
                    version_info = v_info
                    break
                elif version_info is None:
                    version_info = v_info
            
            model_version = version_info.version
        
        # Get detailed version info
        version_details = client.get_model_version(model_name, model_version)
        
        # Get run info for the model
        run_id = version_details.run_id
        run_info = client.get_run(run_id)
        
        return {
            "model_name": model_name,
            "version": version_details.version,
            "current_stage": version_details.current_stage,
            "creation_timestamp": version_details.creation_timestamp,
            "last_updated_timestamp": version_details.last_updated_timestamp,
            "run_id": run_id,
            "run_info": {
                "status": run_info.info.status,
                "start_time": run_info.info.start_time,
                "end_time": run_info.info.end_time,
                "artifact_uri": run_info.info.artifact_uri
            },
            "aliases": version_details.aliases
        }
    
    except Exception as e:
        logger.error(f"Failed to get model version info for '{model_name}' version '{model_version}': {str(e)}")
        raise


def validate_model_features(model: mlflow.pyfunc.PyFuncModel, input_data) -> bool:
    """
    Validate that the input data matches the expected features of the model.
    This is a basic validation that checks if the model can process the input.
    
    Args:
        model: Loaded MLflow PyFunc model
        input_data: Input data to validate
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Try to make a prediction with the model to validate it works with the input
        prediction = model.predict(input_data)
        return True
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Example usage
    logger.info("Model loader module loaded successfully")