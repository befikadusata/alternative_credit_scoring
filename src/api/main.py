"""
FastAPI Application for Credit Scoring API

This module sets up the FastAPI application with endpoints for:
- Health checks
- Credit scoring predictions
- Model information
"""

# Initialize FastAPI app - Moved to top
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown events."""
    logger.info("Starting up Credit Scoring API with Champion-Challenger support...")
    global challenger_traffic_percentage

    challenger_traffic_percentage = float(
        os.getenv("CHALLENGER_TRAFFIC_PERCENTAGE", 0.0)
    )

    with model_lock:
        # Load Champion Model
        try:
            champ_model_name = os.getenv("CHAMPION_MODEL_NAME", "credit_scoring_model")
            champ_model_version = os.getenv("CHAMPION_MODEL_VERSION", "champion")
            _load_model(champ_model_name, champ_model_version, "champion")
        except Exception as e:
            logger.error(f"Failed to load champion model on startup: {str(e)}")

        # Load Challenger Model
        if challenger_traffic_percentage > 0:
            try:
                challenger_model_name = os.getenv("CHALLENGER_MODEL_NAME")
                challenger_model_version = os.getenv("CHALLENGER_MODEL_VERSION")
                if challenger_model_name and challenger_model_version:
                    _load_model(
                        challenger_model_name, challenger_model_version, "challenger"
                    )
                else:
                    logger.warning(
                        "Challenger traffic > 0 but CHALLENGER_MODEL_NAME or CHALLENGER_MODEL_VERSION not set."
                    )
            except Exception as e:
                logger.error(f"Failed to load challenger model on startup: {str(e)}")

    # Initialize Redis client on startup
    if redis_client.client:
        redis_client.client.ping()

    yield  # This is where the application runs

    # Shutdown code would go here if needed
    logger.info("Shutting down Credit Scoring API...")


app = FastAPI(
    title="Credit Scoring API",
    description="API for making credit scoring predictions with Champion-Challenger support",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

import logging
import os
import random
import time
from datetime import datetime, timezone
from threading import Lock

import pandas as pd
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger import jsonlogger

from .model_loader import load_model_from_registry
from .models import (
    APIRootResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelLoadRequest,
    ModelLoadResponse,
    PredictionRequest,
    PredictionResponse,
)
from .redis_client import redis_client  # Import the Redis client

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set up structured logging for predictions
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log_handler.setFormatter(formatter)
prediction_logger = logging.getLogger("prediction_logger")
prediction_logger.addHandler(log_handler)
prediction_logger.setLevel(logging.INFO)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument the app with Prometheus
Instrumentator().instrument(app).expose(app)

# --- Champion-Challenger Model Management ---
model_lock = Lock()
champion_model = None
challenger_model = None
champion_cleaner = None
challenger_cleaner = None
champion_model_info = {}
challenger_model_info = {}
challenger_traffic_percentage = 0.0


def _load_model(model_name: str, model_version: str, model_type: str):
    """Helper to load a model and its data cleaner, and update its info."""
    global champion_model, challenger_model, champion_cleaner, challenger_cleaner, champion_model_info, challenger_model_info

    model, cleaner, feature_names = load_model_from_registry(model_name, model_version)
    model_info = {
        "model_name": model_name,
        "model_version": model_version,
        "load_time": datetime.now(timezone.utc),
        "last_prediction_time": None,
        "feature_names": feature_names,  # Store feature names
    }

    if model_type == "champion":
        champion_model = model
        champion_cleaner = cleaner
        champion_model_info = model_info
    elif model_type == "challenger":
        challenger_model = model
        challenger_cleaner = cleaner
        challenger_model_info = model_info

    logger.info(
        f"Successfully loaded {model_type} model: {model_name} version {model_version}"
    )


@app.get("/")
async def root():
    """Root endpoint for basic information about the API."""
    return APIRootResponse(
        message="Credit Scoring API",
        version="1.1.0",
        status="running",
        model_loaded=champion_model is not None,
        model_name=champion_model_info.get("model_name"),
        model_version=champion_model_info.get("model_version"),
    )


@app.get("/health")
async def health_check():
    """Health check for champion and challenger models."""
    champion_status = "healthy" if champion_model is not None else "unhealthy"
    challenger_status = "healthy" if challenger_model is not None else "not_loaded"

    redis_status = "healthy"
    redis_error = None
    try:
        if not redis_client.client or not redis_client.client.ping():
            redis_status = "unhealthy"
            redis_error = "Redis not reachable"
    except Exception as e:
        redis_status = "unhealthy"
        redis_error = str(e)

    overall_status = "healthy"
    if champion_status == "unhealthy" or redis_status == "unhealthy":
        overall_status = "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc),
        "champion": {"status": champion_status, **champion_model_info},
        "challenger": {"status": challenger_status, **challenger_model_info},
        "redis": {
            "status": redis_status,
            "error": redis_error,
            "host": redis_client.REDIS_HOST,
            "port": redis_client.REDIS_PORT,
        },
        "challenger_traffic_percentage": challenger_traffic_percentage,
    }


@app.get("/model/info")
async def get_model_info():
    """Get info about loaded models."""
    return {
        "champion": champion_model_info,
        "challenger": challenger_model_info,
        "challenger_traffic_percentage": challenger_traffic_percentage,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_credit_risk(request: PredictionRequest):
    """Make a prediction, routing between champion and challenger."""
    if champion_model is None or champion_cleaner is None:
        raise HTTPException(status_code=503, detail="Champion model not loaded")

    use_challenger = (
        challenger_model is not None
        and challenger_cleaner is not None
        and random.random() < (challenger_traffic_percentage / 100.0)
    )

    if use_challenger:
        model = challenger_model
        cleaner = challenger_cleaner
        model_type = "challenger"
        model_info = challenger_model_info
    else:
        model = champion_model
        cleaner = champion_cleaner
        model_type = "champion"
        model_info = champion_model_info

    # --- Feature Store Lookup ---
    features_source = "request_payload"
    input_features = request.input

    if input_features.loan_id:
        cached_features = redis_client.get_features(input_features.loan_id)
        if cached_features:
            input_features = cached_features
            features_source = "redis_cache"
            logger.info(
                f"Using features from Redis cache for loan_id: {input_features.loan_id}"
            )
        else:
            logger.info(
                f"Features for loan_id: {input_features.loan_id} not found in Redis. Using request payload."
            )

    try:
        start_time = time.time()

        # Create a DataFrame from the input features
        input_data_dict = input_features.model_dump(exclude_unset=True)
        # Explicitly remove loan_id before cleaning and prediction
        loan_id_val = input_data_dict.pop("loan_id", None)
        input_df = pd.DataFrame([input_data_dict])

        # Convert all object dtype columns (categorical) to numerical using one-hot encoding
        for col in input_df.select_dtypes(include=["object"]).columns:
            if col in input_df.columns:
                dummies = pd.get_dummies(input_df[col], prefix=col)
                input_df = pd.concat([input_df.drop(columns=[col]), dummies], axis=1)

        # Ensure all columns are float types (for the dummy model, which expects numerical input)
        scaled_df = input_df.astype(float)

        # Align columns with the model's expected feature names
        expected_features = model_info.get("feature_names", [])
        if not expected_features:
            logger.warning(
                "Model feature names not found. Proceeding without feature alignment."
            )
        else:
            scaled_df = scaled_df.reindex(columns=expected_features, fill_value=0)

        prediction_proba = model.predict_proba(scaled_df)
        prediction = model.predict(scaled_df)

        prediction_time = time.time() - start_time
        model_info["last_prediction_time"] = datetime.now(timezone.utc)

        response = PredictionResponse(
            prediction=int(prediction[0]),
            probability_default=float(prediction_proba[0][1]),
            probability_repayment=float(prediction_proba[0][0]),
            risk_level=(
                "high"
                if prediction_proba[0][1] > 0.7
                else "medium" if prediction_proba[0][1] > 0.3 else "low"
            ),
            prediction_time_seconds=prediction_time,
            model_name=model_info["model_name"],
            model_version=model_info["model_version"],
            timestamp=datetime.now(timezone.utc),
        )

        prediction_logger.info(
            "Prediction successful",
            extra={
                "model_type": model_type,
                "features_source": features_source,
                "prediction_details": response.model_dump(),
                "input_features": input_features.model_dump(
                    exclude={"loan_id"}
                ),  # Log features without loan_id
            },
        )

        # --- Write-through cache ---
        if input_features.loan_id:
            redis_client.set_features(input_features.loan_id, input_features)

        return response

    except Exception as e:
        logger.error(f"Prediction with {model_type} model failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_credit_risk_batch(request: BatchPredictionRequest):
    """
    Make credit risk predictions for multiple inputs, routing between champion and challenger.
    """
    if champion_model is None or champion_cleaner is None:
        raise HTTPException(status_code=503, detail="Champion model not loaded")

    use_challenger = (
        challenger_model is not None
        and challenger_cleaner is not None
        and random.random() < (challenger_traffic_percentage / 100.0)
    )

    if use_challenger:
        model = challenger_model
        cleaner = challenger_cleaner
        model_type = "challenger"
        model_info = challenger_model_info
    else:
        model = champion_model
        cleaner = champion_cleaner
        model_type = "champion"
        model_info = champion_model_info

    # --- Feature Store Lookup for Batch ---
    processed_inputs = []
    features_sources = []
    for input_item in request.inputs:
        current_input_features = input_item
        features_source = "request_payload"
        if input_item.loan_id:
            cached_features = redis_client.get_features(input_item.loan_id)
            if cached_features:
                current_input_features = cached_features
                features_source = "redis_cache"
            else:
                logger.info(
                    f"Features for loan_id: {input_item.loan_id} not found in Redis. Using request payload for batch."
                )

        processed_inputs.append(current_input_features)
        features_sources.append(features_source)

    try:
        start_time = time.time()

        # Create a DataFrame from the batch of inputs
        input_data_dicts = []
        for f in processed_inputs:
            input_data_dict = f.model_dump(exclude_unset=True)
            input_data_dict.pop("loan_id", None)  # Explicitly remove loan_id
            input_data_dicts.append(input_data_dict)
        input_df = pd.DataFrame(input_data_dicts)

        # Convert all object dtype columns (categorical) to numerical using one-hot encoding
        for col in input_df.select_dtypes(include=["object"]).columns:
            if col in input_df.columns:
                dummies = pd.get_dummies(input_df[col], prefix=col)
                input_df = pd.concat([input_df.drop(columns=[col]), dummies], axis=1)

        # Ensure all columns are float types (for the dummy model, which expects numerical input)
        scaled_df = input_df.astype(float)

        # Align columns with the model's expected feature names
        expected_features = model_info.get("feature_names", [])
        if not expected_features:
            logger.warning(
                "Model feature names not found. Proceeding without feature alignment."
            )
        else:
            scaled_df = scaled_df.reindex(columns=expected_features, fill_value=0)

        predictions_proba = model.predict_proba(scaled_df)
        predictions = model.predict(scaled_df)

        prediction_time = time.time() - start_time
        model_info["last_prediction_time"] = datetime.now(timezone.utc)

        individual_responses = []
        for i in range(len(predictions)):
            individual_response = PredictionResponse(
                prediction=int(predictions[i]),
                probability_default=float(predictions_proba[i][1]),
                probability_repayment=float(predictions_proba[i][0]),
                risk_level=(
                    "high"
                    if predictions_proba[i][1] > 0.7
                    else "medium" if predictions_proba[i][1] > 0.3 else "low"
                ),
                prediction_time_seconds=0.0,
                model_name=model_info["model_name"],
                model_version=model_info["model_version"],
                timestamp=datetime.now(timezone.utc),
            )
            individual_responses.append(individual_response)

            # --- Write-through cache ---
            if processed_inputs[i].loan_id:
                redis_client.set_features(
                    processed_inputs[i].loan_id, processed_inputs[i]
                )

        batch_response = BatchPredictionResponse(
            predictions=individual_responses,
            total_inputs=len(predictions),
            prediction_time_seconds=prediction_time,
            model_name=model_info["model_name"],
            model_version=model_info["model_version"],
            timestamp=datetime.now(timezone.utc),
        )

        prediction_logger.info(
            "Batch prediction successful",
            extra={
                "model_type": model_type,
                "features_sources": features_sources,
                "batch_details": {
                    "total_inputs": batch_response.total_inputs,
                    "prediction_time_seconds": batch_response.prediction_time_seconds,
                    "model_name": batch_response.model_name,
                    "model_version": batch_response.model_version,
                },
                "individual_predictions": [
                    p.model_dump() for p in batch_response.predictions
                ],
            },
        )

        return batch_response

    except Exception as e:
        logger.error(f"Batch prediction with {model_type} model failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.put("/model/load", response_model=ModelLoadResponse)
async def load_model_endpoint(model_request: ModelLoadRequest):
    """Load or reload a champion or challenger model."""
    model_type = getattr(model_request, "model_type", "champion").lower()
    if model_type not in ["champion", "challenger"]:
        raise HTTPException(
            status_code=400, detail="model_type must be 'champion' or 'challenger'"
        )

    try:
        with model_lock:
            _load_model(
                model_request.model_name, model_request.model_version, model_type
            )

        return ModelLoadResponse(
            message=f"Model {model_request.model_name} version {model_request.model_version} loaded successfully as {model_type}",
            model_name=model_request.model_name,
            model_version=str(model_request.model_version),
            timestamp=datetime.now(timezone.utc),
        )
    except Exception as e:
        logger.error(f"Failed to load {model_type} model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        log_level="info",
    )
