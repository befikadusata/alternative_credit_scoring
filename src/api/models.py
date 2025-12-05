"""
Pydantic Models for API Request and Response Validation

This module defines all the Pydantic models used for request and response validation
in the credit scoring API.
"""

from datetime import datetime
from typing import List, Optional, Union

from pydantic import BaseModel, Field


# Prediction-related models
class PredictionInput(BaseModel):
    """
    Base model for prediction input features.
    These fields should match the features expected by the model.
    """

    loan_id: Optional[str] = Field(
        None,
        description="Unique identifier for the loan, used for "
        "feature store lookup",
    )
    # Loan characteristics
    loan_amnt: float = Field(
        ...,
        description="The listed amount of the loan applied for by the borrower",
        gt=0,
    )
    term: str = Field(
        ..., description="The number of payments on the loan", regex=r"^(36|60) months$"
    )
    int_rate: float = Field(
        ..., description="Interest Rate on the loan", gt=0, le=100
    )
    installment: float = Field(
        ...,
        description="The monthly payment owed by the borrower if the loan originates",
        gt=0,
    )

    # Credit grade
    grade: str = Field(..., description="LC assigned loan grade", regex=r"^[A-G]$")
    sub_grade: str = Field(
        ..., description="LC assigned loan subgrade", regex=r"^[A-G][1-5]$"
    )

    # Employment information
    emp_length: Optional[float] = Field(None, description="Employment length in years")

    # Home ownership
    home_ownership: str = Field(
        ...,
        description="The home ownership status provided by the borrower",
        regex=r"^(OWN|RENT|MORTGAGE|OTHER|NONE|ANY)$",
    )

    # Financial information
    annual_inc: float = Field(
        ...,
        description="The self-reported annual income provided by the borrower",
        ge=0,
    )
    verification_status: str = Field(
        ...,
        description="Indicates if income was verified",
        regex=r"^(Verified|Source Verified|Not Verified)$",
    )

    # Loan purpose
    purpose: str = Field(
        ..., description="A category provided by the borrower for the loan request"
    )

    # Debt-to-income ratio
    dti: float = Field(
        ...,
        description="A ratio calculated using the borrower's total monthly debt payments",
        ge=0,
    )

    # Credit history
    delinq_2yrs: int = Field(
        ...,
        description="The number of 30+ day past-due incidences of delinquency "
        "in the borrower's credit file",
        ge=0,
    )
    inq_last_6mths: int = Field(
        ...,
        description="The number of inquiries by creditors during the last 6 months",
        ge=0,
    )
    open_acc: int = Field(
        ...,
        description="The number of open credit lines in the borrower's credit file",
        ge=0,
    )
    pub_rec: int = Field(..., description="Number of derogatory public records", ge=0)
    revol_bal: int = Field(..., description="Total credit revolving balance", ge=0)
    revol_util: Optional[float] = Field(
        None,
        description="Revolving line utilization rate, or the amount of credit the "
        "borrower is using relative to all available revolving credit",
        ge=0,
        le=100,
    )
    total_acc: int = Field(
        ...,
        description="The total number of credit lines currently in the "
        "borrower's credit file",
        ge=0,
    )

    # Payment status
    initial_list_status: str = Field(
        ..., description="The initial listing status of the loan", regex=r"^[fw]$"
    )
    total_pymnt: float = Field(
        ..., description="Payments received to date for the loan", ge=0
    )
    total_pymnt_inv: float = Field(
        ..., description="Investor received payments to date for the loan", ge=0
    )
    total_rec_prncp: float = Field(
        ..., description="Principal received to date", ge=0
    )
    total_rec_int: float = Field(..., description="Interest received to date", ge=0)
    total_rec_late_fee: float = Field(
        ..., description="Late fees received to date", ge=0
    )
    recoveries: float = Field(..., description="Post charge-off gross recovery", ge=0)
    collection_recovery_fee: float = Field(
        ..., description="Post charge-off collection fee", ge=0
    )
    last_pymnt_amnt: float = Field(
        ..., description="Last month's payment amount", ge=0
    )

    class Config:
        schema_extra = {
            "example": {
                "loan_amnt": 10000.0,
                "term": "36 months",
                "int_rate": 12.12,
                "installment": 335.23,
                "grade": "B",
                "sub_grade": "B1",
                "emp_length": 5.0,
                "home_ownership": "RENT",
                "annual_inc": 65000.0,
                "verification_status": "Verified",
                "purpose": "debt_consolidation",
                "dti": 18.5,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1,
                "open_acc": 10,
                "pub_rec": 0,
                "revol_bal": 15000,
                "revol_util": 65.0,
                "total_acc": 25,
                "initial_list_status": "f",
                "total_pymnt": 4022.76,
                "total_pymnt_inv": 4022.76,
                "total_rec_prncp": 870.85,
                "total_rec_int": 3151.91,
                "total_rec_late_fee": 0.0,
                "recoveries": 0.0,
                "collection_recovery_fee": 0.0,
                "last_pymnt_amnt": 335.23,
            }
        }


class PredictionRequest(BaseModel):
    """
    Request model for single prediction.
    """

    input: PredictionInput = Field(..., description="Input features for the prediction")


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch prediction.
    """

    inputs: List[PredictionInput] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of input features for batch prediction",
    )


class ModelLoadRequest(BaseModel):
    """
    Request model for loading a specific model version.
    """

    model_name: str = Field(..., description="Name of the model in MLflow registry")
    model_version: Union[str, int] = Field(
        ..., description="Version or alias of the model (e.g., 'champion', '1', '2')"
    )
    model_type: str = Field(
        "champion",
        description="Type of model to load ('champion' or 'challenger')",
        regex=r"^(champion|challenger)$",
    )


class PredictionResponse(BaseModel):
    """
    Response model for single prediction.
    """

    prediction: int = Field(
        ..., description="Predicted class (0: low risk, 1: high risk/default)"
    )
    probability_default: float = Field(
        ..., description="Probability of default (class 1)", ge=0, le=1
    )
    probability_repayment: float = Field(
        ..., description="Probability of repayment (class 0)", ge=0, le=1
    )
    risk_level: str = Field(
        ..., description="Risk level classification", regex=r"^(low|medium|high)$"
    )
    prediction_time_seconds: float = Field(
        ..., description="Time taken to make the prediction", ge=0
    )
    model_name: str = Field(..., description="Name of the model used for prediction")
    model_version: str = Field(
        ..., description="Version of the model used for prediction"
    )
    timestamp: datetime = Field(..., description="Timestamp of the prediction")


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch prediction.
    """

    predictions: List[PredictionResponse] = Field(
        ..., description="List of individual prediction responses"
    )
    total_inputs: int = Field(..., description="Total number of inputs processed")
    prediction_time_seconds: float = Field(
        ..., description="Total time taken to make all predictions", ge=0
    )
    model_name: str = Field(..., description="Name of the model used for prediction")
    model_version: str = Field(
        ..., description="Version of the model used for prediction"
    )
    timestamp: datetime = Field(..., description="Timestamp of the batch prediction")


class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint.
    """

    status: str = Field(
        ..., description="Overall health status", regex=r"^(healthy|unhealthy)$"
    )
    timestamp: datetime = Field(..., description="Timestamp of the health check")
    model_loaded: bool = Field(..., description="Whether a model is currently loaded")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    model_version: Optional[str] = Field(
        None, description="Version of the loaded model"
    )
    load_time: Optional[datetime] = Field(
        None, description="Time when the model was loaded"
    )
    last_prediction_time: Optional[datetime] = Field(
        None, description="Time of the last prediction"
    )
    error: Optional[str] = Field(None, description="Error message if any")


class ModelInfoResponse(BaseModel):
    """
    Response model for model information endpoint.
    """

    model_name: Optional[str] = Field(
        None, description="Name of the currently loaded model"
    )
    model_version: Optional[str] = Field(
        None, description="Version of the currently loaded model"
    )
    load_time: Optional[datetime] = Field(
        None, description="Time when the model was loaded"
    )
    model_loaded: bool = Field(..., description="Whether a model is currently loaded")


class ModelLoadResponse(BaseModel):
    """
    Response model for model loading endpoint.
    """

    message: str = Field(..., description="Status message")
    model_name: str = Field(..., description="Name of the loaded model")
    model_version: str = Field(..., description="Version of the loaded model")
    timestamp: datetime = Field(..., description="Timestamp of the load operation")


class APIRootResponse(BaseModel):
    """
    Response model for root endpoint.
    """

    message: str = Field(..., description="API message")
    version: str = Field(..., description="API version")
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether a model is currently loaded")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    model_version: Optional[str] = Field(
        None, description="Version of the loaded model"
    )
