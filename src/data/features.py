"""
Feature engineering module for the credit scoring platform.

This module provides functions for creating predictive features from raw credit data.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureEngineer:
    """
    A class for engineering features from credit data.
    """

    def __init__(self):
        self.feature_configs = {}
        self.logger = logging.getLogger(__name__)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from the input DataFrame.

        Args:
            df: Input DataFrame with raw credit data

        Returns:
            DataFrame with additional engineered features
        """
        self.logger.info("Starting feature engineering process...")
        df = df.copy()

        # Create financial ratio features
        df = self._create_financial_ratios(df)

        # Create categorical feature encodings
        df = self._create_categorical_features(df)

        # Create time-based features
        df = self._create_time_features(df)

        # Create interaction features
        df = self._create_interaction_features(df)

        # Create binned features
        df = self._create_binned_features(df)

        # Create derived risk indicators
        df = self._create_risk_indicators(df)

        self.logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df

    def _create_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial ratio features."""
        # Debt-to-income ratio (already available in some datasets, but let's calculate as an example)
        if "annual_inc" in df.columns and "dti" in df.columns:
            # Calculate debt amount from DTI
            df["debt_amnt"] = df["annual_inc"] * df["dti"] / 100

        # Loan amount to annual income ratio
        if "annual_inc" in df.columns and "loan_amnt" in df.columns:
            df["loan_income_ratio"] = df["loan_amnt"] / (
                df["annual_inc"] + 1e-8
            )  # Add small value to avoid division by zero

        # Installment to income ratio
        if "annual_inc" in df.columns and "installment" in df.columns:
            df["installment_income_ratio"] = (df["installment"] * 12) / (
                df["annual_inc"] + 1e-8
            )

        # Revolving utilization ratio (if not already present)
        if "revol_bal" in df.columns and "tot_hi_cred_lim" in df.columns:
            df["revol_util_ratio"] = df["revol_bal"] / (df["tot_hi_cred_lim"] + 1e-8)

        # Debt-to-income ratio
        if "total_debt" in df.columns and "annual_inc" in df.columns:
            df["debt_income_ratio"] = df["total_debt"] / (df["annual_inc"] + 1e-8)

        return df

    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create and transform categorical features."""
        # Grade-based features
        if "grade" in df.columns:
            # Map grades to numeric values (A=1, B=2, ..., G=7)
            grade_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
            df["grade_numeric"] = df["grade"].map(grade_mapping).fillna(0)

        # Sub-grade based features
        if "sub_grade" in df.columns:
            # Extract numeric part from sub-grade (e.g., A1->1, B2->8, etc.)
            sub_grade_mapping = {
                "A1": 1,
                "A2": 2,
                "A3": 3,
                "A4": 4,
                "A5": 5,
                "B1": 6,
                "B2": 7,
                "B3": 8,
                "B4": 9,
                "B5": 10,
                "C1": 11,
                "C2": 12,
                "C3": 13,
                "C4": 14,
                "C5": 15,
                "D1": 16,
                "D2": 17,
                "D3": 18,
                "D4": 19,
                "D5": 20,
                "E1": 21,
                "E2": 22,
                "E3": 23,
                "E4": 24,
                "E5": 25,
                "F1": 26,
                "F2": 27,
                "F3": 28,
                "F4": 29,
                "F5": 30,
                "G1": 31,
                "G2": 32,
                "G3": 33,
                "G4": 34,
                "G5": 35,
            }
            df["sub_grade_numeric"] = df["sub_grade"].map(sub_grade_mapping).fillna(0)

        # Employment length-based features
        if "emp_length" in df.columns:
            # Create a binary feature for longer-term employment
            df["is_long_term_employed"] = (df["emp_length"] >= 5).astype(int)

        # Home ownership features
        if "home_ownership" in df.columns:
            # One-hot encode home ownership
            home_ownership_dummies = pd.get_dummies(
                df["home_ownership"], prefix="home_ownership"
            )
            df = pd.concat([df, home_ownership_dummies], axis=1)

        # Purpose-based features
        if "purpose" in df.columns:
            # Create some meaningful groupings
            debt_consolidation_purposes = ["debt_consolidation", "credit_card"]
            df["is_debt_consolidation"] = (
                df["purpose"].isin(debt_consolidation_purposes).astype(int)
            )

        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from date columns."""
        # If issue date is available
        if "issue_d" in df.columns and pd.api.types.is_datetime64_any_dtype(
            df["issue_d"]
        ):
            df["issue_month"] = df["issue_d"].dt.month
            df["issue_year"] = df["issue_d"].dt.year
            df["issue_day_of_week"] = df["issue_d"].dt.dayofweek

        # If available date is present
        if "earliest_cr_line" in df.columns and pd.api.types.is_datetime64_any_dtype(
            df["earliest_cr_line"]
        ):
            # Calculate credit history length in years
            df["credit_history_length_years"] = (
                (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
            ).clip(lower=0)

        # If loan term is available
        if "term" in df.columns:
            df["term_numeric"] = df["term"].str.extract("(\d+)").astype(int)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        # Interaction between loan amount and interest rate
        if "loan_amnt" in df.columns and "int_rate" in df.columns:
            df["loan_rate_interaction"] = df["loan_amnt"] * df["int_rate"] / 100

        # Interaction between income and DTI
        if "annual_inc" in df.columns and "dti" in df.columns:
            df["income_dti_interaction"] = df["annual_inc"] * df["dti"] / 100

        # Interaction between grade and loan amount
        if "grade_numeric" in df.columns and "loan_amnt" in df.columns:
            df["grade_loan_interaction"] = df["grade_numeric"] * df["loan_amnt"]

        return df

    def _create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binned categorical features from continuous variables."""
        # Income brackets
        if "annual_inc" in df.columns:
            df["income_bracket"] = pd.cut(
                df["annual_inc"],
                bins=[0, 30000, 50000, 75000, 100000, float("inf")],
                labels=["Low", "Lower_Middle", "Middle", "Upper_Middle", "High"],
                include_lowest=True,
            ).astype(str)

        # Loan amount brackets
        if "loan_amnt" in df.columns:
            df["loan_bracket"] = pd.cut(
                df["loan_amnt"],
                bins=[0, 5000, 15000, 25000, 35000, float("inf")],
                labels=["Very_Small", "Small", "Medium", "Large", "Very_Large"],
                include_lowest=True,
            ).astype(str)

        # Interest rate brackets
        if "int_rate" in df.columns:
            df["int_rate_bracket"] = pd.cut(
                df["int_rate"],
                bins=[0, 8, 12, 16, 20, float("inf")],
                labels=["Very_Low", "Low", "Medium", "High", "Very_High"],
                include_lowest=True,
            ).astype(str)

        return df

    def _create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk indicators from various features."""
        # High-risk indicators
        if "dti" in df.columns:
            df["high_dti_risk"] = (df["dti"] > 25).astype(
                int
            )  # DTI > 25% is considered high risk

        if "int_rate" in df.columns:
            df["high_interest_risk"] = (df["int_rate"] > 15).astype(
                int
            )  # Interest rate > 15% is high risk

        if "revol_util" in df.columns:
            df["high_revol_util_risk"] = (df["revol_util"] > 80).astype(
                int
            )  # Revolving utilization > 80% is high risk

        # Combined risk score
        risk_factors = []
        if "high_dti_risk" in df.columns:
            risk_factors.append(df["high_dti_risk"])
        if "high_interest_risk" in df.columns:
            risk_factors.append(df["high_interest_risk"])
        if "high_revol_util_risk" in df.columns:
            risk_factors.append(df["high_revol_util_risk"])

        if risk_factors:
            df["combined_risk_score"] = sum(risk_factors)

        return df

    def create_target_variable(
        self, df: pd.DataFrame, target_col: str = "default"
    ) -> pd.DataFrame:
        """
        Create or standardize the target variable for classification.

        Args:
            df: Input DataFrame
            target_col: Name of the target column to create/standardize

        Returns:
            DataFrame with standardized target column
        """
        if "loan_status" in df.columns:
            # Map loan status to binary default indicator
            default_mapping = {
                "Charged Off": 1,
                "Default": 1,
                "Late (31-120 days)": 1,
                "Late (16-30 days)": 1,
                "Does not meet the credit policy. Status:Charged Off": 1,
                "Default Receiver": 1,
                "In Grace Period": 0,
                "Current": 0,
                "Fully Paid": 0,
                "Issued": 0,
            }
            df[target_col] = (
                df["loan_status"].map(default_mapping).fillna(0).astype(int)
            )

        return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline to the input data.

    Args:
        df: Input DataFrame with raw credit data

    Returns:
        DataFrame with cleaned and engineered features
    """
    engineer = FeatureEngineer()
    df = engineer.create_features(df)
    return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Example usage would be:
    # df = pd.read_csv('path/to/data.csv')
    # engineered_df = apply_feature_engineering(df)
    pass
