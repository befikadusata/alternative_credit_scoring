"""
Data cleaning and imputation module for the credit scoring platform.

This module provides reusable functions for cleaning credit data,
handling missing values, and performing data quality checks.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
from typing import Tuple, Dict, List, Optional, Union
import re


class DataCleaner:
    """
    A class for cleaning and preprocessing credit data.
    """
    
    def __init__(self):
        self.imputers = {}
        self.label_encoders = {}
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
        
    def clean_loan_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive cleaning to loan data.
        
        Args:
            df: Raw loan data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning process...")
        
        df = df.copy()
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Convert data types
        df = self._fix_data_types(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Validate data ranges
        df = self._validate_data_ranges(df)
        
        self.logger.info(f"Data cleaning completed. Final shape: {df.shape}")
        
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix incorrect data types for specific columns."""
        # Convert percentage fields to numeric
        percentage_cols = [
            'int_rate', 'annual_inc', 'dti', 'revol_util', 
            'emp_length', 'loan_amnt', 'installment'
        ]
        
        for col in percentage_cols:
            if col in df.columns:
                # Remove percentage signs and convert to float
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                    # Handle special cases like '10+ years' for employment length
                    df[col] = df[col].apply(self._convert_employment_length)
        
        # Convert to numeric, coercing errors to NaN
        for col in percentage_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert issue_d to datetime if it exists
        if 'issue_d' in df.columns:
            df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
        
        return df
    
    def _convert_employment_length(self, value):
        """Convert employment length strings to numeric values."""
        if pd.isna(value):
            return np.nan
        
        if isinstance(value, str):
            # Handle different employment length formats
            if 'year' in value.lower():
                # Extract numeric value from strings like '10+ years' or '2 years'
                numbers = re.findall(r'\d+', value)
                if numbers:
                    return float(numbers[0])
            elif 'n/a' in value.lower() or '< 1 year' in value.lower():
                return 0.0
        
        return value
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies."""
        self.logger.info("Handling missing values...")
        
        # Identify columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        for col in missing_cols:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            
            if missing_pct > 50:
                # If more than 50% is missing, consider dropping the column
                self.logger.warning(f"Column {col} has {missing_pct:.2f}% missing values. Consider dropping this column.")
                continue
            
            # For numerical columns, use median imputation
            if df[col].dtype in ['int64', 'float64']:
                if col not in self.imputers:
                    # Use median strategy for numerical columns
                    self.imputers[col] = SimpleImputer(strategy='median')
                    df[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
                else:
                    df[col] = self.imputers[col].transform(df[[col]]).flatten()
            
            # For categorical columns, use most frequent imputation
            elif df[col].dtype == 'object':
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='most_frequent')
                    df[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
                else:
                    df[col] = self.imputers[col].transform(df[[col]]).flatten()
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method for numerical columns."""
        self.logger.info("Removing outliers...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers before removal
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    self.logger.info(f"Found {outliers_count} outliers in {col}")
                
                # Cap outliers rather than removing them to preserve data
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that data falls within expected ranges."""
        self.logger.info("Validating data ranges...")
        
        # Example validation for common credit data fields
        valid_ranges = {
            'int_rate': (0, 30),      # Interest rate between 0-30%
            'annual_inc': (0, 1e8),   # Annual income up to $100M
            'dti': (0, 100),          # DTI should be between 0-100%
            'loan_amnt': (0, 1e6),    # Loan amount up to $1M
        }
        
        for col, (min_val, max_val) in valid_ranges.items():
            if col in df.columns:
                # Identify out-of-range values
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)) & (df[col].notna())
                count = out_of_range.sum()
                
                if count > 0:
                    self.logger.warning(f"Found {count} out-of-range values in {col}")
                    # Cap values to the valid range
                    df.loc[out_of_range, col] = np.clip(df.loc[out_of_range, col], min_val, max_val)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df: DataFrame to encode
            fit: Whether to fit the encoders (True for training, False for test)
            
        Returns:
            Encoded DataFrame
        """
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit and transform the column
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # If encoder doesn't exist and we're not fitting, skip encoding
                    continue
            else:
                # Transform using existing encoder
                # Handle unknown categories by assigning them a default value
                unique_values = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(
                    lambda x: x if str(x) in unique_values else 'UNKNOWN'
                )
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: DataFrame to scale
            fit: Whether to fit the scalers (True for training, False for test)
            
        Returns:
            Scaled DataFrame
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            if col not in self.scalers:
                if fit:
                    self.scalers[col] = StandardScaler()
                    df[col] = self.scalers[col].fit_transform(df[[col]])
                else:
                    # If scaler doesn't exist and we're not fitting, skip scaling
                    continue
            else:
                df[col] = self.scalers[col].transform(df[[col]])
        
        return df


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean data in one function.
    
    Args:
        file_path: Path to the raw data file
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner()
    df = pd.read_csv(file_path)
    df = cleaner.clean_loan_data(df)
    return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Example usage would be:
    # df = load_and_clean_data('path/to/raw/data.csv')
    # cleaned_df = cleaner.clean_loan_data(df)
    pass