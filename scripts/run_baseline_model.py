"""
Baseline Model Training Script

This script runs the baseline model training with default parameters to establish
a performance benchmark for the credit scoring model.
"""

import os
import subprocess
import sys


def run_baseline_training():
    """
    Run the baseline model training with default settings.
    """
    print("Starting baseline model training...")
    print("This will train an XGBoost model as the default baseline.")

    # Set MLflow tracking URI
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

    # Construct the command to run the training script
    cmd = [
        sys.executable,
        "src/models/train.py",
        "--model_type",
        "xgboost",
        "--train_data_path",
        "data/processed/train.csv",
    ]

    try:
        # Execute the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Model training completed successfully!")
        print("Output:")
        print(result.stdout)

        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Model training failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")
        return False
    except FileNotFoundError:
        print("Error: Could not find the training script or data files.")
        print("Make sure you have:")
        print("1. A training data file at data/processed/train.csv")
        print("2. The MLflow server running at http://localhost:5000")
        return False

    return True


if __name__ == "__main__":
    success = run_baseline_training()

    if success:
        print("\nBaseline model training completed successfully!")
        print(
            "The model has been logged to MLflow with parameters, metrics, and artifacts."
        )
    else:
        print("\nBaseline model training failed.")
        sys.exit(1)
