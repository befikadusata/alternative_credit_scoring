import logging
import os
from argparse import ArgumentParser

import pandas as pd
from evidently.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.report import Report
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_drift_analysis(
    reference_data_path: str,
    current_data_path: str,
    output_report_path: str,
    prometheus_url: str,
):
    """
    Runs data and prediction drift analysis using Evidently AI.

    Args:
        reference_data_path: Path to the reference dataset (e.g., training data).
        current_data_path: Path to the current dataset (e.g., production data from logs).
        output_report_path: Path to save the HTML drift report.
        prometheus_url: URL for the Prometheus Pushgateway.
    """
    logger.info(f"Loading reference data from: {reference_data_path}")
    try:
        reference_df = pd.read_csv(reference_data_path)
    except FileNotFoundError:
        logger.error(f"Reference data not found at: {reference_data_path}")
        return

    logger.info(f"Loading current data from: {current_data_path}")
    try:
        current_df = pd.read_csv(current_data_path)
    except FileNotFoundError:
        logger.error(f"Current data not found at: {current_data_path}")
        return

    logger.info("Running drift analysis report...")
    # Assuming the target and prediction columns are named 'target' and 'prediction'
    # These might need to be adjusted based on the actual data logging format.
    report = Report(
        metrics=[
            DataDriftPreset(),
            ClassificationPreset(
                prediction_type="binary",
                target_name="target",
                prediction_name="prediction",
            ),
        ]
    )
    report.run(reference_data=reference_df, current_data=current_df)

    # Save the report
    report_dir = os.path.dirname(output_report_path)
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    report.save_html(output_report_path)
    logger.info(f"Drift analysis report saved to: {output_report_path}")

    # Extract metrics and push to Prometheus
    try:
        report_dict = report.as_dict()

        # Example metrics to extract - keys depend on Evidently's report structure
        drift_score = report_dict["metrics"][0]["result"]["data_drift"]["data"][
            "metrics"
        ]["dataset_drift_score"]
        num_drifted_features = report_dict["metrics"][0]["result"]["data_drift"][
            "data"
        ]["metrics"]["n_drifted_columns"]

        registry = CollectorRegistry()
        g_drift_score = Gauge(
            "data_drift_score",
            "Overall data drift score from Evidently",
            registry=registry,
        )
        g_num_drifted_features = Gauge(
            "num_drifted_features",
            "Number of features that have drifted",
            registry=registry,
        )

        g_drift_score.set(drift_score)
        g_num_drifted_features.set(num_drifted_features)

        if prometheus_url:
            push_to_gateway(
                prometheus_url, job="drift-analysis-batch", registry=registry
            )
            logger.info(
                f"Successfully pushed metrics to Prometheus Pushgateway at: {prometheus_url}"
            )
        else:
            logger.warning(
                "Prometheus Pushgateway URL not provided. Skipping metric push."
            )

    except KeyError as e:
        logger.error(
            f"Could not extract metric from Evidently report: {e}. Report structure might have changed."
        )
    except Exception as e:
        logger.error(f"Failed to push metrics to Prometheus: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run data drift analysis using Evidently AI.")
    parser.add_argument(
        "--reference", required=True, help="Path to the reference dataset (CSV)."
    )
    parser.add_argument(
        "--current", required=True, help="Path to the current dataset (CSV)."
    )
    parser.add_argument(
        "--output",
        default="reports/drift_report.html",
        help="Path to save the HTML drift report.",
    )
    parser.add_argument(
        "--prometheus-url",
        default=os.getenv("PROMETHEUS_PUSHGATEWAY_URL"),
        help="URL of the Prometheus Pushgateway.",
    )

    args = parser.parse_args()

    run_drift_analysis(
        reference_data_path=args.reference,
        current_data_path=args.current,
        output_report_path=args.output,
        prometheus_url=args.prometheus_url,
    )
