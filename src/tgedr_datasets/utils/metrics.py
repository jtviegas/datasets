"""Data quality metrics collection and persistence.

This module provides the MetricsCollector class for gathering data quality
metrics during ETL transformations and persisting them to CSV files.
"""

import csv
import logging
from datetime import datetime, UTC
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects data quality metrics and persists them to CSV files.

    Metrics are stored as name-value pairs and appended as a single row
    to a per-dataset CSV file in wide format (one column per metric).

    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.set("row_count", 100)
        >>> metrics.set("duplicate_id_count", 2)
        >>> metrics.save("metrics", "tickers")
        # Creates/appends metrics/tickers.csv with one row

    """

    def __init__(self) -> None:
        """Initialize an empty metrics collector."""
        self._metrics: dict[str, int] = {}

    def set(self, name: str, value: int) -> None:
        """Set a metric value.

        Args:
            name: Metric name (used as CSV column header).
            value: Metric value.

        """
        self._metrics[name] = value

    def get(self, name: str) -> int | None:
        """Get a metric value by name.

        Args:
            name: Metric name.

        Returns:
            The metric value, or None if not set.

        """
        return self._metrics.get(name)

    def get_all(self) -> dict[str, int]:
        """Get all collected metrics.

        Returns:
            Dictionary of all metric name-value pairs.

        """
        return dict(self._metrics)

    def save(self, metrics_dir: str, dataset_name: str) -> Path:
        """Append metrics as a row to a CSV file for the given dataset.

        Creates the CSV file with headers if it does not exist.
        The CSV has a timestamp column followed by one column per metric.

        Args:
            metrics_dir: Directory where metrics CSV files are stored.
            dataset_name: Name of the dataset (used as the CSV filename).

        Returns:
            Path to the CSV file that was written.

        """
        dir_path = Path(metrics_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        csv_path = dir_path / f"{dataset_name}.csv"

        timestamp = int(datetime.now(UTC).timestamp())
        row = {"timestamp": timestamp, **self._metrics}
        fieldnames = list(row.keys())

        file_exists = csv_path.exists()

        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info("[metrics] saved to %s: %s", csv_path, row)
        return csv_path
