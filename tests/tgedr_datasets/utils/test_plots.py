"""Tests for plot_metrics function."""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import pytest

from tgedr_datasets.utils.plots import plot_metrics


@pytest.fixture
def metrics_csv(tmp_path: Path) -> Path:
    """Create a sample metrics CSV file."""
    csv_path = tmp_path / "test_dataset.csv"
    rows = [
        {"timestamp": 1754600000, "row_count": 100, "duplicate_id_count": 0, "empty_count": 2},
        {"timestamp": 1754686400, "row_count": 105, "duplicate_id_count": 1, "empty_count": 0},
        {"timestamp": 1754772800, "row_count": 98, "duplicate_id_count": 0, "empty_count": 3},
    ]
    fieldnames = ["timestamp", "row_count", "duplicate_id_count", "empty_count"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


@pytest.fixture(autouse=True)
def _cleanup_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestPlotMetrics:
    """Tests for plot_metrics function."""

    def test_creates_plot_file(self, metrics_csv: Path) -> None:
        """Test that plot_metrics creates a PNG file."""
        plot_metrics(str(metrics_csv), "test_dataset")

        plot_file = metrics_csv / "test_dataset_metrics.png"
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_handles_empty_csv(self, tmp_path: Path) -> None:
        """Test that plot_metrics handles an empty CSV gracefully."""
        csv_path = tmp_path / "empty_dataset.csv"
        csv_path.write_text("timestamp,row_count\n")

        plot_metrics(str(tmp_path), "empty_dataset")

        plot_file = tmp_path / "empty_dataset_metrics.png"
        assert not plot_file.exists()

    def test_handles_single_metric_column(self, tmp_path: Path) -> None:
        """Test plot with only one metric column."""
        csv_path = tmp_path / "single.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "row_count"])
            writer.writeheader()
            writer.writerow({"timestamp": 1754600000, "row_count": 100})

        plot_metrics(str(tmp_path), "single")

        plot_file = tmp_path / "single_metrics.png"
        assert plot_file.exists()

    def test_handles_many_metric_columns(self, tmp_path: Path) -> None:
        """Test plot with many metric columns."""
        csv_path = tmp_path / "many.csv"
        fieldnames = ["timestamp", "m1", "m2", "m3", "m4", "m5", "m6"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"timestamp": 1754600000, "m1": 1, "m2": 2, "m3": 3, "m4": 4, "m5": 5, "m6": 6})

        plot_metrics(str(tmp_path), "many")

        plot_file = tmp_path / "many_metrics.png"
        assert plot_file.exists()

    def test_with_cutoff_days(self, metrics_csv: Path) -> None:
        """Test plot with a cutoff filter."""
        plot_metrics(str(metrics_csv), "test_dataset", cutoff_days=365)

        plot_file = metrics_csv / "test_dataset_metrics.png"
        assert plot_file.exists()
