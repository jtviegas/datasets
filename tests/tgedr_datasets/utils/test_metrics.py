"""Tests for MetricsCollector class."""

import csv
from pathlib import Path

import pytest

from tgedr_datasets.utils.metrics import MetricsCollector


@pytest.fixture
def metrics() -> MetricsCollector:
    """Provide a MetricsCollector instance."""
    return MetricsCollector()


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_initialization(self, metrics: MetricsCollector) -> None:
        """Test MetricsCollector starts empty."""
        assert metrics.get_all() == {}

    def test_set_and_get(self, metrics: MetricsCollector) -> None:
        """Test setting and getting a metric."""
        metrics.set("row_count", 42)
        assert metrics.get("row_count") == 42

    def test_get_nonexistent(self, metrics: MetricsCollector) -> None:
        """Test getting a metric that was not set."""
        assert metrics.get("nonexistent") is None

    def test_get_all(self, metrics: MetricsCollector) -> None:
        """Test getting all metrics."""
        metrics.set("row_count", 10)
        metrics.set("duplicate_id_count", 2)
        result = metrics.get_all()
        assert result == {"row_count": 10, "duplicate_id_count": 2}

    def test_get_all_returns_copy(self, metrics: MetricsCollector) -> None:
        """Test that get_all returns a copy, not the internal dict."""
        metrics.set("row_count", 10)
        result = metrics.get_all()
        result["row_count"] = 999
        assert metrics.get("row_count") == 10

    def test_overwrite_metric(self, metrics: MetricsCollector) -> None:
        """Test overwriting an existing metric."""
        metrics.set("row_count", 10)
        metrics.set("row_count", 20)
        assert metrics.get("row_count") == 20

    def test_save_creates_file(self, metrics: MetricsCollector, tmp_path: Path) -> None:
        """Test save creates a CSV file with headers."""
        metrics.set("row_count", 5)
        metrics.set("duplicate_id_count", 1)

        csv_path = metrics.save(str(tmp_path), "test_dataset")

        assert csv_path.exists()
        assert csv_path.name == "test_dataset.csv"

        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert "timestamp" in rows[0]
        assert rows[0]["row_count"] == "5"
        assert rows[0]["duplicate_id_count"] == "1"

    def test_save_appends_row(self, metrics: MetricsCollector, tmp_path: Path) -> None:
        """Test save appends a second row to an existing CSV."""
        metrics.set("row_count", 5)
        metrics.save(str(tmp_path), "test_dataset")

        metrics2 = MetricsCollector()
        metrics2.set("row_count", 10)
        csv_path = metrics2.save(str(tmp_path), "test_dataset")

        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["row_count"] == "5"
        assert rows[1]["row_count"] == "10"

    def test_save_creates_directory(self, metrics: MetricsCollector, tmp_path: Path) -> None:
        """Test save creates the metrics directory if it does not exist."""
        metrics_dir = tmp_path / "nested" / "metrics"
        metrics.set("row_count", 1)

        csv_path = metrics.save(str(metrics_dir), "test_dataset")

        assert csv_path.exists()

    def test_save_returns_path(self, metrics: MetricsCollector, tmp_path: Path) -> None:
        """Test save returns the correct file path."""
        metrics.set("row_count", 1)
        csv_path = metrics.save(str(tmp_path), "my_dataset")
        assert csv_path == tmp_path / "my_dataset.csv"
