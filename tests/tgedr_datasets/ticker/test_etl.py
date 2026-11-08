"""Unit tests for TickerEtl."""

from datetime import UTC, datetime, tzinfo
from unittest.mock import Mock, patch

import pytest
from tgedr_dataops.store.hf_dataset import DataFrameSplits
from tgedr_dataops_abs.great_expectations_validation import ValidationError

from tgedr_datasets.ticker.etl import TickerEtl, _CONTRACT_PATH


@pytest.fixture
def fixed_now() -> datetime:
    """Freeze the ETL's datetime at a fixed instant in UTC."""

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return datetime(2026, 8, 5, 12, tzinfo=tz or UTC)

    with patch("tgedr_datasets.ticker.etl.datetime", FixedDatetime):
        yield datetime(2026, 8, 5, 12, tzinfo=UTC)


@pytest.fixture
def sample_tickers() -> list[str]:
    """Provide sample ticker symbols."""
    return ["AAPL", "MSFT", "GOOGL"]


def test_contract_file_is_packaged() -> None:
    """The ODCS contract must be shipped beside the ETL module."""
    assert _CONTRACT_PATH.exists()


def test_initialization() -> None:
    """Initialize empty state with a null result frame."""
    etl = TickerEtl()

    assert etl._data == []
    assert etl._result is None


def test_extract_splits_and_strips_sources(sample_tickers: list[str]) -> None:
    """Split the configured source string and pass stripped names to the fetcher."""
    etl = TickerEtl(configuration={"source": "  sp500 , nasdaq100 "})
    with patch("tgedr_datasets.ticker.etl.TickerFetcher") as fetcher_class:
        fetcher_class.return_value.fetch.return_value = sample_tickers
        metrics = Mock()
        with patch("tgedr_datasets.ticker.etl.Metrics.instance", return_value=metrics):
            etl.extract()

    assert etl._data == sample_tickers
    fetcher_class.return_value.fetch.assert_called_once_with(["sp500", "nasdaq100"])
    metrics.add_to_gauge.assert_called_once_with(
        name="datasets.ticker.etl.rows",
        value=3,
        attributes={"type": "tickers"},
    )


def test_extract_single_source(sample_tickers: list[str]) -> None:
    """Extract from a single configured source."""
    etl = TickerEtl(configuration={"source": "sp500"})
    with patch("tgedr_datasets.ticker.etl.TickerFetcher") as fetcher_class:
        fetcher_class.return_value.fetch.return_value = sample_tickers
        with patch("tgedr_datasets.ticker.etl.Metrics.instance"):
            etl.extract()

    assert etl._data == sample_tickers
    fetcher_class.return_value.fetch.assert_called_once_with(["sp500"])


def test_transform_populates_rows_and_today(fixed_now: datetime, sample_tickers: list[str]) -> None:
    """Build a frame with each ticker and today's UTC midnight timestamp."""
    etl = TickerEtl()
    etl._data = sample_tickers

    etl.transform()

    expected_epoch = int(fixed_now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    assert etl._result["ticker"].tolist() == sample_tickers
    assert etl._result["actual_time"].tolist() == [expected_epoch] * len(sample_tickers)


def test_transform_empty_input() -> None:
    """Produce an empty frame that still has both columns."""
    etl = TickerEtl()
    etl._data = []

    etl.transform()

    assert etl._result.empty
    assert list(etl._result.columns) == ["ticker", "actual_time"]


def test_validate_transform_accepts_contract_data(sample_tickers: list[str]) -> None:
    """Accept transformed rows that satisfy the ticker contract."""
    etl = TickerEtl()
    etl._data = sample_tickers
    etl.transform()

    etl.validate_transform()


def test_validate_transform_rejects_duplicate_tickers() -> None:
    """Reject duplicate ticker symbols which violate the contract uniqueness."""
    etl = TickerEtl()
    etl._data = ["AAPL", "AAPL"]
    etl.transform()

    with pytest.raises(ValidationError, match="does not meet contract"):
        etl.validate_transform()


@patch("tgedr_datasets.ticker.etl.HuggingFaceDatasetStore")
def test_load_appends_result_to_configured_dataset(
    store_class: Mock, sample_tickers: list[str]
) -> None:
    """Push the transformed rows as a train split with append enabled and return None."""
    etl = TickerEtl(configuration={"dataset": "tickers"})
    etl._data = sample_tickers
    etl.transform()

    result = etl.load()

    assert result is None
    store_class.return_value.update.assert_called_once()
    kwargs = store_class.return_value.update.call_args.kwargs
    assert kwargs["key"] == "tickers"
    assert kwargs["append"] is True
    assert isinstance(kwargs["df"], DataFrameSplits)
    assert kwargs["df"].train.equals(etl._result)


def test_extract_requires_source_configuration() -> None:
    """The inherited configuration decorator reports missing keys."""
    with pytest.raises(Exception, match="missing required configuration parameters"):
        TickerEtl().extract()
