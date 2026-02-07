"""Tests for TickerEtl class."""

import logging
from datetime import datetime, UTC
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from tgedr_datasets.ticker.etl import TickerEtl


@pytest.fixture
def ticker_etl() -> TickerEtl:
    """Provide a TickerEtl instance."""
    return TickerEtl()


@pytest.fixture
def sample_tickers() -> list[str]:
    """Provide sample ticker data."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def test_ticker_etl_initialization() -> None:
    """Test TickerEtl initialization."""
    etl = TickerEtl()
    
    assert etl._data == []
    assert etl._result is None


def test_ticker_etl_initialization_with_config() -> None:
    """Test TickerEtl initialization with configuration."""
    config = {"source": "sp500", "target": "/tmp/test"}
    etl = TickerEtl(configuration=config)
    
    assert etl._data == []
    assert etl._result is None


@patch("tgedr_datasets.ticker.etl.TickerFetcher")
def test_extract_single_source(
    mock_fetcher_class: Mock, sample_tickers: list[str]
) -> None:
    """Test extract method with single source."""

    ticker_etl = TickerEtl(configuration={"source": "sp500"})
    mock_fetcher = Mock()
    mock_fetcher.fetch.return_value = sample_tickers
    mock_fetcher_class.return_value = mock_fetcher
    
    ticker_etl.extract()
    
    assert ticker_etl._data == sample_tickers
    mock_fetcher.fetch.assert_called_once_with(["sp500"])


@patch("tgedr_datasets.ticker.etl.TickerFetcher")
def test_extract_multiple_sources(
    mock_fetcher_class: Mock, sample_tickers: list[str]
) -> None:
    """Test extract method with multiple comma-separated sources."""
    
    ticker_etl = TickerEtl(configuration={"source": "sp500, nasdaq100, dowjones"})
    mock_fetcher = Mock()
    mock_fetcher.fetch.return_value = sample_tickers
    mock_fetcher_class.return_value = mock_fetcher
    
    ticker_etl.extract()
    assert ticker_etl._data == sample_tickers
    mock_fetcher.fetch.assert_called_once_with(["sp500", "nasdaq100", "dowjones"])


@patch("tgedr_datasets.ticker.etl.TickerFetcher")
def test_extract_with_whitespace(
    mock_fetcher_class: Mock, sample_tickers: list[str]
) -> None:
    """Test extract method handles whitespace in source string."""

    ticker_etl = TickerEtl(configuration={"source": "  sp500  ,  nasdaq100  "})
    mock_fetcher = Mock()
    mock_fetcher.fetch.return_value = sample_tickers
    mock_fetcher_class.return_value = mock_fetcher
    
    ticker_etl.extract()
    
    mock_fetcher.fetch.assert_called_once_with(["sp500", "nasdaq100"])


def test_transform_creates_dataframe(ticker_etl: TickerEtl, sample_tickers: list[str]) -> None:
    """Test transform method creates DataFrame with ticker and date columns."""
    ticker_etl._data = sample_tickers
    
    ticker_etl.transform()
    
    assert isinstance(ticker_etl._result, pd.DataFrame)
    assert len(ticker_etl._result) == len(sample_tickers)
    assert "ticker" in ticker_etl._result.columns
    assert "date" in ticker_etl._result.columns


def test_transform_ticker_column(ticker_etl: TickerEtl, sample_tickers: list[str]) -> None:
    """Test transform method correctly populates ticker column."""
    ticker_etl._data = sample_tickers
    
    ticker_etl.transform()
    
    assert ticker_etl._result["ticker"].tolist() == sample_tickers


def test_transform_date_column(ticker_etl: TickerEtl, sample_tickers: list[str]) -> None:
    """Test transform method correctly populates date column with epoch timestamp."""
    ticker_etl._data = sample_tickers
    
    ticker_etl.transform()
    
    # Check that date column exists and has integer values
    assert ticker_etl._result["date"].dtype in [int, "int64"]
    
    # Check that all rows have the same date
    assert ticker_etl._result["date"].nunique() == 1
    
    # Check that the date is approximately today's epoch (within 1 day tolerance)
    today = datetime.now(UTC).date()
    expected_epoch = int(datetime(today.year, today.month, today.day, tzinfo=UTC).timestamp())
    actual_epoch = ticker_etl._result["date"].iloc[0]
    
    assert abs(actual_epoch - expected_epoch) < 86400  # Within 1 day


def test_transform_empty_data(ticker_etl: TickerEtl) -> None:
    """Test transform method with empty data list."""
    ticker_etl._data = []
    
    ticker_etl.transform()
    
    assert isinstance(ticker_etl._result, pd.DataFrame)
    assert len(ticker_etl._result) == 0
    assert "ticker" in ticker_etl._result.columns
    assert "date" in ticker_etl._result.columns


@patch("tgedr_datasets.ticker.etl.ParquetStore")
def test_load_saves_to_parquet(
    mock_store_class: Mock, sample_tickers: list[str]
) -> None:
    """Test load method saves DataFrame to Parquet store."""

    target_path = "/tmp/test_tickers"
    ticker_etl = TickerEtl(configuration={"target_url": target_path})
    mock_store = Mock()
    mock_store_class.return_value = mock_store
    
    # Setup data
    ticker_etl._data = sample_tickers
    ticker_etl.transform()
    
    result = ticker_etl.load()
    
    assert result == target_path
    mock_store.update.assert_called_once()
    
    # Verify the DataFrame was passed
    call_args = mock_store.update.call_args
    assert isinstance(call_args[1]["df"], pd.DataFrame)
    assert call_args[1]["key"] == target_path
    assert call_args[1]["key_fields"] == ["date", "ticker"]



@patch("tgedr_datasets.ticker.etl.ParquetStore")
def test_load_partition_fields(
    mock_store_class: Mock, sample_tickers: list[str]
) -> None:
    """Test load method uses correct partition fields."""

    ticker_etl = TickerEtl(configuration={"target_url": "/tmp/test"})
    mock_store = Mock()
    mock_store_class.return_value = mock_store
    
    ticker_etl._data = sample_tickers
    ticker_etl.transform()
    
    ticker_etl.load()
    
    call_args = mock_store.update.call_args
    assert call_args[1]["key_fields"] == ["date", "ticker"]


@patch("tgedr_datasets.ticker.etl.ParquetStore")
@patch("tgedr_datasets.ticker.etl.TickerFetcher")
def test_full_etl_pipeline(
    mock_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers: list[str],
) -> None:
    """Test complete ETL pipeline from extract to load."""

    ticker_etl = TickerEtl(configuration={"source": "sp500", "target_url": "/tmp/tickers"})
    # Setup mocks
    mock_fetcher = Mock()
    mock_fetcher.fetch.return_value = sample_tickers
    mock_fetcher_class.return_value = mock_fetcher
    
    mock_store = Mock()
    mock_store_class.return_value = mock_store
    
    # Run full pipeline
    ticker_etl.extract()
    ticker_etl.transform()
    result_path = ticker_etl.load()
    
    # Verify results
    assert ticker_etl._data == sample_tickers
    assert isinstance(ticker_etl._result, pd.DataFrame)
    assert len(ticker_etl._result) == len(sample_tickers)
    assert result_path == "/tmp/tickers"
    
    # Verify method calls
    mock_fetcher.fetch.assert_called_once()
    mock_store.update.assert_called_once()


@patch("tgedr_datasets.ticker.etl.TickerFetcher")
def test_extract_logs_ticker_count(
    mock_fetcher_class: Mock, sample_tickers: list[str], caplog
) -> None:
    """Test extract method logs the number of tickers fetched."""
    caplog.set_level(logging.INFO)
    
    ticker_etl = TickerEtl(configuration={"source": "sp500", "target_url": "/tmp/tickers"})
    mock_fetcher = Mock()
    mock_fetcher.fetch.return_value = sample_tickers
    mock_fetcher_class.return_value = mock_fetcher
    
    ticker_etl.extract()
    
    assert "tickers len: 5" in caplog.text


def test_transform_logs_entry_exit(
    ticker_etl: TickerEtl, sample_tickers: list[str], caplog
) -> None:
    """Test transform method logs entry and exit."""
    caplog.set_level(logging.INFO)
    ticker_etl._data = sample_tickers
    
    ticker_etl.transform()
    
    assert "[transform|in]" in caplog.text
    assert "[transform|out]" in caplog.text


@patch("tgedr_datasets.ticker.etl.ParquetStore")
def test_load_logs_target_path(
    mock_store_class: Mock, sample_tickers: list[str], caplog
) -> None:
    """Test load method logs the target path."""
    
    target = "/tmp/test_path"
    ticker_etl = TickerEtl(configuration={"source": "sp500", "target_url": target})
    caplog.set_level(logging.INFO)
    mock_store = Mock()
    mock_store_class.return_value = mock_store
    
    ticker_etl._data = sample_tickers
    ticker_etl.transform()
    
    ticker_etl.load()
    
    assert f"[load|in] ({target})" in caplog.text
    assert f"[load|out] => {target}" in caplog.text


@patch("tgedr_datasets.ticker.etl.TickerFetcher")
def test_extract_with_configuration_injection(
    mock_fetcher_class: Mock, sample_tickers: list[str]
) -> None:
    """Test extract method with configuration injection."""
    mock_fetcher = Mock()
    mock_fetcher.fetch.return_value = sample_tickers
    mock_fetcher_class.return_value = mock_fetcher
    
    config = {"source": "nasdaq100"}
    etl = TickerEtl(configuration=config)
    
    # The @inject_configuration decorator should use config
    etl.extract()
    
    mock_fetcher.fetch.assert_called_once()


@patch("tgedr_datasets.ticker.etl.ParquetStore")
def test_load_with_configuration_injection(
    mock_store_class: Mock, sample_tickers: list[str]
) -> None:
    """Test load method with configuration injection."""
    mock_store = Mock()
    mock_store_class.return_value = mock_store
    
    config = {"target_url": "/tmp/configured_path"}
    etl = TickerEtl(configuration=config)
    etl._data = sample_tickers
    etl.transform()
    
    # The @inject_configuration decorator should use config
    result = etl.load()
    
    mock_store.update.assert_called_once()
