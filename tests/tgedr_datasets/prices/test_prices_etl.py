"""Tests for PricesEtl class."""

import logging
from datetime import datetime, UTC
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from tests.conftest import assert_frames_are_equal
from tgedr_datasets.prices.etl import PricesEtl
from tgedr_datasets.prices.price import Price


@pytest.fixture
def prices_etl() -> PricesEtl:
    """Provide a PricesEtl instance."""
    return PricesEtl()


@pytest.fixture
def sample_prices() -> list[Price]:
    """Provide sample price data."""
    return [
        Price(ticker="AAPL", timestamp=1738800000, open=150.0, high=155.0, low=149.0, close=154.0, volume=1000000),
        Price(ticker="AAPL", timestamp=1738886400, open=154.0, high=156.0, low=153.0, close=155.0, volume=1200000),
        Price(ticker="MSFT", timestamp=1738800000, open=300.0, high=305.0, low=299.0, close=304.0, volume=800000),
        Price(ticker="MSFT", timestamp=1738886400, open=304.0, high=306.0, low=303.0, close=305.0, volume=900000),
    ]


@pytest.fixture
def sample_tickers_df() -> pd.DataFrame:
    """Provide sample ticker DataFrame."""
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "date": [1738800000, 1738800000, 1738800000, 1738800000],
    })


def test_prices_etl_initialization() -> None:
    """Test PricesEtl initialization."""
    etl = PricesEtl()
    
    assert etl._data == []
    assert isinstance(etl._result, pd.DataFrame)
    assert etl._result.empty
    assert isinstance(etl._cutoff_date, int)
    assert etl._cutoff_date > 0


def test_prices_etl_initialization_with_config() -> None:
    """Test PricesEtl initialization with configuration."""
    config = {"tickers_url": "test/tickers", "target_url": "test/prices"}
    etl = PricesEtl(configuration=config)
    
    assert etl._data == []
    assert isinstance(etl._result, pd.DataFrame)



@patch("tgedr_datasets.prices.etl.ParquetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_with_default_cutoff_date(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
) -> None:
    """Test extract method with default cutoff date."""
    config = {"tickers_url": "test/tickers"}
    prices_etl = PricesEtl(configuration=config)
    
    # Mock store
    mock_store = Mock()
    mock_store.get.return_value = sample_tickers_df
    mock_store_class.return_value = mock_store
    
    # Mock price fetcher
    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:2]  # Return 2 prices per ticker
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher
    
    # Initialize with mocked store
    prices_etl._store = mock_store
    
    prices_etl.extract()
    
    assert len(prices_etl._data) == 8  # 4 tickers * 2 prices each
    mock_store.get.assert_called_once_with(key="test/tickers")
    assert mock_fetcher.get_prices.call_count == 4


@patch("tgedr_datasets.prices.etl.ParquetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_with_custom_cutoff_date(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
) -> None:
    """Test extract method with custom cutoff date."""
    custom_date = 1738886400  # Some timestamp
    config = {"tickers_url": "test/tickers", "price_date": custom_date}
    prices_etl = PricesEtl(configuration=config)
    
    # Mock store
    mock_store = Mock()
    mock_store.get.return_value = sample_tickers_df
    mock_store_class.return_value = mock_store
    
    # Mock price fetcher
    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:2]
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher
    
    prices_etl._store = mock_store
    
    prices_etl.extract()
    
    # Verify cutoff date was set to midnight of custom date
    cutoff_dt = datetime.fromtimestamp(prices_etl._cutoff_date, tz=UTC)
    assert cutoff_dt.hour == 0
    assert cutoff_dt.minute == 0
    assert cutoff_dt.second == 0


@patch("tgedr_datasets.prices.etl.ParquetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_filters_latest_tickers(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_prices: list[Price],
) -> None:
    """Test extract filters tickers to only use the latest date."""
    config = {"tickers_url": "test/tickers"}
    prices_etl = PricesEtl(configuration=config)
    
    # DataFrame with multiple dates
    df_with_multiple_dates = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "AAPL", "GOOGL"],
        "date": [1738800000, 1738800000, 1738886400, 1738886400],  # Two different dates
    })
    
    mock_store = Mock()
    mock_store.get.return_value = df_with_multiple_dates
    
    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:1]
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher
    
    prices_etl._store = mock_store
    
    prices_etl.extract()
    
    # Should only fetch prices for tickers with max date (1738886400)
    assert mock_fetcher.get_prices.call_count == 2  # AAPL and GOOGL from latest date
    called_tickers = [call[0][0] for call in mock_fetcher.get_prices.call_args_list]
    assert "AAPL" in called_tickers
    assert "GOOGL" in called_tickers


@patch("tgedr_datasets.prices.etl.ParquetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_logs_info(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
    caplog,
) -> None:
    """Test extract method logs information."""
    caplog.set_level(logging.INFO)
    config = {"tickers_url": "test/tickers"}
    prices_etl = PricesEtl(configuration=config)
    
    mock_store = Mock()
    mock_store.get.return_value = sample_tickers_df
    
    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:2]
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher
    
    prices_etl._store = mock_store
    
    prices_etl.extract()
    
    assert "[extract|in]" in caplog.text
    assert "[extract|out]" in caplog.text
    assert "extracted" in caplog.text


def test_transform_creates_dataframe(prices_etl: PricesEtl, sample_prices: list[Price]) -> None:
    """Test transform method creates DataFrame from prices."""
    prices_etl._data = sample_prices
    
    prices_etl.transform()
    
    assert isinstance(prices_etl._result, pd.DataFrame)
    assert len(prices_etl._result) == len(sample_prices)
    assert not prices_etl._result.empty


def test_transform_adds_metadata_columns(prices_etl: PricesEtl, sample_prices: list[Price]) -> None:
    """Test transform method adds processing_time column."""
    prices_etl._data = sample_prices
    
    prices_etl.transform()
    
    assert "processing_time" in prices_etl._result.columns
    assert prices_etl._result["processing_time"].notna().all()


def test_transform_produces_expected_dataframe(prices_etl: PricesEtl, sample_prices: list[Price]) -> None:
    """Test transform method produces DataFrame with expected structure and values."""
    prices_etl._data = sample_prices
    
    prices_etl.transform()
    
    # Build expected DataFrame
    expected_data = []
    for price in sample_prices:
        expected_data.append({
            "id": price.id,
            "ticker": price.ticker,
            "timestamp": price.timestamp,
            "open": price.open,
            "high": price.high,
            "low": price.low,
            "close": price.close,
            "volume": price.volume,
        })
    expected_df = pd.DataFrame(expected_data)
    
    # Verify shape
    assert prices_etl._result.shape[0] == len(sample_prices)
    
    assert_frames_are_equal(prices_etl._result.drop(columns=["processing_time"]), 
                            expected_df, sort_columns=["ticker", "timestamp"])


def test_transform_empty_data(prices_etl: PricesEtl) -> None:
    """Test transform method with empty data list."""
    prices_etl._data = []
    
    prices_etl.transform()
    
    assert isinstance(prices_etl._result, pd.DataFrame)
    # Should have metadata column even if empty
    assert "processing_time" in prices_etl._result.columns


def test_transform_logs_info(prices_etl: PricesEtl, sample_prices: list[Price], caplog) -> None:
    """Test transform method logs information."""
    caplog.set_level(logging.INFO)
    prices_etl._data = sample_prices
    
    prices_etl.transform()
    
    assert "[transform|in]" in caplog.text
    assert "[transform|out]" in caplog.text
    assert "transformed" in caplog.text


@patch("tgedr_datasets.prices.etl.ParquetStore")
def test_load_calls_store_update(
    mock_store_class: Mock, sample_prices: list[Price]
) -> None:
    """Test load method calls store update with correct parameters."""
    target = "test/prices"
    config = {"target_url": target}
    prices_etl = PricesEtl(configuration=config)
    
    mock_store = Mock()
    mock_store_class.return_value = mock_store
    
    prices_etl._data = sample_prices
    prices_etl.transform()
    
    result = prices_etl.load()
    
    assert result == target
    mock_store.update.assert_called_once()
    call_kwargs = mock_store.update.call_args[1]
    assert isinstance(call_kwargs["df"], pd.DataFrame)
    assert call_kwargs["key"] == target
    assert call_kwargs["key_fields"] == ["id"]


@patch("tgedr_datasets.prices.etl.ParquetStore")
def test_load_logs_info(
    mock_store_class: Mock, sample_prices: list[Price], caplog
) -> None:
    """Test load method logs information."""
    caplog.set_level(logging.INFO)
    target = "test/prices"
    config = {"target_url": target}
    prices_etl = PricesEtl(configuration=config)
    
    mock_store = Mock()
    mock_store_class.return_value = mock_store
    
    prices_etl._data = sample_prices
    prices_etl.transform()
    
    prices_etl.load()
    
    assert f"[load|in] ({target})" in caplog.text
    assert f"[load|out] => {target}" in caplog.text


@patch("tgedr_datasets.prices.etl.ParquetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_full_etl_pipeline(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
) -> None:
    """Test complete ETL pipeline from extract to load."""
    config = {"tickers_url": "test/tickers", "target_url": "test/prices"}
    prices_etl = PricesEtl(configuration=config)
    
    # Mock store for extract
    mock_store = Mock()
    mock_store.get.return_value = sample_tickers_df
    
    # Mock price fetcher
    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:2]
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher
    
    # Mock store for load
    mock_load_store = Mock()
    mock_store_class.return_value = mock_load_store
    
    prices_etl._store = mock_store
    
    # Run full pipeline
    prices_etl.extract()
    prices_etl.transform()
    result = prices_etl.load()
    
    # Verify results
    assert len(prices_etl._data) > 0
    assert not prices_etl._result.empty
    assert result == "test/prices"
    
    # Verify method calls
    mock_store.get.assert_called_once()
    mock_load_store.update.assert_called_once()


@patch("tgedr_datasets.prices.etl.ParquetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_with_configuration_injection(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
) -> None:
    """Test extract method with configuration injection."""
    config = {"tickers_url": "test/configured_tickers", "price_date": 1738886400}
    etl = PricesEtl(configuration=config)
    
    mock_store = Mock()
    mock_store.get.return_value = sample_tickers_df
    mock_store_class.return_value = mock_store
    
    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:1]
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher
    
    etl._store = mock_store
    
    # The @inject_configuration decorator should use config
    etl.extract()
    
    mock_store.get.assert_called_once()


@patch("tgedr_datasets.prices.etl.ParquetStore")
def test_load_with_configuration_injection(
    mock_store_class: Mock, sample_prices: list[Price]
) -> None:
    """Test load method with configuration injection."""
    config = {"target_url": "test/configured_prices"}
    etl = PricesEtl(configuration=config)
    
    mock_store = Mock()
    mock_store_class.return_value = mock_store
    
    etl._data = sample_prices
    etl.transform()
    
    # The @inject_configuration decorator should use config
    result = etl.load()
    
    mock_store.update.assert_called_once()
