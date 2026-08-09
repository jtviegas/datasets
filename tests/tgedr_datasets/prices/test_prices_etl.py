"""Tests for PricesEtl class."""

import logging
from datetime import datetime, UTC, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from tgedr_dataops_abs.great_expectations_validation import ValidationError

from tests.conftest import assert_frames_are_equal
from tgedr_dataops.store.hf_dataset import DataFrameSplits
import tgedr_datasets.prices.etl as etl_module
from tgedr_datasets.prices.etl import PricesEtl, _CONTRACT_PATH
from tgedr_datasets.prices.price import Price


@pytest.fixture
def fixed_today() -> datetime:
    """Pin datetime.now to a fixed Wednesday for deterministic weekday tests."""
    fixed = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)  # Wednesday

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 5, 12, 0, 0, tzinfo=tz or UTC)

    with patch.object(etl_module, "datetime", _FakeDatetime):
        yield fixed


def test_contract_file_is_packaged() -> None:
    """Test that the ODCS data contract file is present next to the module.

    Guards against packaging regressions where the ``*.odcs.yaml`` file is
    omitted from the wheel, which would break contract validation at runtime.
    """
    assert _CONTRACT_PATH.exists(), f"missing data contract: {_CONTRACT_PATH}"


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
    config = {"tickers_dataset": "test/tickers", "target_dataset": "test/prices"}
    etl = PricesEtl(configuration=config)
    
    assert etl._data == []
    assert isinstance(etl._result, pd.DataFrame)



@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_with_default_cutoff_date(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
) -> None:
    """Test extract method with default cutoff date (no target_dataset)."""
    config = {"tickers_dataset": "test/tickers"}
    prices_etl = PricesEtl(configuration=config)
    
    # Mock store
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=sample_tickers_df)
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


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_with_custom_cutoff_date(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
) -> None:
    """Test extract method with custom cutoff date."""
    custom_date = 1738886400  # Some timestamp
    config = {"tickers_dataset": "test/tickers", "price_date": custom_date}
    prices_etl = PricesEtl(configuration=config)
    
    # Mock store
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=sample_tickers_df)
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


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_filters_latest_tickers(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_prices: list[Price],
) -> None:
    """Test extract filters tickers to only use the latest date."""
    config = {"tickers_dataset": "test/tickers"}
    prices_etl = PricesEtl(configuration=config)
    
    # DataFrame with multiple dates
    df_with_multiple_dates = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "AAPL", "GOOGL"],
        "date": [1738800000, 1738800000, 1738886400, 1738886400],  # Two different dates
    })
    
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=df_with_multiple_dates)
    
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


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
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
    config = {"tickers_dataset": "test/tickers"}
    prices_etl = PricesEtl(configuration=config)
    
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=sample_tickers_df)
    
    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:2]
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher
    
    prices_etl._store = mock_store
    
    prices_etl.extract()
    
    assert "[extract|in]" in caplog.text
    assert "[extract|out]" in caplog.text
    assert "extracted" in caplog.text


def _midnight_ts(days_ago: int) -> int:
    """Return the midnight-UTC epoch timestamp for a date N days before 2026-08-05."""
    base = datetime(2026, 8, 5, 0, 0, 0, tzinfo=UTC)
    return int((base - timedelta(days=days_ago)).timestamp())


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
def test_find_missing_weekdays_empty_dataset(mock_store_class: Mock, fixed_today: datetime) -> None:
    """Test _find_missing_weekdays returns today when the dataset is empty."""
    prices_etl = PricesEtl()
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=pd.DataFrame())
    prices_etl._store = mock_store

    result = prices_etl._find_missing_weekdays("test/prices")

    assert result == [_midnight_ts(0)]


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
def test_find_missing_weekdays_no_gaps(mock_store_class: Mock, fixed_today: datetime) -> None:
    """Test _find_missing_weekdays returns empty when all weekdays are present."""
    prices_etl = PricesEtl()
    # Today (Wed) and yesterday (Tue) present
    df = pd.DataFrame({"timestamp": [_midnight_ts(0), _midnight_ts(1)]})
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=df)
    prices_etl._store = mock_store

    result = prices_etl._find_missing_weekdays("test/prices")

    assert result == []


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
def test_find_missing_weekdays_returns_gaps(mock_store_class: Mock, fixed_today: datetime) -> None:
    """Test _find_missing_weekdays returns missing weekday timestamps."""
    prices_etl = PricesEtl()
    # Dataset has an older date (5 days ago, Friday) but today is missing
    df = pd.DataFrame({"timestamp": [_midnight_ts(5)]})
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=df)
    prices_etl._store = mock_store

    result = prices_etl._find_missing_weekdays("test/prices")

    # Yesterday (Tue) should be a missing weekday in the range
    assert _midnight_ts(1) in result
    # Today (Wed) should also be missing
    assert _midnight_ts(0) in result


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
def test_find_missing_weekdays_skips_weekends(mock_store_class: Mock, fixed_today: datetime) -> None:
    """Test _find_missing_weekdays skips weekend dates."""
    prices_etl = PricesEtl()
    # Start from 10 days ago; only today present
    df = pd.DataFrame({"timestamp": [_midnight_ts(0)]})
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=df)
    prices_etl._store = mock_store

    result = prices_etl._find_missing_weekdays("test/prices")

    for ts in result:
        dt = datetime.fromtimestamp(ts, tz=UTC)
        assert dt.weekday() < 5


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_backfills_missing_weekdays(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
    fixed_today: datetime,
) -> None:
    """Test extract backfills missing weekday dates from the prices dataset."""
    config = {"tickers_dataset": "test/tickers", "target_dataset": "test/prices"}
    prices_etl = PricesEtl(configuration=config)

    # Store returns prices first (for _find_missing_weekdays), then tickers
    mock_store = Mock()
    mock_store.get.side_effect = [
        DataFrameSplits(train=pd.DataFrame({"timestamp": [_midnight_ts(5)]})),  # prices (old date, gaps to today)
        DataFrameSplits(train=sample_tickers_df),  # tickers
    ]
    mock_store_class.return_value = mock_store

    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:1]
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher

    prices_etl._store = mock_store

    prices_etl.extract()

    # Should fetch for each missing weekday (yesterday at minimum)
    assert mock_fetcher.get_prices.call_count >= 4  # 4 tickers * at least 1 missing date
    # Verify fetcher was called with a missing date (yesterday)
    called_dates = [call[0][1] for call in mock_fetcher.get_prices.call_args_list]
    assert _midnight_ts(1) in called_dates


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_falls_back_to_today_when_no_missing(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
    fixed_today: datetime,
) -> None:
    """Test extract falls back to today when no missing weekdays are found."""
    config = {"tickers_dataset": "test/tickers", "target_dataset": "test/prices"}
    prices_etl = PricesEtl(configuration=config)

    # Prices dataset has all weekdays present (today and yesterday)
    mock_store = Mock()
    mock_store.get.side_effect = [
        DataFrameSplits(train=pd.DataFrame({"timestamp": [_midnight_ts(0), _midnight_ts(1)]})),  # prices (no gaps)
        DataFrameSplits(train=sample_tickers_df),  # tickers
    ]
    mock_store_class.return_value = mock_store

    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:1]
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher

    prices_etl._store = mock_store

    prices_etl.extract()

    # Should fall back to today: 4 tickers * 1 date
    assert mock_fetcher.get_prices.call_count == 4
    called_dates = [call[0][1] for call in mock_fetcher.get_prices.call_args_list]
    assert all(d == _midnight_ts(0) for d in called_dates)


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_defaults_to_today_when_prices_unreadable(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
    fixed_today: datetime,
) -> None:
    """Test extract defaults to today when the prices dataset cannot be read."""
    config = {"tickers_dataset": "test/tickers", "target_dataset": "test/prices"}
    prices_etl = PricesEtl(configuration=config)

    # Prices read raises, then tickers read succeeds
    mock_store = Mock()
    mock_store.get.side_effect = [
        Exception("boom"),  # prices read fails
        DataFrameSplits(train=sample_tickers_df),  # tickers
    ]
    mock_store_class.return_value = mock_store

    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:1]
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher

    prices_etl._store = mock_store

    prices_etl.extract()

    # Should fall back to today: 4 tickers * 1 date
    assert mock_fetcher.get_prices.call_count == 4
    called_dates = [call[0][1] for call in mock_fetcher.get_prices.call_args_list]
    assert all(d == _midnight_ts(0) for d in called_dates)


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


def test_transform_collects_metrics(prices_etl: PricesEtl, sample_prices: list[Price]) -> None:
    """Test transform collects data quality metrics."""
    prices_etl._data = sample_prices

    prices_etl.transform()

    assert prices_etl._metrics.get("row_count") == 4
    assert prices_etl._metrics.get("duplicate_id_count") == 0
    assert prices_etl._metrics.get("negative_price_count") == 0
    assert prices_etl._metrics.get("zero_volume_count") == 0
    assert prices_etl._metrics.get("ohlc_violations_count") == 0
    assert prices_etl._metrics.get("duplicate_ticker_timestamp") == 0


def test_transform_metrics_with_zero_volume(prices_etl: PricesEtl) -> None:
    """Test transform metrics with zero volume rows."""
    prices = [
        Price(ticker="AAPL", timestamp=1738800000, open=150.0, high=155.0, low=149.0, close=154.0, volume=0),
        Price(ticker="MSFT", timestamp=1738800000, open=300.0, high=305.0, low=299.0, close=304.0, volume=800000),
    ]
    prices_etl._data = prices

    prices_etl.transform()

    assert prices_etl._metrics.get("zero_volume_count") == 1


def test_transform_metrics_with_duplicate_ticker_timestamp(prices_etl: PricesEtl) -> None:
    """Test transform metrics detects duplicate ticker+timestamp."""
    prices = [
        Price(ticker="AAPL", timestamp=1738800000, open=150.0, high=155.0, low=149.0, close=154.0, volume=1000000),
        Price(ticker="AAPL", timestamp=1738800000, open=151.0, high=156.0, low=150.0, close=155.0, volume=1100000),
    ]
    prices_etl._data = prices

    prices_etl.transform()

    assert prices_etl._metrics.get("duplicate_ticker_timestamp") == 1


def test_transform_metrics_empty_data(prices_etl: PricesEtl) -> None:
    """Test transform metrics with empty data."""
    prices_etl._data = []

    prices_etl.transform()

    assert prices_etl._metrics.get("row_count") == 0
    assert prices_etl._metrics.get("duplicate_id_count") == 0
    assert prices_etl._metrics.get("negative_price_count") == 0
    assert prices_etl._metrics.get("zero_volume_count") == 0
    assert prices_etl._metrics.get("ohlc_violations_count") == 0
    assert prices_etl._metrics.get("duplicate_ticker_timestamp") == 0


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
def test_load_saves_metrics(
    mock_store_class: Mock, sample_prices: list[Price], tmp_path
) -> None:
    """Test load method saves metrics to CSV."""
    mock_store = Mock()
    mock_store_class.return_value = mock_store

    config = {"target_dataset": "test/prices", "metrics_dir": str(tmp_path)}
    etl = PricesEtl(configuration=config)
    etl._data = sample_prices
    etl.transform()
    etl.load()

    metrics_file = tmp_path / "prices.csv"
    assert metrics_file.exists()

    import csv
    with metrics_file.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["row_count"] == "4"


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
def test_load_calls_store_update(
   mock_store_class: Mock, sample_prices: list[Price]
) -> None:
    """Test load method calls store update with correct parameters."""
    target = "test/prices"
    config = {"target_dataset": target, "metrics_dir": "/tmp/test_metrics"}

    mock_store = Mock()
    mock_store_class.return_value = mock_store
    prices_etl = PricesEtl(configuration=config)

    prices_etl._data = sample_prices
    prices_etl.transform()

    prices_etl.load()

    mock_store.update.assert_called_once()
    call_kwargs = mock_store.update.call_args[1]
    assert isinstance(call_kwargs["df"], DataFrameSplits)
    assert call_kwargs["key"] == target
    assert call_kwargs["append"] is True


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
def test_load_logs_info(
    mock_store_class: Mock, sample_prices: list[Price], caplog
) -> None:
    """Test load method logs information."""
    caplog.set_level(logging.INFO)
    target = "test/prices"
    config = {"target_dataset": target, "metrics_dir": "/tmp/test_metrics"}
    prices_etl = PricesEtl(configuration=config)

    mock_store = Mock()
    mock_store_class.return_value = mock_store

    prices_etl._data = sample_prices
    prices_etl.transform()

    prices_etl.load()

    assert f"[load|in] ({target})" in caplog.text
    assert f"[load|out] => {target}" in caplog.text


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_full_etl_pipeline(
    mock_price_fetcher_class: Mock,
    mock_store_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
) -> None:
    """Test complete ETL pipeline from extract to load."""
    config = {"tickers_dataset": "test/tickers", "target_dataset": "test/prices", "metrics_dir": "/tmp/test_metrics"}
    prices_etl = PricesEtl(configuration=config)
    
    # Mock store for extract (prices first for _find_missing_weekdays, then tickers)
    mock_store = Mock()
    mock_store.get.side_effect = [
        DataFrameSplits(train=pd.DataFrame({"timestamp": [_midnight_ts(5)]})),  # prices
        DataFrameSplits(train=sample_tickers_df),  # tickers
    ]
    mock_store_class.return_value = mock_store
    prices_etl._store = mock_store
    
    # Mock price fetcher
    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:2]
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher
    
    # Run full pipeline
    prices_etl.extract()
     # Verify method calls
    assert mock_store.get.call_count == 2

    prices_etl.transform()
    prices_etl.load()
    
    # Verify results
    assert len(prices_etl._data) > 0
    assert not prices_etl._result.empty
    
    # Verify method calls
    mock_store.update.assert_called_once()


@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_with_configuration_injection(
    mock_price_fetcher_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_prices: list[Price],
) -> None:
    """Test extract method with configuration injection."""

    config = {"tickers_dataset": "test/configured_tickers", "price_date": 1738886400}
    etl = PricesEtl(configuration=config)
    
    mock_store = Mock()

    mock_store.get.return_value = DataFrameSplits(train=sample_tickers_df)
    etl._store = mock_store
    
    mock_fetcher = Mock()
    mock_fetcher.get_prices.return_value = sample_prices[:1]
    mock_price_fetcher_class.get_instance.return_value = mock_fetcher
    
    # The @inject_configuration decorator should use config
    etl.extract()
    mock_store.get.assert_called_once()


@patch("tgedr_datasets.prices.etl.HuggingFaceDatasetStore")
def test_load_with_configuration_injection(
    mock_store_class: Mock, sample_prices: list[Price]
) -> None:
    """Test load method with configuration injection."""
    config = {"target_dataset": "test/configured_prices", "metrics_dir": "/tmp/test_metrics"}
    etl = PricesEtl(configuration=config)

    mock_store = Mock()
    mock_store_class.return_value = mock_store
    etl._data = sample_prices
    etl.transform()
    # The @inject_configuration decorator should use config
    etl.load()

    mock_store.update.assert_called_once()


def test_validate_transform_success(prices_etl: PricesEtl, sample_prices: list[Price]) -> None:
    """Test validate_transform passes for data meeting the contract."""
    prices_etl._data = sample_prices
    prices_etl.transform()

    prices_etl.validate_transform()


def test_validate_transform_negative_price_fails(prices_etl: PricesEtl, sample_prices: list[Price]) -> None:
    """Test validate_transform raises ValidationError for negative prices."""
    prices_etl._data = sample_prices
    prices_etl.transform()
    prices_etl._result.loc[0, "close"] = -1.0

    with pytest.raises(ValidationError, match="does not meet contract"):
        prices_etl.validate_transform()
