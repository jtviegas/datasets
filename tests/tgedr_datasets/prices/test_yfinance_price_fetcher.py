"""Unit tests for YFinancePriceFetcher class."""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, UTC
import pandas as pd

from tgedr_datasets.prices.price import Price
from tgedr_datasets.prices.yfinance_price_fetcher import YFinancePriceFetcher



@pytest.fixture
def test_date() -> int:
    """Fixture providing a test date timestamp.

    Returns:
        Unix epoch timestamp for 2023-12-01 00:00:00 UTC

    """
    return int(datetime(2023, 12, 1, 0, 0, 0, tzinfo=UTC).timestamp())


@pytest.fixture
def mock_single_day_data() -> pd.DataFrame:
    """Fixture providing mock daily price data for a single day.

    Returns:
        DataFrame with daily OHLCV data for a single day

    """
    dates = pd.date_range("2023-12-01", periods=1, freq="1D", tz=UTC)
    return pd.DataFrame(
        {
            "Open": [180.0],
            "High": [185.0],
            "Low": [179.0],
            "Close": [182.0],
            "Volume": [50000000],
        },
        index=dates,
    )


@pytest.fixture
def mock_daily_data() -> pd.DataFrame:
    """Fixture providing mock daily price data.

    Returns:
        DataFrame with daily OHLCV data for 5 days

    """
    dates = pd.date_range("2023-12-01", periods=5, freq="1D", tz=UTC)
    return pd.DataFrame(
        {
            "Open": [180.0, 181.0, 182.0, 183.0, 184.0],
            "High": [185.0, 186.0, 187.0, 188.0, 189.0],
            "Low": [179.0, 180.0, 181.0, 182.0, 183.0],
            "Close": [182.0, 183.0, 184.0, 185.0, 186.0],
            "Volume": [50000000, 51000000, 52000000, 53000000, 54000000],
        },
        index=dates,
    )


def test_get_prices_single_day_daily_data(
    mock_single_day_data: pd.DataFrame,
    test_date: int,
) -> None:
    """Test get_prices returns daily data for single day (days_window=0)."""
    fetcher = YFinancePriceFetcher()

    with patch("tgedr_datasets.prices.yfinance_price_fetcher.yf.Ticker") as mock_ticker:
        mock_stock = Mock()
        mock_stock.history.return_value = mock_single_day_data
        mock_ticker.return_value = mock_stock

        prices = fetcher.get_prices("AAPL", test_date, days_window=0)

        # Verify yfinance was called with correct parameters
        assert mock_ticker.called
        assert mock_ticker.call_args[0][0] == "AAPL"
        assert mock_stock.history.called

        # Verify correct interval was requested (daily data)
        call_kwargs = mock_stock.history.call_args[1]
        assert call_kwargs["interval"] == "1d"

        # Verify results
        assert len(prices) == 1
        assert all(isinstance(p, Price) for p in prices)
        assert all(p.ticker == "AAPL" for p in prices)
        assert prices[0].close == 182.0
        assert prices[0].volume == 50000000


def test_get_prices_positive_window_daily_data(
    mock_daily_data: pd.DataFrame,
    test_date: int,
) -> None:
    """Test get_prices returns daily data for positive window (future dates)."""
    fetcher = YFinancePriceFetcher()

    with patch("tgedr_datasets.prices.yfinance_price_fetcher.yf.Ticker") as mock_ticker:
        mock_stock = Mock()
        mock_stock.history.return_value = mock_daily_data
        mock_ticker.return_value = mock_stock

        prices = fetcher.get_prices("AAPL", test_date, days_window=5)

        # Verify correct interval was requested (always daily)
        call_kwargs = mock_stock.history.call_args[1]
        assert call_kwargs["interval"] == "1d"

        # Verify results
        assert len(prices) == 5
        assert all(isinstance(p, Price) for p in prices)
        assert prices[0].close == 182.0
        assert prices[4].close == 186.0


def test_get_prices_negative_window_daily_data(
    mock_daily_data: pd.DataFrame,
    test_date: int,
) -> None:
    """Test get_prices returns daily data for negative window (past dates)."""
    fetcher = YFinancePriceFetcher()

    with patch("tgedr_datasets.prices.yfinance_price_fetcher.yf.Ticker") as mock_ticker:
        mock_stock = Mock()
        mock_stock.history.return_value = mock_daily_data
        mock_ticker.return_value = mock_stock

        prices = fetcher.get_prices("AAPL", test_date, days_window=-5)

        # Verify correct interval was requested (always daily)
        call_kwargs = mock_stock.history.call_args[1]
        assert call_kwargs["interval"] == "1d"

        # Verify results
        assert len(prices) == 5
        assert all(isinstance(p, Price) for p in prices)


def test_get_prices_empty_dataframe(test_date: int) -> None:
    """Test get_prices returns empty list when no data is available."""
    fetcher = YFinancePriceFetcher()

    with patch("tgedr_datasets.prices.yfinance_price_fetcher.yf.Ticker") as mock_ticker:
        mock_stock = Mock()
        mock_stock.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_stock

        prices = fetcher.get_prices("INVALID", test_date)

        assert prices == []


def test_get_prices_handles_exception(test_date: int) -> None:
    """Test get_prices handles exceptions gracefully."""
    fetcher = YFinancePriceFetcher()

    with patch("tgedr_datasets.prices.yfinance_price_fetcher.yf.Ticker") as mock_ticker:
        mock_ticker.side_effect = Exception("Network error")

        prices = fetcher.get_prices("AAPL", test_date)

        assert prices == []


def test_get_prices_converts_timestamps_correctly(
    mock_daily_data: pd.DataFrame,
    test_date: int,
) -> None:
    """Test that timestamps are correctly converted from pandas to epoch."""
    fetcher = YFinancePriceFetcher()

    with patch("tgedr_datasets.prices.yfinance_price_fetcher.yf.Ticker") as mock_ticker:
        mock_stock = Mock()
        mock_stock.history.return_value = mock_daily_data
        mock_ticker.return_value = mock_stock

        prices = fetcher.get_prices("AAPL", test_date)

        # Verify timestamps are integers
        assert all(isinstance(p.timestamp, int) for p in prices)

        # Verify timestamps are in ascending order
        timestamps = [p.timestamp for p in prices]
        assert timestamps == sorted(timestamps)


def test_get_prices_ohlcv_data_correct(
    mock_daily_data: pd.DataFrame,
    test_date: int,
) -> None:
    """Test that OHLCV data is correctly extracted from DataFrame."""
    fetcher = YFinancePriceFetcher()

    with patch("tgedr_datasets.prices.yfinance_price_fetcher.yf.Ticker") as mock_ticker:
        mock_stock = Mock()
        mock_stock.history.return_value = mock_daily_data
        mock_ticker.return_value = mock_stock

        prices = fetcher.get_prices("AAPL", test_date, days_window=5)

        # Verify first price data
        assert prices[0].open == 180.0
        assert prices[0].high == 185.0
        assert prices[0].low == 179.0
        assert prices[0].close == 182.0
        assert prices[0].volume == 50000000

        # Verify last price data
        assert prices[4].open == 184.0
        assert prices[4].high == 189.0
        assert prices[4].low == 183.0
        assert prices[4].close == 186.0
        assert prices[4].volume == 54000000


def test_get_prices_ticker_in_price_data(
    mock_daily_data: pd.DataFrame,
    test_date: int,
) -> None:
    """Test that ticker symbol is correctly set in Price objects."""
    fetcher = YFinancePriceFetcher()

    with patch("tgedr_datasets.prices.yfinance_price_fetcher.yf.Ticker") as mock_ticker:
        mock_stock = Mock()
        mock_stock.history.return_value = mock_daily_data
        mock_ticker.return_value = mock_stock

        prices = fetcher.get_prices("GOOGL", test_date)

        assert all(p.ticker == "GOOGL" for p in prices)


@patch("tgedr_datasets.prices.yfinance_price_fetcher.logger")
def test_get_prices_logs_debug_info(
    mock_logger: Mock,
    mock_daily_data: pd.DataFrame,
    test_date: int,
) -> None:
    """Test that get_prices logs appropriate debug and info messages."""
    fetcher = YFinancePriceFetcher()

    with patch("tgedr_datasets.prices.yfinance_price_fetcher.yf.Ticker") as mock_ticker:
        mock_stock = Mock()
        mock_stock.history.return_value = mock_daily_data
        mock_ticker.return_value = mock_stock

        fetcher.get_prices("AAPL", test_date)

        # Verify logging calls
        assert mock_logger.debug.called
        assert mock_logger.info.called


@patch("tgedr_datasets.prices.yfinance_price_fetcher.logger")
def test_get_prices_logs_error_on_exception(
    mock_logger: Mock,
    test_date: int,
) -> None:
    """Test that get_prices logs errors when exceptions occur."""
    fetcher = YFinancePriceFetcher()

    with patch("tgedr_datasets.prices.yfinance_price_fetcher.yf.Ticker") as mock_ticker:
        mock_ticker.side_effect = Exception("Network error")

        fetcher.get_prices("AAPL", test_date)

        # Verify error was logged
        assert mock_logger.exception.called


def test_get_prices_date_range_calculation_single_day(
    mock_single_day_data: pd.DataFrame,
    test_date: int,
) -> None:
    """Test that date range is correctly calculated for single day."""
    fetcher = YFinancePriceFetcher()

    with patch("tgedr_datasets.prices.yfinance_price_fetcher.yf.Ticker") as mock_ticker:
        mock_stock = Mock()
        mock_stock.history.return_value = mock_single_day_data
        mock_ticker.return_value = mock_stock

        fetcher.get_prices("AAPL", test_date, days_window=0)

        # Verify start and end dates
        call_kwargs = mock_stock.history.call_args[1]
        start = call_kwargs["start"]
        end = call_kwargs["end"]

        # Start should be beginning of day
        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0

        # End should be start of next day
        assert (end - start).days == 1
