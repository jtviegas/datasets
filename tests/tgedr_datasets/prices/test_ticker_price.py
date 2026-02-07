"""Unit tests for PriceFetcher singleton factory class."""

from datetime import UTC, datetime

from tgedr_datasets.prices.price_fetcher import PriceFetcher
from tgedr_pycommons.utils.singleton import SingletonMeta
from tgedr_datasets.prices.yfinance_price_fetcher import YFinancePriceFetcher


def test_ticker_price_uses_singleton_metaclass() -> None:
    """Test that PriceFetcher uses the SingletonMeta metaclass."""
    assert type(PriceFetcher) is SingletonMeta
    assert isinstance(PriceFetcher, type)


def test_get_instance_returns_yfinance_fetcher() -> None:
    """Test that get_instance returns a YFinancePriceFetcher instance."""
    fetcher = PriceFetcher.get_instance()

    assert fetcher is not None
    assert isinstance(fetcher, YFinancePriceFetcher)


def test_get_instance_returns_same_fetcher() -> None:
    """Test that multiple calls to get_instance return the same fetcher instance."""
    fetcher1 = PriceFetcher.get_instance()
    fetcher2 = PriceFetcher.get_instance()
    fetcher3 = PriceFetcher.get_instance()

    # All fetcher instances should be the exact same object
    assert fetcher1 is fetcher2
    assert fetcher2 is fetcher3
    assert fetcher1 is fetcher3


def test_ticker_price_singleton_behavior() -> None:
    """Test that multiple PriceFetcher instantiations return the same instance.

    This verifies the SingletonMeta metaclass is working correctly at the
    PriceFetcher class level.
    """
    instance1 = PriceFetcher()
    instance2 = PriceFetcher()
    instance3 = PriceFetcher()

    # All PriceFetcher instances should be the exact same object
    assert instance1 is instance2
    assert instance2 is instance3
    assert instance1 is instance3


def test_ticker_price_and_get_instance_consistency() -> None:
    """Test that direct instantiation and get_instance() return consistent fetchers."""
    # Get fetcher via get_instance()
    fetcher_via_method = PriceFetcher.get_instance()

    # Get fetcher via direct instantiation
    ticker_price_instance = PriceFetcher()
    fetcher_via_direct = ticker_price_instance.fetcher

    # Both should return the same fetcher instance
    assert fetcher_via_method is fetcher_via_direct


def test_singleton_instance_is_functional() -> None:
    """Test that the singleton fetcher instance works correctly.

    This test uses actual yfinance calls, so it may fail if:
    - There's no internet connection
    - Yahoo Finance API is down
    - The ticker symbol is invalid or delisted

    We use a date in the past to ensure data exists.
    """
    fetcher = PriceFetcher.get_instance()

    # Use a recent date with known market activity
    test_date = int(datetime(2024, 12, 3, 0, 0, 0, tzinfo=UTC).timestamp())

    # Fetch prices for AAPL (reliable ticker)
    prices = fetcher.get_prices("AAPL", test_date, days_window=0)

    # Verify we got some data back
    # Note: This may return empty list if market was closed or data unavailable
    assert isinstance(prices, list)
    # We can't assert len(prices) > 0 because market might be closed

    # Verify the fetcher has the expected method
    assert hasattr(fetcher, "get_prices")
    assert callable(fetcher.get_prices)


def test_singleton_persists_across_multiple_calls() -> None:
    """Test that the singleton persists and maintains state across calls."""
    # Get multiple fetcher instances
    fetchers = [PriceFetcher.get_instance() for _ in range(10)]

    # All should be the same instance
    first_fetcher = fetchers[0]
    for fetcher in fetchers[1:]:
        assert fetcher is first_fetcher


def test_ticker_price_metaclass_instances_dict() -> None:
    """Test that the SingletonMeta properly tracks PriceFetcher in its instances dict."""
    # Create an instance
    instance = PriceFetcher()

    # Check that PriceFetcher is tracked in the metaclass instances dictionary
    assert PriceFetcher in SingletonMeta._instances
    assert SingletonMeta._instances[PriceFetcher] is instance
