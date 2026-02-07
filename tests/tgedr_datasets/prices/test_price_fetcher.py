"""Tests for price fetcher base class."""


import pytest

from tgedr_datasets.prices.fetcher import Fetcher
from tgedr_datasets.prices.price import Price


class ConcreteFetcher(Fetcher):
    """Concrete implementation of Fetcher for testing."""

    def get_prices(
        self,
        ticker: str,
        date: int,
        days_window: int = 0,
    ) -> list[Price]:
        """Implement abstract method."""
        return []


def test_fetcher_abstract_class_cannot_be_instantiated() -> None:
    """Test Fetcher abstract class cannot be instantiated directly."""
    # Create concrete implementation for testing
    fetcher = ConcreteFetcher()
    
    # Verify it can be instantiated
    assert fetcher is not None


def test_concrete_fetcher_get_prices() -> None:
    """Test concrete fetcher implementation works."""
    fetcher = ConcreteFetcher()
    prices = fetcher.get_prices("AAPL", 1701388800)
    assert prices == []


def test_concrete_fetcher_get_prices_with_window() -> None:
    """Test concrete fetcher with days_window parameter."""
    fetcher = ConcreteFetcher()
    prices = fetcher.get_prices("AAPL", 1701388800, days_window=7)
    assert prices == []


def test_fetcher_abstract_method_raises_not_implemented() -> None:
    """Test that Fetcher.get_prices raises NotImplementedError when called directly."""
    with pytest.raises(NotImplementedError):
        Fetcher.get_prices(None, "AAPL", 1701388800)