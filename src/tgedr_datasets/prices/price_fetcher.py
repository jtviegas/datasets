"""Singleton factory for obtaining price fetcher instances.

This module provides the PriceFetcher singleton class that serves as a
centralized factory for obtaining price fetcher implementations. Currently
returns a shared YFinancePriceFetcher instance.
"""
from tgedr_pycommons.utils.singleton import SingletonMeta

from tgedr_datasets.prices.yfinance_price_fetcher import YFinancePriceFetcher


class PriceFetcher(metaclass=SingletonMeta):
    """Singleton factory for obtaining price fetcher instances.

    This class provides a centralized way to obtain a price fetcher instance
    without needing to manage instantiation. It ensures a single shared instance
    of YFinancePriceFetcher is used throughout the application, reducing overhead
    and maintaining consistency.

    The singleton behavior is implemented via the SingletonMeta metaclass, which
    ensures that multiple calls to PriceFetcher() return the same instance.

    Usage:
        >>> fetcher = PriceFetcher.get_instance()
        >>> prices = fetcher.get_prices("AAPL", timestamp, days_window=5)

    Note:
        While this class uses a metaclass to ensure singleton behavior, the
        recommended way to obtain a price fetcher is through the get_instance()
        class method for clarity and consistency.

    """

    def __init__(self) -> None:
        """Initialize the PriceFetcher singleton with a YFinancePriceFetcher instance.

        This method is called only once due to the SingletonMeta metaclass.
        Subsequent instantiations return the existing instance without calling __init__.
        """
        self.fetcher = YFinancePriceFetcher()

    @classmethod
    def get_instance(cls) -> YFinancePriceFetcher:
        """Get the singleton instance of the price fetcher.

        Returns a shared YFinancePriceFetcher instance. On first call, creates
        the PriceFetcher singleton and its internal fetcher; subsequent calls
        return the same fetcher instance.

        Returns:
            YFinancePriceFetcher: The singleton price fetcher instance.

        Example:
            >>> fetcher = PriceFetcher.get_instance()
            >>> prices = fetcher.get_prices("AAPL", 1701388800)
            >>> # Multiple calls return the same fetcher
            >>> fetcher2 = PriceFetcher.get_instance()
            >>> fetcher is fetcher2
            True

        """
        return cls().fetcher
