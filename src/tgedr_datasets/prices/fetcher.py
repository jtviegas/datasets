"""Abstract base class for stock price data fetchers.

This module provides the Fetcher abstract base class that defines the interface
for fetching stock price data from various data sources.
"""
from abc import ABC, abstractmethod
from tgedr_datasets.prices.price import Price


class Fetcher(ABC):
    """Abstract base class for fetching stock price data.

    This class defines the interface that all concrete price fetcher implementations
    must follow. It enforces a consistent API across different data sources while
    allowing each implementation to handle its specific data source requirements.

    The Open-Closed Principle is applied here: this interface is closed for
    modification but open for extension through concrete implementations.
    """

    @abstractmethod
    def get_prices(
        self,
        ticker: str,
        date: int,
        days_window: int = 0,
    ) -> list[Price]:
        """Fetch daily price data for a given stock ticker around a specific date.

        This abstract method must be implemented by all concrete fetcher classes.
        Each implementation will fetch price data from its specific data source.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "GOOGL")
            date: Reference date as UTC epoch timestamp (seconds). This is the
                 center point for the data retrieval.
            days_window: Number of days to extend the window. Default is 0.
                        - If 0: Retrieves daily prices for the single day
                        - If positive: Extends window forward to more recent dates
                        - If negative: Extends window backward to past dates
                        The window extends symmetrically around the date parameter.

        Returns:
            List of Price objects sorted by timestamp, or empty list on error.
            Returns daily OHLCV data.

        Raises:
            NotImplementedError: If called directly on the abstract base class.

        """
        raise NotImplementedError
