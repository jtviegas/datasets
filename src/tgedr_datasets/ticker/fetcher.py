"""Tickers symbol fetcher for retrieving lists of available stock tickers.

This module provides the Tickers class that retrieves ticker symbols
from various sources including S&P 500, NASDAQ, NYSE, and other major indices.
"""

import logging
from io import StringIO
from typing import Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class TickerSourceError(Exception):  # noqa: D101   # pragma: no cover
    pass


TickerSource = Literal["sp500", "nasdaq100", "dowjones", "russell1000"]


class TickerFetcher:
    """Fetcher for retrieving lists of stock ticker symbols.

    This class provides methods to fetch ticker symbols from various sources
    including major market indices. The primary method is fetch() which
    retrieves tickers from a specified source.

    Sources:
        - sp500: S&P 500 companies
        - nasdaq100: NASDAQ-100 companies
        - dowjones: Dow Jones Industrial Average companies
        - russell1000: Russell 1000 companies

    Example:
        >>> fetcher = TickerFetcher()
        >>> sp500_tickers = fetcher.fetch(source="sp500")
        >>> print(f"Found {len(sp500_tickers)} S&P 500 tickers")

    """

    def __init__(self) -> None:
        """Initialize the Tickers fetcher."""
        self._sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        self._nasdaq100_url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
        self._dowjones_url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        self._russell1000_url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def _fetch_html_tables(self, url: str) -> list[pd.DataFrame]:
        """Fetch HTML tables from a URL with proper headers.

        Args:
            url: The URL to fetch tables from.

        Returns:
            list[pd.DataFrame]: List of DataFrames containing the HTML tables.

        Raises:
            Exception: If the request fails or parsing fails.

        """
        response = requests.get(url, headers=self._headers, timeout=10)
        response.raise_for_status()
        return pd.read_html(StringIO(response.text))

    def fetch(self, source: TickerSource | list[TickerSource] = "sp500") -> list[str]:
        """Fetch ticker symbols from the specified source(s).

        Retrieves a list of ticker symbols from various market indices and
        sources. The default source is the S&P 500. If a list of sources is
        provided, returns combined unique tickers from all specified sources.

        Args:
            source: The source(s) to fetch tickers from. Can be a single source
                   or a list of sources. Options:
                   - "sp500": S&P 500 companies (default)
                   - "nasdaq100": NASDAQ-100 companies
                   - "dowjones": Dow Jones Industrial Average companies
                   - "russell1000": Russell 1000 companies

        Returns:
            list[str]: List of unique ticker symbols, sorted alphabetically.
                      If multiple sources are specified, returns deduplicated
                      combined tickers from all sources.

        Example:
            >>> fetcher = Tickers()
            >>> # Fetch S&P 500 tickers
            >>> tickers = fetcher.fetch(source="sp500")
            >>> # Fetch from multiple sources
            >>> combined = fetcher.fetch(source=["sp500", "nasdaq100"])
            >>> # Fetch NASDAQ-100 tickers
            >>> nasdaq_tickers = fetcher.fetch(source="nasdaq100")

        """
        # Handle list of sources
        if isinstance(source, list):
            all_tickers = set()
            for src in source:
                tickers = self._fetch_single_source(src)
                all_tickers.update(tickers)
            return sorted(all_tickers)

        # Handle single source
        return self._fetch_single_source(source)

    def _fetch_single_source(self, source: TickerSource) -> list[str]:
        """Fetch ticker symbols from a single source.

        Args:
            source: The source to fetch tickers from.

        Returns:
            list[str]: List of ticker symbols.

        Raises:
            ValueError: If the source is unknown.

        """
        if source == "sp500":
            return self._fetch_sp500()
        if source == "nasdaq100":
            return self._fetch_nasdaq100()
        if source == "dowjones":
            return self._fetch_dowjones()
        if source == "russell1000":
            return self._fetch_russell1000()

        raise ValueError(f"Unknown source: {source}")  # noqa: EM102, TRY003

    def _fetch_sp500(self) -> list[str]:
        """Fetch S&P 500 ticker symbols from Wikipedia.

        Returns:
            list[str]: List of S&P 500 ticker symbols.

        """
        logger.info("Fetching S&P 500 tickers from Wikipedia")

        # Read the first table from the Wikipedia page
        tables = self._fetch_html_tables(self._sp500_url)
        sp500_table = tables[0]

        # Extract ticker symbols (column name is 'Symbol')
        tickers = sp500_table["Symbol"].tolist()

        # Clean up ticker symbols (some may have notes or formatting)
        tickers = [ticker.strip().replace(".", "-") for ticker in tickers if isinstance(ticker, str)]

        logger.info("Successfully fetched %d S&P 500 tickers", len(tickers))
        return tickers

    def _fetch_nasdaq100(self) -> list[str]:
        """Fetch NASDAQ-100 ticker symbols from Wikipedia.

        Returns:
            list[str]: List of NASDAQ-100 ticker symbols.

        """
        logger.info("Fetching NASDAQ-100 tickers from Wikipedia")

        # Read tables from the Wikipedia page
        tables = self._fetch_html_tables(self._nasdaq100_url)

        # The NASDAQ-100 table is typically the 4th table (index 3)
        # but we'll search for the one with 'Ticker' column
        nasdaq_table = None
        for table in tables:
            if "Ticker" in table.columns:
                nasdaq_table = table
                break

        if nasdaq_table is None:
            msg = "Could not find NASDAQ-100 ticker table"
            raise TickerSourceError(msg)

        # Extract ticker symbols
        tickers = nasdaq_table["Ticker"].tolist()

        # Clean up ticker symbols
        tickers = [ticker.strip() for ticker in tickers if isinstance(ticker, str) and ticker.strip()]

        logger.info("Successfully fetched %d NASDAQ-100 tickers", len(tickers))
        return tickers

    def _fetch_dowjones(self) -> list[str]:
        """Fetch Dow Jones Industrial Average ticker symbols from Wikipedia.

        Returns:
            list[str]: List of Dow Jones ticker symbols.

        """
        logger.info("Fetching Dow Jones tickers from Wikipedia")

        # Read tables from the Wikipedia page
        tables = self._fetch_html_tables(self._dowjones_url)

        # Find the table with 'Symbol' column
        dowjones_table = None
        for table in tables:
            if "Symbol" in table.columns:
                dowjones_table = table
                break

        if dowjones_table is None:
            msg = "Could not find Dow Jones ticker table"
            raise TickerSourceError(msg)

        # Extract ticker symbols
        tickers = dowjones_table["Symbol"].tolist()

        # Clean up ticker symbols
        tickers = [ticker.strip() for ticker in tickers if isinstance(ticker, str) and ticker.strip()]

        logger.info("Successfully fetched %d Dow Jones tickers", len(tickers))
        return tickers

    def _fetch_russell1000(self) -> list[str]:
        """Fetch Russell 1000 ticker symbols from Wikipedia.

        Note: The Russell 1000 Wikipedia page may not have a complete ticker list.
        This method attempts to extract available tickers but may return fewer
        than the full 1000 companies.

        Returns:
            list[str]: List of Russell 1000 ticker symbols (may be incomplete).

        """
        logger.info("Fetching Russell 1000 tickers from Wikipedia")
        logger.warning("Russell 1000 ticker list from Wikipedia may be incomplete")

        # Read tables from the Wikipedia page
        tables = self._fetch_html_tables(self._russell1000_url)

        # Try to find a table with ticker information
        # The Wikipedia page structure varies, so we'll look for common column names
        russell_table = None
        ticker_column = None

        for table in tables:
            for col_name in ["Ticker", "Symbol", "Ticker symbol"]:
                if col_name in table.columns:
                    russell_table = table
                    ticker_column = col_name
                    break
            if russell_table is not None:
                break

        if russell_table is None or ticker_column is None:
            msg = "Could not find Russell 1000 ticker table with ticker symbols"
            raise TickerSourceError(msg)

        tickers = russell_table[ticker_column].tolist()

        # Clean up ticker symbols
        tickers = [ticker.strip() for ticker in tickers if isinstance(ticker, str) and ticker.strip()]

        logger.info("Successfully fetched %d Russell 1000 tickers", len(tickers))
        if len(tickers) < 1000:
            logger.warning("Fetched fewer than 1000 tickers (%d) - Wikipedia data may be incomplete", len(tickers))

        return tickers

    def fetch_all(self) -> dict[str, list[str]]:
        """Fetch tickers from all available sources.

        Attempts to fetch tickers from all supported sources and returns
        a dictionary with source names as keys and ticker lists as values.

        Returns:
            dict[str, list[str]]: Dictionary mapping source names to ticker lists.
                                 Sources with failed fetches will have empty lists.

        Example:
            >>> fetcher = Tickers()
            >>> all_tickers = fetcher.fetch_all()
            >>> for source, tickers in all_tickers.items():
            ...     print(f"{source}: {len(tickers)} tickers")

        """
        logger.info("Fetching tickers from all sources")

        results = {
            "sp500": self.fetch(source="sp500"),
            "nasdaq100": self.fetch(source="nasdaq100"),
            "dowjones": self.fetch(source="dowjones"),
            "russell1000": self.fetch(source="russell1000"),
        }

        total_unique = len({ticker for tickers in results.values() for ticker in tickers})
        logger.info("Fetched total of %d unique tickers across all sources", total_unique)

        return results

    def fetch_combined(self) -> list[str]:
        """Fetch and combine unique tickers from all sources.

        Retrieves tickers from all available sources and returns a deduplicated
        list of unique ticker symbols.

        Returns:
            list[str]: Sorted list of unique ticker symbols from all sources.

        Example:
            >>> fetcher = Tickers()
            >>> all_unique_tickers = fetcher.fetch_combined()
            >>> print(f"Total unique tickers: {len(all_unique_tickers)}")

        """
        logger.info("Fetching and combining tickers from all sources")

        all_tickers = self.fetch_all()
        unique_tickers = set()

        for source, tickers in all_tickers.items():
            unique_tickers.update(tickers)
            logger.debug("Added %d tickers from %s", len(tickers), source)

        sorted_tickers = sorted(unique_tickers)
        logger.info("Combined total: %d unique tickers", len(sorted_tickers))

        return sorted_tickers
