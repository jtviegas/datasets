"""Yahoo Finance price fetcher implementation.

This module provides a concrete implementation of TickerPriceFetcher
using the yfinance library to fetch stock price data from Yahoo Finance.
"""

import logging
from datetime import datetime, UTC, timedelta
import yfinance as yf
from tgedr_datasets.prices.fetcher import Fetcher
from tgedr_datasets.prices.price import Price



logger = logging.getLogger(__name__)


class YFinancePriceFetcher(Fetcher):
    """Fetches stock price data from Yahoo Finance using yfinance library.

    This fetcher retrieves daily OHLCV (Open, High, Low, Close, Volume) data for stocks
    from Yahoo Finance.

    The fetcher handles the conversion between epoch timestamps and datetime objects,
    and formats the data into PriceData objects for consistent handling across
    different data sources.
    """

    __DAY_INTERVAL = "1d"

    def get_prices(
        self,
        ticker: str,
        date: int,
        days_window: int = 0,
    ) -> list[Price]:
        """Fetch daily price data from Yahoo Finance.

        Retrieves daily OHLCV (Open, High, Low, Close, Volume) data for the specified
        ticker around the given date. Always returns daily aggregated data, not intraday.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "GOOGL")
            date: Reference date as UTC epoch timestamp (seconds)
            days_window: Number of days to extend the window (default: 0)
                        - 0: Single day of daily data
                        - Positive: Extends forward in time (more recent dates)
                        - Negative: Extends backward in time (past dates)

        Returns:
            List of PriceData objects with daily price data, sorted by timestamp.
            Returns empty list on error.

        """
        logger.debug("[get_prices|in] => ticker=%s, date=%d, days_window=%d",
                    ticker, date, days_window)

        try:
            # Convert epoch timestamp to datetime
            reference_dt = datetime.fromtimestamp(date, tz=UTC)

            # Calculate date range for daily data
            start_dt = reference_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt = start_dt + timedelta(days=1)

            if days_window < 0:
                start_dt = end_dt + timedelta(days=days_window)
            elif days_window > 0:
                end_dt = start_dt + timedelta(days=days_window)

            logger.debug("Fetching daily data (%s) from %s to %s",
                        self.__DAY_INTERVAL, start_dt, end_dt)

            # Fetch daily data from yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_dt, end=end_dt, interval=self.__DAY_INTERVAL)  # noqa: PD901

            if df.empty:
                logger.warning("No price data found for %s", ticker)
                return []

            # Convert DataFrame to PriceData objects
            prices = []
            for index, row in df.iterrows():
                # Convert pandas Timestamp to epoch
                timestamp = int(index.timestamp())

                price_data = Price(
                    ticker=ticker,
                    timestamp=timestamp,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                )
                prices.append(price_data)

            logger.info("Found %d price points for %s", len(prices), ticker)
            logger.debug("[get_prices|out] => %d price points", len(prices))
            return prices  # noqa: TRY300

        except Exception as e:
            msg = f"Error fetching prices from Yahoo Finance: {e}"
            logger.exception(msg)
            return []
