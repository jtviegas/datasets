"""Price data model for stock prices.

This module defines the Price dataclass used to represent stock price
information at a specific point in time.
"""

from dataclasses import dataclass
import hashlib
import logging


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Price:
    """Represents stock price data at a specific timestamp.

    This immutable dataclass holds price information for a stock at a particular
    point in time, including OHLCV (Open, High, Low, Close, Volume) data.

    Attributes:
        ticker: Stock ticker symbol (e.g., "AAPL", "GOOGL")
        timestamp: Unix epoch timestamp in seconds (UTC)
        open: Opening price for the period
        high: Highest price during the period
        low: Lowest price during the period
        close: Closing price for the period
        volume: Trading volume during the period
        id: Unique identifier composed of ticker hash concatenated with timestamp

    """

    ticker: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        """Validate price data after initialization.

        Raises:
            ValueError: If price data violates basic market data constraints

        """
        # Validate non-negative prices
        if self.open < 0:
            msg = f"open price must be non-negative, got {self.open}"
            raise ValueError(msg)
        if self.high < 0:
            msg = f"high price must be non-negative, got {self.high}"
            raise ValueError(msg)
        if self.low < 0:
            msg = f"low price must be non-negative, got {self.low}"
            raise ValueError(msg)
        if self.close < 0:
            msg = f"close price must be non-negative, got {self.close}"
            raise ValueError(msg)

        # Validate volume is non-negative
        if self.volume < 0:
            msg = f"volume must be non-negative, got {self.volume}"
            raise ValueError(msg)

        # Validate OHLC relationships
        if self.high < self.low:
            msg = f"[__post_init] {self.ticker}: high price ({self.high}) must be >= low price ({self.low})"
            logger.info(msg)
        if self.high < self.open:
            msg = f"[__post_init] {self.ticker}: high price ({self.high}) must be >= open price ({self.open})"
            logger.info(msg)
        if self.high < self.close:
            msg = f"[__post_init] {self.ticker}: high price ({self.high}) must be >= close price ({self.close})"
            logger.info(msg)
        if self.low > self.open:
            msg = f"[__post_init] {self.ticker}: low price ({self.low}) must be <= open price ({self.open})"
            logger.info(msg)
        if self.low > self.close:
            msg = f"[__post_init] {self.ticker}: low price ({self.low}) must be <= close price ({self.close})"
            logger.info(msg)

    @property
    def id(self) -> int:
        """Generate a unique identifier from ticker and timestamp.

        Combines the hash of the ticker symbol with the timestamp to create
        a unique integer identifier for this price data point. Uses Python's
        built-in hash on a tuple to ensure consistent and collision-resistant IDs.

        Returns:
            int: Unique identifier as integer

        """
        # Create deterministic hash using SHA-256
        hash_input = f"{self.ticker}:{self.timestamp}".encode()
        hash_digest = hashlib.sha256(hash_input).hexdigest()
        # Convert first 16 characters of hex to integer (64 bits)
        return int(hash_digest[:16], 16)

    @staticmethod
    def from_dict(data: dict[str, object]) -> "Price":
        """Create a Price instance from a dictionary.

        Args:
            data: Dictionary with keys matching Price attributes:
                  'ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume'.

        Returns:
            Price: New Price instance populated from the dictionary.

        """
        return Price(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
        )

    def to_pd_df_row(self) -> dict[str, object]:
        """Convert Price instance to a dictionary suitable for pandas DataFrame row.

        Returns:
            dict[str, object]: Dictionary with all Price fields including the id property.
                              Keys match the attribute names and can be used directly with
                              pandas DataFrame constructor.

        Example:
            >>> price = Price(
            ...     ticker="AAPL",
            ...     timestamp=1701388800,
            ...     open=180.50,
            ...     high=182.75,
            ...     low=179.80,
            ...     close=181.25,
            ...     volume=50000000
            ... )
            >>> row = price.to_pd_df_row()
            >>> import pandas as pd
            >>> df = pd.DataFrame([row])

        """
        return {
            "id": self.id,
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
