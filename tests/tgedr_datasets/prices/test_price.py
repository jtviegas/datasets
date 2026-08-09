"""Unit tests for Price class."""

from datetime import datetime, UTC

import pytest

from tgedr_datasets.prices.price import Price



@pytest.fixture
def valid_price() -> Price:
    """Provide a valid Price instance."""
    return Price(
        ticker="AAPL",
        timestamp=1701388800,
        open=180.50,
        high=182.75,
        low=179.80,
        close=181.25,
        volume=50000000,
    )


def test_price_creation(valid_price: Price) -> None:
    """Test basic Price instantiation."""
    assert valid_price.ticker == "AAPL"
    assert valid_price.timestamp == 1701388800
    assert valid_price.open == 180.50
    assert valid_price.high == 182.75
    assert valid_price.low == 179.80
    assert valid_price.close == 181.25
    assert valid_price.volume == 50000000


def test_price_is_immutable(valid_price: Price) -> None:
    """Test Price is immutable (frozen dataclass)."""
    with pytest.raises(AttributeError):
        valid_price.open = 200.0  # type: ignore[misc]


def test_price_id_property(valid_price: Price) -> None:
    """Test Price.id property generates unique identifier."""
    id1 = valid_price.id
    id2 = valid_price.id
    assert id1 == id2  # Should be consistent

    # Different ticker should have different id
    price2 = Price(
        ticker="MSFT",
        timestamp=1701388800,
        open=180.50,
        high=182.75,
        low=179.80,
        close=181.25,
        volume=50000000,
    )
    assert price2.id != valid_price.id


def test_price_id_different_timestamp(valid_price: Price) -> None:
    """Test Price.id differs with different timestamp."""
    price2 = Price(
        ticker="AAPL",
        timestamp=1701388801,  # Different timestamp
        open=180.50,
        high=182.75,
        low=179.80,
        close=181.25,
        volume=50000000,
    )
    assert price2.id != valid_price.id


def test_price_from_dict(valid_price: Price) -> None:
    """Test creating Price from dictionary."""
    data = {
        "ticker": "AAPL",
        "timestamp": 1701388800,
        "open": 180.50,
        "high": 182.75,
        "low": 179.80,
        "close": 181.25,
        "volume": 50000000,
    }
    price = Price.from_dict(data)
    assert price == valid_price


def test_price_to_pd_df_row(valid_price: Price) -> None:
    """Test converting Price to pandas DataFrame row dictionary."""
    row = valid_price.to_pd_df_row()

    # Verify all fields are present
    assert "id" in row
    assert "ticker" in row
    assert "timestamp" in row
    assert "open" in row
    assert "high" in row
    assert "low" in row
    assert "close" in row
    assert "volume" in row

    # Verify values
    assert row["ticker"] == "AAPL"
    assert row["timestamp"] == 1701388800
    assert row["open"] == 180.50
    assert row["high"] == 182.75
    assert row["low"] == 179.80
    assert row["close"] == 181.25
    assert row["volume"] == 50000000
    assert row["id"] == valid_price.id


def test_price_negative_open_raises_error() -> None:
    """Test negative open price raises ValueError."""
    with pytest.raises(ValueError, match="open price must be non-negative"):
        Price(
            ticker="AAPL",
            timestamp=1701388800,
            open=-100.0,
            high=182.75,
            low=179.80,
            close=181.25,
            volume=50000000,
        )


def test_price_negative_high_raises_error() -> None:
    """Test negative high price raises ValueError."""
    with pytest.raises(ValueError, match="high price must be non-negative"):
        Price(
            ticker="AAPL",
            timestamp=1701388800,
            open=180.50,
            high=-182.75,
            low=179.80,
            close=181.25,
            volume=50000000,
        )


def test_price_negative_low_raises_error() -> None:
    """Test negative low price raises ValueError."""
    with pytest.raises(ValueError, match="low price must be non-negative"):
        Price(
            ticker="AAPL",
            timestamp=1701388800,
            open=180.50,
            high=182.75,
            low=-179.80,
            close=181.25,
            volume=50000000,
        )


def test_price_negative_close_raises_error() -> None:
    """Test negative close price raises ValueError."""
    with pytest.raises(ValueError, match="close price must be non-negative"):
        Price(
            ticker="AAPL",
            timestamp=1701388800,
            open=180.50,
            high=182.75,
            low=179.80,
            close=-181.25,
            volume=50000000,
        )


def test_price_negative_volume_raises_error() -> None:
    """Test negative volume raises ValueError."""
    with pytest.raises(ValueError, match="volume must be non-negative"):
        Price(
            ticker="AAPL",
            timestamp=1701388800,
            open=180.50,
            high=182.75,
            low=179.80,
            close=181.25,
            volume=-50000000,
        )


def test_price_high_less_than_low_logs_warning(caplog) -> None:
    """Test high < low logs warning instead of raising error."""
    import logging
    caplog.set_level(logging.INFO)
    
    price = Price(
        ticker="AAPL",
        timestamp=1701388800,
        open=180.50,
        high=175.0,  # Less than low
        low=179.80,
        close=181.25,
        volume=50000000,
    )
    
    # Price should be created successfully
    assert price.high == 175.0
    assert price.low == 179.80
    
    # Should log warning message
    assert "high price" in caplog.text
    assert "must be >= low price" in caplog.text


def test_price_high_less_than_open_logs_warning(caplog) -> None:
    """Test high < open logs warning instead of raising error."""
    import logging
    caplog.set_level(logging.INFO)
    
    price = Price(
        ticker="AAPL",
        timestamp=1701388800,
        open=180.50,
        high=179.0,  # Less than open
        low=178.0,  # Valid low price
        close=179.0,
        volume=50000000,
    )
    
    # Price should be created successfully
    assert price.high == 179.0
    assert price.open == 180.50
    
    # Should log warning message
    assert "high price" in caplog.text
    assert "must be >= open price" in caplog.text


def test_price_high_less_than_close_logs_warning(caplog) -> None:
    """Test high < close logs warning instead of raising error."""
    import logging
    caplog.set_level(logging.INFO)
    
    price = Price(
        ticker="AAPL",
        timestamp=1701388800,
        open=180.0,
        high=180.0,  # Less than close
        low=179.0,  # Valid low price
        close=181.25,
        volume=50000000,
    )
    
    # Price should be created successfully
    assert price.high == 180.0
    assert price.close == 181.25
    
    # Should log warning message
    assert "high price" in caplog.text
    assert "must be >= close price" in caplog.text


def test_price_low_greater_than_open_logs_warning(caplog) -> None:
    """Test low > open logs warning instead of raising error."""
    import logging
    caplog.set_level(logging.INFO)
    
    price = Price(
        ticker="AAPL",
        timestamp=1701388800,
        open=180.50,
        high=182.75,
        low=181.0,  # Greater than open
        close=181.25,
        volume=50000000,
    )
    
    # Price should be created successfully
    assert price.low == 181.0
    assert price.open == 180.50
    
    # Should log warning message
    assert "low price" in caplog.text
    assert "must be <= open price" in caplog.text


def test_price_low_greater_than_close_logs_warning(caplog) -> None:
    """Test low > close logs warning instead of raising error."""
    import logging
    caplog.set_level(logging.INFO)
    
    price = Price(
        ticker="AAPL",
        timestamp=1701388800,
        open=182.0,
        high=182.75,
        low=181.25,  # Greater than close
        close=180.50,
        volume=50000000,
    )
    
    # Price should be created successfully
    assert price.low == 181.25
    assert price.close == 180.50
    
    # Should log warning message
    assert "low price" in caplog.text
    assert "must be <= close price" in caplog.text


def test_price_equality(valid_price: Price) -> None:
    """Test Price equality comparison."""
    price2 = Price(
        ticker="AAPL",
        timestamp=1701388800,
        open=180.50,
        high=182.75,
        low=179.80,
        close=181.25,
        volume=50000000,
    )
    assert valid_price == price2


def test_price_inequality(valid_price: Price) -> None:
    """Test Price inequality comparison."""
    price2 = Price(
        ticker="MSFT",  # Different ticker
        timestamp=1701388800,
        open=180.50,
        high=182.75,
        low=179.80,
        close=181.25,
        volume=50000000,
    )
    assert valid_price != price2

def test_price_hash_equal_for_equal_prices(valid_price: Price) -> None:
    """Test that equal prices have the same hash."""
    price2 = Price(
        ticker="AAPL",
        timestamp=valid_price.timestamp,
        open=180.50,
        high=182.75,
        low=179.80,
        close=181.25,
        volume=50000000,
    )
    assert hash(valid_price) == hash(price2)


def test_price_hash_differs_for_different_prices(valid_price: Price) -> None:
    """Test that different prices have different hashes."""
    price2 = Price(
        ticker="MSFT",  # Different ticker
        timestamp=valid_price.timestamp,
        open=180.50,
        high=182.75,
        low=179.80,
        close=181.25,
        volume=50000000,
    )
    assert hash(valid_price) != hash(price2)


def test_price_hashable_in_set(valid_price: Price) -> None:
    """Test that Price instances can be used in a set (hashable)."""
    price2 = Price(
        ticker="AAPL",
        timestamp=valid_price.timestamp,
        open=180.50,
        high=182.75,
        low=179.80,
        close=181.25,
        volume=50000000,
    )
    s = {valid_price, price2}
    assert len(s) == 1  # Equal prices collapse to one element


def test_price_equality_with_non_price(valid_price: Price) -> None:
    """Test Price equality against a non-Price object returns NotImplemented."""
    assert valid_price.__eq__("not a price") is NotImplemented

def test_price_zero_volume() -> None:
    """Test Price with zero volume is valid."""
    price = Price(
        ticker="AAPL",
        timestamp=1701388800,
        open=180.50,
        high=182.75,
        low=179.80,
        close=181.25,
        volume=0,
    )
    assert price.volume == 0


def test_price_zero_open_high_low_close() -> None:
    """Test Price with all zero prices is valid."""
    price = Price(
        ticker="AAPL",
        timestamp=1701388800,
        open=0.0,
        high=0.0,
        low=0.0,
        close=0.0,
        volume=0,
    )
    assert price.open == 0.0
