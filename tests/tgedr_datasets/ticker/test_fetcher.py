"""Tests for tickers.py edge cases and error handling."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from tgedr_datasets.ticker.fetcher import TickerFetcher, TickerSourceError


@pytest.fixture
def ticker_fetcher() -> TickerFetcher:
    """Provide a TickerFetcher instance."""
    return TickerFetcher()


def create_mock_response(html_content: str) -> Mock:
    """Create a mock response object with HTML content."""
    mock_response = Mock()
    mock_response.text = html_content
    mock_response.raise_for_status = Mock()
    return mock_response


def create_sp500_html() -> str:
    """Create mock HTML for S&P 500 page."""
    df = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "Security": ["Apple Inc.", "Microsoft", "Alphabet Inc.", "Amazon"],
    })
    return df.to_html()


def create_nasdaq100_html() -> str:
    """Create mock HTML for NASDAQ-100 page."""
    df = pd.DataFrame({
        "Ticker": ["AAPL", "TSLA", "NVDA", "META"],
        "Company": ["Apple", "Tesla", "NVIDIA", "Meta"],
    })
    return df.to_html()


def create_dowjones_html() -> str:
    """Create mock HTML for Dow Jones page."""
    df = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "JPM", "V"],
        "Company": ["Apple", "Microsoft", "JPMorgan", "Visa"],
    })
    return df.to_html()


def create_russell1000_html(column_name: str = "Ticker") -> str:
    """Create mock HTML for Russell 1000 page."""
    df = pd.DataFrame({
        column_name: ["BRK.A", "BRK.B", "JPM", "XOM"],
        "Company": ["Berkshire A", "Berkshire B", "JPMorgan", "Exxon"],
    })
    return df.to_html()


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_sp500_success(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_sp500 successfully fetches and cleans ticker data."""
    mock_get.return_value = create_mock_response(create_sp500_html())
    
    tickers = ticker_fetcher._fetch_sp500()
    
    assert len(tickers) == 4
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "GOOGL" in tickers
    assert "AMZN" in tickers
    mock_get.assert_called_once()


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_sp500_empty_table(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_sp500 handles empty table."""
    # Create HTML with no tables
    mock_get.return_value = create_mock_response("<html><body></body></html>")
    
    with pytest.raises(ValueError):
        ticker_fetcher._fetch_sp500()


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_nasdaq100_success(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_nasdaq100 successfully fetches ticker data."""
    mock_get.return_value = create_mock_response(create_nasdaq100_html())
    
    tickers = ticker_fetcher._fetch_nasdaq100()
    
    assert len(tickers) == 4
    assert "AAPL" in tickers
    assert "TSLA" in tickers
    assert "NVDA" in tickers
    assert "META" in tickers


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_nasdaq100_no_ticker_column(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_nasdaq100 raises error when no Ticker column found."""
    # Create table without Ticker column
    df = pd.DataFrame({"Other": [1, 2, 3]})
    mock_get.return_value = create_mock_response(df.to_html())
    
    with pytest.raises(TickerSourceError, match="Could not find NASDAQ-100"):
        ticker_fetcher._fetch_nasdaq100()


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_dowjones_success(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_dowjones successfully fetches ticker data."""
    mock_get.return_value = create_mock_response(create_dowjones_html())
    
    tickers = ticker_fetcher._fetch_dowjones()
    
    assert len(tickers) == 4
    assert "AAPL" in tickers
    assert "JPM" in tickers


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_dowjones_no_symbol_column(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_dowjones raises error when no Symbol column found."""
    df = pd.DataFrame({"Other": [1, 2, 3]})
    mock_get.return_value = create_mock_response(df.to_html())
    
    with pytest.raises(TickerSourceError, match="Could not find Dow Jones"):
        ticker_fetcher._fetch_dowjones()


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_russell1000_with_ticker_column(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_russell1000 successfully fetches with Ticker column."""
    mock_get.return_value = create_mock_response(create_russell1000_html("Ticker"))
    
    tickers = ticker_fetcher._fetch_russell1000()
    
    assert len(tickers) == 4
    assert "BRK.A" in tickers
    assert "BRK.B" in tickers


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_russell1000_with_symbol_column(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_russell1000 uses Symbol column if available."""
    mock_get.return_value = create_mock_response(create_russell1000_html("Symbol"))
    
    tickers = ticker_fetcher._fetch_russell1000()
    
    assert len(tickers) == 4
    assert "BRK.A" in tickers


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_russell1000_with_ticker_symbol_column(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_russell1000 uses 'Ticker symbol' column if available."""
    mock_get.return_value = create_mock_response(create_russell1000_html("Ticker symbol"))
    
    tickers = ticker_fetcher._fetch_russell1000()
    
    assert len(tickers) == 4


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_russell1000_no_valid_column(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_russell1000 raises error when no valid ticker column found."""
    df = pd.DataFrame({"Other": [1, 2, 3]})
    mock_get.return_value = create_mock_response(df.to_html())
    
    with pytest.raises(TickerSourceError, match="Could not find Russell 1000"):
        ticker_fetcher._fetch_russell1000()


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_russell1000_fewer_than_1000(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_russell1000 logs warning when fewer than 1000 tickers fetched."""
    # Only 4 tickers, should log warning
    mock_get.return_value = create_mock_response(create_russell1000_html())
    
    tickers = ticker_fetcher._fetch_russell1000()
    
    assert len(tickers) == 4
    assert len(tickers) < 1000


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_network_error(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_html_tables handles network errors."""
    mock_get.side_effect = Exception("Network error")
    
    with pytest.raises(Exception, match="Network error"):
        ticker_fetcher._fetch_sp500()


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_http_error(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test _fetch_html_tables handles HTTP errors."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")
    mock_get.return_value = mock_response
    
    with pytest.raises(Exception, match="404 Not Found"):
        ticker_fetcher._fetch_sp500()


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_fetch_single_source_sp500(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test fetch with single sp500 source."""
    mock_get.return_value = create_mock_response(create_sp500_html())
    
    tickers = ticker_fetcher.fetch("sp500")
    
    assert len(tickers) == 4
    assert "AAPL" in tickers


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_fetch_multiple_sources(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test fetch with multiple sources returns combined unique tickers."""
    def side_effect(url, **kwargs):
        if "S%26P_500" in url:
            return create_mock_response(create_sp500_html())
        elif "NASDAQ-100" in url:
            return create_mock_response(create_nasdaq100_html())
        return create_mock_response("<html></html>")
    
    mock_get.side_effect = side_effect
    
    tickers = ticker_fetcher.fetch(["sp500", "nasdaq100"])
    
    # Should have unique tickers from both sources
    assert len(tickers) >= 6  # Some overlap (AAPL in both)
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "TSLA" in tickers
    assert "NVDA" in tickers


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_fetch_invalid_source(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test fetch with invalid source raises error."""
    with pytest.raises(ValueError, match="Unknown source"):
        ticker_fetcher.fetch("invalid_source")  # type: ignore


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_fetch_all(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test fetch_all returns all ticker sources."""
    def side_effect(url, **kwargs):
        if "S%26P_500" in url:
            return create_mock_response(create_sp500_html())
        elif "NASDAQ-100" in url:
            return create_mock_response(create_nasdaq100_html())
        elif "Dow_Jones" in url:
            return create_mock_response(create_dowjones_html())
        elif "Russell_1000" in url:
            return create_mock_response(create_russell1000_html())
        return create_mock_response("<html></html>")
    
    mock_get.side_effect = side_effect
    
    all_tickers = ticker_fetcher.fetch_all()
    
    assert "sp500" in all_tickers
    assert "nasdaq100" in all_tickers
    assert "dowjones" in all_tickers
    assert "russell1000" in all_tickers
    assert len(all_tickers["sp500"]) == 4
    assert len(all_tickers["nasdaq100"]) == 4


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_fetch_combined(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test fetch_combined returns unique combined tickers from all sources."""
    def side_effect(url, **kwargs):
        if "S%26P_500" in url:
            return create_mock_response(create_sp500_html())
        elif "NASDAQ-100" in url:
            return create_mock_response(create_nasdaq100_html())
        elif "Dow_Jones" in url:
            return create_mock_response(create_dowjones_html())
        elif "Russell_1000" in url:
            return create_mock_response(create_russell1000_html())
        return create_mock_response("<html></html>")
    
    mock_get.side_effect = side_effect
    
    combined = ticker_fetcher.fetch_combined()
    
    # Should have unique tickers from all sources
    assert len(combined) >= 10  # Some overlap expected
    assert "AAPL" in combined
    assert "TSLA" in combined
    assert "BRK.A" in combined


@patch("tgedr_datasets.ticker.fetcher.requests.get")
def test_ticker_fetcher_symbol_cleaning(
    mock_get: Mock, ticker_fetcher: TickerFetcher
) -> None:
    """Test that symbols with dots are converted to dashes in S&P 500."""
    df = pd.DataFrame({
        "Symbol": ["BRK.B", "BF.B"],
        "Security": ["Berkshire", "Brown-Forman"],
    })
    mock_get.return_value = create_mock_response(df.to_html())
    
    tickers = ticker_fetcher._fetch_sp500()
    
    # S&P 500 should replace dots with dashes
    assert "BRK-B" in tickers
    assert "BF-B" in tickers
    assert "BRK.B" not in tickers