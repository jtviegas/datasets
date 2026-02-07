"""Unit tests for AlphaVantageNewsFetcher class."""
import pytest
import requests
from unittest.mock import Mock, patch
from datetime import datetime, UTC

from tgedr_datasets.article.alpha_vantage_fetcher import AlphaVantageNewsFetcher


@pytest.fixture
def mock_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Fixture providing a test API key via environment variable.

    Args:
        monkeypatch: pytest fixture for mocking environment

    Returns:
        The test API key string

    """
    api_key = "test_api_key"
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", api_key)
    return api_key


@pytest.fixture
def test_timestamps() -> dict[str, int]:
    """Fixture providing consistent test timestamps.

    Returns:
        Dictionary with start_date and end_date epoch timestamps

    """
    return {
        "start_date": int(datetime(2023, 12, 1, tzinfo=UTC).timestamp()),
        "end_date": int(datetime(2023, 12, 7, tzinfo=UTC).timestamp()),
    }


@pytest.fixture
def sample_feed_item() -> dict[str, any]:
    """Fixture providing a sample Alpha Vantage feed item.

    Returns:
        Dictionary representing a news feed item

    """
    return {
        "title": "Company X Reports Strong Earnings",
        "summary": "Company X exceeded expectations",
        "url": "https://example.com/article1",
        "time_published": "20231207T100000",
        "source": "Financial Times"
    }


def test_initialization_with_api_key(mock_api_key: str) -> None:
    """Test AlphaVantageNewsFetcher initialization with API key.

    Args:
        mock_api_key: Fixture providing test API key

    """
    fetcher = AlphaVantageNewsFetcher()

    assert fetcher._api_key == mock_api_key
    assert fetcher._base_url == "https://www.alphavantage.co/query"
    assert fetcher._session is not None


def test_initialization_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test AlphaVantageNewsFetcher raises error when API key is missing.

    Args:
        monkeypatch: pytest fixture for mocking environment

    """
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ALPHAVANTAGE_API_KEY environment variable not set"):
        AlphaVantageNewsFetcher()


@patch("tgedr_datasets.article.alpha_vantage_fetcher.requests.Session")
def test_get_articles_success(
    mock_session_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
    sample_feed_item: dict[str, any],
) -> None:
    """Test successful fetching of news articles from Alpha Vantage.

    Args:
        mock_session_class: Mocked requests Session class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range
        sample_feed_item: Fixture with sample feed data

    """
    # Mock response data
    mock_response = Mock()
    mock_response.json.return_value = {"feed": [sample_feed_item]}
    mock_response.raise_for_status = Mock()

    # Setup mock session
    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    fetcher = AlphaVantageNewsFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"],
        max_articles=50
    )

    assert len(articles) == 1
    assert articles[0].title == "Company X Reports Strong Earnings"
    assert articles[0].source == "Financial Times"
    assert articles[0].url == "https://example.com/article1"



@patch("tgedr_datasets.article.alpha_vantage_fetcher.requests.Session")
def test_get_articles_without_ticker_sentiment(
    mock_session_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles with articles that don't have ticker sentiment.

    Args:
        mock_session_class: Mocked requests Session class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_response = Mock()
    mock_response.json.return_value = {
        "feed": [
            {
                "title": "Market News",
                "summary": "General market update",
                "url": "https://example.com/article2",
                "time_published": "20231207T110000",
                "source": "Bloomberg",
            }
        ]
    }
    mock_response.raise_for_status = Mock()

    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    fetcher = AlphaVantageNewsFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    assert len(articles) == 1



@patch("tgedr_datasets.article.alpha_vantage_fetcher.requests.Session")
def test_get_articles_with_default_end_date(
    mock_session_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles uses current time when end_date is None.

    Args:
        mock_session_class: Mocked requests Session class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_response = Mock()
    mock_response.json.return_value = {"feed": []}
    mock_response.raise_for_status = Mock()

    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    fetcher = AlphaVantageNewsFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles("AAPL", test_timestamps["start_date"], end_date=None)

    assert isinstance(articles, list)
    assert len(articles) == 0


@patch("tgedr_datasets.article.alpha_vantage_fetcher.requests.Session")
def test_get_articles_handles_request_exception(
    mock_session_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles handles request exceptions gracefully.

    Args:
        mock_session_class: Mocked requests Session class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_session = Mock()
    mock_session.get.side_effect = requests.exceptions.RequestException("Network error")
    mock_session_class.return_value = mock_session

    fetcher = AlphaVantageNewsFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    assert articles == []


@patch("tgedr_datasets.article.alpha_vantage_fetcher.requests.Session")
@patch("tgedr_datasets.article.alpha_vantage_fetcher.logger")
def test_get_articles_handles_invalid_timestamp(
    mock_logger: Mock,
    mock_session_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles skips articles with invalid timestamps.

    Args:
        mock_logger: Mocked logger
        mock_session_class: Mocked requests Session class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_response = Mock()
    mock_response.json.return_value = {
        "feed": [
            {
                "title": "Invalid Article",
                "summary": "Bad timestamp",
                "url": "https://example.com/invalid",
                "time_published": "invalid_date",
                "source": "Test Source",
            },
            {
                "title": "Valid Article",
                "summary": "Good timestamp",
                "url": "https://example.com/valid",
                "time_published": "20231207T120000",
                "source": "Reuters",
            },
        ]
    }
    mock_response.raise_for_status = Mock()

    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    fetcher = AlphaVantageNewsFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    # Should only get the valid article
    assert len(articles) == 1
    assert articles[0].title == "Valid Article"

    # Check that warning was logged
    mock_logger.warning.assert_called()
    warning_call = str(mock_logger.warning.call_args)
    assert "Skipping Alpha Vantage article due to error" in warning_call


@patch("tgedr_datasets.article.alpha_vantage_fetcher.requests.Session")
def test_get_articles_with_no_feed_in_response(
    mock_session_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles handles response without feed key.

    Args:
        mock_session_class: Mocked requests Session class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_response = Mock()
    mock_response.json.return_value = {"error": "Rate limit exceeded"}
    mock_response.raise_for_status = Mock()

    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    fetcher = AlphaVantageNewsFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    assert articles == []
