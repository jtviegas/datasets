"""Unit tests for FinnhubNewsFetcher class."""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, UTC

from tgedr_datasets.article.finnhub_fetcher import FinnhubNewsFetcher




@pytest.fixture
def mock_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Fixture providing a test API key via environment variable.

    Args:
        monkeypatch: pytest fixture for mocking environment

    Returns:
        The test API key string

    """
    api_key = "test_finnhub_api_key"
    monkeypatch.setenv("FINNHUB_API_KEY", api_key)
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
def sample_finnhub_item() -> dict[str, any]:
    """Fixture providing a sample Finnhub news item.

    Returns:
        Dictionary representing a Finnhub news article

    """
    return {
        "datetime": int(datetime(2023, 12, 7, 10, 0, 0, tzinfo=UTC).timestamp()),
        "headline": "Apple Reports Strong Q4 Earnings",
        "summary": "Apple Inc exceeded revenue expectations in Q4",
        "url": "https://example.com/apple-earnings",
        "source": "Financial Times",
    }


@patch("tgedr_datasets.article.finnhub_fetcher.finnhub.Client")
def test_initialization_with_api_key(mock_client_class: Mock, mock_api_key: str) -> None:
    """Test FinnhubNewsFetcher initialization with API key.

    Args:
        mock_client_class: Mocked finnhub.Client class
        mock_api_key: Fixture providing test API key

    """
    fetcher = FinnhubNewsFetcher()

    assert fetcher._client is not None
    mock_client_class.assert_called_once_with(api_key=mock_api_key)


def test_initialization_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test FinnhubNewsFetcher raises error when API key is missing.

    Args:
        monkeypatch: pytest fixture for mocking environment

    """
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)

    with pytest.raises(ValueError, match="FINNHUB_API_KEY environment variable must be set"):
        FinnhubNewsFetcher()


@patch("tgedr_datasets.article.finnhub_fetcher.finnhub.Client")
@patch("tgedr_datasets.article.finnhub_fetcher.time.sleep")
def test_get_articles_success(
    mock_sleep: Mock,
    mock_client_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
    sample_finnhub_item: dict[str, any],
) -> None:
    """Test successful fetching of news articles from Finnhub.

    Args:
        mock_sleep: Mocked time.sleep function
        mock_client_class: Mocked finnhub.Client class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range
        sample_finnhub_item: Fixture with sample Finnhub data

    """
    # Setup mock client
    mock_client = Mock()
    mock_client.company_news.return_value = [sample_finnhub_item]
    mock_client_class.return_value = mock_client

    fetcher = FinnhubNewsFetcher()

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"],
    )

    # Verify sleep was called
    mock_sleep.assert_called_once_with(1)

    # Verify client.company_news was called with correct params
    mock_client.company_news.assert_called_once_with(
        "AAPL",
        _from="2023-12-01",
        to="2023-12-07"
    )

    assert len(articles) == 1
    assert articles[0].title == "Apple Reports Strong Q4 Earnings"
    assert articles[0].source == "Financial Times"
    assert articles[0].url == "https://example.com/apple-earnings"
    assert articles[0].description == "Apple Inc exceeded revenue expectations in Q4"
    assert articles[0].query == "AAPL"


@patch("tgedr_datasets.article.finnhub_fetcher.finnhub.Client")
@patch("tgedr_datasets.article.finnhub_fetcher.time.sleep")
def test_get_articles_with_empty_summary(
    mock_sleep: Mock,
    mock_client_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles with articles that don't have summary field.

    Args:
        mock_sleep: Mocked time.sleep function
        mock_client_class: Mocked finnhub.Client class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_client = Mock()
    mock_client.company_news.return_value = [
        {
            "datetime": int(datetime(2023, 12, 7, 11, 0, 0, tzinfo=UTC).timestamp()),
            "headline": "Market News Update",
            "url": "https://example.com/market-update",
            "source": "Bloomberg",
            # No 'summary' field
        }
    ]
    mock_client_class.return_value = mock_client

    fetcher = FinnhubNewsFetcher()

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    assert len(articles) == 1
    assert articles[0].description == ""


@patch("tgedr_datasets.article.finnhub_fetcher.finnhub.Client")
@patch("tgedr_datasets.article.finnhub_fetcher.time.sleep")
def test_get_articles_with_default_end_date(
    mock_sleep: Mock,
    mock_client_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles uses current time when end_date is None.

    Args:
        mock_sleep: Mocked time.sleep function
        mock_client_class: Mocked finnhub.Client class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_client = Mock()
    mock_client.company_news.return_value = []
    mock_client_class.return_value = mock_client

    fetcher = FinnhubNewsFetcher()

    articles = fetcher.get_articles("AAPL", test_timestamps["start_date"], end_date=None)

    # Verify company_news was called
    mock_client.company_news.assert_called_once()
    call_args = mock_client.company_news.call_args
    assert call_args[0][0] == "AAPL"
    assert call_args[1]["_from"] == "2023-12-01"
    # Verify date format is YYYY-MM-DD
    assert len(call_args[1]["to"]) == 10

    assert isinstance(articles, list)
    assert len(articles) == 0


@patch("tgedr_datasets.article.finnhub_fetcher.finnhub.Client")
@patch("tgedr_datasets.article.finnhub_fetcher.time.sleep")
def test_get_articles_handles_request_exception(
    mock_sleep: Mock,
    mock_client_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles handles exceptions gracefully.

    Args:
        mock_sleep: Mocked time.sleep function
        mock_client_class: Mocked finnhub.Client class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_client = Mock()
    mock_client.company_news.side_effect = Exception("Network error")
    mock_client_class.return_value = mock_client

    fetcher = FinnhubNewsFetcher()

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    assert articles == []


@patch("tgedr_datasets.article.finnhub_fetcher.finnhub.Client")
@patch("tgedr_datasets.article.finnhub_fetcher.time.sleep")
@patch("tgedr_datasets.article.finnhub_fetcher.logger")
def test_get_articles_handles_invalid_article_data(
    mock_logger: Mock,
    mock_sleep: Mock,
    mock_client_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles skips articles with missing required fields.

    Args:
        mock_logger: Mocked logger
        mock_sleep: Mocked time.sleep function
        mock_client_class: Mocked finnhub.Client class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_client = Mock()
    mock_client.company_news.return_value = [
        {
            # Missing 'headline' field
            "datetime": int(datetime(2023, 12, 7, 10, 0, 0, tzinfo=UTC).timestamp()),
            "url": "https://example.com/invalid",
            "source": "Test Source",
        },
        {
            "datetime": int(datetime(2023, 12, 7, 11, 0, 0, tzinfo=UTC).timestamp()),
            "headline": "Valid Article",
            "summary": "Good data",
            "url": "https://example.com/valid",
            "source": "Reuters",
        },
    ]
    mock_client_class.return_value = mock_client

    fetcher = FinnhubNewsFetcher()

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    # Should only get the valid article
    assert len(articles) == 1
    assert articles[0].title == "Valid Article"

    # Check that exception was logged
    mock_logger.exception.assert_called()


@patch("tgedr_datasets.article.finnhub_fetcher.finnhub.Client")
@patch("tgedr_datasets.article.finnhub_fetcher.time.sleep")
def test_get_articles_with_multiple_articles(
    mock_sleep: Mock,
    mock_client_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles handles multiple articles correctly.

    Args:
        mock_sleep: Mocked time.sleep function
        mock_client_class: Mocked finnhub.Client class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_client = Mock()
    mock_client.company_news.return_value = [
        {
            "datetime": int(datetime(2023, 12, 7, 10, 0, 0, tzinfo=UTC).timestamp()),
            "headline": "First Article",
            "summary": "First summary",
            "url": "https://example.com/first",
            "source": "Source 1",
        },
        {
            "datetime": int(datetime(2023, 12, 7, 11, 0, 0, tzinfo=UTC).timestamp()),
            "headline": "Second Article",
            "summary": "Second summary",
            "url": "https://example.com/second",
            "source": "Source 2",
        },
        {
            "datetime": int(datetime(2023, 12, 7, 12, 0, 0, tzinfo=UTC).timestamp()),
            "headline": "Third Article",
            "summary": "Third summary",
            "url": "https://example.com/third",
            "source": "Source 3",
        },
    ]
    mock_client_class.return_value = mock_client

    fetcher = FinnhubNewsFetcher()

    articles = fetcher.get_articles(
        "GOOGL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    assert len(articles) == 3
    assert articles[0].title == "First Article"
    assert articles[1].title == "Second Article"
    assert articles[2].title == "Third Article"
    # Verify all articles have the correct ticker
    assert all(article.query == "GOOGL" for article in articles)


@patch("tgedr_datasets.article.finnhub_fetcher.finnhub.Client")
@patch("tgedr_datasets.article.finnhub_fetcher.time.sleep")
def test_get_articles_date_conversion(
    mock_sleep: Mock,
    mock_client_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test that epoch timestamps are correctly converted to YYYY-MM-DD format.

    Args:
        mock_sleep: Mocked time.sleep function
        mock_client_class: Mocked finnhub.Client class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_client = Mock()
    mock_client.company_news.return_value = []
    mock_client_class.return_value = mock_client

    fetcher = FinnhubNewsFetcher()

    fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    # Verify the date conversion
    call_args = mock_client.company_news.call_args
    assert call_args[0][0] == "AAPL"
    assert call_args[1]["_from"] == "2023-12-01"
    assert call_args[1]["to"] == "2023-12-07"


@patch("tgedr_datasets.article.finnhub_fetcher.finnhub.Client")
@patch("tgedr_datasets.article.finnhub_fetcher.time.sleep")
def test_get_articles_with_http_error(
    mock_sleep: Mock,
    mock_client_class: Mock,
    mock_api_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles handles HTTP errors gracefully.

    Args:
        mock_sleep: Mocked time.sleep function
        mock_client_class: Mocked finnhub.Client class
        mock_api_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_client = Mock()
    mock_client.company_news.side_effect = Exception("403 Forbidden")
    mock_client_class.return_value = mock_client

    fetcher = FinnhubNewsFetcher()

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    assert articles == []
