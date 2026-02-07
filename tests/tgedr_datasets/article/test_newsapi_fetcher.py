"""Unit tests for NewsApiFetcher class."""
import pytest
import requests
from unittest.mock import Mock, patch
from datetime import datetime, UTC

from tgedr_datasets.article.newsapi_fetcher import NewsApiFetcher


@pytest.fixture
def mock_newsapi_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Fixture providing a test NewsAPI key via environment variable.

    Args:
        monkeypatch: pytest fixture for mocking environment

    Returns:
        The test API key string

    """
    api_key = "test_newsapi_key"
    monkeypatch.setenv("NEWSAPI_API_KEY", api_key)
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
def sample_newsapi_article() -> dict[str, any]:
    """Fixture providing a sample NewsAPI article.

    Returns:
        Dictionary representing a NewsAPI article

    """
    return {
        "title": "Apple Stock Rises",
        "description": "Apple sees growth in Q4",
        "url": "https://example.com/apple-news",
        "publishedAt": "2023-12-07T10:00:00Z",
        "source": {"name": "Reuters"},
    }


def test_initialization_with_api_key(mock_newsapi_key: str) -> None:
    """Test NewsApiFetcher initialization with API key.

    Args:
        mock_newsapi_key: Fixture providing test API key

    """
    fetcher = NewsApiFetcher()

    assert fetcher._base_url == "https://newsapi.org/v2"
    assert fetcher._session is not None
    assert "X-Api-Key" in fetcher._session.headers


def test_initialization_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test NewsApiFetcher raises error when API key is missing.

    Args:
        monkeypatch: pytest fixture for mocking environment

    """
    monkeypatch.delenv("NEWSAPI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NEWSAPI_API_KEY environment variable not set"):
        NewsApiFetcher()


@patch("tgedr_datasets.article.newsapi_fetcher.requests.Session")
def test_get_articles_success(
    mock_session_class: Mock,
    mock_newsapi_key: str,
    test_timestamps: dict[str, int],
    sample_newsapi_article: dict[str, any],
) -> None:
    """Test successful fetching of news articles from NewsAPI.

    Args:
        mock_session_class: Mocked requests Session class
        mock_newsapi_key: Fixture providing test API key
        test_timestamps: Fixture with test date range
        sample_newsapi_article: Fixture with sample article data

    """
    mock_response = Mock()
    mock_response.json.return_value = {"articles": [sample_newsapi_article]}
    mock_response.raise_for_status = Mock()

    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    fetcher = NewsApiFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"],
        extra_query="Apple Inc."
    )

    assert len(articles) == 1
    assert articles[0].title == "Apple Stock Rises"
    assert articles[0].source == "Reuters"
    assert articles[0].description == "Apple sees growth in Q4"


@patch("tgedr_datasets.article.newsapi_fetcher.requests.Session")
def test_get_articles_without_extra_query(
    mock_session_class: Mock,
    mock_newsapi_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles constructs query without extra_query.

    Args:
        mock_session_class: Mocked requests Session class
        mock_newsapi_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_response = Mock()
    mock_response.json.return_value = {"articles": []}
    mock_response.raise_for_status = Mock()

    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    fetcher = NewsApiFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"],
        extra_query=None
    )

    # Verify the query was constructed with ticker only
    call_args = mock_session.get.call_args
    assert "AAPL" in call_args[1]["params"]["q"]
    assert isinstance(articles, list)


@patch("tgedr_datasets.article.newsapi_fetcher.requests.Session")
def test_get_stock_news_filters_removed_articles(
    mock_session_class: Mock,
    mock_newsapi_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test that articles with [Removed] title are filtered out.

    Args:
        mock_session_class: Mocked requests Session class
        mock_newsapi_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_response = Mock()
    mock_response.json.return_value = {
        "articles": [
            {
                "title": "[Removed]",
                "description": "Removed article",
                "url": "https://example.com/removed",
                "publishedAt": "2023-12-07T10:00:00Z",
                "source": {"name": "Test Source"},
            },
            {
                "title": "Valid Article",
                "description": "Valid content",
                "url": "https://example.com/valid",
                "publishedAt": "2023-12-07T11:00:00Z",
                "source": {"name": "Reuters"},
            },
        ]
    }
    mock_response.raise_for_status = Mock()

    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    fetcher = NewsApiFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    assert len(articles) == 1
    assert articles[0].title == "Valid Article"


@patch("tgedr_datasets.article.newsapi_fetcher.requests.Session")
def test_get_articles_filters_null_titles(
    mock_session_class: Mock,
    mock_newsapi_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test that articles with null or empty titles are filtered out.

    Args:
        mock_session_class: Mocked requests Session class
        mock_newsapi_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_response = Mock()
    mock_response.json.return_value = {
        "articles": [
            {
                "title": None,
                "description": "No title article",
                "url": "https://example.com/notitle",
                "publishedAt": "2023-12-07T10:00:00Z",
                "source": {"name": "Test Source"},
            },
            {
                "title": "Good Article",
                "description": "Valid content",
                "url": "https://example.com/good",
                "publishedAt": "2023-12-07T11:00:00Z",
                "source": {"name": "Reuters"},
            },
        ]
    }
    mock_response.raise_for_status = Mock()

    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    fetcher = NewsApiFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    assert len(articles) == 1
    assert articles[0].title == "Good Article"


@patch("tgedr_datasets.article.newsapi_fetcher.requests.Session")
def test_get_articles_with_default_end_date(
    mock_session_class: Mock,
    mock_newsapi_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles uses current time when end_date is None.

    Args:
        mock_session_class: Mocked requests Session class
        mock_newsapi_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_response = Mock()
    mock_response.json.return_value = {"articles": []}
    mock_response.raise_for_status = Mock()

    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    fetcher = NewsApiFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        end_date=None
    )

    assert isinstance(articles, list)
    assert len(articles) == 0


@patch("tgedr_datasets.article.newsapi_fetcher.requests.Session")
def test_get_articles_respects_max_articles_limit(
    mock_session_class: Mock,
    mock_newsapi_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles respects the API's 100 article limit.

    Args:
        mock_session_class: Mocked requests Session class
        mock_newsapi_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_response = Mock()
    mock_response.json.return_value = {"articles": []}
    mock_response.raise_for_status = Mock()

    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    fetcher = NewsApiFetcher()
    fetcher._session = mock_session

    # Request more than API limit
    fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"],
        max_articles=150
    )

    # Verify that pageSize is capped at 100
    call_args = mock_session.get.call_args
    assert call_args[1]["params"]["pageSize"] == 100


@patch("tgedr_datasets.article.newsapi_fetcher.requests.Session")
def test_get_articles_handles_request_exception(
    mock_session_class: Mock,
    mock_newsapi_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles handles request exceptions gracefully.

    Args:
        mock_session_class: Mocked requests Session class
        mock_newsapi_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_session = Mock()
    mock_session.get.side_effect = requests.exceptions.RequestException("Network error")
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    fetcher = NewsApiFetcher()
    fetcher._session = mock_session

    articles = fetcher.get_articles(
        "AAPL",
        test_timestamps["start_date"],
        test_timestamps["end_date"]
    )

    assert articles == []


@patch("tgedr_datasets.article.newsapi_fetcher.requests.Session")
@patch("tgedr_datasets.article.newsapi_fetcher.logger")
def test_get_articles_handles_invalid_timestamp(
    mock_logger: Mock,
    mock_session_class: Mock,
    mock_newsapi_key: str,
    test_timestamps: dict[str, int],
) -> None:
    """Test get_articles skips articles with invalid timestamps.

    Args:
        mock_logger: Mocked logger
        mock_session_class: Mocked requests Session class
        mock_newsapi_key: Fixture providing test API key
        test_timestamps: Fixture with test date range

    """
    mock_response = Mock()
    mock_response.json.return_value = {
        "articles": [
            {
                "title": "Invalid Article",
                "description": "Bad timestamp",
                "url": "https://example.com/invalid",
                "publishedAt": "invalid_date",
                "source": {"name": "Test Source"},
            },
            {
                "title": "Valid Article",
                "description": "Good timestamp",
                "url": "https://example.com/valid",
                "publishedAt": "2023-12-07T12:00:00Z",
                "source": {"name": "Reuters"},
            },
        ]
    }
    mock_response.raise_for_status = Mock()

    mock_session = Mock()
    mock_session.get.return_value = mock_response
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    fetcher = NewsApiFetcher()
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
    assert "Skipping article due to error" in warning_call
