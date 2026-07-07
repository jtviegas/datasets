"""Unit tests for ArticlesAggregator class."""
import pytest
from unittest.mock import Mock, patch

from tgedr_datasets.article.alpha_vantage_fetcher import AlphaVantageNewsFetcher
from tgedr_datasets.article.articles_aggregator import ArticlesAggregator
from tgedr_datasets.article.article import Article


@pytest.fixture
def mock_api_keys(monkeypatch: pytest.MonkeyPatch) -> str:
    """Fixture providing a test NewsAPI key via environment variable.

    Args:
        monkeypatch: pytest fixture for mocking environment

    Returns:
        The test API key string

    """
    monkeypatch.setenv("NEWSAPI_API_KEY", "test_newsapi_key")
    monkeypatch.setenv("FINNHUB_API_KEY", "test_finnhub_api_key")
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test_alphavantage_api_key")


@pytest.fixture
def sample_articles() -> dict[str, Article]:
    """Fixture providing sample news articles for testing.

    Returns:
        Dictionary with named sample articles

    """
    return {
        "article1": Article(
            title="Article 1",
            description="AAPL reports strong earnings",
            url="https://example.com/1",
            timestamp=1701950400,
            source="Source 1",
                query="AAPL",
        ),
        "article2": Article(
            title="Article 2",
            description="Apple Inc. announces new product",
            url="https://example.com/2",
            timestamp=1701954000,
            source="Source 2",
                query="AAPL",
        ),
        "later_article": Article(
            title="Later Article",
            description="Apple Inc. stock rises",
            url="https://example.com/later",
            timestamp=1701954000,
            source="Source 1",
                query="AAPL",
        ),
        "earlier_article": Article(
            title="Earlier Article",
            description="AAPL market analysis",
            url="https://example.com/earlier",
            timestamp=1701950400,
            source="Source 2",
                query="AAPL",
        ),
        "duplicate": Article(
            title="Article 1",
            description="AAPL duplicate article",
            url="https://example.com/duplicate",
            timestamp=1701950400,
            source="Source 1",
                query="AAPL",
        ),
    }


@pytest.fixture
def test_timestamp() -> int:
    """Fixture providing consistent test timestamp.

    Returns:
        Epoch timestamp for testing

    """
    return 1702000000


def test_initialization(mock_api_keys) -> None:
    """Test ArticlesAggregator initializes with fetchers."""
    with patch("tgedr_datasets.article.articles_aggregator.AlphaVantageNewsFetcher"):
        aggregator = ArticlesAggregator()

        assert hasattr(aggregator, "_fetchers")
        assert isinstance(aggregator._fetchers, list)
        assert len(aggregator._fetchers) == 3


@patch("tgedr_datasets.article.articles_aggregator.AlphaVantageNewsFetcher")
def test_get_news_aggregates_from_multiple_sources(
    mock_alphavantage_class: Mock,
    sample_articles: dict[str, Article],
    test_timestamp: int,
    mock_api_keys
) -> None:
    """Test that get_news aggregates articles from AlphaVantage fetcher.

    Args:
        mock_alphavantage_class: Mocked AlphaVantageNewsFetcher class
        sample_articles: Fixture with sample articles
        test_timestamp: Fixture with test timestamp

    """
    mock_resolve_company = Mock()
    mock_resolve_company.return_value = "Apple Inc."

    # Setup mock fetcher
    mock_av_fetcher = Mock(spec=AlphaVantageNewsFetcher)
    mock_av_fetcher.get_articles.return_value = [sample_articles["article1"], sample_articles["article2"]]
    mock_alphavantage_class.return_value = mock_av_fetcher

    aggregator = ArticlesAggregator()
    aggregator.resolve_company_name = mock_resolve_company

    articles = aggregator.get_news("AAPL", test_timestamp, past_window_days=3)

    assert len(articles) == 2
    assert articles[0].title == "Article 1"
    assert articles[1].title == "Article 2"
    # Verify articles are sorted by timestamp
    assert articles[0].timestamp <= articles[1].timestamp


@patch("tgedr_datasets.article.articles_aggregator.AlphaVantageNewsFetcher")
def test_get_news_removes_duplicates(
    mock_alphavantage_class: Mock,
    sample_articles: dict[str, Article],
    test_timestamp: int,
    mock_api_keys
) -> None:
    """Test that get_news removes duplicate articles by URL.

    Args:
        mock_alphavantage_class: Mocked AlphaVantageNewsFetcher class
        sample_articles: Fixture with sample articles
        test_timestamp: Fixture with test timestamp
        mock_api_keys: Fixture with mock API keys
    """
    mock_resolve_company = Mock()
    mock_resolve_company.return_value = "Apple Inc."

    # Create duplicate articles with same URL
    article_dup1 = Article(
        title="Article 1",
        description="AAPL duplicate news article",
        url="https://example.com/duplicate",
        timestamp=1701950400,
        source="Source 1",
                query="AAPL",
    )
    article_dup2 = Article(
        title="Article 1 Duplicate",
        description="Apple Inc. same article duplicate",
        url="https://example.com/duplicate",
        timestamp=1701954000,
        source="Source 2",
                query="AAPL",
    )

    # Setup mock fetcher - return both duplicates from same fetcher
    mock_av_fetcher = Mock(spec=AlphaVantageNewsFetcher)
    mock_av_fetcher.get_articles.return_value = [article_dup1, article_dup2]
    mock_alphavantage_class.return_value = mock_av_fetcher

    aggregator = ArticlesAggregator()
    aggregator.resolve_company_name = mock_resolve_company

    articles = aggregator.get_news("AAPL", test_timestamp, past_window_days=3)

    # Should only have one article since they have the same URL
    assert len(articles) == 1
    # Dict-based dedupe keeps the last article for duplicate URLs.
    assert articles[0].url == "https://example.com/duplicate"
    assert articles[0].title == "Article 1 Duplicate"
    assert articles[0].source == "Source 2"
    assert articles[0].timestamp == 1701954000


@patch("tgedr_datasets.article.articles_aggregator.AlphaVantageNewsFetcher")
def test_get_news_sorts_by_timestamp(
    mock_alphavantage_class: Mock,
    sample_articles: dict[str, Article],
    test_timestamp: int,
    mock_api_keys
) -> None:
    """Test that get_news sorts articles by timestamp in ascending order.

    Args:
        mock_alphavantage_class: Mocked AlphaVantageNewsFetcher class
        mock_resolve_company: Mocked resolve_company_name function
        sample_articles: Fixture with sample articles
        test_timestamp: Fixture with test timestamp

    """
    mock_resolve_company = Mock()
    mock_resolve_company.return_value = "Apple Inc."

    # Setup mock fetcher - return both articles in unsorted order
    mock_av_fetcher = Mock(spec=AlphaVantageNewsFetcher)
    mock_av_fetcher.get_articles.return_value = [
        sample_articles["later_article"],
        sample_articles["earlier_article"],
    ]
    mock_alphavantage_class.return_value = mock_av_fetcher

    aggregator = ArticlesAggregator()
    aggregator.resolve_company_name = mock_resolve_company
    articles = aggregator.get_news("AAPL", test_timestamp, past_window_days=3)

    # Verify articles are sorted by timestamp (earlier first)
    assert len(articles) == 2
    assert articles[0].title == "Earlier Article"
    assert articles[1].title == "Later Article"
    assert articles[0].timestamp < articles[1].timestamp


@patch("tgedr_datasets.article.articles_aggregator.AlphaVantageNewsFetcher")
def test_get_news_calculates_correct_date_range(
    mock_alphavantage_class: Mock,
    mock_api_keys,
    test_timestamp: int,
) -> None:
    """Test that get_news calculates the correct start date based on window.

    Args:
        mock_alphavantage_class: Mocked AlphaVantageNewsFetcher class
        mock_resolve_company: Mocked resolve_company_name function
        test_timestamp: Fixture with test timestamp

    """
    mock_resolve_company = Mock()
    mock_resolve_company.return_value = "Apple Inc."

    mock_av_fetcher = Mock(spec=AlphaVantageNewsFetcher)
    mock_av_fetcher.get_articles.return_value = []
    mock_alphavantage_class.return_value = mock_av_fetcher

    aggregator = ArticlesAggregator()
    aggregator.resolve_company_name = mock_resolve_company

    past_window_days = 5
    aggregator.get_news("AAPL", test_timestamp, past_window_days=past_window_days)

    # Verify the start_date passed to fetchers
    expected_start_date = test_timestamp - (past_window_days * 86400)

    mock_av_fetcher.get_articles.assert_called_once()
    call_args = mock_av_fetcher.get_articles.call_args
    assert call_args[1]["start_date"] == expected_start_date
    assert call_args[1]["end_date"] == test_timestamp


@patch("tgedr_datasets.article.articles_aggregator.AlphaVantageNewsFetcher")
def test_get_news_passes_company_name_to_fetchers(
    mock_alphavantage_class: Mock,
    test_timestamp: int,
    mock_api_keys
) -> None:
    """Test that get_news passes resolved company name to fetchers.

    Args:
        mock_alphavantage_class: Mocked AlphaVantageNewsFetcher class
        mock_resolve_company: Mocked resolve_company_name function
        test_timestamp: Fixture with test timestamp

    """
    mock_resolve_company = Mock()
    mock_resolve_company.return_value = "Apple Inc."

    mock_av_fetcher = Mock(spec=AlphaVantageNewsFetcher)
    mock_av_fetcher.get_articles.return_value = []
    mock_alphavantage_class.return_value = mock_av_fetcher

    aggregator = ArticlesAggregator()
    aggregator.resolve_company_name = mock_resolve_company

    aggregator.get_news("AAPL", test_timestamp)

    # Verify company name was resolved and passed
    mock_resolve_company.assert_called_once_with("AAPL")

    call_args = mock_av_fetcher.get_articles.call_args
    assert call_args[1]["extra_query"] == "Apple Inc."


@patch("tgedr_datasets.article.articles_aggregator.AlphaVantageNewsFetcher")
def test_get_news_handles_fetcher_returning_empty_list(
    mock_alphavantage_class: Mock,
    test_timestamp: int,
    mock_api_keys
) -> None:
    """Test that get_news handles fetchers returning empty lists (on internal errors).

    Args:
        mock_alphavantage_class: Mocked AlphaVantageNewsFetcher class
        mock_resolve_company: Mocked resolve_company_name function
        test_timestamp: Fixture with test timestamp

    """
    mock_resolve_company = Mock()
    mock_resolve_company.return_value = "Apple Inc."

    article = Article(
        title="Article from working fetcher",
        description="AAPL news from working fetcher",
        url="https://example.com/article",
        timestamp=1701950400,
        source="Source",
                query="AAPL",
    )

    # Setup mock fetcher - returns article
    mock_av_fetcher = Mock(spec=AlphaVantageNewsFetcher)
    mock_av_fetcher.get_articles.return_value = [article]
    mock_alphavantage_class.return_value = mock_av_fetcher

    aggregator = ArticlesAggregator()
    aggregator.resolve_company_name = mock_resolve_company
    articles = aggregator.get_news("AAPL", test_timestamp)

    # Should get articles from the fetcher
    assert len(articles) == 1
    assert articles[0].title == "Article from working fetcher"


@patch("tgedr_datasets.article.articles_aggregator.AlphaVantageNewsFetcher")
def test_get_news_with_default_window(
    mock_alphavantage_class: Mock,
    test_timestamp: int,
    mock_api_keys
) -> None:
    """Test that get_news uses default 3-day window when not specified.

    Args:
        mock_alphavantage_class: Mocked AlphaVantageNewsFetcher class
        mock_resolve_company: Mocked resolve_company_name function
        test_timestamp: Fixture with test timestamp

    """
    mock_resolve_company = Mock()
    mock_resolve_company.return_value = "Apple Inc."

    mock_av_fetcher = Mock(spec=AlphaVantageNewsFetcher)
    mock_av_fetcher.get_articles.return_value = []
    mock_alphavantage_class.return_value = mock_av_fetcher

    aggregator = ArticlesAggregator()
    aggregator.resolve_company_name = mock_resolve_company

    # Test without specifying window
    aggregator.get_news("AAPL", test_timestamp)

    # Verify default 3-day window was used
    expected_start_date = test_timestamp - (3 * 86400)

    call_args = mock_av_fetcher.get_articles.call_args
    assert call_args[1]["start_date"] == expected_start_date


@patch("tgedr_datasets.article.articles_aggregator.yf.Ticker")
def test_resolve_company_name_success(mock_ticker_class: Mock, mock_api_keys) -> None:
    """Test resolve_company_name successfully resolves company name."""
    # Mock the ticker info
    mock_ticker = Mock()
    mock_ticker.info = {"longName": "Apple Inc."}
    mock_ticker_class.return_value = mock_ticker

    aggregator = ArticlesAggregator()
    result = aggregator.resolve_company_name("AAPL")

    assert result == "Apple Inc."
    mock_ticker_class.assert_called_once_with("AAPL")


@patch("tgedr_datasets.article.articles_aggregator.yf.Ticker")
def test_resolve_company_name_fallback_to_ticker(mock_ticker_class: Mock, mock_api_keys) -> None:
    """Test resolve_company_name falls back to ticker when longName not available."""
    # Mock ticker with no longName
    mock_ticker = Mock()
    mock_ticker.info = {}
    mock_ticker_class.return_value = mock_ticker

    aggregator = ArticlesAggregator()
    result = aggregator.resolve_company_name("AAPL")

    assert result == "AAPL"
    mock_ticker_class.assert_called_once_with("AAPL")


@patch("tgedr_datasets.article.articles_aggregator.yf.Ticker")
def test_resolve_company_name_exception_returns_none(mock_ticker_class: Mock, mock_api_keys) -> None:
    """Test resolve_company_name returns None when yfinance raises exception."""
    # Mock ticker to raise exception
    mock_ticker_class.side_effect = Exception("Network error")

    aggregator = ArticlesAggregator()
    result = aggregator.resolve_company_name("INVALID")

    assert result is None
    mock_ticker_class.assert_called_once_with("INVALID")
