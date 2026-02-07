"""Articles aggregator for combining multiple news sources.

This module provides the ArticlesAggregator class that fetches news from
multiple sources (Alpha Vantage, NewsAPI, and Finnhub), deduplicates results,
and returns a unified sorted list of articles.
"""

from tgedr_datasets.article.alpha_vantage_fetcher import AlphaVantageNewsFetcher
from tgedr_datasets.article.finnhub_fetcher import FinnhubNewsFetcher
from tgedr_datasets.article.newsapi_fetcher import NewsApiFetcher
from tgedr_datasets.article.article import Article
from tgedr_datasets.article.fetcher import ArticleFetcher
import yfinance as yf


class ArticlesAggregator:
    """Aggregates articles from multiple fetcher sources.

    This class combines articles from various sources (currently Alpha Vantage, NewsAPI,
    and Finnhub), removes duplicate articles based on URL, and returns a unified
    sorted result set. This provides comprehensive news coverage by leveraging
    multiple data sources.

    New fetchers can be added by
    modifying the __init__ method without changing the aggregation logic.

    Attributes:
        _fetchers: List of ArticleFetcher implementations to query

    """

    def __init__(self) -> None:
        """Initialize the articles aggregator with multiple fetcher sources.

        Currently initializes with:
        - AlphaVantageNewsFetcher: financial news
        - NewsApiFetcher: General financial news coverage
        - FinnhubNewsFetcher: Company news from Finnhub (60 calls/min free tier)
        """
        self._fetchers: list = [
            AlphaVantageNewsFetcher(),
            NewsApiFetcher(),
            FinnhubNewsFetcher(),
        ]

    def get_news(
        self,
        query: str,
        timestamp: int,
        past_window_days: int = 3,
        extra_query: str | None = None,
    ) -> list[Article]:
        """Fetch and aggregate news from all configured sources.

        Queries all registered fetchers for news within the specified time window,
        deduplicates articles by URL, and returns a sorted unified result set.

        Args:
            query: Stock ticker symbol or search query (e.g., "AAPL", "GOOGL")
            timestamp: Reference timestamp as UTC epoch (seconds). News is fetched
                      backwards from this point.
            past_window_days: Number of days to look back from timestamp.
                             Default is 3 days.
            extra_query: Optional extra query string. Can be used as company name for better search results.

        Returns:
            List of unique Article objects sorted by timestamp in ascending order.
            Empty list if no articles found or if all fetchers fail.

        """
        start_date = timestamp - past_window_days * 86400  # Convert days to seconds
        all_articles: list[Article] = []
        company_name = extra_query if extra_query is not None else self.resolve_company_name(query)

        for fetcher in self._fetchers:
            articles = fetcher.get_articles(
                query=query,
                start_date=start_date,
                end_date=timestamp,
                extra_query=company_name,
                max_articles=100,
            )
            # Filter articles to only include those mentioning query or company in description
            filtered_articles = ArticleFetcher.filter_by_description(articles, query, company_name)
            all_articles.extend(filtered_articles)

        # Remove duplicates based on URL
        unique_articles = {article.url: article for article in all_articles}
        result = list(unique_articles.values())
        result.sort(key=lambda article: article.timestamp)
        return result

    def resolve_company_name(self, ticker: str) -> str | None:
        """Resolve company name from ticker symbol using Yahoo Finance.

        Attempts to fetch the full company name for a given stock ticker
        symbol. Returns None if the lookup fails.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "GOOGL")

        Returns:
            Full company name if found, None otherwise

        """
        try:
            stock = yf.Ticker(ticker)
            company_name = stock.info.get("longName", ticker)
            return company_name  # noqa: TRY300
        except Exception:  # noqa: BLE001
            # Catch all exceptions as yfinance can raise various types
            # (network errors, parsing errors, invalid tickers, etc.)
            return None

