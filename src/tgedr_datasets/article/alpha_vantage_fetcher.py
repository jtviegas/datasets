"""Alpha Vantage article fetcher implementation.

This module provides a concrete implementation of ArticleFetcher that fetches
news articles from the Alpha Vantage API.
"""

import os
import logging

import requests
from datetime import datetime, UTC

from tgedr_datasets.article.article import Article
from tgedr_datasets.article.fetcher import ArticleFetcher


logger = logging.getLogger(__name__)


class AlphaVantageNewsFetcher(ArticleFetcher):
    """Fetches news articles with from Alpha Vantage API.

    This fetcher retrieves news articles aspecific to the requested ticker.

    Attributes:
        _api_key: Alpha Vantage API key from environment variable
        _base_url: Base URL for Alpha Vantage API
        _session: Requests session for HTTP connections

    """

    def __init__(self) -> None:
        """Initialize Alpha Vantage article fetcher.

        Validates that the ALPHAVANTAGE_API_KEY environment variable is set
        and initializes the HTTP session.

        Raises:
            ValueError: If ALPHAVANTAGE_API_KEY environment variable is not set.

        """
        if not os.getenv("ALPHAVANTAGE_API_KEY"):
            msg = "ALPHAVANTAGE_API_KEY environment variable not set"
            raise ValueError(msg)
        self._api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        self._base_url = "https://www.alphavantage.co/query"
        self._session = requests.Session()

    def get_articles(
        self,
        query: str,
        start_date: int,
        end_date: int | None = None,
        extra_query: str | None = None,  # noqa: ARG002
        max_articles: int = 50,
    ) -> list[Article]:
        """Fetch news articles with sentiment analysis from Alpha Vantage.

        Retrieves news articles for the specified ticker within the date range,
        including sentiment scores and relevance ratings when available.

        Args:
            query: Stock ticker symbol (e.g., "AAPL", "GOOGL")
            start_date: Start of date range as UTC epoch timestamp (seconds)
            end_date: End of date range as UTC epoch timestamp (seconds).
                     If None, defaults to current time.
            extra_query: Not used by Alpha Vantage API (API uses query only which is a ticker)
            max_articles: Maximum number of articles to fetch. Default is 50.

        Returns:
            List of Article objects with sentiment and relevance data,
            sorted by timestamp, or empty list on error.

        """
        logger.debug(
            "[get_articles|in] query=%s, start_date=%s, end_date=%s, max_articles=%s",
            query, start_date, end_date, max_articles
        )

        if end_date is None:
            end_date = int(datetime.now(UTC).timestamp())

        dt_start = datetime.fromtimestamp(start_date, tz=UTC).strftime("%Y%m%dT%H%M")
        dt_end = datetime.fromtimestamp(end_date, tz=UTC).strftime("%Y%m%dT%H%M")

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": query,
            "apikey": self._api_key,
            "limit": max_articles,
            "time_from": dt_start,
            "time_to": dt_end,
        }

        try:
            response = self._session.get(
                self._base_url,
                params=params,
                verify=os.getenv("SSL_CERT_VERIFY", "true") != "false",
            )
            response.raise_for_status()

            data = response.json()
            articles = []

            if "feed" in data:
                for item in data["feed"]:
                    try:
                        news_article = Article(
                            title=item["title"],
                            description=item.get("summary", ""),
                            url=item["url"],
                            timestamp=int(datetime.fromisoformat(item["time_published"]).timestamp()),
                            source=item["source"],
                            query=query,
                        )
                        articles.append(news_article)
                    except ValueError as e:
                        logger.warning(
                            "Skipping Alpha Vantage article due to error: %s", e
                        )
            logger.info("[get_articles|out] => %d articles", len(articles))
            return articles  # noqa: TRY300

        except requests.exceptions.RequestException as x:
            msg = f"error while fetching news from Alpha Vantage: {x}"
            logger.exception(msg)
            return []
