"""NewsAPI article fetcher implementation.

This module provides a concrete implementation of ArticleFetcher that fetches
general financial news from NewsAPI. It constructs intelligent search queries
combining ticker symbols, company names, and financial keywords.
"""

import os
import logging

import requests
from datetime import datetime, UTC

from tgedr_datasets.article.article import Article
from tgedr_datasets.article.fetcher import ArticleFetcher



logger = logging.getLogger(__name__)


class NewsApiFetcher(ArticleFetcher):
    """Fetches general financial news articles from NewsAPI.

    This fetcher retrieves news articles using keyword-based search that combines
    ticker symbols, company names, and financial terms.

    The search query construction uses boolean logic to find relevant articles:
    - Includes: ticker symbol OR company name
    - AND financial keywords: stock, shares, earnings, revenue

    Attributes:
        _base_url: Base URL for NewsAPI
        _session: Requests session with API key in headers

    """

    def __init__(self) -> None:
        """Initialize NewsAPI article fetcher.

        Validates that the NEWSAPI_API_KEY environment variable is set
        and configures the HTTP session with authentication headers.

        Raises:
            ValueError: If NEWSAPI_API_KEY environment variable is not set.

        """
        if not os.getenv("NEWSAPI_API_KEY"):
            msg = "NEWSAPI_API_KEY environment variable not set"
            raise ValueError(msg)
        self._base_url = "https://newsapi.org/v2"
        self._session = requests.Session()
        self._session.headers.update({"X-Api-Key": os.getenv("NEWSAPI_API_KEY")})

    def get_articles(
        self,
        query: str,
        start_date: int,
        end_date: int | None = None,
        extra_query: str | None = None,
        max_articles: int = 100,
    ) -> list[Article]:
        """Fetch news articles from NewsAPI using keyword search.

        Constructs a search query combining ticker, company name, and financial
        keywords to find relevant articles. Filters out removed or invalid articles.

        Args:
            query: Search query string (e.g., "AAPL", "Apple Inc")
            start_date: Start of date range as UTC epoch timestamp (seconds)
            end_date: End of date range as UTC epoch timestamp (seconds).
                     If None, defaults to current time.
            extra_query: Optional extra query string for enhanced search queries.
                         Improves search accuracy when provided.
            max_articles: Maximum number of articles to fetch. API limit is 100.

        Returns:
            List of Article objects without sentiment data,
            sorted by timestamp, or empty list on error.

        """
        logger.debug(
            "[get_articles|in] query=%s, start_date=%s, end_date=%s, extra_query=%s, max_articles=%s",
            query, start_date, end_date, extra_query, max_articles
        )

        # Construct search query
        query = (
            f'({query} OR "{extra_query}") AND (stock OR shares OR earnings OR revenue)'
            if extra_query
            else f"{query} AND (stock OR shares OR earnings OR revenue)"
        )

        if end_date is None:
            end_date = int(datetime.now(UTC).timestamp())

        params = {
            "q": query,
            "searchIn": "title,description",
            "from": datetime.fromtimestamp(start_date, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "to": datetime.fromtimestamp(end_date, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "sortBy": "publishedAt",
            "pageSize": min(max_articles, 100),  # API limit is 100 per request
            "language": "en",
        }

        try:
            response = self._session.get(
                f"{self._base_url}/everything",
                params=params,
                verify=os.getenv("SSL_CERT_VERIFY", "true") != "false",
            )
            response.raise_for_status()

            data = response.json()
            articles = []

            for article in data.get("articles", []):
                if article["title"] and article["title"] != "[Removed]":
                    try:
                        news_article = Article(
                            title=article["title"],
                            description=article.get("description", ""),
                            url=article["url"],
                            timestamp=int(datetime.fromisoformat(article["publishedAt"]).timestamp()),
                            source=article["source"]["name"],
                            query=query,
                        )
                        articles.append(news_article)
                    except ValueError as e:
                        logger.warning("Skipping article due to error: %s", e)

            logger.info(
                "Found %d articles for %s from %s to %s",
                len(articles), query, start_date, end_date
            )
            logger.debug("[get_stock_news|out] => %d articles", len(articles))
            return articles  # noqa: TRY300

        except requests.exceptions.RequestException as x:
            msg = f"error while fetching news from NewsAPI: {x}"
            logger.exception(msg)
            return []
