"""Finnhub article fetcher implementation.

This module provides an article fetcher that retrieves news from Finnhub's
free News API. Finnhub offers a generous free tier with 60 API calls per minute.
"""

import logging
import os
import time
from datetime import datetime, UTC


import finnhub
from tgedr_datasets.article.article import Article
from tgedr_datasets.article.fetcher import ArticleFetcher

logger = logging.getLogger(__name__)


class FinnhubNewsFetcher(ArticleFetcher):
    """Fetches news from Finnhub API.

    This fetcher retrieves news articles for specific stock tickers using the
    Finnhub News API. It requires a FINNHUB_API_KEY environment variable.

    Free tier limits: 60 API calls per minute

    Attributes:
        _session: Requests session with API key configured
        _base_url: Finnhub API base URL

    Example:
        >>> fetcher = FinnhubNewsFetcher()
        >>> articles = fetcher.get_articles("AAPL", start_date=1701388800, end_date=1701475200)
        >>> print(f"Found {len(articles)} articles")

    """

    def __init__(self) -> None:
        """Initialize Finnhub article fetcher with API key from environment.

        Raises:
            ValueError: If FINNHUB_API_KEY environment variable is not set

        """
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            msg = "FINNHUB_API_KEY environment variable must be set"
            raise ValueError(msg)
        self._client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
        logger.info("Initialized FinnhubNewsFetcher")

    def get_articles(
        self,
        query: str,
        start_date: int,
        end_date: int | None = None,
        extra_query: str | None = None,  # noqa: ARG002
        max_articles: int = 100,  # noqa: ARG002
    ) -> list[Article]:
        """Fetch news articles for a query from Finnhub API.

        Args:
            query: Stock query symbol
            start_date: Start date as UTC epoch timestamp in seconds
            end_date: End date as UTC epoch timestamp in seconds (defaults to current time)
            extra_query: Additional query string (unused for Finnhub API)
            max_articles: Maximum articles to retrieve (unused - Finnhub controls limit)

        Returns:
            list[Article]: List of news articles, empty list on error

        """
        logger.debug("Fetching news for query=%s from Finnhub", query)

        if end_date is None:
            end_date = int(datetime.now(UTC).timestamp())

        # Convert epoch timestamps to Finnhub date format (YYYY-MM-DD)
        start_date_str = datetime.fromtimestamp(start_date, tz=UTC).strftime("%Y-%m-%d")
        end_date_str = datetime.fromtimestamp(end_date, tz=UTC).strftime("%Y-%m-%d")

        try:
            # Add 1-second delay to respect rate limits
            time.sleep(1)
            data = self._client.company_news(query, _from=start_date_str, to=end_date_str)

            articles = []
            for item in data:
                try:
                    # Parse Finnhub response
                    # Expected fields: datetime (epoch), headline, summary, url, source
                    article = Article(
                        title=item["headline"],
                        description=item.get("summary", ""),
                        url=item["url"],
                        timestamp=item["datetime"],  # Already in epoch format
                        source=item["source"],
                        query=query
                    )
                    articles.append(article)

                except (KeyError, TypeError, ValueError):
                    logger.exception("Error parsing Finnhub article: %s", item)
                    continue
            logger.debug("Retrieved %d articles for query=%s from Finnhub", len(articles), query)

        except Exception:
            logger.exception("Failed to fetch news from Finnhub for query=%s", query)
            return []
        else:
            return articles
