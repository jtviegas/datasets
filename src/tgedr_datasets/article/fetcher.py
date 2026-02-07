"""Abstract base class for article fetchers.

This module defines the ArticleFetcher abstract base class that establishes
the interface contract for all article fetcher implementations. Following the
Open-Closed Principle, new fetcher implementations can extend this base class
without modifying existing code.
"""

from abc import ABC, abstractmethod

from tgedr_datasets.article.article import Article


class ArticleFetcher(ABC):
    """Abstract base class for fetching news articles related to stock tickers.

    This class defines the interface that all concrete article fetcher implementations
    must follow. It enforces a consistent API across different news sources while
    allowing each implementation to handle its specific data source requirements.

    The Open-Closed Principle is applied here: this interface is closed for
    modification but open for extension through concrete implementations.
    """

    @abstractmethod
    def get_articles(
        self,
        query: str,
        start_date: int,
        end_date: int | None = None,
        extra_query: str | None = None,
        max_articles: int = 100,
    ) -> list[Article]:
        """Fetch news articles for a given stock query within a date range.

        This abstract method must be implemented by all concrete fetcher classes.
        Each implementation will fetch articles from its specific data source.

        Args:
            query: query term (e.g., ticker symbol like "AAPL" or company name like "Apple Inc.").
            start_date: Start of date range as UTC epoch timestamp (seconds)
            end_date: End of date range as UTC epoch timestamp (seconds).
                     If None, defaults to current time.
            extra_query: Optional extra query string for enhanced search queries.
                         Some APIs can use this for better results.
            max_articles: Maximum number of articles to fetch. Default is 100.

        Returns:
            List of Article objects sorted by timestamp, or empty list on error.

        Raises:
            NotImplementedError: If called directly on the abstract base class.

        """
        raise NotImplementedError

    @staticmethod
    def filter_by_description(
        articles: list[Article],
        query: str,
        extra_query: str | None = None,
    ) -> list[Article]:
        """Filter articles whose description contains the query or extra query.

        Checks each article's description field to see if it contains the query
        symbol or extra query (case-sensitive). Articles matching either
        criterion are included in the results.

        Args:
            articles: List of Article instances to filter.
            query: query term to search for (e.g., "AAPL", "GOOGL").
            extra_query: Optional extra query string to search for (e.g., "Apple Inc.").
                    If None, only the query is used for filtering.

        Returns:
            List of Article instances whose description contains the query
            or extra query. Returns empty list if no articles match.
            The original order of articles is preserved.

        Example:
            >>> articles = [
            ...     Article(title="News", description="Apple Inc. reports earnings", ...),
            ...     Article(title="News", description="Market update today", ...),
            ... ]
            >>> filtered = ArticleFetcher.filter_by_description(articles, "AAPL", "Apple Inc.")
            >>> len(filtered)  # Returns 1, only the first article matches
            1

        """
        if not articles:
            return []

        filtered_articles = []

        for article in articles:
            # Check if description contains query or extra query (case-sensitive)
            text_to_match = article.description
            if article.description is None:
                text_to_match = article.title

            if query in text_to_match or (extra_query and extra_query in text_to_match):
                filtered_articles.append(article)

        return filtered_articles

