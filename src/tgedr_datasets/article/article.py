"""News article data models.

This module defines data structures for representing news articles with
market timing classifications and sentiment analysis.
"""

from dataclasses import dataclass
import hashlib
from datetime import datetime, UTC

@dataclass
class Article:
    """Data class for news articles with epoch timestamp in published_at field.

    Attributes:
        title: Article headline
        description: Article description or summary
        url: URL to the full article
        timestamp: Unix epoch timestamp in seconds (UTC)
        source: News source name
        query: Query string that found the article (e.g., stock ticker)
        id: Unique identifier composed of query hash concatenated with timestamp

    """

    title: str
    description: str
    url: str
    timestamp: int
    source: str
    query: str
    # sentiment and relevance fields removed

    @property
    def id(self) -> int:
        """Generate a unique identifier from query, source, url and timestamp.

        Combines the hash of the query, source, url and timestamp to create
        a unique integer identifier for this news article. The URL is included
        because it is both stable (the same article always has the same URL) and
        unique (different articles have different URLs), which prevents hash
        collisions while keeping ids stable across re-fetches.

        Returns:
            int: Unique identifier as integer

        """
        # Create deterministic hash using SHA-256
        hash_input = f"{self.query}:{self.source}:{self.url}:{self.timestamp}".encode()
        hash_digest = hashlib.sha256(hash_input).hexdigest()
        # Convert first 16 characters of hex to integer (64 bits)
        return int(hash_digest[:16], 16)

    def __eq__(self, other: object) -> bool:
        """Compare two Article instances for equality.

        Two articles are equal when all their fields (title, description, url,
        timestamp, source, query) match.

        Args:
            other: The object to compare against.

        Returns:
            bool: True if the other object is an Article with identical fields.
        """
        if not isinstance(other, Article):
            return NotImplemented
        return (
            self.title == other.title
            and self.description == other.description
            and self.url == other.url
            and self.timestamp == other.timestamp
            and self.source == other.source
            and self.query == other.query
        )

    def __hash__(self) -> int:
        """Compute a hash for the Article instance.

        Returns:
            int: Hash based on the immutable fields of the article.
        """
        return hash((self.title, self.description, self.url, self.timestamp, self.source, self.query))

    def to_pd_df_row(self) -> dict[str, object]:
        """Convert Article instance to a dictionary suitable for pandas DataFrame row.

        Returns:
            dict[str, object]: Dictionary with all Article fields including the id property.
                              Keys match the attribute names and can be used directly with
                              pandas DataFrame constructor.

        Example:
            >>> article = Article(
            ...     title="Market Update",
            ...     description="Stock markets rise",
            ...     url="https://example.com/article",
            ...     timestamp=1701388800,
            ...     source="Financial Times",
            ...     query="AAPL",
            ... )
            >>> row = article.to_pd_df_row()
            >>> import pandas as pd
            >>> df = pd.DataFrame([row])

        """
        return {
            "id": self.id,
            "query": self.query,
            "actual_time": self.timestamp,
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "source": self.source,
        }

    def __str__(self) -> str:
        """Return a formatted string representation of the news article.

        Returns:
            str: Formatted string with all article properties, with timestamp in human-readable format.

        """
        # Format timestamp as human-readable date
        formatted_date = datetime.fromtimestamp(self.timestamp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        return (
            f"Article: {formatted_date} - {self.title}\n"
            f"  query: {self.query}\n"
            f"  description: {self.description}\n"
            f"  source: {self.source} | url: {self.url}\n"
        )
