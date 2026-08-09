"""Unit tests for Article class."""

from datetime import datetime, UTC

import pytest

from tgedr_datasets.article.article import Article


@pytest.fixture
def valid_article() -> Article:
    """Provide a valid Article instance with all fields."""
    return Article(
        title="Apple Stock Surges",
        description="AAPL rises on strong quarterly results",
        url="https://example.com/news/apple-surge",
        timestamp=1701388800,
        source="Financial Times",
        query="AAPL",
    )


@pytest.fixture
def article_without_scores() -> Article:
    """Provide an Article without sentiment and relevance scores."""
    return Article(
        title="Market Update",
        description="General market news",
        url="https://example.com/news/market",
        timestamp=1701388800,
        source="MarketWatch",
        query="MSFT",
    )


def test_article_creation(valid_article: Article) -> None:
    """Test basic Article instantiation."""
    assert valid_article.title == "Apple Stock Surges"
    assert valid_article.description == "AAPL rises on strong quarterly results"
    assert valid_article.url == "https://example.com/news/apple-surge"
    assert valid_article.timestamp == 1701388800
    assert valid_article.source == "Financial Times"
    assert valid_article.query == "AAPL"


# sentiment and relevance were removed from Article; optional-score tests removed


def test_article_id_property(valid_article: Article) -> None:
    """Test Article.id property generates unique identifier."""
    id1 = valid_article.id
    id2 = valid_article.id
    assert id1 == id2  # Should be consistent

    # Different query should have different id
    article2 = Article(
        title="Microsoft News",
        description="MSFT update",
        url="https://example.com/news/msft",
        timestamp=1701388800,
        source="Reuters",
        query="MSFT",
    )
    assert article2.id != valid_article.id


def test_article_id_different_timestamp(valid_article: Article) -> None:
    """Test Article.id differs with different timestamp."""
    article2 = Article(
        title="Apple Stock Surges",
        description="AAPL rises on strong quarterly results",
        url="https://example.com/news/apple-surge",
        timestamp=1701388801,  # Different timestamp
        source="Financial Times",
        query="AAPL",
    )
    assert article2.id != valid_article.id

def test_article_id_different_source(valid_article: Article) -> None:
    """Test Article.id differs with different source."""
    article2 = Article(
        title="Apple Stock Surges",
        description="AAPL rises on strong quarterly results",
        url="https://example.com/news/apple-surge",
        timestamp=valid_article.timestamp,
        source="Reuters",  # Different source
        query="AAPL",
    )
    assert article2.id != valid_article.id

def test_article_to_pd_df_row(valid_article: Article) -> None:
    """Test converting Article to pandas DataFrame row dictionary."""
    row = valid_article.to_pd_df_row()

    # Verify all fields are present
    assert "id" in row
    assert "query" in row
    assert "timestamp" in row
    assert "title" in row
    assert "description" in row
    assert "url" in row
    assert "source" in row

    # Verify values
    assert row["query"] == "AAPL"
    assert row["timestamp"] == 1701388800
    assert row["title"] == "Apple Stock Surges"
    assert row["description"] == "AAPL rises on strong quarterly results"
    assert row["url"] == "https://example.com/news/apple-surge"
    assert row["source"] == "Financial Times"
    assert row["id"] == valid_article.id


# removed tests for None sentiment/relevance


def test_article_str_representation(valid_article: Article) -> None:
    """Test Article string representation."""
    article_str = str(valid_article)

    # Check key components are present
    assert "Article:" in article_str
    assert valid_article.title in article_str
    assert "query: AAPL" in article_str
    assert valid_article.description in article_str
    assert "Financial Times" in article_str
    # sentiment/relevance not part of string representation anymore


def test_article_str_representation_with_timestamp() -> None:
    """Test Article string representation includes formatted timestamp."""
    # Use a known timestamp: 2023-12-01 00:00:00 UTC
    article = Article(
        title="Test Article",
        description="Test description",
        url="https://example.com",
        timestamp=1701388800,
        source="TestSource",
        query="TEST",
    )
    article_str = str(article)

    # Formatted date should appear in string
    assert "2023-12-01" in article_str
    assert "UTC" in article_str


def test_article_str_representation_without_scores() -> None:
    """Test Article string representation with None sentiment/relevance."""
    article = Article(
        title="Test",
        description="Desc",
        url="https://example.com",
        timestamp=1701388800,
        source="Test",
        query="TEST",
    )
    article_str = str(article)

    # sentiment/relevance removed; ensure their labels are not present
    assert "sentiment" not in article_str
    assert "relevance" not in article_str


def test_article_equality() -> None:
    """Test Article equality comparison."""
    article1 = Article(
        title="Test",
        description="Desc",
        url="https://example.com",
        timestamp=1701388800,
        source="Test",
        query="AAPL",
    )
    article2 = Article(
        title="Test",
        description="Desc",
        url="https://example.com",
        timestamp=1701388800,
        source="Test",
        query="AAPL",
    )
    assert article1 == article2


def test_article_inequality_title() -> None:
    """Test Article inequality with different title."""
    article1 = Article(
        title="Title 1",
        description="Desc",
        url="https://example.com",
        timestamp=1701388800,
        source="Test",
        query="AAPL",
    )
    article2 = Article(
        title="Title 2",
        description="Desc",
        url="https://example.com",
        timestamp=1701388800,
        source="Test",
        query="AAPL",
    )
    assert article1 != article2


def test_article_mutability() -> None:
    """Test Article fields can be modified (not frozen)."""
    article = Article(
        title="Original",
        description="Desc",
        url="https://example.com",
        timestamp=1701388800,
        source="Test",
        query="AAPL",
    )

    # Article should be mutable (unlike Price which is frozen)
    article.title = "Modified"
    assert article.title == "Modified"

