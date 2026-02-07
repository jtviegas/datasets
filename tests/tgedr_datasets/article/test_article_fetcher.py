"""Tests for article fetcher base class edge cases."""

import pytest

from tgedr_datasets.article.article import Article
from tgedr_datasets.article.fetcher import ArticleFetcher



class ConcreteArticleFetcher(ArticleFetcher):
    """Concrete implementation for testing abstract methods."""

    def get_articles(
        self,
        query: str,
        start_date: int,
        end_date: int | None = None,
        extra_query: str | None = None,
        max_articles: int = 100,
    ) -> list[Article]:
        """Implement abstract method."""
        return []


def test_article_fetcher_filter_by_description_empty_list() -> None:
    """Test filter_by_description with empty list returns empty list."""
    result = ArticleFetcher.filter_by_description([], "AAPL")
    assert result == []


def test_article_fetcher_filter_by_description_ticker_only() -> None:
    """Test filter_by_description matches on query only."""
    articles = [
        Article(
            title="AAPL News",
            description="Apple Inc. AAPL reports earnings",
            url="https://example.com/1",
            timestamp=1701388800,
            source="Finance",
            query="AAPL",
        ),
        Article(
            title="Microsoft News",
            description="MSFT reports earnings",
            url="https://example.com/2",
            timestamp=1701388800,
            source="Finance",
            query="MSFT",
        ),
    ]
    
    result = ArticleFetcher.filter_by_description(articles, "AAPL")
    assert len(result) == 1
    assert result[0].query == "AAPL"


def test_article_fetcher_filter_by_description_company_name() -> None:
    """Test filter_by_description matches on company name."""
    articles = [
        Article(
            title="Apple Inc. News",
            description="Apple Inc. reports earnings",
            url="https://example.com/1",
            timestamp=1701388800,
            source="Finance",
            query="AAPL",
        ),
        Article(
            title="Microsoft News",
            description="Microsoft Corp reports earnings",
            url="https://example.com/2",
            timestamp=1701388800,
            source="Finance",
            query="MSFT",
        ),
    ]
    
    result = ArticleFetcher.filter_by_description(articles, "AAPL", "Apple Inc.")
    assert len(result) == 1
    assert result[0].query == "AAPL"



def test_article_fetcher_filter_by_description_case_sensitive() -> None:
    """Test filter_by_description is case-sensitive."""
    articles = [
        Article(
            title="Apple News",
            description="apple inc news",  # lowercase
            url="https://example.com/1",
            timestamp=1701388800,
            source="Finance",
            query="AAPL",
        ),
    ]
    
    # Should not match because AAPL is uppercase
    result = ArticleFetcher.filter_by_description(articles, "aapl")
    assert len(result) == 0


def test_article_fetcher_filter_by_description_multiple_matches() -> None:
    """Test filter_by_description returns all matching articles."""
    articles = [
        Article(
            title="AAPL Earnings",
            description="AAPL reports earnings",
            url="https://example.com/1",
            timestamp=1701388800,
            source="Finance",
            query="AAPL",
        ),
        Article(
            title="AAPL Stock",
            description="AAPL stock rises",
            url="https://example.com/2",
            timestamp=1701388801,
            source="Finance",
            query="AAPL",
        ),
    ]
    
    result = ArticleFetcher.filter_by_description(articles, "AAPL")
    assert len(result) == 2


def test_article_fetcher_filter_by_description_preserves_order() -> None:
    """Test filter_by_description preserves original article order."""
    articles = [
        Article(
            title="First AAPL",
            description="First article about AAPL",
            url="https://example.com/1",
            timestamp=1701388800,
            source="Finance",
            query="AAPL",
        ),
        Article(
            title="Other",
            description="Other article",
            url="https://example.com/2",
            timestamp=1701388801,
            source="Finance",
            query="MSFT",
        ),
        Article(
            title="Second AAPL",
            description="Second article about AAPL",
            url="https://example.com/3",
            timestamp=1701388802,
            source="Finance",
            query="AAPL",
        ),
    ]
    
    result = ArticleFetcher.filter_by_description(articles, "AAPL")
    assert len(result) == 2
    assert result[0].url == "https://example.com/1"
    assert result[1].url == "https://example.com/3"


def test_article_fetcher_filter_by_description_company_with_spaces() -> None:
    """Test filter_by_description handles company names with spaces."""
    articles = [
        Article(
            title="News",
            description="Apple Inc. reports",
            url="https://example.com/1",
            timestamp=1701388800,
            source="Finance",
            query="AAPL",
        ),
    ]
    
    result = ArticleFetcher.filter_by_description(articles, "AAPL", "Apple Inc.")
    assert len(result) == 1


def test_article_fetcher_filter_by_description_no_matches() -> None:
    """Test filter_by_description returns empty list when no matches."""
    articles = [
        Article(
            title="Microsoft News",
            description="Microsoft reports",
            url="https://example.com/1",
            timestamp=1701388800,
            source="Finance",
            query="MSFT",
        ),
    ]
    
    result = ArticleFetcher.filter_by_description(articles, "AAPL")
    assert len(result) == 0


def test_article_fetcher_filter_by_description_none_description() -> None:
    """Test filter_by_description with None description falls back to title."""
    articles = [
        Article(
            title="AAPL News Article",
            description=None,  # No description, should check title
            url="https://example.com/1",
            timestamp=1701388800,
            source="Finance",
            query="AAPL",
        ),
    ]
    
    result = ArticleFetcher.filter_by_description(articles, "AAPL")
    assert len(result) == 1
    assert result[0].title == "AAPL News Article"


def test_article_fetcher_filter_by_description_none_description_no_match() -> None:
    """Test filter_by_description with None description when no match in title."""
    articles = [
        Article(
            title="Microsoft News",
            description=None,
            url="https://example.com/1",
            timestamp=1701388800,
            source="Finance",
            query="MSFT",
        ),
    ]
    
    result = ArticleFetcher.filter_by_description(articles, "AAPL")
    assert len(result) == 0


def test_concrete_fetcher_implementation() -> None:
    """Test ConcreteArticleFetcher can be instantiated."""
    fetcher = ConcreteArticleFetcher()
    articles = fetcher.get_articles("AAPL", 1701388800)
    assert articles == []


def test_article_fetcher_abstract_method_raises_not_implemented() -> None:
    """Test that ArticleFetcher.get_stock_news raises NotImplementedError when called directly."""
    with pytest.raises(NotImplementedError):
        ArticleFetcher.get_articles(None, "AAPL", 1701388800)
