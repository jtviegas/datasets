"""Unit tests for ArticlesEtl class."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from datasets import Dataset
from tgedr_dataops.store.hf_dataset import DataFrameSplits
from tgedr_dataops_abs.etl import EtlException
from tgedr_dataops_abs.great_expectations_validation import ValidationError

from tgedr_datasets.article.article import Article
from tgedr_datasets.article.etl import ArticlesEtl, _CONTRACT_PATH


@pytest.fixture
def sample_articles() -> list[Article]:
    """Provide distinct articles for transformation tests."""
    return [
        Article("Apple News", "Apple announces a product", "https://example.com/1", 1, "Source", "AAPL"),
        Article("Google News", "Google reports earnings", "https://example.com/2", 2, "Source", "GOOGL"),
    ]


def test_contract_file_is_packaged() -> None:
    """The contract file must be included alongside the module."""
    assert _CONTRACT_PATH.exists()


def test_initialization() -> None:
    """Initialize internal state and the current Arrow-backed store."""
    with patch("tgedr_datasets.article.etl.HuggingFaceDatasetFileArrowBasedStore") as store_class:
        etl = ArticlesEtl()

    assert etl._data == []
    assert etl._result.empty
    assert etl._store is store_class.return_value
    assert isinstance(etl._processing_time, int)


def test_get_tickers_selects_latest_unique_tickers() -> None:
    """Select distinct tickers from the most recent ticker timestamp."""
    etl = ArticlesEtl()
    etl._store = Mock()
    etl._store.get.return_value = DataFrameSplits(
        train=Dataset.from_pandas(pd.DataFrame({
            "ticker": ["OLD", "AAPL", "AAPL", "MSFT"],
            "actual_time": [1, 2, 2, 2],
        })),
    )

    assert etl._get_tickers("tickers") == ["AAPL", "MSFT"]
    etl._store.get.assert_called_once_with(key="tickers")


def test_extract_uses_configured_tickers_and_aggregates(sample_articles: list[Article]) -> None:
    """Extract one-day news for each ticker in the latest ticker split."""
    timestamp = 1_754_064_000
    etl = ArticlesEtl(configuration={"tickers_dataset": "tickers", "timestamp": timestamp})
    etl._store = Mock()
    etl._store.get.return_value = DataFrameSplits(
        train=Dataset.from_pandas(pd.DataFrame({"ticker": ["AAPL", "MSFT"], "actual_time": [1, 1]})),
    )
    with patch("tgedr_datasets.article.etl.ArticlesAggregator") as aggregator_class:
        aggregator = aggregator_class.return_value
        aggregator.get_news.side_effect = [[sample_articles[0]], [sample_articles[1]]]

        etl.extract()

    assert etl._data == sample_articles
    assert etl._processing_time == timestamp
    assert aggregator.get_news.call_args_list == [
        (("AAPL", timestamp, 1),),
        (("MSFT", timestamp, 1),),
    ]


def test_transform_empty_data() -> None:
    """Transform empty input without emitting a metric."""
    etl = ArticlesEtl()
    with patch("tgedr_datasets.article.etl.Metrics.instance") as metrics_instance:
        etl.transform()

    assert etl._result.empty
    assert "processing_time" in etl._result.columns
    metrics_instance.assert_not_called()


def test_transform_deduplicates_sorts_and_counts(sample_articles: list[Article]) -> None:
    """Deduplicate article IDs, sort rows, and increment the article counter."""
    etl = ArticlesEtl()
    etl._data = [sample_articles[1], sample_articles[0], sample_articles[0]]
    metrics = Mock()
    with patch("tgedr_datasets.article.etl.Metrics.instance", return_value=metrics):
        etl.transform()

    assert etl._result["id"].is_unique
    assert etl._result["id"].tolist() == sorted(etl._result["id"].tolist())
    assert set(etl._result["query"]) == {"AAPL", "GOOGL"}
    assert etl._result["processing_time"].eq(etl._processing_time).all()
    metrics.add_to_counter.assert_called_once_with(
        name="datasets.articles.etl.rows",
        value=2,
        attributes={"type": "articles"},
    )


def test_validate_transform_accepts_contract_data(sample_articles: list[Article]) -> None:
    """Accept data that meets the article contract."""
    etl = ArticlesEtl()
    etl._data = sample_articles
    etl.transform()

    etl.validate_transform()


def test_validate_transform_rejects_null_title(sample_articles: list[Article]) -> None:
    """Reject a transformed row with a null title."""
    etl = ArticlesEtl()
    etl._data = sample_articles
    etl.transform()
    etl._result.loc[0, "title"] = None

    with pytest.raises(ValidationError, match="does not meet contract"):
        etl.validate_transform()


def test_load_updates_configured_dataset() -> None:
    """Convert the result to a Dataset and update the configured store split."""
    etl = ArticlesEtl(configuration={"target_dataset": "articles"})
    etl._store = Mock()
    etl._result = pd.DataFrame({"id": [1], "title": ["Headline"]})

    etl.load()

    etl._store.save.assert_called_once()
    kwargs = etl._store.save.call_args.kwargs
    assert kwargs["key"] == "articles"
    assert kwargs["split"] == "train"
    assert kwargs["append"] is True
    assert isinstance(kwargs["df"], Dataset)
    assert kwargs["df"].to_pandas().equals(etl._result)


def test_configuration_injection_requires_required_values() -> None:
    """The inherited configuration decorator rejects missing dataset keys."""
    etl = ArticlesEtl()

    with pytest.raises(EtlException, match="missing required configuration parameters"):
        etl.extract()
