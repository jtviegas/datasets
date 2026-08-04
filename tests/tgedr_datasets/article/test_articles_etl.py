"""Unit tests for ArticlesEtl class."""
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from pandas.testing import assert_frame_equal

from tgedr_dataops.store.hf_dataset import DataFrameSplits
from tgedr_datasets.article.etl import ArticlesEtl
from tgedr_datasets.article.article import Article


@pytest.fixture
def sample_tickers_df() -> pd.DataFrame:
    """Fixture providing sample tickers DataFrame."""
    return pd.DataFrame({
        "ticker": ["AAPL", "GOOGL", "MSFT"],
        "date": [1702000000, 1702000000, 1701900000]  # Two tickers on latest date
    })


@pytest.fixture
def sample_articles() -> list[Article]:
    """Fixture providing sample articles."""
    return [
        Article(
            title="Apple News",
            description="Apple announces new product",
            url="https://example.com/1",
            timestamp=1702000000,
            source="TestSource",
            query="AAPL",
        ),
        Article(
            title="Google News",
            description="Google reports earnings",
            url="https://example.com/2",
            timestamp=1702000000,
            source="TestSource",
            query="GOOGL",
        ),
    ]


@patch("tgedr_datasets.article.etl.HuggingFaceDatasetStore")
def test_articles_etl_initialization(mock_store_class: Mock) -> None:
    """Test ArticlesEtl initializes correctly."""
    mock_store = Mock()
    mock_store_class.return_value = mock_store

    etl = ArticlesEtl()

    assert etl._data == []
    assert etl._new_data.empty
    assert etl._store == mock_store
    assert isinstance(etl._processing_time, int)


@patch("tgedr_datasets.article.etl.ArticlesAggregator")
def test_extract_with_tickers_and_articles(
    mock_aggregator_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_articles: list[Article],
) -> None:
    """Test extract method processes tickers and aggregates articles."""
    config = {"tickers_dataset": "test_tickers_dataset"}
    etl = ArticlesEtl(configuration=config)

    # Setup mocks
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=sample_tickers_df)
    etl._store = mock_store  # Override the store

    mock_aggregator = Mock()
    mock_aggregator.get_news.side_effect = [
        [sample_articles[0]],  # For AAPL
        [sample_articles[1]],  # For GOOGL
    ]
    mock_aggregator_class.return_value = mock_aggregator

    etl.extract()

    # Verify store.get was called
    mock_store.get.assert_called_once_with(key="test_tickers_dataset")

    # Verify aggregator was called for each ticker on latest date
    assert mock_aggregator.get_news.call_count == 2
    mock_aggregator.get_news.assert_any_call("AAPL", etl._processing_time, 2)
    mock_aggregator.get_news.assert_any_call("GOOGL", etl._processing_time, 2)

    # Verify articles were added to _data
    assert len(etl._data) == 2
    assert etl._data[0].query == "AAPL"
    assert etl._data[1].query == "GOOGL"


@patch("tgedr_datasets.article.etl.ArticlesAggregator")
def test_extract_with_base_date_parameter(
    mock_aggregator_class: Mock,
    sample_tickers_df: pd.DataFrame,
) -> None:
    """Test extract method uses base_date parameter when provided."""
    config = {"tickers_dataset": "test_tickers_dataset", "base_date": 1702100000}
    etl = ArticlesEtl(configuration=config)

    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=sample_tickers_df.sort_values("ticker", ascending=False))  # Ensure latest date is first
    etl._store = mock_store

    mock_aggregator = Mock()
    mock_aggregator.get_news.return_value = []
    mock_aggregator_class.return_value = mock_aggregator

    etl.extract()

    # Verify aggregator called with custom base_date from config
    mock_aggregator.get_news.assert_any_call("AAPL", etl._processing_time, 2)


@patch("tgedr_datasets.article.etl.ArticlesAggregator")
def test_transform_creates_dataframe(
    mock_aggregator_class: Mock,
    sample_articles: list[Article],
) -> None:
    """Test transform method creates DataFrame."""
    mock_aggregator_class.return_value = Mock()

    etl = ArticlesEtl()
    etl._data = sample_articles

    etl.transform()

    # Verify DataFrame was created
    assert not etl._new_data.empty
    assert etl._new_data.shape[0] == 2

    # Verify columns exist
    expected_columns = ["id", "query", "timestamp", "title", "description", "url", "source", "processing_time"]
    for col in expected_columns:
        assert col in etl._new_data.columns

    # Verify processing_time was added
    assert all(etl._new_data["processing_time"] == etl._processing_time)

    # Verify article data is sorted by id
    assert etl._new_data.iloc[0]["query"] == "AAPL"
    assert etl._new_data.iloc[1]["query"] == "GOOGL"


@patch("tgedr_datasets.article.etl.ArticlesAggregator")
def test_transform_empty_data(
    mock_aggregator_class: Mock,
) -> None:
    """Test transform method handles empty article data."""
    mock_aggregator_class.return_value = Mock()

    etl = ArticlesEtl()
    etl._data = []  # No articles

    with pytest.raises(KeyError, match="id"):
        etl.transform()

def test_load_calls_store_update() -> None:
    """Test load method calls store.update with a single DataFrameSplits train split."""
    mock_store = Mock()

    etl = ArticlesEtl(configuration={"target_dataset": "test_target_dataset"})
    etl._store = mock_store  # Override the store

    # Set up some test data
    data = pd.DataFrame({
        "id": [1, 2],
        "query": ["AAPL", "GOOGL"],
        "processing_time": [[1712345600], [1712345600]]
    })
    etl._new_data = data

    target_dataset = "test_target_dataset"
    etl.load()

    # Verify store.update was called once with a DataFrameSplits
    mock_store.update.assert_called_once()
    call = mock_store.update.call_args
    assert call[1]["key"] == target_dataset
    assert call[1]["append"] is True
    assert isinstance(call[1]["df"], DataFrameSplits)
    assert_frame_equal(call[1]["df"].train, data)
