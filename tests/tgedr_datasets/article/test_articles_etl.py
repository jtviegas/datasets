"""Unit tests for ArticlesEtl class."""
import time
from datetime import datetime, UTC
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from pandas.testing import assert_frame_equal

from tgedr_dataops.store.hf_dataset import DataFrameSplits
from tgedr_dataops_abs.great_expectations_validation import ValidationError
from tgedr_datasets.article.etl import ArticlesEtl, _CONTRACT_PATH
from tgedr_datasets.article.article import Article


def test_contract_file_is_packaged() -> None:
    """Test that the ODCS data contract file is present next to the module.

    Guards against packaging regressions where the ``*.odcs.yaml`` file is
    omitted from the wheel, which would break contract validation at runtime.
    """
    assert _CONTRACT_PATH.exists(), f"missing data contract: {_CONTRACT_PATH}"


@pytest.fixture
def sample_tickers_df() -> pd.DataFrame:
    """Fixture providing sample tickers DataFrame."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "ticker": ["AAPL", "GOOGL", "MSFT"],
        "date": [int(time.time()), int(time.time()), int(time.time())]  # Two tickers on latest date
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
    config = {"tickers_dataset": "test_tickers_dataset", "target_dataset": "test_articles"}
    etl = ArticlesEtl(configuration=config)

    # Setup mocks: articles first (for _find_missing_dates), then tickers
    mock_store = Mock()
    mock_store.get.side_effect = [
        DataFrameSplits(train=pd.DataFrame({"timestamp": [int(time.time())]})),  # articles (no gaps)
        DataFrameSplits(train=sample_tickers_df),  # tickers
    ]
    etl._store = mock_store  # Override the store

    mock_aggregator = Mock()
    mock_aggregator.get_news.side_effect = [
        [sample_articles[0]],  # For AAPL
        [sample_articles[1]],  # For GOOGL
        [],  # For MSFT (no articles)
    ]
    mock_aggregator_class.return_value = mock_aggregator

    etl.extract()

    # Verify store.get was called twice (articles + tickers)
    assert mock_store.get.call_count == 2

    # Verify aggregator was called for each ticker on latest date
    assert mock_aggregator.get_news.call_count == 3
    mock_aggregator.get_news.assert_any_call("AAPL", etl._processing_time, 2)
    mock_aggregator.get_news.assert_any_call("GOOGL", etl._processing_time, 2)
    mock_aggregator.get_news.assert_any_call("MSFT", etl._processing_time, 2)

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
    config = {"tickers_dataset": "test_tickers_dataset", "target_dataset": "test_articles", "base_date": int(time.time())}
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

    etl.transform()

    assert etl._metrics.get("row_count") == 0
    assert etl._metrics.get("duplicate_id_count") == 0
    assert etl._metrics.get("empty_title_count") == 0
    assert etl._metrics.get("empty_description_count") == 0


@patch("tgedr_datasets.article.etl.ArticlesAggregator")
def test_transform_collects_metrics(
    mock_aggregator_class: Mock,
    sample_articles: list[Article],
) -> None:
    """Test transform collects data quality metrics."""
    mock_aggregator_class.return_value = Mock()

    etl = ArticlesEtl()
    etl._data = sample_articles

    etl.transform()

    assert etl._metrics.get("row_count") == 2
    assert etl._metrics.get("duplicate_id_count") == 0
    assert etl._metrics.get("empty_title_count") == 0
    assert etl._metrics.get("empty_description_count") == 0


@patch("tgedr_datasets.article.etl.ArticlesAggregator")
def test_transform_deduplicates_by_id(
    mock_aggregator_class: Mock,
) -> None:
    """Test transform deduplicates articles with the same id."""
    mock_aggregator_class.return_value = Mock()

    # Two articles with identical fields (same id) and one unique
    ts = int(time.time())
    articles = [
        Article(title="Apple News", description="Apple announces", url="https://example.com/1", timestamp=ts, source="S", query="AAPL"),
        Article(title="Apple News", description="Apple announces", url="https://example.com/1", timestamp=ts, source="S", query="AAPL"),
        Article(title="Google News", description="Google reports", url="https://example.com/2", timestamp=ts, source="S", query="GOOGL"),
    ]
    etl = ArticlesEtl()
    etl._data = articles

    etl.transform()

    # Duplicate id should be removed
    assert etl._new_data.shape[0] == 2
    assert etl._new_data["id"].is_unique
    assert etl._metrics.get("duplicate_id_count") == 0


@patch("tgedr_datasets.article.etl.HuggingFaceDatasetStore")
def test_find_missing_dates_empty_dataset(mock_store_class: Mock) -> None:
    """Test _find_missing_dates returns today when the dataset is empty."""
    etl = ArticlesEtl()
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=pd.DataFrame())
    etl._store = mock_store

    result = etl._find_missing_dates("test/articles")

    assert result == [int(datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())]


@patch("tgedr_datasets.article.etl.HuggingFaceDatasetStore")
def test_find_missing_dates_unreadable_dataset(mock_store_class: Mock) -> None:
    """Test _find_missing_dates returns today when the dataset cannot be read."""
    etl = ArticlesEtl()
    mock_store = Mock()
    mock_store.get.side_effect = Exception("boom")
    etl._store = mock_store

    result = etl._find_missing_dates("test/articles")

    assert result == [int(datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())]


@patch("tgedr_datasets.article.etl.HuggingFaceDatasetStore")
def test_find_missing_dates_no_gaps(mock_store_class: Mock) -> None:
    """Test _find_missing_dates returns empty when all dates are present."""
    etl = ArticlesEtl()
    now = int(time.time())
    df = pd.DataFrame({"timestamp": [now, now - 86400]})
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=df)
    etl._store = mock_store

    result = etl._find_missing_dates("test/articles")

    assert result == []


@patch("tgedr_datasets.article.etl.HuggingFaceDatasetStore")
def test_find_missing_dates_returns_gaps(mock_store_class: Mock) -> None:
    """Test _find_missing_dates returns missing date timestamps."""
    etl = ArticlesEtl()
    now = int(time.time())
    # Dataset has an older date (3 days ago) but today is present; yesterday is missing
    df = pd.DataFrame({"timestamp": [now - 86400 * 3, now]})
    mock_store = Mock()
    mock_store.get.return_value = DataFrameSplits(train=df)
    etl._store = mock_store

    result = etl._find_missing_dates("test/articles")

    # Should have missing dates (yesterday and day before yesterday at minimum)
    assert len(result) >= 2
    # All results should be midnight timestamps
    for ts in result:
        dt = datetime.fromtimestamp(ts, tz=UTC)
        assert dt.hour == 0 and dt.minute == 0 and dt.second == 0


@patch("tgedr_datasets.article.etl.ArticlesAggregator")
@patch("tgedr_datasets.article.etl.HuggingFaceDatasetStore")
def test_extract_backfills_missing_dates(
    mock_store_class: Mock,
    mock_aggregator_class: Mock,
    sample_tickers_df: pd.DataFrame,
    sample_articles: list[Article],
) -> None:
    """Test extract backfills missing dates from the articles dataset."""
    config = {"tickers_dataset": "test/tickers", "target_dataset": "test/articles"}
    etl = ArticlesEtl(configuration=config)

    old_ts = int(time.time()) - 86400 * 3  # 3 days ago
    mock_store = Mock()
    mock_store.get.side_effect = [
        DataFrameSplits(train=pd.DataFrame({"timestamp": [old_ts]})),  # articles (gaps to today)
        DataFrameSplits(train=sample_tickers_df),  # tickers
    ]
    mock_store_class.return_value = mock_store

    mock_aggregator = Mock()
    mock_aggregator.get_news.return_value = sample_articles[:1]
    mock_aggregator_class.return_value = mock_aggregator

    etl._store = mock_store

    etl.extract()

    # Should fetch for each missing date (at least yesterday)
    assert mock_aggregator.get_news.call_count >= 3  # 3 tickers * at least 1 missing date


@patch("tgedr_datasets.article.etl.ArticlesAggregator")
def test_transform_metrics_with_empty_fields(
    mock_aggregator_class: Mock,
) -> None:
    """Test transform metrics with empty title and description."""
    mock_aggregator_class.return_value = Mock()

    articles = [
        Article(title="", description="Some desc", url="https://example.com/1", timestamp=1702000000, source="S", query="AAPL"),
        Article(title="Title", description="", url="https://example.com/2", timestamp=1702000000, source="S", query="GOOGL"),
        Article(title="  ", description="  ", url="https://example.com/3", timestamp=1702000000, source="S", query="MSFT"),
    ]
    etl = ArticlesEtl()
    etl._data = articles

    etl.transform()

    assert etl._metrics.get("row_count") == 3
    assert etl._metrics.get("empty_title_count") == 2
    assert etl._metrics.get("empty_description_count") == 2


def test_load_saves_metrics(tmp_path) -> None:
    """Test load method saves metrics to CSV."""
    mock_store = Mock()

    etl = ArticlesEtl(configuration={"target_dataset": "test_target", "metrics_dir": str(tmp_path)})
    etl._store = mock_store

    data = pd.DataFrame({
        "id": [1, 2],
        "query": ["AAPL", "GOOGL"],
        "processing_time": [1712345600, 1712345600],
    })
    etl._new_data = data
    etl._metrics.set("row_count", 2)

    etl.load()

    metrics_file = tmp_path / "articles.csv"
    assert metrics_file.exists()

    import csv
    with metrics_file.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["row_count"] == "2"


def test_load_calls_store_update() -> None:
    """Test load method calls store.update with a single DataFrameSplits train split."""
    mock_store = Mock()

    etl = ArticlesEtl(configuration={"target_dataset": "test_target_dataset", "metrics_dir": "/tmp/test_metrics"})
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


@patch("tgedr_datasets.article.etl.ArticlesAggregator")
def test_validate_transform_success(
    mock_aggregator_class: Mock,
    sample_articles: list[Article],
) -> None:
    """Test validate_transform passes for data meeting the contract."""
    mock_aggregator_class.return_value = Mock()

    etl = ArticlesEtl()
    etl._data = sample_articles
    etl.transform()

    etl.validate_transform()


@patch("tgedr_datasets.article.etl.ArticlesAggregator")
def test_validate_transform_null_title_fails(
    mock_aggregator_class: Mock,
    sample_articles: list[Article],
) -> None:
    """Test validate_transform raises ValidationError for null titles."""
    mock_aggregator_class.return_value = Mock()

    etl = ArticlesEtl()
    etl._data = sample_articles
    etl.transform()
    etl._new_data.loc[0, "title"] = None

    with pytest.raises(ValidationError, match="does not meet contract"):
        etl.validate_transform()
