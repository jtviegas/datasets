"""Unit tests for PricesEtl."""

from datetime import UTC, datetime, tzinfo
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from datasets import Dataset
from tgedr_dataops.store.data_splits import DatasetSplits
from tgedr_dataops_abs.etl import EtlException
from tgedr_dataops_abs.great_expectations_validation import ValidationError

import tgedr_datasets.prices.etl as etl_module
from tgedr_datasets.prices.etl import PricesEtl, _CONTRACT_PATH
from tgedr_datasets.prices.price import Price


@pytest.fixture
def fixed_now() -> datetime:
    """Freeze the ETL clock at noon UTC on a fixed date."""
    fixed = datetime(2026, 8, 5, 12, tzinfo=UTC)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return fixed if tz is None else fixed.astimezone(tz)

    with patch.object(etl_module, "datetime", FixedDatetime):
        yield fixed


@pytest.fixture
def sample_prices() -> list[Price]:
    """Provide valid daily prices for extraction and transform tests."""
    return [
        Price("AAPL", 1_754_323_200, 150.0, 155.0, 149.0, 154.0, 1_000_000),
        Price("MSFT", 1_754_323_200, 300.0, 305.0, 299.0, 304.0, 800_000),
    ]


def ticker_splits(tickers: list[str], timestamps: list[int]) -> DatasetSplits:
    """Build a realistic Arrow-backed ticker split."""
    return DatasetSplits(
        train=Dataset.from_pandas(pd.DataFrame({"ticker": tickers, "actual_time": timestamps}))
    )


def test_contract_file_is_packaged() -> None:
    """The ODCS contract must be present beside the ETL module."""
    assert _CONTRACT_PATH.exists()


def test_initialization() -> None:
    """Initialize clean state and the Arrow-backed store."""
    with patch("tgedr_datasets.prices.etl.HuggingFaceDatasetFileArrowBasedStore") as store_class:
        etl = PricesEtl()

    assert etl._data == []
    assert etl._result.empty
    assert etl._store is store_class.return_value


def test_get_tickers_uses_latest_timestamp_and_deduplicates() -> None:
    """Select unique tickers at the greatest actual_time from the Arrow split."""
    etl = PricesEtl()
    etl._store = Mock()
    etl._store.get.return_value = ticker_splits(
        ["OLD", "AAPL", "AAPL", "MSFT"], [1, 2, 2, 2]
    )

    assert etl._get_tickers("tickers") == ["AAPL", "MSFT"]
    etl._store.get.assert_called_once_with(key="tickers")


@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_uses_midnight_today_when_date_omitted(
    fetcher_class: Mock, sample_prices: list[Price], fixed_now: datetime
) -> None:
    """Use today's UTC midnight when no price_date is configured."""
    etl = PricesEtl(configuration={"tickers_dataset": "tickers", "target_dataset": "prices"})
    etl._store = Mock()
    etl._store.get.return_value = ticker_splits(["AAPL", "MSFT"], [1, 1])
    fetcher = fetcher_class.get_instance.return_value
    fetcher.get_prices.side_effect = [[sample_prices[0]], [sample_prices[1]]]

    etl.extract()

    midnight = int(fixed_now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    assert etl._data == sample_prices
    assert fetcher.get_prices.call_args_list == [
        (("AAPL", midnight),),
        (("MSFT", midnight),),
    ]


@patch("tgedr_datasets.prices.etl.PriceFetcher")
def test_extract_normalizes_configured_date_to_utc_midnight(fetcher_class: Mock) -> None:
    """Normalize a configured epoch timestamp to the start of its UTC date."""
    date_at_noon = int(datetime(2026, 8, 5, 12, tzinfo=UTC).timestamp())
    etl = PricesEtl(configuration={
        "tickers_dataset": "tickers",
        "target_dataset": "prices",
        "price_date": date_at_noon,
    })
    etl._store = Mock()
    etl._store.get.return_value = ticker_splits(["AAPL"], [1])

    etl.extract()

    expected = int(datetime(2026, 8, 5, tzinfo=UTC).timestamp())
    fetcher_class.get_instance.return_value.get_prices.assert_called_once_with("AAPL", expected)


def test_extract_requires_tickers_dataset_configuration() -> None:
    """Report missing required configuration through the inherited decorator."""
    with pytest.raises(EtlException, match="missing required configuration parameters"):
        PricesEtl().extract()


def test_transform_creates_rows_and_emits_metric(sample_prices: list[Price]) -> None:
    """Convert prices to rows and emit a row-count metric."""
    etl = PricesEtl()
    etl._data = sample_prices
    metrics = Mock()
    with patch("tgedr_datasets.prices.etl.Metrics.instance", return_value=metrics):
        etl.transform()

    assert etl._result["ticker"].tolist() == ["AAPL", "MSFT"]
    assert etl._result["id"].tolist() == [price.id for price in sample_prices]
    assert etl._result["processing_time"].notna().all()
    metrics.add_to_counter.assert_called_once_with(
        name="datasets.prices.etl.rows",
        value=2,
        attributes={"type": "prices"},
    )


def test_transform_handles_empty_input() -> None:
    """Create an empty result with processing metadata and emit a zero metric."""
    etl = PricesEtl()
    metrics = Mock()
    with patch("tgedr_datasets.prices.etl.Metrics.instance", return_value=metrics):
        etl.transform()

    assert etl._result.empty
    assert "processing_time" in etl._result.columns
    metrics.add_to_counter.assert_called_once_with(
        name="datasets.prices.etl.rows",
        value=0,
        attributes={"type": "prices"},
    )


def test_validate_transform_accepts_valid_rows(sample_prices: list[Price]) -> None:
    """Accept rows meeting the prices contract."""
    etl = PricesEtl()
    etl._data = sample_prices
    etl.transform()

    etl.validate_transform()


def test_validate_transform_rejects_negative_price(sample_prices: list[Price]) -> None:
    """Reject a result that violates the non-negative price contract."""
    etl = PricesEtl()
    etl._data = sample_prices
    etl.transform()
    etl._result.loc[0, "close"] = -1

    with pytest.raises(ValidationError, match="does not meet contract"):
        etl.validate_transform()


def test_load_saves_arrow_dataset_and_returns_none() -> None:
    """Update the result as an Arrow-backed train split and return None."""
    etl = PricesEtl(configuration={"target_dataset": "prices"})
    etl._store = Mock()
    etl._result = pd.DataFrame({"id": [1], "ticker": ["AAPL"]})

    result = etl.load()

    assert result is None
    etl._store.update.assert_called_once()
    kwargs = etl._store.update.call_args.kwargs
    assert kwargs["key"] == "prices"
    assert kwargs["split"] == "train"
    assert kwargs["key_fields"] == ["id"]
    assert isinstance(kwargs["df"], Dataset)
    assert kwargs["df"].to_pandas().equals(etl._result)
