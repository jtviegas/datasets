"""Unit tests for contract_validation module."""

from pathlib import Path

import pandas as pd
import pytest
from tgedr_dataops_abs.great_expectations_validation import ValidationError

from tgedr_datasets.quality import contract_validation
from tgedr_datasets.quality.contract_validation import (
    load_expectations,
    validate_df_against_contract,
)

_PACKAGE_ROOT = Path(contract_validation.__file__).parent.parent
TICKER_CONTRACT = _PACKAGE_ROOT / "ticker" / "ticker.odcs.yaml"
PRICES_CONTRACT = _PACKAGE_ROOT / "prices" / "prices.odcs.yaml"
ARTICLES_CONTRACT = _PACKAGE_ROOT / "article" / "articles.odcs.yaml"


@pytest.fixture
def valid_tickers_df() -> pd.DataFrame:
    """Provide a valid tickers DataFrame."""
    return pd.DataFrame({"ticker": ["AAPL", "MSFT"], "date": [1701388800, 1701388800]})


@pytest.fixture
def valid_prices_df() -> pd.DataFrame:
    """Provide a valid prices DataFrame."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "ticker": ["AAPL", "AAPL"],
            "timestamp": [1701388800, 1701475200],
            "open": [180.5, 181.0],
            "high": [182.75, 183.0],
            "low": [179.8, 180.0],
            "close": [181.25, 182.0],
            "volume": [50000000, 40000000],
            "processing_time": [1701500000, 1701500000],
        }
    )


@pytest.fixture
def valid_articles_df() -> pd.DataFrame:
    """Provide a valid articles DataFrame."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "query": ["AAPL", "MSFT"],
            "timestamp": [1701388800, 1701388801],
            "title": ["Apple surges", "Microsoft news"],
            "description": ["desc1", "desc2"],
            "url": ["https://example.com/1", "https://example.com/2"],
            "source": ["FT", "Reuters"],
            "processing_time": [1701500000, 1701500000],
        }
    )


@pytest.mark.parametrize("contract", [TICKER_CONTRACT, PRICES_CONTRACT, ARTICLES_CONTRACT])
def test_load_expectations_returns_well_formed_suite(contract: Path) -> None:
    """Test each contract embeds a well-formed GX expectations suite."""
    expectations = load_expectations(contract)

    assert "expectation_suite_name" in expectations
    assert isinstance(expectations["expectations"], list)
    assert len(expectations["expectations"]) > 0
    for expectation in expectations["expectations"]:
        assert "expectation_type" in expectation
        assert "kwargs" in expectation


def test_load_expectations_file_not_found() -> None:
    """Test load_expectations raises ValidationError for a missing file."""
    with pytest.raises(ValidationError, match="could not read data contract"):
        load_expectations("/nonexistent/contract.odcs.yaml")


def test_load_expectations_no_expectations(tmp_path: Path) -> None:
    """Test load_expectations raises ValidationError when no expectations are embedded."""
    contract = tmp_path / "contract.odcs.yaml"
    contract.write_text("apiVersion: v3.1.0\nkind: DataContract\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="no expectations found"):
        load_expectations(contract)


def test_load_expectations_malformed_expectations(tmp_path: Path) -> None:
    """Test load_expectations raises ValidationError for malformed expectations."""
    contract = tmp_path / "contract.odcs.yaml"
    contract.write_text(
        "customProperties:\n  - property: expectations\n    value: not_a_dict\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="malformed expectations"):
        load_expectations(contract)


def test_validate_tickers_success(valid_tickers_df: pd.DataFrame) -> None:
    """Test tickers DataFrame validates successfully against its contract."""
    result = validate_df_against_contract(valid_tickers_df, TICKER_CONTRACT)

    assert result["success"] is True


def test_validate_tickers_duplicate_ticker_fails(valid_tickers_df: pd.DataFrame) -> None:
    """Test duplicate tickers fail the uniqueness expectation."""
    valid_tickers_df.loc[1, "ticker"] = "AAPL"

    with pytest.raises(ValidationError, match="does not meet contract"):
        validate_df_against_contract(valid_tickers_df, TICKER_CONTRACT)


def test_validate_prices_success(valid_prices_df: pd.DataFrame) -> None:
    """Test prices DataFrame validates successfully against its contract."""
    result = validate_df_against_contract(valid_prices_df, PRICES_CONTRACT)

    assert result["success"] is True


def test_validate_prices_negative_price_fails(valid_prices_df: pd.DataFrame) -> None:
    """Test negative close price fails the minimum expectation."""
    valid_prices_df.loc[0, "close"] = -1.0

    with pytest.raises(ValidationError, match="does not meet contract"):
        validate_df_against_contract(valid_prices_df, PRICES_CONTRACT)


def test_validate_articles_success(valid_articles_df: pd.DataFrame) -> None:
    """Test articles DataFrame validates successfully against its contract."""
    result = validate_df_against_contract(valid_articles_df, ARTICLES_CONTRACT)

    assert result["success"] is True


def test_validate_articles_null_title_fails(valid_articles_df: pd.DataFrame) -> None:
    """Test null title fails the not-null expectation."""
    valid_articles_df.loc[0, "title"] = None

    with pytest.raises(ValidationError, match="does not meet contract"):
        validate_df_against_contract(valid_articles_df, ARTICLES_CONTRACT)
