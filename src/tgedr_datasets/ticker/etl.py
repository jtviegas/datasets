"""ETL process for fetching, transforming, and loading news articles.

This module provides the NewsEtl class which orchestrates the extraction,
transformation, and loading of news articles from external sources into
a Parquet data store.
"""

import logging
from pathlib import Path
from typing import Any
from datetime import datetime, UTC
import pandas as pd
from tgedr_dataops_abs.etl import Etl
from tgedr_dataops.store.hf_dataset import DataFrameSplits, HuggingFaceDatasetStore

from tgedr_datasets.quality.contract_validation import validate_df_against_contract
from tgedr_datasets.ticker.fetcher import TickerFetcher

logger = logging.getLogger(__name__)

_CONTRACT_PATH = Path(__file__).parent / "ticker.odcs.yaml"


class TickerEtl(Etl):
    """ETL process for fetching, transforming, and loading tickers."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:  # pragma: no cover
        """Initialize TickerEtl with optional configuration.

        Args:
            configuration : dict[str, Any]
                source for configuration injection

        """
        super().__init__(configuration)
        self._data: list[str] = []
        self._result: pd.DataFrame = None

    @Etl.inject_configuration
    def extract(self, source: str) -> None:
        """Extract ticker data from specified sources.

        Args:
            source: Comma-separated string of data sources to fetch tickers from.

        """
        logger.info(f"[extract|in] ({source})")

        sources = [s.strip() for s in source.split(",")]
        self._data = TickerFetcher().fetch(sources)

        logger.info("[extract|out] tickers len: %d", len(self._data))

    def transform(self) -> None:
        """Transform extracted ticker data into a DataFrame with date information."""
        logger.info("[transform|in]")

        today = datetime.now(UTC).date()
        epoch: int = int(datetime(today.year, today.month, today.day, tzinfo=UTC).timestamp())

        self._result = pd.DataFrame(self._data, columns=["ticker"])
        self._result["date"] = epoch

        logger.info("[transform|out]")

    def validate_transform(self) -> None:
        """Validate the transformed ticker data against its data contract.

        Raises
        ------
        ValidationError
            If the transformed data does not meet the contract expectations.
        """
        logger.info("[validate_transform|in]")

        validate_df_against_contract(self._result, _CONTRACT_PATH)

        logger.info("[validate_transform|out]")

    @Etl.inject_configuration
    def load(self, dataset: str) -> str:
        """Load transformed ticker data into a Parquet store.

        Args:
            dataset: Name of the dataset where the Parquet data will be saved.

        Returns:
            The target path where data was saved.

        """
        logger.info(f"[load|in] ({dataset})")

        dfs: DataFrameSplits = DataFrameSplits(train=self._result)
        HuggingFaceDatasetStore().update(df=dfs, key=dataset, append=True)

        logger.info(f"[load|out] => {dataset}")


