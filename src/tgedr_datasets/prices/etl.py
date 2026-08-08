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
from tgedr_datasets.prices.price import Price  # noqa: TC001
from tgedr_datasets.prices.price_fetcher import PriceFetcher
from tgedr_datasets.quality.contract_validation import validate_df_against_contract


logger = logging.getLogger(__name__)

_CONTRACT_PATH = Path(__file__).parent / "prices.odcs.yaml"


class PricesEtl(Etl):
    """ETL process for fetching, transforming, and loading ticker prices."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:  # pragma: no cover
        """Initialize the PricesEtl instance with optional configuration.

        Args:
            configuration: Optional configuration dictionary for the ETL process.
        """
        super().__init__(configuration)
        self._data: list[Price] = []
        self._result: pd.DataFrame = pd.DataFrame()
        self._cutoff_date: int = int(datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        self._store: HuggingFaceDatasetStore = HuggingFaceDatasetStore()


    @Etl.inject_configuration
    def extract(self, tickers_dataset: str, price_date: int |None = None) -> None:
        """Extract ticker data and fetch prices for each ticker.

        Args:
            tickers_dataset: URL or key to the tickers dataset.
            price_date: Optional cutoff timestamp for price data. If not provided,
                       uses the current date at midnight UTC.
        """
        logger.info(f"[extract|in] ({tickers_dataset}, {price_date})")

        if price_date is not None:
            cutoff_ts_dt = datetime.fromtimestamp(price_date, tz=UTC)
            self._cutoff_date: int = int(cutoff_ts_dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())

        df_tickers = self._store.get(key=tickers_dataset).train
        max_date: int = df_tickers["date"].max()
        tickers = df_tickers[df_tickers["date"] == max_date]["ticker"].tolist()
        max_data_formatted = datetime.fromtimestamp(max_date, tz=UTC).strftime("%Y-%m-%d")
        logger.info(f"[extract] tickers max date: {max_data_formatted} len: {len(tickers)}")
        logger.info(f"Fetching ticker prices with cutoff date: {datetime.fromtimestamp(self._cutoff_date, tz=UTC).strftime('%Y-%m-%d')}")
        for ticker in tickers:
            self._data.extend(PriceFetcher.get_instance().get_prices(ticker, self._cutoff_date))

        logger.info("[extract|out] extracted %d prices", len(self._data))

    def transform(self) -> None:
        """Transform extracted ticker data into a DataFrame with date information."""
        logger.info("[transform|in]")

        for price in self._data:
            self._result = pd.concat([self._result, pd.DataFrame([price.to_pd_df_row()])], ignore_index=True)
        self._result["processing_time"] = int(datetime.now(UTC).timestamp())

        logger.info("[transform|out] transformed %d prices", self._result.shape[0])

    def validate_transform(self) -> None:
        """Validate the transformed price data against its data contract.

        Raises
        ------
        ValidationError
            If the transformed data does not meet the contract expectations.
        """
        logger.info("[validate_transform|in]")

        validate_df_against_contract(self._result, _CONTRACT_PATH)

        logger.info("[validate_transform|out]")

    @Etl.inject_configuration
    def load(self, target_dataset: str) -> str:
        """Load transformed price data into the target data store.

        Args:
            target_dataset: URL or key to the target data store location.

        Returns:
            The target URL where the data was loaded.
        """
        logger.info(f"[load|in] ({target_dataset})")

        dfs: DataFrameSplits = DataFrameSplits(train=self._result)
        HuggingFaceDatasetStore().update(df=dfs, key=target_dataset, append=True)

        logger.info(f"[load|out] => {target_dataset}")
