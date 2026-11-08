"""ETL process for fetching, transforming, and loading ticker prices."""

import logging
from pathlib import Path
from typing import Any
from datetime import datetime, UTC
import pandas as pd
from datasets import Dataset
from tgedr_dataops_abs.etl4gh import Etl4GH
from tgedr_dataops.store.hf_dataset_file_arrow_based import HuggingFaceDatasetFileArrowBasedStore
from tgedr_datasets.prices.price import Price  # noqa: TC001
from tgedr_datasets.prices.price_fetcher import PriceFetcher
from tgedr_datasets.quality.contract_validation import validate_df_against_contract
from tgedr_observability.metrics import Metrics


logger = logging.getLogger(__name__)

_CONTRACT_PATH = Path(__file__).parent / "prices.odcs.yaml"

class PricesEtl(Etl4GH):
    """ETL process for fetching, transforming, and loading ticker prices."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:  # pragma: no cover
        """Initialize the PricesEtl instance with optional configuration.

        Args:
            configuration: Optional configuration dictionary for the ETL process.
        """
        super().__init__(configuration)
        self._data: list[Price] = []
        self._result: pd.DataFrame = pd.DataFrame()
        self._store: HuggingFaceDatasetFileArrowBasedStore = HuggingFaceDatasetFileArrowBasedStore()

    def _get_tickers(self, tickers_dataset: str) -> list[str]:
            logger.info(f"[extract|in] ({tickers_dataset})")
            df_all_tickers = self._store.get(key=tickers_dataset).train.to_pandas()
            max_date: int = df_all_tickers["actual_time"].max()
            df_last_tickers = df_all_tickers[df_all_tickers["actual_time"] == max_date]
            # drop duplicates in case we ran tickers twice in the same day (timestamp)
            df_tickers = df_last_tickers.drop_duplicates(subset=["ticker"], keep="first")
            result: list[str] = df_tickers["ticker"].tolist()
            logger.info(f"[extract|out] result length: {len(result)}")
            return result

    @Etl4GH.inject_configuration
    def extract(self, tickers_dataset: str, target_dataset: str, price_date: int | None = None) -> None:
        """Extract ticker prices for a given date from a tickers dataset.

        Args:
            tickers_dataset (str): The dataset key containing the tickers to fetch prices for.
            target_dataset (str): The dataset key to load the prices into.
            price_date (int | None): The date to fetch prices for. If None, uses current date.
        """
        logger.info(f"[extract|in] ({tickers_dataset}, {target_dataset}, {price_date})")
        prices_epoch: int = int(datetime.now(UTC).timestamp()) if price_date is None else price_date
        prices_date: int = int(
            datetime.fromtimestamp(prices_epoch, tz=UTC)
                .replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        )
        tickers = self._get_tickers(tickers_dataset)
        for ticker in tickers:
            self._data.extend(PriceFetcher.get_instance().get_prices(ticker, prices_date))
        logger.info("[extract|out] extracted %d prices", len(self._data))

    def transform(self) -> None:
        """Transform extracted ticker data into a DataFrame with date information."""
        logger.info("[transform|in]")
        for price in self._data:
            self._result = pd.concat([self._result, pd.DataFrame([price.to_pd_df_row()])], ignore_index=True)
        self._result["processing_time"] = int(datetime.now(UTC).timestamp())
        Metrics.instance().add_to_counter(
            name="datasets.prices.etl.rows",
            value=self._result.shape[0],
            attributes={"type": "prices"},
        )
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

    @Etl4GH.inject_configuration
    def load(self, target_dataset: str) -> None:
        """Load transformed price data into the target data store.

        Args:
            target_dataset (str): The dataset key to load the prices into.

        Returns:
            None
        """
        logger.info(f"[load|in] ({target_dataset})")
        self._store.update(df=Dataset.from_pandas(self._result), key=target_dataset, key_fields=["id"], split="train")
        logger.info("[load|out]")
