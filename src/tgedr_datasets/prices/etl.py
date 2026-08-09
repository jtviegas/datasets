"""ETL process for fetching, transforming, and loading news articles.

This module provides the NewsEtl class which orchestrates the extraction,
transformation, and loading of news articles from external sources into
a Parquet data store.
"""

import logging
from pathlib import Path
from typing import Any
from datetime import datetime, UTC, timedelta
import pandas as pd
from tgedr_dataops_abs.etl import Etl
from tgedr_dataops.store.hf_dataset import DataFrameSplits, HuggingFaceDatasetStore
from tgedr_datasets.prices.price import Price  # noqa: TC001
from tgedr_datasets.prices.price_fetcher import PriceFetcher
from tgedr_datasets.quality.contract_validation import validate_df_against_contract
from tgedr_datasets.utils.metrics import MetricsCollector


logger = logging.getLogger(__name__)

_CONTRACT_PATH = Path(__file__).parent / "prices.odcs.yaml"
_CATCHUP_BATCH_SIZE = 5


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
        self._metrics = MetricsCollector()


    @Etl.inject_configuration
    def extract(self, tickers_dataset: str, target_dataset: str, price_date: int | None = None) -> None:
        """Extract ticker data and fetch prices for each ticker.

        Args:
            tickers_dataset: URL or key to the tickers dataset.
            target_dataset: URL or key to the prices dataset, used to find missing
                weekday dates when no explicit price_date is provided.
            price_date: Optional cutoff timestamp for price data. If not provided,
                       missing weekday dates are backfilled from the prices dataset.
        """
        logger.info(f"[extract|in] ({tickers_dataset}, {target_dataset}, {price_date})")

        if price_date is not None:
            cutoff_ts_dt = datetime.fromtimestamp(price_date, tz=UTC)
            self._cutoff_date: int = int(cutoff_ts_dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
            dates_to_fetch: list[int] = [self._cutoff_date]
        else:
            dates_to_fetch = self._find_missing_weekdays(target_dataset)
            dates_to_fetch = [self._cutoff_date] if not dates_to_fetch else dates_to_fetch[:_CATCHUP_BATCH_SIZE]

        df_tickers = self._store.get(key=tickers_dataset).train
        max_date: int = df_tickers["date"].max()
        tickers = df_tickers[df_tickers["date"] == max_date]["ticker"].tolist()
        max_data_formatted = datetime.fromtimestamp(max_date, tz=UTC).strftime("%Y-%m-%d")
        logger.info(f"[extract] tickers max date: {max_data_formatted} len: {len(tickers)}")
        logger.info(f"Fetching ticker prices for %d date(s): {[datetime.fromtimestamp(d, tz=UTC).strftime('%Y-%m-%d') for d in dates_to_fetch]}", len(dates_to_fetch))
        for ticker in tickers:
            for date in dates_to_fetch:
                self._data.extend(PriceFetcher.get_instance().get_prices(ticker, date))

        logger.info("[extract|out] extracted %d prices", len(self._data))

    def transform(self) -> None:
        """Transform extracted ticker data into a DataFrame with date information."""
        logger.info("[transform|in]")

        for price in self._data:
            self._result = pd.concat([self._result, pd.DataFrame([price.to_pd_df_row()])], ignore_index=True)
        self._result["processing_time"] = int(datetime.now(UTC).timestamp())

        self._collect_metrics()

        logger.info("[transform|out] transformed %d prices", self._result.shape[0])

    def _collect_metrics(self) -> None:
        """Collect data quality metrics from the transformed DataFrame."""
        result = self._result
        if result.empty:
            self._metrics.set("row_count", 0)
            self._metrics.set("duplicate_id_count", 0)
            self._metrics.set("negative_price_count", 0)
            self._metrics.set("zero_volume_count", 0)
            self._metrics.set("ohlc_violations_count", 0)
            self._metrics.set("duplicate_ticker_timestamp", 0)
            return

        self._metrics.set("row_count", len(result))
        self._metrics.set("duplicate_id_count", int(result.duplicated(subset=["id"]).sum()))

        negative = ((result["open"] < 0) | (result["high"] < 0) | (result["low"] < 0) | (result["close"] < 0)).sum()
        self._metrics.set("negative_price_count", int(negative))

        self._metrics.set("zero_volume_count", int((result["volume"] == 0).sum()))

        ohlc_violations = (
            (result["high"] < result["low"])
            | (result["high"] < result["open"])
            | (result["high"] < result["close"])
            | (result["low"] > result["open"])
            | (result["low"] > result["close"])
        ).sum()
        self._metrics.set("ohlc_violations_count", int(ohlc_violations))

        self._metrics.set("duplicate_ticker_timestamp", int(result.duplicated(subset=["ticker", "timestamp"]).sum()))

    def _find_missing_weekdays(self, prices_dataset: str) -> list[int]:
        """Find weekday dates not yet present in the prices dataset.

        Reads the existing prices dataset and returns the midnight-UTC epoch
        timestamps of all weekdays (Monday-Friday) from the first date found in
        the dataset up to today that are missing from the dataset. If the dataset
        is empty or has no data, the current date is returned so it can be
        processed.

        Args:
            prices_dataset: URL or key to the prices dataset.

        Returns:
            List of midnight-UTC epoch timestamps for missing weekday dates.
        """
        logger.info(f"[_find_missing_weekdays|in] ({prices_dataset})")

        today_midnight = int(datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())

        try:
            prices_df = self._store.get(key=prices_dataset).train
        except Exception:  # noqa: BLE001
            logger.warning("[_find_missing_weekdays] could not read prices dataset %s, defaulting to today", prices_dataset)
            return [today_midnight]

        if prices_df is None or prices_df.empty or "timestamp" not in prices_df.columns:
            logger.info("[_find_missing_weekdays] prices dataset empty, defaulting to today")
            return [today_midnight]

        existing_ts = set(prices_df["timestamp"].astype(int).tolist())
        first_ts = int(prices_df["timestamp"].min())
        start_dt = datetime.fromtimestamp(first_ts, tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

        missing: list[int] = []
        current = start_dt
        while current <= end_dt:
            if current.weekday() < 5:  # Monday-Friday
                ts = int(current.timestamp())
                if ts not in existing_ts:
                    missing.append(ts)
            current += timedelta(days=1)

        if not missing:
            logger.info("[_find_missing_weekdays] no missing weekdays found")
            return []

        logger.info("[_find_missing_weekdays|out] => %d missing weekdays", len(missing))
        return missing

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
    def load(self, target_dataset: str, metrics_dir: str = "metrics") -> str:
        """Load transformed price data into the target data store.

        Args:
            target_dataset: URL or key to the target data store location.
            metrics_dir: Directory to save metrics CSV files.

        Returns:
            The target URL where the data was loaded.
        """
        logger.info(f"[load|in] ({target_dataset})")

        dfs: DataFrameSplits = DataFrameSplits(train=self._result)
        HuggingFaceDatasetStore().update(df=dfs, key=target_dataset, append=True)

        self._metrics.save(metrics_dir, "prices")

        logger.info(f"[load|out] => {target_dataset}")
