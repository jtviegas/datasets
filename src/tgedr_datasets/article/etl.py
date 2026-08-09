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
from tgedr_datasets.article.article import Article  # noqa: TC001
from tgedr_datasets.article.articles_aggregator import ArticlesAggregator
from tgedr_datasets.utils.metrics import MetricsCollector
from tgedr_dataops.store.hf_dataset import DataFrameSplits, HuggingFaceDatasetStore
from tgedr_datasets.quality.contract_validation import validate_df_against_contract

logger = logging.getLogger(__name__)

_CONTRACT_PATH = Path(__file__).parent / "articles.odcs.yaml"


class ArticlesEtl(Etl):
    """ETL class for processing and storing news articles.

    This class handles the extraction, transformation, and loading of news articles
    from external sources into a Parquet data store.
    """

    __SEED = 53
    _CATCHUP_BATCH_SIZE = 3

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialize the ArticlesEtl instance with configuration and setup internal state.

        Args:
            configuration (dict[str, Any] | None): Optional configuration dictionary.
        """
        super().__init__(configuration)

        self._data: list[Article] = []
        self._store = HuggingFaceDatasetStore()
        self._processing_time: int = int(datetime.now(UTC).timestamp())
        self._new_data: pd.DataFrame = pd.DataFrame()
        self._new_train: pd.DataFrame = None
        self._new_validation: pd.DataFrame = None
        self._metrics = MetricsCollector()


    @Etl.inject_configuration
    def extract(self, tickers_dataset: str, target_dataset: str, base_date: int | None = None) -> None:
        """Extract news articles for the latest tickers from the provided URL.

        Args:
            tickers_dataset (str): URL or key to retrieve tickers data.
            target_dataset (str): URL or key to the articles dataset, used to find
                missing dates when no explicit base_date is provided.
            base_date (int | None): Optional base date for extraction.

        This method fetches the latest tickers, retrieves news articles for each,
        and stores them in the internal data list.
        """
        logger.info(f"[extract|in] ({tickers_dataset}, {target_dataset}, {base_date})")

        if base_date is not None:
            dates_to_fetch: list[int] = [base_date]
        else:
            dates_to_fetch = self._find_missing_dates(target_dataset)
            dates_to_fetch = dates_to_fetch[: self._CATCHUP_BATCH_SIZE] if dates_to_fetch else [self._processing_time]

        logger.info(f"[extract] fetching dates: {dates_to_fetch}")

        df_all_tickers = self._store.get(key=tickers_dataset).train
        max_date: int = df_all_tickers["actual_time"].max()
        df_last_tickers = df_all_tickers[df_all_tickers["actual_time"] == max_date]
         # drop duplicates in case we ran tickers twice in the same day (timestamp)
        df_tickers = df_last_tickers.drop_duplicates(subset=["ticker"], keep="first")
        tickers = df_tickers["ticker"].tolist()
        max_date_formatted = datetime.fromtimestamp(max_date, tz=UTC).strftime("%Y-%m-%d")
        logger.info(f"[extract] tickers max date: {max_date_formatted} len: {len(tickers)}")

        articles_aggregator = ArticlesAggregator()
        for date in dates_to_fetch:
            self._processing_time = date
            processing_ts_formatted = datetime.fromtimestamp(self._processing_time, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[extract] getting news to: {processing_ts_formatted} from 2 days before")
            for ticker in tickers:
                self._data.extend(articles_aggregator.get_news(ticker, self._processing_time, 2))

        logger.info("[extract|out] extracted %d articles", len(self._data))

    def transform(self) -> None:
        """Transform the extracted articles into a DataFrame.

        This method processes the list of articles, converts each to a DataFrame row,
        concatenates them, and adds a processing time column.
        """
        logger.info("[transform|in]")

        for article in self._data:
            self._new_data = pd.concat([self._new_data, pd.DataFrame([article.to_pd_df_row()])], ignore_index=True)
        self._new_data["processing_time"] = self._processing_time
        if not self._new_data.empty:
            # Deduplicate by id to avoid unique-constraint violations in validation.
            # The same article can be fetched multiple times (e.g. across tickers or
            # sources) with identical fields, producing the same id.
            self._new_data = self._new_data.drop_duplicates(subset=["id"], keep="first")
            self._new_data = self._new_data.sort_values(by="id")

        self._collect_metrics()

        logger.info("[transform|out] transformed %d articles", self._new_data.shape[0])

    def _collect_metrics(self) -> None:
        """Collect data quality metrics from the transformed DataFrame."""
        data = self._new_data
        if data.empty or "id" not in data.columns:
            self._metrics.set("row_count", 0)
            self._metrics.set("duplicate_id_count", 0)
            self._metrics.set("empty_title_count", 0)
            self._metrics.set("empty_description_count", 0)
            return

        self._metrics.set("row_count", len(data))
        self._metrics.set("duplicate_id_count", int(data.duplicated(subset=["id"]).sum()))
        self._metrics.set("empty_title_count", int(data["title"].isna().sum() + (data["title"].str.strip() == "").sum()))
        self._metrics.set(
            "empty_description_count", int(data["description"].isna().sum() + (data["description"].str.strip() == "").sum())
        )

    def _find_missing_dates(self, articles_dataset: str) -> list[int]:
        """Find dates not yet present in the articles dataset.

        Reads the existing articles dataset and returns the midnight-UTC epoch
        timestamps of all dates from the first article timestamp found in the
        dataset up to today that are missing from the dataset. If the dataset
        is empty or has no data, the current date is returned so it can be
        processed.

        Args:
            articles_dataset: URL or key to the articles dataset.

        Returns:
            List of midnight-UTC epoch timestamps for missing dates.
        """
        logger.info(f"[_find_missing_dates|in] ({articles_dataset})")

        today_midnight = int(datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())

        try:
            articles_df = self._store.get(key=articles_dataset).train
        except Exception:  # noqa: BLE001
            logger.warning("[_find_missing_dates] could not read articles dataset %s, defaulting to today", articles_dataset)
            return [today_midnight]

        if articles_df is None or articles_df.empty or "actual_time" not in articles_df.columns:
            logger.info("[_find_missing_dates] articles dataset empty, defaulting to today")
            return [today_midnight]

        existing_ts = set()
        for ts in articles_df["actual_time"].astype(int).tolist():
            dt = datetime.fromtimestamp(ts, tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            existing_ts.add(int(dt.timestamp()))
        first_ts = int(articles_df["actual_time"].min())
        start_dt = datetime.fromtimestamp(first_ts, tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

        missing: list[int] = []
        current = start_dt
        while current <= end_dt:
            ts = int(current.timestamp())
            if ts not in existing_ts:
                missing.append(ts)
            current += timedelta(days=1)

        if not missing:
            logger.info("[_find_missing_dates] no missing dates found")
            return []

        logger.info("[_find_missing_dates|out] => %d missing dates: %s", len(missing), missing)
        return missing

    def validate_transform(self) -> None:
        """Validate the transformed articles data against its data contract.

        Raises
        ------
        ValidationError
            If the transformed data does not meet the contract expectations.
        """
        logger.info("[validate_transform|in]")

        validate_df_against_contract(self._new_data, _CONTRACT_PATH)

        logger.info("[validate_transform|out]")

    @Etl.inject_configuration
    def load(self, target_dataset: str, metrics_dir: str = "metrics") -> str:
        """Load the transformed articles into the Parquet store.

        Args:
            target_dataset (str): The key or name of the target dataset where to store the data.
            metrics_dir: Directory to save metrics CSV files.

        Returns:
            str: The target dataset where the data was stored.

        This method updates the Parquet store with the transformed DataFrame.
        """
        logger.info(f"[load|in] ({target_dataset})")

        dfs: DataFrameSplits = DataFrameSplits(train=self._new_data)
        self._store.update(df=dfs, key=target_dataset, append=True)

        self._metrics.save(metrics_dir, "articles")

        logger.info(f"[load|out] => {target_dataset}")
