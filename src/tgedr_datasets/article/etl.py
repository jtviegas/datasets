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
from datasets import Dataset
from tgedr_datasets.article.article import Article  # noqa: TC001
from tgedr_datasets.article.articles_aggregator import ArticlesAggregator
from tgedr_observability.metrics import Metrics
from tgedr_dataops.store.hf_dataset_file_arrow_based import HuggingFaceDatasetFileArrowBasedStore
from tgedr_dataops_abs.etl4gh import Etl4GH

from tgedr_datasets.quality.contract_validation import validate_df_against_contract

logger = logging.getLogger(__name__)

_CONTRACT_PATH = Path(__file__).parent / "articles.odcs.yaml"


class ArticlesEtl(Etl4GH):
    """ETL class for processing and storing news articles.

    This class handles the extraction, transformation, and loading of news articles
    from external sources into a HF dataset.
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
        self._store = HuggingFaceDatasetFileArrowBasedStore()
        self._processing_time: int = int(datetime.now(UTC).timestamp())
        self._result: pd.DataFrame = pd.DataFrame()

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
    def extract(self, tickers_dataset: str, timestamp: int | None = None) -> None:
        """Extract news articles for the given tickers dataset.

        Args:
            tickers_dataset (str): The dataset key containing the tickers to fetch
                news for.
            timestamp (int | None): The timestamp to use for processing. If None, the current time is used.
        """
        logger.info(f"[extract|in] ({tickers_dataset})")
        tickers: list[str] = self._get_tickers(tickers_dataset)
        articles_aggregator = ArticlesAggregator()
        if timestamp is not None:
            self._processing_time = timestamp
        processing_ts_formatted = datetime.fromtimestamp(self._processing_time, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[extract] getting news to: {processing_ts_formatted} from 1 days before")
        for ticker in tickers:
            self._data.extend(articles_aggregator.get_news(ticker, self._processing_time, 1))
        logger.info("[extract|out] extracted %d articles", len(self._data))

    def transform(self) -> None:
        """Transform the extracted articles into a DataFrame.

        This method processes the list of articles, converts each to a DataFrame row,
        concatenates them, and adds a processing time column.
        """
        logger.info("[transform|in]")

        for article in self._data:
            self._result = pd.concat([self._result, pd.DataFrame([article.to_pd_df_row()])], ignore_index=True)
        self._result["processing_time"] = self._processing_time
        if not self._result.empty:
            # Deduplicate by id to avoid unique-constraint violations in validation.
            # The same article can be fetched multiple times (e.g. across tickers or
            # sources) with identical fields, producing the same id.
            self._result = self._result.drop_duplicates(subset=["id"], keep="first")
            self._result = self._result.sort_values(by="id")
            Metrics.instance().add_to_counter(
                name="datasets.articles.etl.rows",
                value=self._result.shape[0],
                attributes={"type": "articles"},
            )
        logger.info("[transform|out] transformed %d articles", self._result.shape[0])


    def validate_transform(self) -> None:
        """Validate the transformed articles data against its data contract.

        Raises
        ------
        ValidationError
            If the transformed data does not meet the contract expectations.
        """
        logger.info("[validate_transform|in]")
        validate_df_against_contract(self._result, _CONTRACT_PATH)
        logger.info("[validate_transform|out]")

    @Etl4GH.inject_configuration
    def load(self, target_dataset: str) -> str:
        """Load the transformed articles into the target dataset.

        Args:
            target_dataset (str): The dataset key to load the articles into.

        Returns:
            str: The target dataset key.
        """
        logger.info(f"[load|in] ({target_dataset})")
        self._store.save(df=Dataset.from_pandas(self._result), key=target_dataset, split="train", append=True)
        logger.info("[load|out]")
