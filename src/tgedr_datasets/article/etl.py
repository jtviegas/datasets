"""ETL process for fetching, transforming, and loading news articles.

This module provides the NewsEtl class which orchestrates the extraction,
transformation, and loading of news articles from external sources into
a Parquet data store.
"""

import logging
from typing import Any
from datetime import datetime, UTC
import pandas as pd
from tgedr_dataops_abs.etl import Etl
from tgedr_dataops.store.parquet_store import ParquetStore
from tgedr_datasets.article.article import Article  # noqa: TC001
from tgedr_datasets.article.articles_aggregator import ArticlesAggregator



logger = logging.getLogger(__name__)


class ArticlesEtl(Etl):
    """ETL class for processing and storing news articles.

    This class handles the extraction, transformation, and loading of news articles
    from external sources into a Parquet data store.
    """

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialize the ArticlesEtl instance with configuration and setup internal state.

        Args:
            configuration (dict[str, Any] | None): Optional configuration dictionary.
        """
        super().__init__(configuration)
        self._data: list[Article] = []
        self._result: pd.DataFrame = pd.DataFrame()
        self._store = ParquetStore()
        self._processing_time: int = int(datetime.now(UTC).timestamp())

    @Etl.inject_configuration
    def extract(self, tickers_url: str, base_date: int |None = None) -> None:
        """Extract news articles for the latest tickers from the provided URL.

        Args:
            tickers_url (str): URL or key to retrieve tickers data.
            base_date (int | None): Optional base date for extraction.

        This method fetches the latest tickers, retrieves news articles for each,
        and stores them in the internal data list.
        """
        logger.info(f"[extract|in] ({tickers_url}, {base_date})")

        df_tickers = self._store.get(key=tickers_url)
        max_date: int = df_tickers["date"].max()
        tickers = df_tickers[df_tickers["date"] == max_date]["ticker"].tolist()
        max_data_formatted = datetime.fromtimestamp(max_date, tz=UTC).strftime("%Y-%m-%d")
        logger.info(f"[extract] tickers max date: {max_data_formatted} len: {len(tickers)}")

        processing_ts_formatted = datetime.fromtimestamp(self._processing_time, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        articles_aggregator = ArticlesAggregator()
        logger.info(f"[extract] getting news to: {processing_ts_formatted} from one day before")
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

        logger.info("[transform|out] transformed %d articles", self._result.shape[0])

    @Etl.inject_configuration
    def load(self, target_url: str) -> str:
        """Load the transformed articles into the Parquet store.

        Args:
            target_url (str): The key or URL where to store the data.

        Returns:
            str: The target URL where the data was stored.

        This method updates the Parquet store with the transformed DataFrame.
        """
        logger.info(f"[load|in] ({target_url})")

        self._store.update(df=self._result, key=target_url, key_fields=["id"])

        logger.info(f"[load|out] => {target_url}")
        return target_url
