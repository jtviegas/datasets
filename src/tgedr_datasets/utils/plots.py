
"""Module for plotting distributions and trends of article data.

This module contains utility functions to visualize article data from Parquet files,
including distributions per ticker and articles over processing time.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

logger = logging.getLogger(__name__)

def plot_distribution_of_articles_per_ticker(data_url: str, plot_url: str) -> None:
    """Plot the distribution of the number of articles per ticker.

    Args:
        data_url: URL or file path to the Parquet file containing article data.
        plot_url: URL or file path where the generated plot will be saved.

    This function reads article data from a Parquet file, groups the articles by ticker,
    counts the number of articles for each ticker, and then creates a bar chart to visualize
    the distribution of article counts across different tickers. The resulting plot is saved
    to the specified location.
    """
    logger.info(f"[plot_distribution_of_articles_per_ticker|in] ({data_url}, {plot_url})")
    # Read the articles from the parquet file
    df = pd.read_parquet(data_url)  # noqa: PD901

    # Group by ticker and count the number of articles (using "id" as unique identifier)
    article_counts = df.groupby("query")["id"].count()

    # Plot the distribution as a bar chart
    plt.figure(figsize=(12, 6))
    article_counts.plot(kind="bar", color="skyblue")
    plt.title("Distribution of Number of Articles per Ticker")
    plt.xlabel("Ticker")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_url, dpi=300, bbox_inches="tight")
    logger.info("[plot_distribution_of_articles_per_ticker|out]")

def plot_articles_per_processing_time(data_url: str, plot_url: str) -> None:
    """Plot the total number of articles per processing time.

    Args:
        data_url: URL or file path to the Parquet file containing article data.
        plot_url: URL or file path where the generated plot will be saved.

    This function reads article data from a Parquet file, groups the articles by their processing time,
    counts the number of articles for each processing time, and then creates a line chart to visualize
    the trend of article counts over time. The resulting plot is saved to the specified location.
    """
    logger.info(f"[plot_articles_per_processing_time|in] ({data_url}, {plot_url})")
    # Read the articles from the parquet file
    df = pd.read_parquet(data_url)  # noqa: PD901

    # Limit to the latest 90 days
    cutoff_time = time.time() - 90 * 24 * 3600  # 90 days ago
    df_filtered = df[df["processing_time"] >= cutoff_time]

    # Group by processing_time and count the number of articles
    articles_per_time = df_filtered.groupby("processing_time")["id"].count()

    # Convert processing_time index to datetime for better plotting
    articles_per_time.index = pd.to_datetime(articles_per_time.index, unit="s")

    # Plot as a line plot
    plt.figure(figsize=(12, 6))
    articles_per_time.plot(kind="line", marker="o", color="orange")
    plt.title("Total Number of Articles per Processing Time")
    plt.xlabel("Processing Time")
    plt.ylabel("Number of Articles")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(plot_url, dpi=300, bbox_inches="tight")
    logger.info("[plot_articles_per_processing_time|out]")

def do_plots(articles_url: str, plots_url: str) -> None:
    """Generate plots for article data distributions and trends.

    Args:
        articles_url: URL or file path to the Parquet file containing article data.
        plots_url: URL or file path to the directory where the generated plots will be saved.

    This function calls the plotting functions to create visualizations of article distributions
    per ticker and articles over processing time, saving the plots to the specified directory.
    """
    logger.info(f"[do_plots|in] ({articles_url}, {plots_url})")
    plot_distribution_of_articles_per_ticker(data_url=articles_url, plot_url=f"{plots_url}/article_distribution.png")
    plot_articles_per_processing_time(data_url=articles_url, plot_url=f"{plots_url}/articles_per_processing_time.png")
    logger.info("[do_plots|out]")
