"""Module for plotting distributions and trends of ticker data.

This module contains utility functions to visualize ticker data from Parquet files,
including distributions per ticker and tickers over processing time.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

logger = logging.getLogger(__name__)


def plot_items_distribution(df: pd.DataFrame, item_col: str, item_name: str, plot_parent_url: str = ".") -> None:
    """Plot the distribution of items from a DataFrame.

    This function groups the DataFrame by the specified item column, counts the occurrences,
    and plots a bar chart of the distribution, saving it to the given URL.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    item_col : str
        The column name in the DataFrame to group by.
    item_name : str
        The name of the items for labeling the plot.
    plot_parent_url : str, optional
        The URL or file path where the plot will be saved. Default is the current directory.

    Returns
    -------
    None
    """
    logger.info(f"[plot_items_distribution|in] ({df.shape}, {item_col}, {item_name}, {plot_parent_url})")

    # Clear any existing plots
    plt.clf()

    # Group by the specified item column and count the number of items
    df_counts = df.groupby(item_col).size()
    df_counts = df_counts.rename(lambda x: x[:6])

    # Plot the distribution as a bar chart
    plt.figure(figsize=(12, 6))
    df_counts.plot(kind="bar", color="skyblue")
    plt.title(f"Distribution of {item_name}")
    plt.xlabel(item_col)
    plt.ylabel(f"Number of {item_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_parent_url}/{item_name}_distribution.png", dpi=300, bbox_inches="tight")
    # plt.show()  # noqa: ERA001
    logger.info("[plot_items_distribution|out]")


def plot_items_per_time(
    df: pd.DataFrame,
    item_name: str,
    cutoff_days: int | None = 90,
    processing_time_col: str = "processing_time",
    plot_parent_url: str = ".",
) -> None:
    """Plot the number of items over processing time.

    This function filters the DataFrame by cutoff days if specified, groups by processing time,
    and plots a line chart of the number of items over time, saving it to a file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    item_name : str
        The name of the items for labeling the plot.
    cutoff_days : int | None, optional
        Number of days to look back from current time for filtering data. If None, no cutoff is applied. Default is 90.
    processing_time_col : str, optional
        The column name for processing time in the DataFrame. Default is "processing_time".
    plot_parent_url : str, optional
        The URL or file path where the plot will be saved. Default is the current directory.

    Returns
    -------
    None
    """
    logger.info(
        f"[plot_items_per_time|in] ({df.shape}, {item_name}, {cutoff_days}, {processing_time_col}, {plot_parent_url})"
    )
    # Clear any existing plots
    plt.clf()

    if cutoff_days is not None:
        cutoff_time = time.time() - cutoff_days * 24 * 3600
        df_filtered = df[df[processing_time_col] >= cutoff_time]
    else:
        df_filtered = df

    # Group by processing_time and count the number of items
    items_per_time = df_filtered.groupby(processing_time_col).size()

    # Convert processing_time index to datetime for better plotting
    items_per_time.index = pd.to_datetime(items_per_time.index, unit="s")

    # Plot as a line plot
    plt.figure(figsize=(12, 6))
    items_per_time.plot(kind="line", marker="o", color="orange")
    plt.title(f"Total Number of {item_name} per Time")
    plt.xlabel("Processing Time")
    plt.ylabel(f"Number of {item_name}")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{plot_parent_url}/{item_name}_per_time.png", dpi=300, bbox_inches="tight")
    # plt.show()  # noqa: ERA001
    logger.info("[plot_items_per_time|out]")


def do_plots(base_data_url: str, dataset_name: str, base_plots_url: str = ".") -> None:
    """Generate plots for ticker data distributions and trends.

    Args:
        base_data_url: URL or file path to the Parquet file containing ticker data.
        dataset_name: Name of the dataset to plot (e.g., "articles", "prices", "tickers").
        base_plots_url: URL or file path to the directory where the generated plots will be saved.

    This function calls the plotting functions to create visualizations of ticker distributions
    per ticker and tickers over processing time, saving the plots to the specified directory.
    """
    logger.info(f"[do_plots|in] ({base_data_url}, {dataset_name}, {base_plots_url})")
    dataset_spec = {
        "articles": {"item_col": "query", "item_name": "articles", "processing_time_col": "processing_time"},
        "prices": {"item_col": "ticker", "item_name": "prices", "processing_time_col": "processing_time"},
        "tickers": {"item_col": "ticker", "item_name": "tickers", "processing_time_col": "actual_time"},
    }
    spec = dataset_spec.get(dataset_name)
    df = pd.read_parquet(f"{base_data_url}/{dataset_name}.parquet")  # noqa: PD901
    plot_items_distribution(
        df,
        item_col=spec["item_col"],
        item_name=spec["item_name"],
        plot_parent_url=base_plots_url,
    )
    plot_items_per_time(
        df,
        item_name=spec["item_name"],
        processing_time_col=spec["processing_time_col"],
        plot_parent_url=base_plots_url,
    )
    logger.info("[do_plots|out]")


def plot_metrics(metrics_dir: str, dataset_name: str, cutoff_days: int | None = None) -> None:
    """Plot all count metrics for a dataset on a single chart.

    Reads the CSV file for the given dataset, converts the timestamp column to
    datetime, and overlays every metric column as a line on one shared axes,
    saving a single plot per dataset.

    Parameters
    ----------
    metrics_dir : str
        Directory containing the metrics CSV files.
    dataset_name : str
        Name of the dataset (e.g., "tickers", "prices", "articles").
    cutoff_days : int | None, optional
        Number of days to look back. If None, all data is plotted. Default is None.

    """
    logger.info(f"[plot_metrics|in] ({metrics_dir}, {dataset_name}, {cutoff_days})")

    csv_path = f"{metrics_dir}/{dataset_name}.csv"
    metrics_df = pd.read_csv(csv_path)

    if metrics_df.empty:
        logger.warning("[plot_metrics] empty metrics file: %s", csv_path)
        return

    metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"], unit="s", utc=True)

    if cutoff_days is not None:
        cutoff_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=cutoff_days)
        metrics_df = metrics_df[metrics_df["timestamp"] >= cutoff_time]

    metric_cols = [col for col in metrics_df.columns if col != "timestamp"]

    if not metric_cols:
        logger.warning("[plot_metrics] no metric columns found in %s", csv_path)
        return

    plt.clf()
    _, ax = plt.subplots(figsize=(12, 6))
    for col in metric_cols:
        ax.plot(metrics_df["timestamp"], metrics_df[col], marker="o", linewidth=1.5, markersize=4, label=col)

    ax.set_title(f"{dataset_name} — metrics")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Count")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/{dataset_name}_metrics.png", dpi=300, bbox_inches="tight")
    logger.info(f"[plot_metrics|out] => {metrics_dir}/{dataset_name}_metrics.png")
