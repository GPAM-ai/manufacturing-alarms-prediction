"""
Functions to build and clean the daily alarm time series.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.data_prep.db_preparation import save_plot

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format
plt.style.use("default")

def build_time_series(db, timestamp_col='timestamp', dropna_cols=None):
    """Convert a column to datetime, drop invalid/NaN rows, and sort chronologically.

    Args:
        db (pandas DataFrame)
        timestamp_col (str): Name of the timestamp column
        dropna_cols (list of str or None): Columns to check for NaNs after conversion
            (defaults to [timestamp_col])
    Returns:
        db (pandas DataFrame): Time-sorted DataFrame with a reset index
    """
    db = db.copy()
    db[timestamp_col] = pd.to_datetime(db[timestamp_col], errors='coerce')

    dropna_cols = dropna_cols or [timestamp_col]
    n_before = db.shape[0]
    db = db.dropna(subset=dropna_cols)
    print(f"[LOG] Rows dropped due to invalid '{timestamp_col}'/NaNs: {n_before - db.shape[0]}")

    # Keep the timestamp as datetime. ``sort_db`` is intended for nominal
    # categories and would convert this column to an unordered Categorical,
    # breaking chronological min/max and downstream ``.dt`` operations.
    db = db.sort_values(timestamp_col).reset_index(drop=True)
    print(f"[LOG] Time series sorted: {db[timestamp_col].min()} -> {db[timestamp_col].max()}")

    return db

def get_daily_counts(df_serie, timestamp_col='timestamp', fill_gaps=True):
    """Aggregate row counts per day.

    Args:
        df_serie (pandas DataFrame): Must contain a datetime column
        timestamp_col (str)
        fill_gaps (bool): If True, include days with zero occurrences (resample).
            If False, only include days that already have at least one record.
    Returns:
        daily (pandas Series): Indexed by date, values are daily counts
    """
    if fill_gaps:
        daily = df_serie.set_index(timestamp_col).resample('D').size()
    else:
        daily = df_serie.groupby(df_serie[timestamp_col].dt.date).size()

    print(f"[LOG] Daily series length: {len(daily)} "
          f"({(daily > 0).sum()} days with data, {(daily == 0).sum()} empty days)")

    return daily

def plot_daily_series(daily_series, smooth_window=3, title="Number of Alarms per Day",
                       trim_percentiles=None, figsize=(16, 8), save_path=None):
    """Plot a daily series, optionally smoothed and/or percentile-trimmed.

    Args:
        daily_series (pandas Series): Indexed by date, e.g. from get_daily_counts
        smooth_window (int or None): Rolling average window (None to skip smoothing)
        title (str)
        trim_percentiles (tuple of (low, high) or None): Percentile bounds (0-100)
            to trim to before smoothing, e.g. (10, 90)
        figsize (tuple)
        save_path (str or None): Path to save the figure (defaults to None)
    Returns:
        plotted_series (pandas Series): The (possibly trimmed) series that was plotted
    """
    series = daily_series

    plt.figure(figsize=figsize)

    if trim_percentiles is not None:
        low, high = trim_percentiles
        p_low = daily_series.quantile(low / 100)
        p_high = daily_series.quantile(high / 100)
        series = daily_series[(daily_series >= p_low) & (daily_series <= p_high)]
        plt.plot(daily_series, alpha=0.2, label='Original data', color='lightgray')
        plt.plot(series, alpha=0.3, label=f'Trimmed data ({low}-{high}%)', color='lightblue')
        print(f"[LOG] Trimmed to [{low}-{high}] percentiles: "
              f"{len(daily_series) - len(series)} points removed "
              f"({(len(daily_series) - len(series)) / len(daily_series) * 100:.1f}%)")
    else:
        plt.plot(series, alpha=0.6, label='Raw data', color='lightblue')

    if smooth_window:
        smoothed = series.rolling(window=smooth_window).mean()
        plt.plot(smoothed, label=f'{smooth_window}-Day Rolling Average', color='darkblue', linewidth=2)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Quantity of Alarms")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        save_plot(save_path)

    plt.show()

    return series

def remove_outliers_iqr(daily_series, iqr_multiplier=1.5):
    """Remove outliers by IQR method and fill the gaps by linear interpolation.

    This produces a "clean" daily series (e.g. df_daily_clean).

    Args:
        daily_series (pandas Series): Indexed by datetime
        iqr_multiplier (float): Multiplier applied to the IQR to set outlier bounds
    Returns:
        cleaned (pandas Series): Series with outliers replaced via interpolation
        n_outliers (int): Total number of points treated as outliers
    """
    def _clean_group(group):
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        is_outlier = (group < lower) | (group > upper)
        cleaned_group = group.copy()
        cleaned_group[is_outlier] = np.nan
        cleaned_group = (
            cleaned_group
            .interpolate(method='linear')
            .bfill()
            .ffill()
        )

        return cleaned_group, int(is_outlier.sum())

    cleaned, n_outliers = _clean_group(daily_series)
    print(f"[LOG] {n_outliers} outlier(s) interpolated")

    return cleaned, n_outliers

def plot_series_by_year(daily_series, smooth_window=3, save_path=None):
    """Plot a daily series split into one subplot per year.

    Args:
        daily_series (pandas Series): Indexed by datetime
        smooth_window (int or None): Rolling average window (None to skip smoothing)
        save_path (str or None): Path to save the figure (defaults to None)
    """
    years = sorted(daily_series.index.year.unique())
    n_years = len(years)

    fig, axes = plt.subplots(n_years, 1, figsize=(16, 5 * n_years))
    if n_years == 1:
        axes = [axes]

    smoothed = daily_series.rolling(window=smooth_window).mean() if smooth_window else None

    for idx, year in enumerate(years):
        year_data = daily_series[daily_series.index.year == year]
        axes[idx].plot(year_data.index, year_data.values, alpha=0.6, label='Raw data', color='lightblue')
        if smoothed is not None:
            year_smooth = smoothed[smoothed.index.year == year]
            axes[idx].plot(year_smooth.index, year_smooth.values,
                            label=f'{smooth_window}-Day Rolling Average', color='darkblue', linewidth=2)
        axes[idx].set_title(f"Alarms per Day - {year}", fontsize=14, fontweight='bold')
        axes[idx].set_xlabel("Date")
        axes[idx].set_ylabel("Quantity of Alarms")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
