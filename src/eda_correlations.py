"""
Functions to elaborate multiple ways of correlation.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.db_preparation import save_plot

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format
plt.style.use("default")

# Endogen variables use

def n_cols_correlation(db, cols, plot=True, figsize=(8, 6), title="Correlation between Variables", save_path=None):
    """Summarize N columns: per-column counts, correlation matrix/heatmap,
    and pairwise equality between columns (useful for spotting redundant features).

    Args:
        db (pandas DataFrame)
        cols (iterable of str): Column names to correlate (2 or more)
        plot (bool): Whether to plot the correlation heatmap
        figsize (tuple)
        title (str): Heatmap title
    Returns:
        correlation (pandas DataFrame): Correlation matrix between cols
    """
    cols = list(cols)
    if len(cols) < 2:
        raise ValueError("cols must contain at least 2 columns")

    for col in cols:
        print(f"[LOG] Counts by '{col}':")
        print(db.groupby([col]).size())
        print()

    correlation = db[cols].corr()
    print("[LOG] Correlation matrix:")
    print(correlation)

    if plot:
        plt.figure(figsize=figsize)
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(title)
        if save_path:
            save_plot(save_path)
        plt.show()

    # Pairwise equality check (helps explain strong correlations, e.g. two near-duplicate columns)
    total_rows = len(db)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a, col_b = cols[i], cols[j]
            equal_rows = (db[col_a] == db[col_b]).sum()
            print(f"[LOG] {col_a} == {col_b}: {equal_rows} of {total_rows} rows "
                  f"({(equal_rows / total_rows) * 100:.2f}%)")

    return correlation

def x_y_correlation(db, x, y, plot=True, figsize=(6, 5), save_path=None):
    """Summarize the relationship between exactly two columns: counts,
    correlation coefficient, and equality check (via n_cols_correlation),
    plus a scatter plot — more informative than a 2x2 heatmap for a single pair.

    Args:
        db (pandas DataFrame)
        x (str): First column name
        y (str): Second column name
        plot (bool): Whether to plot the scatter plot
        figsize (tuple)
        save_path (str or None): If given, path to save the scatter plot
    Returns:
        corr_value (float): Pearson correlation coefficient between x and y
    """
    correlation = n_cols_correlation(db, [x, y], plot=False)
    corr_value = correlation.loc[x, y]

    if plot:
        plt.figure(figsize=figsize)
        sns.scatterplot(data=db, x=x, y=y, alpha=0.5)
        plt.title(f"{x} vs {y} (corr = {corr_value:.2f})")
        plt.tight_layout()
        save_plot(save_path)
        plt.show()

    return corr_value

def plot_acf_pacf(series, max_lags=90, save_dir=None):
    """Plot the ACF and PACF of a time series, highlighting statistically
    significant lags against a 95% confidence interval.

    Args:
        series (pandas Series): Time series (e.g. daily alarm counts), no NaNs/Infs
        max_lags (int): Maximum number of lags to compute/plot
        save_dir (str or None): Directory to save the figures to (created if missing)
    Returns:
        acf_vals (np.ndarray), pacf_vals (np.ndarray)
    """
    from statsmodels.tsa.stattools import acf, pacf

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    n = len(series)
    ci = 1.96 / np.sqrt(n)

    # --- ACF ---
    acf_vals, _ = acf(series, nlags=max_lags, alpha=0.05)
    lags_axis = np.arange(max_lags + 1)
    acf_body_max = np.nanmax(np.abs(acf_vals[1:]))
    acf_yrange = max(acf_body_max * 1.4 if np.isfinite(acf_body_max) and acf_body_max > 0 else ci * 3, ci * 3)

    fig, ax = plt.subplots(figsize=(18, 5))
    colors_acf = ['#cf1a1a' if abs(v) > ci else '#d4a5a5' for v in acf_vals]
    ax.bar(lags_axis, acf_vals, color=colors_acf, alpha=0.85, width=0.6)
    ax.axhline(ci, color='#660505', linewidth=1.3, linestyle='--', label=f'95% CI ({ci:.4f})')
    ax.axhline(-ci, color='#660505', linewidth=1.3, linestyle='--')
    ax.axhline(0, color='black', linewidth=0.8)

    for lag, label in [(7, '7d'), (14, '14d'), (21, '21d'), (30, '30d'), (60, '60d'), (90, '90d')]:
        if lag <= max_lags:
            ax.axvline(lag, color='steelblue', linewidth=1, linestyle=':', alpha=0.7)
            ax.text(lag + 0.4, acf_yrange * 0.82, label, color='steelblue', fontsize=8.5)

    ax.set_ylim(-acf_yrange, acf_yrange)
    ax.set_xlim(-1, max_lags + 1)
    ax.set_title("ACF - Daily Alarm Count (Direct + Indirect effects)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/acf_alarms.png", dpi=130, bbox_inches='tight')
    plt.show()

    sig_acf = [i for i in range(1, max_lags + 1) if abs(acf_vals[i]) > ci]
    print(f"[LOG] ACF significant lags: {sig_acf}")

    # --- PACF ---
    max_pacf_lags = min(max_lags, n // 2 - 2)
    pacf_vals, _ = pacf(series, nlags=max_pacf_lags, alpha=0.05, method='ywm')
    lags_pacf = np.arange(len(pacf_vals))
    pacf_body_max = np.nanmax(np.abs(pacf_vals[1:]))
    pacf_yrange = max(pacf_body_max * 1.4 if np.isfinite(pacf_body_max) and pacf_body_max > 0 else ci * 3, ci * 3)

    fig, ax = plt.subplots(figsize=(18, 5))
    colors_pacf = ['#cf1a1a' if abs(v) > ci else '#d4a5a5' for v in pacf_vals]
    ax.bar(lags_pacf, pacf_vals, color=colors_pacf, alpha=0.85, width=0.6)
    ax.axhline(ci, color='#660505', linewidth=1.3, linestyle='--', label=f'95% CI ({ci:.4f})')
    ax.axhline(-ci, color='#660505', linewidth=1.3, linestyle='--')
    ax.axhline(0, color='black', linewidth=0.8)

    for lag, label in [(7, '7d'), (14, '14d'), (30, '30d'), (60, '60d')]:
        if lag < len(pacf_vals):
            ax.axvline(lag, color='steelblue', linewidth=1, linestyle=':', alpha=0.7)
            ax.text(lag + 0.4, pacf_yrange * 0.82, label, color='steelblue', fontsize=8.5)

    ax.set_ylim(-pacf_yrange, pacf_yrange)
    ax.set_xlim(-1, max_pacf_lags + 1)
    ax.set_title("PACF - Daily Alarm Count (Direct effects only)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Partial Autocorrelation")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/pacf_alarms.png", dpi=130, bbox_inches='tight')
    plt.show()

    sig_pacf = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > ci]
    print(f"[LOG] PACF significant lags: {sig_pacf}")
    print(f"[LOG] Suggested AR order: {sig_pacf[-1] if sig_pacf else 'none'}")

    return acf_vals, pacf_vals

def acf_by_year_heatmap(series, year_lags=30, key_lags=(1, 2, 3, 5, 7, 10, 14, 21, 28, 30),
                         save_dir=None):
    """Plot per-year ACF subplots and a year x lag ACF heatmap (checks whether the
    temporal structure is stable across years).

    Args:
        series (pandas Series): Time series, no NaNs/Infs
        year_lags (int): Number of lags for the per-year ACF plots
        key_lags (iterable of int): Lags summarized in the heatmap
        save_dir (str or None): Directory to save the figures to
    Returns:
        heatmap_data (pandas DataFrame): ACF values per (year, lag)
        acf_by_year (dict): year -> array of ACF values
    """
    from statsmodels.tsa.stattools import acf

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    years = sorted(series.index.year.unique())

    fig, axes = plt.subplots(len(years), 1, figsize=(18, len(years) * 3.2), constrained_layout=True)
    fig.suptitle("ACF per Year - Daily Alarm Count\n(checks temporal structure stability)",
                 fontsize=14, fontweight='bold')
    if len(years) == 1:
        axes = [axes]

    acf_by_year = {}

    for ax, year in zip(axes, years):
        s_yr = series[series.index.year == year]
        if len(s_yr) < year_lags + 10:
            ax.set_title(f"{year} - insufficient data (n={len(s_yr)})")
            continue

        ci_yr = 1.96 / np.sqrt(len(s_yr))
        nlags_yr = min(year_lags, len(s_yr) // 2 - 1)
        acf_yr = acf(s_yr, nlags=nlags_yr, fft=True)
        acf_by_year[year] = acf_yr

        lags_yr = np.arange(len(acf_yr))
        body_max = np.nanmax(np.abs(acf_yr[1:]))
        yr_yrange = max(body_max * 1.4 if np.isfinite(body_max) and body_max > 0 else ci_yr * 3, ci_yr * 3)

        colors_yr = ['#cf1a1a' if abs(v) > ci_yr else '#d4a5a5' for v in acf_yr]
        ax.bar(lags_yr, acf_yr, color=colors_yr, alpha=0.85, width=0.6)
        ax.axhline(ci_yr, color='#660505', linewidth=1.1, linestyle='--', label=f'95% CI ({ci_yr:.3f})')
        ax.axhline(-ci_yr, color='#660505', linewidth=1.1, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.7)

        for lag, lbl in [(7, '7d'), (14, '14d'), (30, '30d')]:
            if lag < len(acf_yr):
                ax.axvline(lag, color='steelblue', linewidth=1, linestyle=':', alpha=0.7)
                ax.text(lag + 0.3, yr_yrange * 0.82, lbl, color='steelblue', fontsize=8)

        ax.set_ylim(-yr_yrange, yr_yrange)
        ax.set_xlim(-0.5, nlags_yr + 0.5)

        n_sig = sum(1 for i in range(1, len(acf_yr)) if abs(acf_yr[i]) > ci_yr)
        ax.set_title(f"{year} - n={len(s_yr)} days - {n_sig} significant lag(s)",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("ACF")
        ax.legend(fontsize=8, loc='upper right')

    if save_dir:
        plt.savefig(f"{save_dir}/acf_by_year.png", dpi=130, bbox_inches='tight')
    plt.show()

    key_lags = list(key_lags)
    heatmap_data = pd.DataFrame(index=years, columns=[f'lag {k}d' for k in key_lags], dtype=float)
    for year, acf_yr in acf_by_year.items():
        for lag in key_lags:
            if lag < len(acf_yr):
                heatmap_data.loc[year, f'lag {lag}d'] = round(float(acf_yr[lag]), 4)

    fig, ax = plt.subplots(figsize=(14, max(3, len(years) * 0.9)))
    sns.heatmap(heatmap_data.astype(float), annot=True, fmt='.3f', cmap='coolwarm', center=0,
                linewidths=0.5, ax=ax, vmin=-0.6, vmax=0.6)
    ax.set_title("ACF Value per Year x Lag\n(divergence across rows = non-stationary temporal structure)",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Year")
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/acf_heatmap_year_lag.png", dpi=130, bbox_inches='tight')
    plt.show()

    print("[LOG] ACF @ key lags per year:")
    print(heatmap_data.to_string())

    return heatmap_data, acf_by_year
