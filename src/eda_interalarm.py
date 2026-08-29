"""Inter-alarm-time analysis by machine or another categorical group."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.db_preparation import save_plot


def compute_inter_alarm_time(
    db,
    group_col='Maq',
    timestamp_col='TIMESTAMP',
    new_col='IAT_min',
    timezone=None,
    positive_only=True,
):
    """Compute minutes since the previous alarm in each group.

    Timestamps are parsed as UTC because the project alarm file explicitly
    stores UTC values. ``timezone`` can be supplied when a local timestamp is
    also needed for calendar summaries; it does not change elapsed times.

    Args:
        db (pandas DataFrame): Row-level alarm data.
        group_col (str): Machine or other grouping column.
        timestamp_col (str): Timestamp column.
        new_col (str): Output inter-alarm-time column.
        timezone (str or None): Optional timezone used in ``timestamp_col``.
        positive_only (bool): Replace zero/negative intervals with NaN.
    Returns:
        pandas DataFrame: A sorted copy with ``new_col`` added.
    """
    missing = [col for col in (group_col, timestamp_col) if col not in db.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    result = db.copy()
    timestamps = pd.to_datetime(result[timestamp_col], errors='coerce', utc=True)
    if timezone:
        timestamps = timestamps.dt.tz_convert(timezone)
    result[timestamp_col] = timestamps
    result = result.dropna(subset=[group_col, timestamp_col])
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    result[new_col] = (
        result.groupby(group_col, observed=True)[timestamp_col]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    if positive_only:
        invalid = result[new_col].notna() & result[new_col].le(0)
        result.loc[invalid, new_col] = pd.NA
    else:
        invalid = pd.Series(False, index=result.index)

    print(
        f"[LOG] Inter-alarm time '{new_col}' computed for "
        f"{result[group_col].nunique()} group(s): "
        f"{result[new_col].notna().sum():,} valid interval(s), "
        f"{int(invalid.sum()):,} non-positive interval(s) ignored"
    )
    return result


def inter_alarm_statistics(db, group_col='Maq', iat_col='IAT_min'):
    """Return descriptive inter-alarm-time statistics for each group."""
    missing = [col for col in (group_col, iat_col) if col not in db.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    valid = db.loc[db[iat_col].gt(0), [group_col, iat_col]].copy()
    stats = valid.groupby(group_col, observed=True)[iat_col].agg(
        Median='median',
        Mean='mean',
        Std='std',
        Minimum='min',
        Maximum='max',
        N='count',
    )
    stats = stats.sort_values('Median')
    print(f"[LOG] Inter-alarm statistics computed for {len(stats)} group(s)")
    return stats


def inter_alarm_by_year(
    db,
    group_col='Maq',
    timestamp_col='TIMESTAMP',
    iat_col='IAT_min',
    statistic='median',
):
    """Build a group-by-year pivot of an inter-alarm statistic."""
    missing = [
        col for col in (group_col, timestamp_col, iat_col)
        if col not in db.columns
    ]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    valid = db.loc[db[iat_col].gt(0), [group_col, timestamp_col, iat_col]].copy()
    valid['Year'] = valid[timestamp_col].dt.year
    pivot = valid.pivot_table(
        index=group_col,
        columns='Year',
        values=iat_col,
        aggfunc=statistic,
    )
    print(
        f"[LOG] Inter-alarm {statistic} pivot built: "
        f"{pivot.shape[0]} group(s) x {pivot.shape[1]} year(s)"
    )
    return pivot


def quick_recurrence_counts(
    db,
    threshold_min=60,
    group_col='Maq',
    iat_col='IAT_min',
):
    """Count valid intervals shorter than ``threshold_min`` by group."""
    if threshold_min <= 0:
        raise ValueError("threshold_min must be positive")

    valid = db.loc[db[iat_col].gt(0), [group_col, iat_col]].copy()
    summary = valid.groupby(group_col, observed=True)[iat_col].agg(
        Valid_intervals='size',
        Quick_recurrences=lambda values: values.lt(threshold_min).sum(),
    )
    summary['Proportion'] = (
        summary['Quick_recurrences'].div(summary['Valid_intervals']).mul(100)
    )
    summary = summary.sort_values('Quick_recurrences', ascending=False)
    print(
        f"[LOG] Quick recurrences (< {threshold_min} min): "
        f"{summary['Quick_recurrences'].sum():,} of "
        f"{summary['Valid_intervals'].sum():,} valid intervals"
    )
    return summary


def plot_inter_alarm_overview(
    db,
    group_col='Maq',
    timestamp_col='TIMESTAMP',
    iat_col='IAT_min',
    upper_quantile=0.95,
    figsize=(16, 12),
    save_path=None,
    show=True,
):
    """Plot an interval boxplot and the annual median interval heatmap.

    Missing group-year combinations stay blank instead of being represented as
    zero-minute intervals.

    Returns:
        tuple: ``(boxplot_data, annual_pivot, figure)``.
    """
    if not 0 < upper_quantile <= 1:
        raise ValueError("upper_quantile must be in (0, 1]")

    valid = db.loc[db[iat_col].gt(0)].copy()
    if valid.empty:
        raise ValueError("No positive inter-alarm intervals are available")

    cutoff = valid[iat_col].quantile(upper_quantile)
    box_data = valid.loc[valid[iat_col].le(cutoff)].copy()
    group_order = (
        box_data.groupby(group_col, observed=True)[iat_col]
        .median()
        .sort_values()
        .index
    )
    annual = inter_alarm_by_year(
        valid,
        group_col=group_col,
        timestamp_col=timestamp_col,
        iat_col=iat_col,
    )

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    sns.boxplot(
        data=box_data,
        x=group_col,
        y=iat_col,
        order=group_order,
        hue=group_col,
        palette='coolwarm',
        legend=False,
        ax=axes[0],
        fliersize=2,
    )
    axes[0].set_title(
        'Inter-alarm-time distribution by machine\n'
        f'(up to percentile {upper_quantile * 100:.0f})',
        fontsize=13,
        fontweight='bold',
    )
    axes[0].set_xlabel('Machine')
    axes[0].set_ylabel('Interval (minutes)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    sns.heatmap(
        annual,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd_r',
        ax=axes[1],
        cbar_kws={'label': 'Median interval (minutes)'},
    )
    axes[1].set_title(
        'Median inter-alarm time by machine and year\n'
        '(lower values indicate more frequent alarms)',
        fontsize=13,
        fontweight='bold',
    )
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Machine')

    plt.tight_layout()
    save_plot(save_path)
    if show:
        plt.show()
    return box_data, annual, fig


def plot_quick_recurrence_counts(
    summary,
    threshold_min=60,
    figsize=(14, 5),
    save_path=None,
    show=True,
):
    """Plot the absolute number of quick recurrences by group."""
    if 'Quick_recurrences' not in summary.columns:
        raise KeyError("summary must contain 'Quick_recurrences'")

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        summary.index.astype(str),
        summary['Quick_recurrences'],
        color='#cf1a1a',
        alpha=0.85,
    )
    ax.set_title(
        f'Quick recurrences (< {threshold_min} min) by group',
        fontsize=13,
        fontweight='bold',
    )
    ax.set_xlabel(summary.index.name or 'Group')
    ax.set_ylabel('Occurrences')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.bar_label(bars, fmt='%.0f', padding=2, fontsize=8)
    plt.tight_layout()
    save_plot(save_path)
    if show:
        plt.show()
    return fig
