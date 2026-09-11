"""Functions to analyze alarm recurrence patterns by machine and shift."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from src.data_prep.db_preparation import save_plot
from src.eda.eda_interalarm import compute_inter_alarm_time


DEFAULT_SHIFT_BOUNDS = (
    (6, 14, '1T (06h-14h)'),
    (14, 22, '2T (14h-22h)'),
    (22, 6, '3T (22h-06h)'),
)
DEFAULT_SHIFT_ORDER = [definition[2] for definition in DEFAULT_SHIFT_BOUNDS]


def assign_shift(
    db,
    hour_col=None,
    timestamp_col='TIMESTAMP',
    shift_col='Shift',
    shift_bounds=DEFAULT_SHIFT_BOUNDS,
    timezone='America/Recife',
):
    """Assign a work-shift label from an hour or timestamp column.

    A shift crossing midnight is represented by a start greater than its end,
    for example ``(22, 6, '3T')``.
    """
    result = db.copy()
    if hour_col is not None:
        if hour_col not in result.columns:
            raise KeyError(f"Missing required column: {hour_col}")
        hours = pd.to_numeric(result[hour_col], errors='coerce')
    else:
        if timestamp_col not in result.columns:
            raise KeyError(f"Missing required column: {timestamp_col}")
        timestamps = pd.to_datetime(result[timestamp_col], errors='coerce', utc=True)
        if timezone:
            timestamps = timestamps.dt.tz_convert(timezone)
        result[timestamp_col] = timestamps
        hours = timestamps.dt.hour

    labels = pd.Series(pd.NA, index=result.index, dtype='object')
    for start, end, label in shift_bounds:
        mask = hours.ge(start) & hours.lt(end) if start < end else hours.ge(start) | hours.lt(end)
        labels.loc[mask] = label
    result[shift_col] = labels

    unmatched = result[shift_col].isna().sum()
    if unmatched:
        raise ValueError(
            f"Shift definitions left {unmatched} row(s) without a shift label"
        )
    print(f"[LOG] Shifts assigned: {result[shift_col].value_counts().to_dict()}")
    return result


def recurrence_rate_by_group(
    db,
    group_col='Maq',
    iat_col='IAT_min',
    thresholds=(15, 30, 60),
    denominator='alarms',
):
    """Compute quick-recurrence proportions for each group and threshold.

    ``denominator='alarms'`` reproduces the reviewed analysis: the first alarm
    in each group remains in the denominator even though it has no preceding
    interval. Use ``'valid_intervals'`` to condition only on measured IATs.
    """
    missing = [col for col in (group_col, iat_col) if col not in db.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")
    if denominator not in {'alarms', 'valid_intervals'}:
        raise ValueError("denominator must be 'alarms' or 'valid_intervals'")

    thresholds = tuple(sorted(set(thresholds)))
    if not thresholds or any(limit <= 0 for limit in thresholds):
        raise ValueError("thresholds must contain positive values")

    rows = []
    for group, subset in db.groupby(group_col, observed=True):
        valid_iat = subset.loc[subset[iat_col].gt(0), iat_col]
        total = len(subset) if denominator == 'alarms' else len(valid_iat)
        for limit in thresholds:
            recurrences = int(valid_iat.lt(limit).sum())
            rows.append({
                group_col: group,
                'Threshold_min': limit,
                'Recurrences': recurrences,
                'Total': total,
                'Proportion': recurrences / total * 100 if total else 0.0,
            })

    result = pd.DataFrame(rows)
    print(
        f"[LOG] Recurrence rates computed for {db[group_col].nunique()} group(s) "
        f"at thresholds {list(thresholds)} min"
    )
    return result


def recurrence_by_shift(
    db,
    group_col='Maq',
    iat_col='IAT_min',
    shift_col='Shift',
    threshold_min=30,
    shift_order=None,
    denominator='alarms',
):
    """Compute recurrence percentage for every group-by-shift combination."""
    missing = [col for col in (group_col, iat_col, shift_col) if col not in db.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")
    if threshold_min <= 0:
        raise ValueError("threshold_min must be positive")
    if denominator not in {'alarms', 'valid_intervals'}:
        raise ValueError("denominator must be 'alarms' or 'valid_intervals'")

    shift_order = list(shift_order or DEFAULT_SHIFT_ORDER)
    rows = []
    for group, group_data in db.groupby(group_col, observed=True):
        row = {group_col: group}
        for shift in shift_order:
            subset = group_data.loc[group_data[shift_col].eq(shift)]
            valid_iat = subset.loc[subset[iat_col].gt(0), iat_col]
            total = len(subset) if denominator == 'alarms' else len(valid_iat)
            row[shift] = valid_iat.lt(threshold_min).sum() / total * 100 if total else 0.0
        rows.append(row)

    pivot = pd.DataFrame(rows).set_index(group_col).reindex(columns=shift_order)
    print(
        f"[LOG] Recurrence by shift computed at {threshold_min} min: "
        f"{pivot.shape[0]} group(s) x {pivot.shape[1]} shift(s)"
    )
    return pivot


def plot_recurrence_rate(
    recurrence,
    group_col='Maq',
    threshold_min=30,
    figsize=(10, 7),
    save_path=None,
    show=True,
):
    """Plot ranked recurrence percentages for one threshold."""
    plotted = recurrence.loc[
        recurrence['Threshold_min'].eq(threshold_min)
    ].sort_values('Proportion', ascending=False)
    if plotted.empty:
        raise ValueError(f"No rows found for threshold {threshold_min} min")

    props = plotted['Proportion'].to_numpy()
    prop_range = props.max() - props.min()
    normalized = (props - props.min()) / prop_range if prop_range else np.full(len(props), 0.5)
    colors = [plt.cm.RdYlGn_r(value * 0.85 + 0.05) for value in normalized]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        plotted[group_col].astype(str),
        props,
        color=colors,
        edgecolor='white',
        linewidth=0.6,
        height=0.72,
    )
    mean_prop = plotted['Proportion'].mean()
    ax.axvline(mean_prop, color='#333333', linewidth=1.8, linestyle='--',
               label=f'Mean: {mean_prop:.1f}%')

    for bar, row in zip(bars, plotted.itertuples(index=False)):
        ax.text(
            row.Proportion + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f'{row.Proportion:.1f}% ({row.Recurrences:,} of {row.Total:,})',
            va='center',
            fontsize=9,
            fontweight='bold',
        )

    ax.set_xlabel('Quick-recurrence proportion (%)')
    ax.set_title(
        f'Quick recurrence rate by {group_col} (threshold: {threshold_min} min)\n'
        'A recurrence is an alarm occurring shortly after the previous alarm '
        'in the same group',
        fontsize=12,
        fontweight='bold',
    )
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlim(0, max(props.max() * 1.28, 1))
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    save_plot(save_path)
    if show:
        plt.show()
    return plotted


def plot_recurrence_thresholds(
    recurrence,
    group_col='Maq',
    thresholds=(15, 30, 60),
    figsize=(13, 6),
    save_path=None,
    show=True,
):
    """Compare recurrence rates across multiple thresholds."""
    thresholds = tuple(thresholds)
    groups = sorted(recurrence[group_col].dropna().unique())
    if not groups:
        raise ValueError("recurrence contains no groups")

    x = np.arange(len(groups))
    width = min(0.8 / len(thresholds), 0.25)
    colors = plt.cm.YlOrRd(np.linspace(0.45, 0.9, len(thresholds)))
    fig, ax = plt.subplots(figsize=figsize)

    for index, (threshold, color) in enumerate(zip(thresholds, colors)):
        subset = recurrence.loc[recurrence['Threshold_min'].eq(threshold)]
        values = subset.set_index(group_col)['Proportion'].reindex(groups)
        if values.isna().any():
            raise ValueError(f"Incomplete recurrence data for threshold {threshold}")
        offset = (index - (len(thresholds) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=f'< {threshold} min', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Quick recurrence rate by group and threshold', fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_plot(save_path)
    if show:
        plt.show()
    return fig


def plot_recurrence_by_shift(
    pivot,
    threshold_min=30,
    figsize=(12, 7),
    annot=True,
    save_path=None,
    show=True,
):
    """Plot a heatmap of quick-recurrence percentages by group and shift."""
    if pivot.empty:
        raise ValueError("pivot must not be empty")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=annot,
        fmt='.1f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Recurrence rate (%)'},
        ax=ax,
    )
    ax.set_title(
        f'Quick recurrence rate by machine and shift (< {threshold_min} min)',
        fontsize=13,
        fontweight='bold',
    )
    ax.set_xlabel('Shift')
    ax.set_ylabel(pivot.index.name or 'Machine')
    plt.tight_layout()
    save_plot(save_path)
    if show:
        plt.show()
    return fig


def build_recurrence_analysis(
    db,
    group_col='Maq',
    timestamp_col='TIMESTAMP',
    thresholds=(15, 30, 60),
    shift_col='Shift',
    timezone='America/Recife',
):
    """Run the reusable data-preparation portion of recurrence analysis.

    Returns a dictionary containing the row-level data with IAT/shift labels,
    the rate table, and one group-by-shift pivot per threshold.
    """
    prepared = compute_inter_alarm_time(
        db,
        group_col=group_col,
        timestamp_col=timestamp_col,
        timezone=timezone,
    )
    prepared = assign_shift(
        prepared,
        timestamp_col=timestamp_col,
        shift_col=shift_col,
        timezone=timezone,
    )
    rates = recurrence_rate_by_group(
        prepared,
        group_col=group_col,
        thresholds=thresholds,
    )
    by_shift = {
        threshold: recurrence_by_shift(
            prepared,
            group_col=group_col,
            shift_col=shift_col,
            threshold_min=threshold,
        )
        for threshold in thresholds
    }
    return {'data': prepared, 'rates': rates, 'by_shift': by_shift}
