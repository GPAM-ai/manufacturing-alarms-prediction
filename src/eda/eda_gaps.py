"""Detection, classification, and visualization of zero-alarm hiatuses."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

from src.data_prep.db_preparation import save_plot


CATEGORY_COLORS = {
    'COVID-19': '#8b0000',
    'Plant stoppage': '#e07b39',
    'Carnival': '#7b4fa6',
    'Easter': '#2ca02c',
    'Holiday': '#1a6bcf',
    'Unresolved': '#333333',
    'Unclassified': '#999999',
}

# Dates transcribed from the reviewed EDA. The two largest hiatuses are kept
# explicitly unresolved, matching the report's methodological caveat.
DEFAULT_GAP_ANNOTATIONS = (
    ('2020-03-26', '2020-05-05', 'COVID-19', 'COVID-19 shutdown', False),
    ('2021-01-23', '2021-02-13', 'Plant stoppage', 'Plant stoppage', False),
    ('2021-02-26', '2021-03-18', 'Plant stoppage', 'Plant stoppage', False),
    ('2022-07-28', '2022-10-05', 'Unresolved', 'Unresolved hiatus (70 days)', True),
    ('2023-03-03', '2023-04-09', 'Plant stoppage', 'Plant stoppage', False),
    ('2023-08-05', '2023-11-02', 'Unresolved', 'Unresolved hiatus (90 days)', True),
    ('2020-02-21', '2020-02-24', 'Carnival', 'Carnival', False),
    ('2021-02-15', '2021-02-17', 'Carnival', 'Carnival', False),
    ('2022-02-28', '2022-03-01', 'Carnival', 'Carnival', False),
    ('2023-02-17', '2023-02-21', 'Carnival', 'Carnival', False),
    ('2024-02-11', '2024-02-14', 'Carnival', 'Carnival', False),
    ('2021-03-30', '2021-04-02', 'Easter', 'Easter', False),
    ('2022-04-19', '2022-04-21', 'Easter', 'Easter', False),
    ('2023-04-18', '2023-04-21', 'Easter', 'Easter', False),
    ('2024-03-30', '2024-03-31', 'Easter', 'Easter', False),
)


def build_daily_alarm_series(
    db,
    timestamp_col='TIMESTAMP',
    timezone='America/Recife',
):
    """Aggregate row-level alarms to a continuous daily count series."""
    if timestamp_col not in db.columns:
        raise KeyError(f"Missing required column: {timestamp_col}")

    timestamps = pd.to_datetime(db[timestamp_col], errors='coerce', utc=True)
    if timezone:
        timestamps = timestamps.dt.tz_convert(timezone)
    valid = timestamps.dropna()
    if valid.empty:
        raise ValueError("No valid timestamps are available")

    daily = pd.Series(1, index=pd.DatetimeIndex(valid)).resample('D').sum()
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(full_range, fill_value=0).astype(int)
    daily.name = 'Alarms'
    print(
        f"[LOG] Daily alarm series: {len(daily):,} days, "
        f"{daily.eq(0).sum():,} zero-alarm day(s)"
    )
    return daily


def identify_gaps(daily_series, min_days=1):
    """Identify every contiguous run of zero-valued days.

    Unlike the original loop, this implementation also retains a hiatus that
    reaches the final day of the series.
    """
    if min_days < 1:
        raise ValueError("min_days must be at least 1")
    if not isinstance(daily_series.index, pd.DatetimeIndex):
        raise TypeError("daily_series must have a DatetimeIndex")
    if daily_series.empty:
        return pd.DataFrame(columns=['Start', 'End', 'Duration_days'])

    series = daily_series.sort_index().copy()
    full_range = pd.date_range(series.index.min(), series.index.max(), freq='D', tz=series.index.tz)
    series = series.reindex(full_range, fill_value=0)
    zero_mask = series.eq(0)
    run_id = zero_mask.ne(zero_mask.shift(fill_value=False)).cumsum()

    rows = []
    for _, run in series.loc[zero_mask].groupby(run_id.loc[zero_mask]):
        duration = len(run)
        if duration >= min_days:
            rows.append({
                'Start': run.index.min(),
                'End': run.index.max(),
                'Duration_days': duration,
            })
    gaps = pd.DataFrame(rows, columns=['Start', 'End', 'Duration_days'])
    if not gaps.empty:
        gaps = gaps.sort_values(['Duration_days', 'Start'], ascending=[False, True]).reset_index(drop=True)
    print(f"[LOG] Identified {len(gaps)} gap(s) lasting at least {min_days} day(s)")
    return gaps


def classify_gaps(gaps, annotations=DEFAULT_GAP_ANNOTATIONS):
    """Classify gaps by maximum overlap with manually validated date ranges."""
    result = gaps.copy()
    if result.empty:
        for col in ('Category', 'Label', 'Color', 'Requires_validation'):
            result[col] = pd.Series(dtype='object')
        return result

    parsed_annotations = []
    for start, end, category, label, requires_validation in annotations:
        parsed_annotations.append({
            'Start': pd.Timestamp(start),
            'End': pd.Timestamp(end),
            'Category': category,
            'Label': label,
            'Requires_validation': bool(requires_validation),
        })

    categories = []
    labels = []
    validation = []
    for row in result.itertuples(index=False):
        gap_start = pd.Timestamp(row.Start).tz_localize(None)
        gap_end = pd.Timestamp(row.End).tz_localize(None)
        best = None
        best_overlap = 0
        for annotation in parsed_annotations:
            overlap_start = max(gap_start, annotation['Start'])
            overlap_end = min(gap_end, annotation['End'])
            overlap = max(0, (overlap_end - overlap_start).days + 1)
            if overlap > best_overlap:
                best = annotation
                best_overlap = overlap
        if best is None:
            categories.append('Unclassified')
            labels.append('Unclassified')
            validation.append(True)
        else:
            categories.append(best['Category'])
            labels.append(best['Label'])
            validation.append(best['Requires_validation'])

    result['Category'] = categories
    result['Label'] = labels
    result['Color'] = result['Category'].map(CATEGORY_COLORS).fillna('#999999')
    result['Requires_validation'] = validation
    print(f"[LOG] Gap categories: {result['Category'].value_counts().to_dict()}")
    return result


def gap_summary(gaps):
    """Summarize number and total duration of gaps by category."""
    if gaps.empty:
        return pd.DataFrame(columns=['Gaps', 'Total_days', 'Longest_gap'])
    group_col = 'Category' if 'Category' in gaps.columns else None
    if group_col is None:
        return pd.DataFrame({
            'Gaps': [len(gaps)],
            'Total_days': [gaps['Duration_days'].sum()],
            'Longest_gap': [gaps['Duration_days'].max()],
        }, index=['All'])
    return gaps.groupby(group_col)['Duration_days'].agg(
        Gaps='size',
        Total_days='sum',
        Longest_gap='max',
    ).sort_values('Total_days', ascending=False)


def plot_gaps_overview(
    daily_series,
    gaps,
    figsize=(18, 10),
    save_path=None,
    show=True,
):
    """Plot the daily series with highlighted gaps and a duration ranking."""
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes[0].fill_between(daily_series.index, daily_series.values, alpha=0.5, color='#1a6bcf')
    for row in gaps.itertuples(index=False):
        color = getattr(row, 'Color', '#cf1a1a')
        axes[0].axvspan(row.Start, row.End, alpha=0.25, color=color)
    axes[0].set_title('Daily alarm series with zero-alarm gaps highlighted', fontweight='bold')
    axes[0].set_ylabel('Alarms/day')
    axes[0].grid(alpha=0.3)

    ordered = gaps.sort_values('Duration_days', ascending=True)
    y = range(len(ordered))
    colors = ordered['Color'] if 'Color' in ordered.columns else '#cf1a1a'
    axes[1].barh(list(y), ordered['Duration_days'], color=colors, alpha=0.85)
    axes[1].set_yticks(list(y))
    axes[1].set_yticklabels([
        f"{pd.Timestamp(row.Start).date()} -> {pd.Timestamp(row.End).date()}"
        for row in ordered.itertuples(index=False)
    ], fontsize=8)
    axes[1].set_xlabel('Duration (days)')
    axes[1].set_title('Gap duration', fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_plot(save_path)
    if show:
        plt.show()
    return fig


def plot_classified_gap_series(
    daily_series,
    gaps,
    rolling_window=7,
    annotate_min_days=20,
    figsize=(18, 6),
    save_path=None,
    show=True,
):
    """Plot the complete daily series with classified hiatuses and annotations."""
    series = daily_series.copy()
    plot_index = series.index.tz_localize(None) if series.index.tz is not None else series.index
    rolling = series.rolling(rolling_window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=figsize)
    for row in gaps.itertuples(index=False):
        start = pd.Timestamp(row.Start).tz_localize(None)
        end = pd.Timestamp(row.End).tz_localize(None)
        color = getattr(row, 'Color', '#999999')
        ax.axvspan(start, end, alpha=0.25, color=color)

    ax.fill_between(plot_index, series.values, alpha=0.25, color='#1a6bcf')
    ax.plot(plot_index, series.values, color='#1a6bcf', linewidth=0.7, alpha=0.7)
    ax.plot(plot_index, rolling.values, color='#1a1a2e', linewidth=2,
            label=f'{rolling_window}-day moving average')

    max_value = max(float(series.max()), 1.0)
    for row in gaps.loc[gaps['Duration_days'].ge(annotate_min_days)].itertuples(index=False):
        start = pd.Timestamp(row.Start).tz_localize(None)
        end = pd.Timestamp(row.End).tz_localize(None)
        midpoint = start + (end - start) / 2
        label = getattr(row, 'Label', 'Gap')
        color = getattr(row, 'Color', '#999999')
        ax.annotate(
            f'{label}\n({row.Duration_days} days)',
            xy=(midpoint, 0),
            xytext=(midpoint, max_value * 0.75),
            ha='center',
            fontsize=8,
            fontweight='bold',
            color=color,
            bbox={'boxstyle': 'round,pad=0.3', 'fc': 'white', 'ec': color, 'alpha': 0.9},
            arrowprops={'arrowstyle': '->', 'color': color, 'lw': 1.2},
        )

    categories = gaps['Category'].dropna().unique() if 'Category' in gaps.columns else []
    legend = [
        mpatches.Patch(color=CATEGORY_COLORS.get(category, '#999999'), alpha=0.7, label=category)
        for category in categories
    ]
    legend.append(plt.Line2D([0], [0], color='#1a1a2e', linewidth=2,
                             label=f'{rolling_window}-day moving average'))
    ax.legend(handles=legend, fontsize=9, loc='upper left', ncol=3)
    ax.set_ylabel('Alarms/day')
    ax.set_title('Daily alarm series - diagnosed and classified gaps', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.25)
    ax.set_xlim(plot_index.min(), plot_index.max())

    plt.tight_layout()
    save_plot(save_path)
    if show:
        plt.show()
    return fig
