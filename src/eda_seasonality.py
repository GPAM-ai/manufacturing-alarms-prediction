"""Calendar seasonality analysis for row-level alarm data."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.db_preparation import save_plot


WEEKDAY_LABELS_PT = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']
MONTH_LABELS_PT = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                   'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']


def add_calendar_columns(
    db,
    timestamp_col='TIMESTAMP',
    timezone='America/Recife',
    hour_col='Hour',
    weekday_col='Weekday',
    month_col='Month',
):
    """Parse timestamps and add hour, weekday (Monday=0), and month columns."""
    if timestamp_col not in db.columns:
        raise KeyError(f"Missing required column: {timestamp_col}")

    result = db.copy()
    timestamps = pd.to_datetime(result[timestamp_col], errors='coerce', utc=True)
    if timezone:
        timestamps = timestamps.dt.tz_convert(timezone)
    result[timestamp_col] = timestamps
    result = result.dropna(subset=[timestamp_col]).copy()
    result[hour_col] = result[timestamp_col].dt.hour
    result[weekday_col] = result[timestamp_col].dt.dayofweek
    result[month_col] = result[timestamp_col].dt.month
    result = result.sort_values(timestamp_col).reset_index(drop=True)
    print(
        f"[LOG] Calendar columns added to {len(result):,} alarm rows "
        f"(timezone={timezone or 'UTC'})"
    )
    return result


def calendar_seasonality_tables(
    db,
    machine_col='Maq',
    hour_col='Hour',
    weekday_col='Weekday',
    month_col='Month',
):
    """Build complete count tables used by the calendar-seasonality plots."""
    missing = [
        col for col in (machine_col, hour_col, weekday_col, month_col)
        if col not in db.columns
    ]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    hourly = db.groupby(hour_col, observed=True).size().reindex(range(24), fill_value=0)
    weekday = db.groupby(weekday_col, observed=True).size().reindex(range(7), fill_value=0)
    monthly = db.groupby(month_col, observed=True).size().reindex(range(1, 13), fill_value=0)
    hour_weekday = (
        db.groupby([weekday_col, hour_col], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    machine_hour = (
        db.groupby([machine_col, hour_col], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    machine_hour_pct = machine_hour.div(machine_hour.sum(axis=1).replace(0, pd.NA), axis=0).mul(100)

    tables = {
        'hourly': hourly,
        'weekday': weekday,
        'monthly': monthly,
        'hour_weekday': hour_weekday,
        'machine_hour_pct': machine_hour_pct,
    }
    print("[LOG] Calendar-seasonality tables built")
    return tables


def peak_periods(tables):
    """Return the peak hour, weekday, and month from seasonality tables."""
    peaks = {
        'Peak_hour': int(tables['hourly'].idxmax()),
        'Peak_weekday': WEEKDAY_LABELS_PT[int(tables['weekday'].idxmax())],
        'Peak_month': MONTH_LABELS_PT[int(tables['monthly'].idxmax()) - 1],
    }
    print(f"[LOG] Peak calendar periods: {peaks}")
    return peaks


def plot_calendar_seasonality(
    tables,
    figsize=(20, 18),
    save_path=None,
    show=True,
):
    """Plot hour, weekday, month, hour-by-weekday, and machine-hour profiles."""
    required = {'hourly', 'weekday', 'monthly', 'hour_weekday', 'machine_hour_pct'}
    missing = required.difference(tables)
    if missing:
        raise KeyError(f"Missing seasonality table(s): {sorted(missing)}")

    hourly = tables['hourly']
    weekday = tables['weekday']
    monthly = tables['monthly']
    hour_weekday = tables['hour_weekday'].copy()
    hour_weekday.index = WEEKDAY_LABELS_PT

    fig = plt.figure(figsize=figsize)
    fig.suptitle('Alarm calendar seasonality', fontsize=17, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.bar(hourly.index, hourly.values, color='#cf1a1a', alpha=0.85, edgecolor='white')
    peak_hour = int(hourly.idxmax())
    ax1.axvline(peak_hour, color='navy', linestyle='--', label=f'Peak: {peak_hour:02d}h')
    ax1.set_title('Distribution by hour of day', fontweight='bold')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Alarms')
    ax1.set_xticks(range(24))
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2 = fig.add_subplot(3, 2, 2)
    bars = ax2.bar(WEEKDAY_LABELS_PT, weekday.values, color='#1a6bcf', alpha=0.85)
    ax2.bar_label(bars, fmt='%.0f', padding=2, fontsize=8)
    ax2.set_title('Distribution by weekday', fontweight='bold')
    ax2.set_xlabel('Weekday')
    ax2.set_ylabel('Alarms')
    ax2.grid(axis='y', alpha=0.3)

    ax3 = fig.add_subplot(3, 2, (3, 4))
    sns.heatmap(
        hour_weekday,
        cmap='YlOrRd',
        ax=ax3,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'label': 'Alarms'},
    )
    ax3.set_title('Hour of day x weekday', fontweight='bold')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Weekday')

    ax4 = fig.add_subplot(3, 2, 5)
    ax4.bar(MONTH_LABELS_PT, monthly.values, color='#2ca02c', alpha=0.85)
    ax4.set_title('Distribution by month', fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Alarms')
    ax4.grid(axis='y', alpha=0.3)

    ax5 = fig.add_subplot(3, 2, 6)
    sns.heatmap(
        tables['machine_hour_pct'],
        cmap='Blues',
        ax=ax5,
        linewidths=0.2,
        cbar_kws={'label': '% of machine alarms'},
    )
    ax5.set_title('Hourly concentration by machine (%)', fontweight='bold')
    ax5.set_xlabel('Hour')
    ax5.set_ylabel('Machine')

    plt.tight_layout()
    save_plot(save_path)
    if show:
        plt.show()
    return fig


def analyze_calendar_seasonality(
    db,
    timestamp_col='TIMESTAMP',
    machine_col='Maq',
    timezone='America/Recife',
):
    """Prepare alarm data and return calendar tables plus peak periods."""
    prepared = add_calendar_columns(
        db,
        timestamp_col=timestamp_col,
        timezone=timezone,
    )
    tables = calendar_seasonality_tables(prepared, machine_col=machine_col)
    return {'data': prepared, 'tables': tables, 'peaks': peak_periods(tables)}
