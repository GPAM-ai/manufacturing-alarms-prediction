"""Additive decomposition into trend, weekly, monthly, and residual terms."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data_prep.db_preparation import save_plot


def compute_additive_decomposition(
    daily_series,
    trend_window=30,
    interpolate_zero_days=True,
):
    """Decompose a daily alarm series with two additive seasonal components.

    The decomposition is sequential:
    ``interpolated = trend + weekly + monthly + residual``.
    Zero-alarm hiatuses may be interpolated for estimation while the original
    observations are retained in the ``Observed`` column.
    """
    if trend_window < 2:
        raise ValueError("trend_window must be at least 2")
    if not isinstance(daily_series.index, pd.DatetimeIndex):
        raise TypeError("daily_series must have a DatetimeIndex")
    if daily_series.empty:
        raise ValueError("daily_series must not be empty")

    observed = pd.to_numeric(daily_series, errors='coerce').astype(float).sort_index()
    full_range = pd.date_range(
        observed.index.min(),
        observed.index.max(),
        freq='D',
        tz=observed.index.tz,
    )
    observed = observed.reindex(full_range)
    working = observed.mask(observed.eq(0)) if interpolate_zero_days else observed.copy()
    interpolated = working.interpolate(method='linear').bfill().ffill()
    if interpolated.isna().any():
        raise ValueError("The daily series could not be fully interpolated")

    trend = interpolated.rolling(trend_window, center=True, min_periods=1).mean()
    detrended = interpolated - trend
    weekly_profile = detrended.groupby(detrended.index.dayofweek).mean()
    weekly = pd.Series(
        detrended.index.dayofweek.map(weekly_profile),
        index=detrended.index,
        dtype=float,
    )
    after_weekly = detrended - weekly
    monthly_profile = after_weekly.groupby(after_weekly.index.month).mean()
    monthly = pd.Series(
        after_weekly.index.month.map(monthly_profile),
        index=after_weekly.index,
        dtype=float,
    )
    residual = after_weekly - monthly

    decomposition = pd.DataFrame({
        'Observed': observed,
        'Interpolated': interpolated,
        'Trend': trend,
        'Weekly': weekly,
        'Monthly': monthly,
        'Residual': residual,
    })
    reconstruction_error = (
        decomposition['Interpolated']
        - decomposition[['Trend', 'Weekly', 'Monthly', 'Residual']].sum(axis=1)
    ).abs().max()
    print(
        f"[LOG] Additive decomposition computed: trend_window={trend_window}, "
        f"maximum reconstruction error={reconstruction_error:.3e}"
    )
    return decomposition


def decomposition_summary(decomposition):
    """Return component amplitudes and residual dispersion."""
    required = {'Trend', 'Weekly', 'Monthly', 'Residual'}
    missing = required.difference(decomposition.columns)
    if missing:
        raise KeyError(f"Missing decomposition column(s): {sorted(missing)}")
    return pd.Series({
        'Trend_range': decomposition['Trend'].max() - decomposition['Trend'].min(),
        'Weekly_amplitude': decomposition['Weekly'].max() - decomposition['Weekly'].min(),
        'Monthly_amplitude': decomposition['Monthly'].max() - decomposition['Monthly'].min(),
        'Residual_std': decomposition['Residual'].std(),
    }, name='Value')


def plot_additive_decomposition(
    decomposition,
    figsize=(16, 15),
    save_path=None,
    show=True,
):
    """Plot observed/trend, weekly, monthly, and residual components."""
    required = {'Observed', 'Trend', 'Weekly', 'Monthly', 'Residual'}
    missing = required.difference(decomposition.columns)
    if missing:
        raise KeyError(f"Missing decomposition column(s): {sorted(missing)}")

    index = decomposition.index.tz_localize(None) if decomposition.index.tz is not None else decomposition.index
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    axes[0].fill_between(index, decomposition['Observed'], alpha=0.25, color='#1a6bcf')
    axes[0].plot(index, decomposition['Observed'], color='#1a6bcf', linewidth=0.7,
                 alpha=0.7, label='Observed')
    axes[0].plot(index, decomposition['Trend'], color='#cf1a1a', linewidth=2.2,
                 label='Trend')
    axes[0].set_title('Daily alarms and long-term trend', fontweight='bold')
    axes[0].set_ylabel('Alarms/day')
    axes[0].legend()

    component_specs = (
        ('Weekly', '#1a6bcf', 'Weekly component'),
        ('Monthly', '#e07b39', 'Monthly component'),
        ('Residual', '#777777', 'Residual component'),
    )
    for ax, (column, color, title) in zip(axes[1:], component_specs):
        values = decomposition[column].to_numpy()
        ax.plot(index, values, color=color, linewidth=1.0, alpha=0.8)
        ax.fill_between(index, 0, values, where=values >= 0, color='#2ca02c', alpha=0.25)
        ax.fill_between(index, 0, values, where=values < 0, color='#cf1a1a', alpha=0.25)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Deviation')

    for ax in axes:
        ax.grid(alpha=0.25)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[-1].tick_params(axis='x', rotation=45)
    axes[-1].set_xlim(index.min(), index.max())

    plt.tight_layout()
    save_plot(save_path)
    if show:
        plt.show()
    return fig
