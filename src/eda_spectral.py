"""FFT spectral analysis and conditional seasonal patterns."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq

from src.db_preparation import save_plot


DEFAULT_FREQUENCY_BANDS = (
    ('2-4 days', 2, 4.5, '#888888'),
    ('Weekly (5-9d)', 5, 9, '#1a6bcf'),
    ('Biweekly (10-17d)', 10, 17, '#7b4fa6'),
    ('Monthly (18-35d)', 18, 35, '#e07b39'),
    ('Bimonthly (36-90d)', 36, 90, '#d4a017'),
    ('Quarterly (91-180d)', 91, 180, '#2ca02c'),
    ('Semiannual (181-270d)', 181, 270, '#cf1a1a'),
    ('Annual (271-400d)', 271, 400, '#5a2d82'),
    ('Pluriannual (>400d)', 400, np.inf, '#333333'),
)


def prepare_spectral_series(daily_series, interpolate_zero_days=True):
    """Return continuous, interpolated, and detrended daily series.

    Zero-alarm days are treated as missing only for FFT estimation, matching
    the reviewed EDA. The original values remain untouched outside this return.
    """
    if not isinstance(daily_series.index, pd.DatetimeIndex):
        raise TypeError("daily_series must have a DatetimeIndex")
    if daily_series.empty:
        raise ValueError("daily_series must not be empty")

    numeric = pd.to_numeric(daily_series, errors='coerce').astype(float).sort_index()
    full_range = pd.date_range(
        numeric.index.min(),
        numeric.index.max(),
        freq='D',
        tz=numeric.index.tz,
    )
    numeric = numeric.reindex(full_range)
    working = numeric.mask(numeric.eq(0)) if interpolate_zero_days else numeric.copy()
    interpolated = working.interpolate(method='linear').bfill().ffill()
    if interpolated.isna().any():
        raise ValueError("The daily series could not be fully interpolated")
    detrended = pd.Series(
        signal.detrend(interpolated.to_numpy()),
        index=interpolated.index,
        name='Detrended',
    )
    print(
        f"[LOG] Spectral series prepared: {len(detrended):,} observations, "
        f"interpolate_zero_days={interpolate_zero_days}"
    )
    return interpolated.rename('Interpolated'), detrended


def compute_fft_spectrum(daily_series, interpolate_zero_days=True, sampling_days=1.0):
    """Compute the positive-frequency FFT power spectrum of a daily series.

    Returns:
        tuple: ``(spectrum, interpolated_series, detrended_series)``.
    """
    if sampling_days <= 0:
        raise ValueError("sampling_days must be positive")
    interpolated, detrended = prepare_spectral_series(
        daily_series,
        interpolate_zero_days=interpolate_zero_days,
    )
    n = len(detrended)
    frequencies = rfftfreq(n, d=sampling_days)
    power = np.abs(rfft(detrended.to_numpy())) ** 2
    positive = frequencies > 0
    spectrum = pd.DataFrame({
        'Frequency_per_day': frequencies[positive],
        'Period_days': 1 / frequencies[positive],
        'Power': power[positive],
    }).sort_values('Period_days').reset_index(drop=True)
    total_power = spectrum['Power'].sum()
    spectrum['Power_pct'] = spectrum['Power'].div(total_power).mul(100) if total_power else 0.0
    print(f"[LOG] FFT spectrum computed: {len(spectrum):,} positive frequency bins")
    return spectrum, interpolated, detrended


def summarize_frequency_bands(spectrum, bands=DEFAULT_FREQUENCY_BANDS):
    """Summarize total power and strongest period within predefined bands."""
    if spectrum.empty:
        raise ValueError("spectrum must not be empty")
    total_power = spectrum['Power'].sum()
    rows = []
    for name, min_period, max_period, color in bands:
        mask = spectrum['Period_days'].ge(min_period) & spectrum['Period_days'].lt(max_period)
        subset = spectrum.loc[mask]
        rows.append({
            'Band': name,
            'Min_period_days': min_period,
            'Max_period_days': max_period,
            'Power_pct': subset['Power'].sum() / total_power * 100 if total_power else 0.0,
            'Peak_period_days': (
                subset.loc[subset['Power'].idxmax(), 'Period_days'] if not subset.empty else np.nan
            ),
            'Color': color,
        })
    summary = pd.DataFrame(rows)
    print("[LOG] Spectral power summarized by frequency band")
    return summary


def plot_power_spectrum(
    spectrum,
    band_summary=None,
    reference_periods=(7, 14, 30, 182, 365),
    figsize=(14, 6),
    save_path=None,
    show=True,
):
    """Plot the FFT power spectrum with colored frequency bands."""
    if spectrum.empty:
        raise ValueError("spectrum must not be empty")
    band_summary = band_summary if band_summary is not None else summarize_frequency_bands(spectrum)
    positive_power = spectrum.loc[spectrum['Power'].gt(0), 'Power']
    floor = positive_power.min() if not positive_power.empty else 1e-12

    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(spectrum['Period_days'], spectrum['Power'], color='#bbbbbb', linewidth=0.7)
    for row in band_summary.itertuples(index=False):
        upper = row.Max_period_days
        mask = spectrum['Period_days'].ge(row.Min_period_days)
        if np.isfinite(upper):
            mask &= spectrum['Period_days'].lt(upper)
        subset = spectrum.loc[mask]
        if subset.empty:
            continue
        ax.fill_between(
            subset['Period_days'],
            floor,
            subset['Power'],
            alpha=0.35,
            color=row.Color,
            label=f'{row.Band} ({row.Power_pct:.1f}%)',
        )
        ax.semilogy(subset['Period_days'], subset['Power'], color=row.Color, linewidth=1.2)

    reference_colors = ['#1a6bcf', '#7b4fa6', '#e07b39', '#cf1a1a', '#5a2d82']
    for period, color in zip(reference_periods, reference_colors):
        if spectrum['Period_days'].min() <= period <= spectrum['Period_days'].max():
            ax.axvline(period, color=color, linewidth=1.2, linestyle='--', alpha=0.7)
            ax.text(period, spectrum['Power'].max() * 0.35, f'{period}d', rotation=90,
                    ha='center', va='top', fontsize=8, color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Period (days, log scale)')
    ax.set_ylabel('Spectral power (log scale)')
    ax.set_title('FFT power spectrum - daily alarm series', fontweight='bold')
    ax.set_xlim(max(1.5, spectrum['Period_days'].min()), spectrum['Period_days'].max())
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.2, which='both')
    plt.tight_layout()
    save_plot(save_path)
    if show:
        plt.show()
    return fig


def conditional_seasonal_pattern(daily_series, period='weekday', exclude_zero_days=True):
    """Compute mean alarm volume conditioned on weekday or calendar month."""
    if period not in {'weekday', 'month'}:
        raise ValueError("period must be 'weekday' or 'month'")
    series = pd.to_numeric(daily_series, errors='coerce').dropna()
    if exclude_zero_days:
        series = series.loc[series.gt(0)]
    if series.empty:
        raise ValueError("No observations remain for the conditional pattern")

    if period == 'weekday':
        values = series.groupby(series.index.dayofweek).mean().reindex(range(7))
        labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    else:
        values = series.groupby(series.index.month).mean().reindex(range(1, 13))
        labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    result = pd.DataFrame({'Label': labels, 'Mean_alarms': values.to_numpy()})
    result['Deviation_from_mean'] = result['Mean_alarms'] - series.mean()
    return result


def plot_conditional_patterns(
    daily_series,
    exclude_zero_days=True,
    figsize=(14, 5),
    save_path=None,
    show=True,
):
    """Plot weekly and monthly conditional mean alarm patterns."""
    weekly = conditional_seasonal_pattern(daily_series, 'weekday', exclude_zero_days)
    monthly = conditional_seasonal_pattern(daily_series, 'month', exclude_zero_days)
    overall_mean = daily_series.loc[daily_series.gt(0)].mean() if exclude_zero_days else daily_series.mean()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, data, title, color in (
        (axes[0], weekly, 'Weekly pattern', '#1a6bcf'),
        (axes[1], monthly, 'Monthly pattern', '#e07b39'),
    ):
        colors = [color if value >= overall_mean else '#cccccc' for value in data['Mean_alarms']]
        bars = ax.bar(data['Label'], data['Mean_alarms'], color=colors, edgecolor='white')
        ax.axhline(overall_mean, color='#333333', linestyle='--', label=f'Mean: {overall_mean:.1f}')
        amplitude = data['Mean_alarms'].max() - data['Mean_alarms'].min()
        ax.set_title(
            f'{title}\nAmplitude: {amplitude:.1f} alarms '
            f'({amplitude / overall_mean * 100:.1f}% of mean)',
            fontweight='bold',
        )
        ax.bar_label(bars, labels=[f'{value:+.1f}' for value in data['Deviation_from_mean']],
                     padding=2, fontsize=8)
        ax.set_ylabel('Mean alarms/day')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_plot(save_path)
    if show:
        plt.show()
    return weekly, monthly, fig
