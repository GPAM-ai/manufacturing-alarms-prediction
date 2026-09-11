"""Build daily feature datasets for forecasting rapid alarm recurrences.

The target is the daily number of alarms whose positive inter-alarm time
(IAT), measured within the same machine, is shorter than 30 minutes. The
pipeline cleans and orders the event records, aggregates alarm and recurrence
counts by local calendar day, marks zero-record periods, and creates calendar
and lagged-volume features.

Two datasets are produced:
  - Simple: calendar variables, gap-history indicators, and lagged total alarm
    volumes.
  - Complete: all simple features, monthly SELIC and durable-goods IPCA values,
    and lagged alarm volumes for TR10, TR17, and TR19.

The module also prepares model-ready rows and expanding-window validation
splits.
"""

import numpy as np
import pandas as pd

from src.data_prep.db_preparation import clean_db
from src.eda.eda_exogen_variables import load_ipca_duraveis, load_selic
from src.eda.eda_interalarm import compute_inter_alarm_time


TARGET_COL = 'recurrences_lt_30m'
TARGET_AVAILABLE_COL = 'target_available'
DEFAULT_IAT_THRESHOLD_MIN = 30
DEFAULT_LAG_DAYS = {
    '1d': 1,
    '3d': 3,
    '7d': 7,
    '14d': 14,
    '21d': 21,
    '28d': 28,
}
DEFAULT_KEY_MACHINES = ('TR10', 'TR17', 'TR19')
NON_FEATURE_COLUMNS = {'Date', TARGET_COL, TARGET_AVAILABLE_COL}


def build_calendar_features(dates):
    """Build calendar features known in advance for each target date."""
    dates = pd.DatetimeIndex(dates)
    feat = pd.DataFrame({'Date': dates})
    feat['Year'] = dates.year
    feat['Month'] = dates.month
    feat['Day'] = dates.day
    feat['DayOfWeek'] = dates.dayofweek
    feat['WeekOfYear'] = dates.isocalendar().week.to_numpy(dtype=int)
    feat['IsWeekend'] = (dates.dayofweek >= 5).astype(int)
    return feat


def _continuous_daily_index(timestamps):
    """Return a complete normalized daily index for valid timestamps."""
    valid = pd.DatetimeIndex(timestamps.dropna())
    if valid.empty:
        raise ValueError('No valid timestamps are available')
    return pd.date_range(
        valid.min().normalize(),
        valid.max().normalize(),
        freq='D',
        tz=valid.tz,
    )


def build_daily_recurrence_base(
    db_alarms,
    timestamp_col='TIMESTAMP',
    machine_col='Maq',
    iat_threshold_min=DEFAULT_IAT_THRESHOLD_MIN,
    timezone='America/Recife',
    clean_first=True,
):
    """Aggregate event-level alarms into daily volumes and a recurrence target.

    The target counts events with positive IAT below ``iat_threshold_min``. The
    returned daily table covers the complete local-date range, and zero-volume
    days receive ``target_available = 0`` so they can be handled explicitly.

    Returns:
        tuple: ``(daily_base, machine_daily, alarm_events_with_iat)``.
    """
    if iat_threshold_min <= 0:
        raise ValueError('iat_threshold_min must be positive')

    events = db_alarms.copy()
    if clean_first:
        events = clean_db(events)
        timestamp_col = timestamp_col.lower()
        machine_col = machine_col.lower()

    required = {timestamp_col, machine_col}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f'Missing required column(s): {sorted(missing)}')

    events = compute_inter_alarm_time(
        events,
        group_col=machine_col,
        timestamp_col=timestamp_col,
        new_col='IAT_min',
        timezone=timezone,
        positive_only=True,
    )
    events['Date'] = events[timestamp_col].dt.normalize()
    daily_index = _continuous_daily_index(events[timestamp_col])

    alarm_volume = (
        events.groupby('Date', observed=True)
        .size()
        .reindex(daily_index, fill_value=0)
        .astype(int)
        .rename('AlarmVolume')
    )
    recurrence_mask = events['IAT_min'].gt(0) & events['IAT_min'].lt(iat_threshold_min)
    recurrence_target = (
        events.loc[recurrence_mask]
        .groupby('Date', observed=True)
        .size()
        .reindex(daily_index, fill_value=0)
        .astype(int)
        .rename(TARGET_COL)
    )
    machine_daily = (
        events.groupby(['Date', machine_col], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=daily_index, fill_value=0)
        .astype(int)
    )

    daily_base = pd.DataFrame({
        'Date': daily_index,
        'AlarmVolume': alarm_volume.to_numpy(),
        TARGET_COL: recurrence_target.to_numpy(),
    })
    daily_base[TARGET_AVAILABLE_COL] = daily_base['AlarmVolume'].gt(0).astype(int)

    print(
        f"[LOG] Daily recurrence base built: {len(daily_base):,} days, "
        f"{int(daily_base[TARGET_AVAILABLE_COL].sum()):,} target-available days, "
        f"{int(recurrence_target.sum()):,} recurrences below "
        f"{iat_threshold_min} minutes"
    )
    return daily_base, machine_daily, events


def build_gap_features(daily_base, volume_col='AlarmVolume'):
    """Describe recent zero-record periods using lagged flags and counts."""
    if volume_col not in daily_base.columns:
        raise KeyError(f'Missing required column: {volume_col}')

    is_gap = daily_base[volume_col].eq(0).astype(int)
    run_id = is_gap.ne(is_gap.shift(fill_value=0)).cumsum()
    run_length = is_gap.groupby(run_id).cumsum()

    gap_feat = pd.DataFrame({'Date': daily_base['Date']})
    gap_feat['GapFlag_lag1d'] = is_gap.shift(1)
    gap_feat['GapRunLength_lag1d'] = run_length.shift(1)
    gap_feat['GapDays_last7d'] = is_gap.shift(1).rolling(7, min_periods=1).sum()
    gap_feat['GapDays_last30d'] = is_gap.shift(1).rolling(30, min_periods=1).sum()
    return gap_feat


def add_temporal_lags(feat, columns, lag_days=None):
    """Create the configured day lags for selected numeric columns."""
    lag_days = lag_days or DEFAULT_LAG_DAYS
    if not lag_days or any(days < 1 for days in lag_days.values()):
        raise ValueError('All lag values must be positive integers')

    missing = [column for column in columns if column not in feat.columns]
    if missing:
        raise KeyError(f'Missing lag source column(s): {missing}')

    result = feat.copy()
    for column in columns:
        for label, days in lag_days.items():
            result[f'{column}_lag{label}'] = result[column].shift(days)
    print(f"[LOG] Temporal lags added for {columns}: {list(lag_days.keys())}")
    return result


def build_simple_features(daily_base, lag_days=None):
    """Build calendar, gap, and lagged total-volume features."""
    required = {'Date', 'AlarmVolume', TARGET_COL, TARGET_AVAILABLE_COL}
    missing = required.difference(daily_base.columns)
    if missing:
        raise KeyError(f'Missing daily-base column(s): {sorted(missing)}')

    output = daily_base[['Date', TARGET_COL, TARGET_AVAILABLE_COL]].copy()
    output = output.merge(
        build_calendar_features(pd.DatetimeIndex(daily_base['Date'])),
        on='Date',
        how='left',
    )
    output = output.merge(build_gap_features(daily_base), on='Date', how='left')

    volume_lags = add_temporal_lags(
        daily_base[['Date', 'AlarmVolume']],
        columns=['AlarmVolume'],
        lag_days=lag_days,
    ).drop(columns='AlarmVolume')
    output = output.merge(volume_lags, on='Date', how='left')
    print(f"[LOG] Simple feature table built: {output.shape[0]:,} rows x {output.shape[1]} columns")
    return output


def add_available_exogenous_features(
    feat,
    selic_path,
    ipca_duraveis_path,
    availability_lag_months=1,
):
    """Add monthly SELIC and durable-goods IPCA aligned to each target date.

    With the default one-month delay, dates in month M receive values from the
    completed month M-1. Missing edge values remain missing.
    """
    if availability_lag_months < 1:
        raise ValueError('availability_lag_months must be at least 1')

    selic = (
        load_selic(selic_path)
        .groupby('AnoMes', observed=True)['Media_Selic']
        .mean()
    )
    ipca = (
        load_ipca_duraveis(
            ipca_duraveis_path,
            date_col='Data',
            value_col='IPCA_Duraveis',
        )
        .groupby('AnoMes', observed=True)['IPCA_Duraveis']
        .mean()
    )

    result = feat.copy()
    target_periods = pd.PeriodIndex(result['Date'].dt.tz_localize(None), freq='M')
    available_periods = target_periods - availability_lag_months
    result['SELIC_available'] = [selic.get(period, np.nan) for period in available_periods]
    result['IPCA_Duraveis_available'] = [ipca.get(period, np.nan) for period in available_periods]
    print(
        '[LOG] Available exogenous features added: SELIC and IPCA_Duraveis '
        f'(lag={availability_lag_months} completed month(s))'
    )
    return result


def add_machine_volume_lags(
    feat,
    machine_daily,
    machines=DEFAULT_KEY_MACHINES,
    lag_days=None,
):
    """Add lagged daily alarm volumes for selected high-recurrence machines."""
    missing = [machine for machine in machines if machine not in machine_daily.columns]
    if missing:
        raise KeyError(f'Missing machine(s) in daily volume table: {missing}')

    machine_source = machine_daily.loc[:, list(machines)].copy()
    machine_source.columns = [f'{machine}_AlarmVolume' for machine in machines]
    machine_source.insert(0, 'Date', machine_source.index)
    machine_source = add_temporal_lags(
        machine_source,
        columns=[column for column in machine_source.columns if column != 'Date'],
        lag_days=lag_days,
    )
    machine_source = machine_source.drop(
        columns=[f'{machine}_AlarmVolume' for machine in machines]
    )
    result = feat.merge(machine_source, on='Date', how='left')
    print(f'[LOG] Lagged machine volumes added for {list(machines)}')
    return result


def build_feature_datasets(
    db_alarms,
    selic_path,
    ipca_duraveis_path,
    timestamp_col='TIMESTAMP',
    machine_col='Maq',
    iat_threshold_min=DEFAULT_IAT_THRESHOLD_MIN,
    lag_days=None,
    key_machines=DEFAULT_KEY_MACHINES,
    timezone='America/Recife',
):
    """Build the daily recurrence base and both feature tables."""
    lag_days = lag_days or DEFAULT_LAG_DAYS
    daily_base, machine_daily, alarm_events = build_daily_recurrence_base(
        db_alarms,
        timestamp_col=timestamp_col,
        machine_col=machine_col,
        iat_threshold_min=iat_threshold_min,
        timezone=timezone,
    )
    simple = build_simple_features(daily_base, lag_days=lag_days)
    complete = add_available_exogenous_features(
        simple,
        selic_path=selic_path,
        ipca_duraveis_path=ipca_duraveis_path,
    )
    complete = add_machine_volume_lags(
        complete,
        machine_daily=machine_daily,
        machines=key_machines,
        lag_days=lag_days,
    )
    return {
        'simple': simple,
        'complete': complete,
        'daily_base': daily_base,
        'machine_daily': machine_daily,
        'alarm_events': alarm_events,
    }


def get_feature_columns(dataset):
    """Return predictor columns, excluding target, date, and quality metadata."""
    return [column for column in dataset.columns if column not in NON_FEATURE_COLUMNS]


def prepare_model_dataset(
    dataset,
    drop_unavailable_targets=True,
    drop_incomplete_features=True,
):
    """Return a chronologically sorted, model-ready copy of a feature table."""
    required = {'Date', TARGET_COL, TARGET_AVAILABLE_COL}
    missing = required.difference(dataset.columns)
    if missing:
        raise KeyError(f'Missing model column(s): {sorted(missing)}')

    result = dataset.copy().sort_values('Date').reset_index(drop=True)
    feature_columns = get_feature_columns(result)
    unavailable = int(result[TARGET_AVAILABLE_COL].eq(0).sum())
    if drop_unavailable_targets:
        result = result.loc[result[TARGET_AVAILABLE_COL].eq(1)].copy()
    if drop_incomplete_features:
        result = result.dropna(subset=feature_columns).copy()
    result = result.reset_index(drop=True)
    print(
        f"[LOG] Model dataset prepared: {len(result):,} rows x "
        f"{len(feature_columns)} features; {unavailable:,} unavailable target day(s) "
        f"{'excluded' if drop_unavailable_targets else 'retained'}"
    )
    return result


def make_expanding_window_splits(
    model_dataset,
    date_col='Date',
    initial_train_days=730,
    validation_days=365,
    step_days=365,
    min_validation_rows=30,
):
    """Create expanding-window train/validation splits ordered by date.

    Returns:
        tuple: ``(splits, summary)`` where each split is a pair of positional
        NumPy index arrays compatible with scikit-learn estimators.
    """
    if initial_train_days < 1 or validation_days < 1 or step_days < 1:
        raise ValueError('Window sizes must be positive')
    if min_validation_rows < 1:
        raise ValueError('min_validation_rows must be positive')
    if date_col not in model_dataset.columns:
        raise KeyError(f'Missing required column: {date_col}')

    dates = pd.to_datetime(model_dataset[date_col], errors='raise')
    if not dates.is_monotonic_increasing:
        raise ValueError('model_dataset must be sorted chronologically')
    if dates.duplicated().any():
        raise ValueError('model_dataset must contain at most one row per date')

    start = dates.min().normalize()
    last_date = dates.max().normalize()
    validation_start = start + pd.Timedelta(days=initial_train_days)
    splits = []
    rows = []
    fold = 1

    while validation_start <= last_date:
        validation_end = min(
            validation_start + pd.Timedelta(days=validation_days - 1),
            last_date,
        )
        train_mask = dates.lt(validation_start)
        validation_mask = dates.ge(validation_start) & dates.le(validation_end)
        train_index = np.flatnonzero(train_mask.to_numpy())
        validation_index = np.flatnonzero(validation_mask.to_numpy())

        if len(validation_index) >= min_validation_rows:
            splits.append((train_index, validation_index))
            rows.append({
                'Fold': fold,
                'Train_start': dates.iloc[train_index[0]],
                'Train_end': dates.iloc[train_index[-1]],
                'Validation_start': dates.iloc[validation_index[0]],
                'Validation_end': dates.iloc[validation_index[-1]],
                'Train_rows': len(train_index),
                'Validation_rows': len(validation_index),
            })
            fold += 1

        validation_start += pd.Timedelta(days=step_days)

    if not splits:
        raise ValueError('No valid expanding-window split could be created')
    summary = pd.DataFrame(rows)
    print(f'[LOG] Expanding-window validation created: {len(splits)} fold(s)')
    return splits, summary
