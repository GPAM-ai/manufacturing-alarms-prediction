"""Exogen variables testing and correlation analysis with alarm counts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Reuses the generic PNG-saving helper from data preparation.
from src.data_prep.db_preparation import save_plot

# Loading and standardizing exogenous variables

def parse_brl_number(value):
    """Convert a Brazilian-formatted number (e.g. '1.234,56') to float.

    Args:
        value (str, float, int, or None): Value to convert. Non-string values
            are passed straight to float().
    Returns:
        float or None: Parsed value, or None if conversion fails.
    """
    if isinstance(value, str):
        value = value.replace('.', '').replace(',', '.')
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def load_gdp(filepath, date_col='Data', value_col='Valor'):
    """Load and standardize the GDP (PIB) exogenous variable.

    Handles this dataset's specific date format ('YYYY.MM' -> first day of
    that month) and Brazilian number formatting.

    Args:
        filepath (str): Path to the GDP CSV file
        date_col (str): Name of the date column in the raw file
        value_col (str): Name of the value column in the raw file
    Returns:
        db (pandas DataFrame): Columns ['AnoMes', 'PIB'], one row per month
    """
    db = pd.read_csv(filepath)

    db[date_col] = db[date_col].astype(str).str.replace('.', '-', regex=False) + '-01'
    db[date_col] = pd.to_datetime(db[date_col], format='%Y-%m-%d', errors='coerce')
    db['AnoMes'] = db[date_col].dt.to_period('M')

    db[value_col] = db[value_col].apply(parse_brl_number)

    db = db[['AnoMes', value_col]].rename(columns={value_col: 'PIB'})
    print(f"[LOG] GDP loaded: {len(db)} months, {db['PIB'].isna().sum()} missing values")

    return db

def load_selic(filepath, sep=';', date_col='data', value_col='valor'):
    """Load and standardize the SELIC (interest rate) exogenous variable,
    aggregating daily records into a monthly mean.

    Args:
        filepath (str): Path to the SELIC CSV file
        sep (str): CSV separator
        date_col (str): Name of the date column in the raw file (format dd/mm/YYYY)
        value_col (str): Name of the value column in the raw file
    Returns:
        db (pandas DataFrame): Columns ['AnoMes', 'Media_Selic'], one row per month
    """
    db = pd.read_csv(filepath, sep=sep)

    db[date_col] = pd.to_datetime(db[date_col], format='%d/%m/%Y', errors='coerce')
    db[value_col] = db[value_col].apply(parse_brl_number)
    db['AnoMes'] = db[date_col].dt.to_period('M')

    monthly = db.groupby('AnoMes')[value_col].mean().reset_index(name='Media_Selic')
    print(f"[LOG] SELIC loaded: {len(db)} daily records -> {len(monthly)} months "
          f"({db[value_col].isna().sum()} unparsable values)")

    return monthly

def load_trade_balance(filepath, sep=';', date_col='data', value_col='valor'):
    """Load and standardize the Trade Balance (Balança Comercial) exogenous variable.

    Args:
        filepath (str): Path to the Trade Balance CSV file
        sep (str): CSV separator
        date_col (str): Name of the date column in the raw file (format dd/mm/YYYY)
        value_col (str): Name of the value column in the raw file
    Returns:
        db (pandas DataFrame): Columns ['AnoMes', 'Trade_Balance'], one row per month
    """
    db = pd.read_csv(filepath, sep=sep)

    db[date_col] = pd.to_datetime(db[date_col], format='%d/%m/%Y', errors='coerce')
    db[value_col] = db[value_col].apply(parse_brl_number)
    db['AnoMes'] = db[date_col].dt.to_period('M')

    db = db[['AnoMes', value_col]].rename(columns={value_col: 'Trade_Balance'})
    print(f"[LOG] Trade Balance loaded: {len(db)} months, "
          f"{db['Trade_Balance'].isna().sum()} missing values")

    return db

def load_ipca_duraveis(filepath, date_col='data', value_col='IPCA_Duraveis'):
    """Load and standardize the IPCA de Duráveis (durable-goods inflation
    index) exogenous variable. Already monthly, no special date format.

    Args:
        filepath (str): Path to the IPCA de Duráveis CSV file
        date_col (str): Name of the date column in the raw file
        value_col (str): Name of the value column in the raw file
    Returns:
        db (pandas DataFrame): Columns ['AnoMes', 'IPCA_Duraveis'], one row per month
    """
    db = pd.read_csv(filepath)

    db[date_col] = pd.to_datetime(db[date_col], errors='coerce')
    db['AnoMes'] = db[date_col].dt.to_period('M')

    db[value_col] = db[value_col].apply(parse_brl_number)

    db = db[['AnoMes', value_col]].rename(columns={value_col: 'IPCA_Duraveis'})
    print(f"[LOG] IPCA de Duráveis loaded: {len(db)} months, "
          f"{db['IPCA_Duraveis'].isna().sum()} missing values")

    return db

def monthly_alarm_counts(df_alarmes, timestamp_col='TIMESTAMP'):
    """Aggregate alarm data into monthly totals.

    Accepts either row-level alarm data, which are counted per month, or an
    already aggregated daily Series, whose values are summed per month. The
    latter is required for an outlier-cleaned/interpolated daily series.

    Args:
        df_alarmes (pandas DataFrame or Series): Row-level alarms or daily totals
        timestamp_col (str): Datetime column used for a DataFrame input
    Returns:
        db (pandas DataFrame): Columns ['AnoMes', 'Qty_Alarms'], one row per month
    """
    if isinstance(df_alarmes, pd.Series):
        if not isinstance(df_alarmes.index, pd.DatetimeIndex):
            raise TypeError("A daily Series must have a DatetimeIndex")
        daily = pd.to_numeric(df_alarmes, errors='coerce')
        dates = daily.index
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        periods = dates.to_period('M')
        db = (
            daily.groupby(periods)
            .sum(min_count=1)
            .rename_axis('AnoMes')
            .reset_index(name='Qty_Alarms')
        )
    else:
        if timestamp_col not in df_alarmes.columns:
            raise KeyError(f"Missing timestamp column: {timestamp_col}")
        timestamps = df_alarmes[timestamp_col]
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps, errors='coerce', utc=True)
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_localize(None)
        periods = timestamps.dt.to_period('M')
        db = (
            df_alarmes.groupby(periods)
            .size()
            .rename_axis('AnoMes')
            .reset_index(name='Qty_Alarms')
        )
    print(f"[LOG] Monthly alarm counts: {len(db)} months")

    return db

def build_economic_dataset(df_alarmes, gdp_path, selic_path, trade_balance_path,
                            ipca_duraveis_path, timestamp_col='TIMESTAMP'):
    """Load all four exogenous variables, aggregate alarms to monthly counts,
    and merge everything into a single monthly dataset.

    Args:
        df_alarmes (pandas DataFrame): Row-level alarm data with a datetime column
        gdp_path (str): Path to the GDP CSV file
        selic_path (str): Path to the SELIC CSV file
        trade_balance_path (str): Path to the Trade Balance CSV file
        ipca_duraveis_path (str): Path to the IPCA de Duráveis CSV file
        timestamp_col (str)
    Returns:
        db (pandas DataFrame): Monthly dataset with columns
            ['AnoMes', 'Qty_Alarms', 'Media_Selic', 'PIB', 'Trade_Balance', 'IPCA_Duraveis']
    """
    alarms = monthly_alarm_counts(df_alarmes, timestamp_col=timestamp_col)
    gdp = load_gdp(gdp_path)
    selic = load_selic(selic_path)
    trade_balance = load_trade_balance(trade_balance_path)
    ipca_duraveis = load_ipca_duraveis(ipca_duraveis_path)

    db = alarms.merge(selic, on='AnoMes', how='inner')
    db = db.merge(gdp, on='AnoMes', how='inner')
    db = db.merge(trade_balance, on='AnoMes', how='inner')
    db = db.merge(ipca_duraveis, on='AnoMes', how='inner')

    print(f"[LOG] Economic dataset built: {len(db)} months "
          f"({db['AnoMes'].min()} to {db['AnoMes'].max()})")

    return db

# Correlation analysis

def correlation_pvalue(db, x, y, method='pearson'):
    """Compute the correlation between two columns along with its p-value and
    a plain-language significance check (alpha = 0.05).

    Args:
        db (pandas DataFrame)
        x (str): First column name
        y (str): Second column name
        method (str): 'pearson' or 'spearman'
    Returns:
        corr (float), p_value (float)
    """
    if method == 'pearson':
        corr, p_value = pearsonr(db[x], db[y])
    elif method == 'spearman':
        corr, p_value = spearmanr(db[x], db[y])
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    print(f"[LOG] {method.capitalize()} correlation ({x} vs {y}): {corr:.3f} (p={p_value:.5f})")
    print("[LOG] " + ("Statistically significant (p < 0.05)" if p_value < 0.05
                       else "Not statistically significant"))

    return corr, p_value

def plot_regression(db, x, y, title=None, xlabel=None, ylabel=None, figsize=(10, 6),
                     save_path=None):
    """Plot a scatter plot with a linear regression trend line and 95% CI between
    two variables (e.g. an exogenous variable vs the alarm count).

    Args:
        db (pandas DataFrame)
        x (str): Column for the x-axis
        y (str): Column for the y-axis
        title (str or None)
        xlabel (str or None)
        ylabel (str or None)
        figsize (tuple)
        save_path (str or None): If given, path to save the figure
    """
    plt.figure(figsize=figsize)
    sns.regplot(x=x, y=y, data=db,
                scatter_kws={'alpha': 0.7, 's': 60},
                line_kws={'color': 'red', 'linestyle': '--'},
                ci=95)

    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.title(title or f'Relationship between {x} and {y}')
    plt.grid(True, alpha=0.3)

    save_plot(save_path)
    plt.show()

def add_lag_feature(db, column, lag, new_col=None):
    """Add a lagged version of a column (e.g. SELIC 3 months ago).

    Args:
        db (pandas DataFrame)
        column (str): Column to lag
        lag (int): Number of periods to shift (positive = past values)
        new_col (str or None): Name for the new column (defaults to '<column>_Lag<lag>')
    Returns:
        db (pandas DataFrame): Same DataFrame with the new lagged column added
    """
    new_col = new_col or f'{column}_Lag{lag}'
    db[new_col] = db[column].shift(lag)
    print(f"[LOG] Added lag feature '{new_col}' (lag={lag})")

    return db

def lag_correlation_analysis(db, target, predictors, lags=range(-3, 4), method='pearson',
                              plot=True, figsize=(10, 6), save_path=None):
    """Compute the correlation between a target variable and each predictor across
    a range of lags, to check whether predictors lead or lag the target.

    Args:
        db (pandas DataFrame)
        target (str): Target column (e.g. 'Qty_Alarms')
        predictors (list of str): Predictor columns to test (e.g. exogenous variables)
        lags (iterable of int): Lags to test, in periods (e.g. months). Positive
            lag shifts the predictor forward (tests if it precedes the target).
        method (str): Correlation method passed to pandas .corr()
        plot (bool): Whether to plot a bar chart of correlation by lag
        figsize (tuple)
        save_path (str or None): If given, path to save the figure
    Returns:
        df_lags (pandas DataFrame): Columns ['Lag', 'Variable', 'Corr']
    """
    results = {'Lag': [], 'Variable': [], 'Corr': []}

    for var in predictors:
        for lag in lags:
            shifted_predictor = db[var].shift(lag)
            corr = db[target].corr(shifted_predictor, method=method)
            results['Lag'].append(lag)
            results['Variable'].append(var)
            results['Corr'].append(corr)

    df_lags = pd.DataFrame(results)
    print(f"[LOG] Lag correlation ({method}) computed for {len(predictors)} variable(s) "
          f"across lags {min(lags)} to {max(lags)}")

    if plot:
        plt.figure(figsize=figsize)
        sns.barplot(data=df_lags, x='Lag', y='Corr', hue='Variable')
        plt.title(f'{method.capitalize()} Correlation with Lags')
        plt.ylabel('Correlation')
        plt.xlabel('Lag (periods) -> Positive indicates Predictor Precedes Target')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if save_path:
            save_plot(save_path)
        plt.show()

    return df_lags
