"""Functions to analyze and visualize features' relevance in database.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Reuses the generic data-preparation and PNG-saving helpers.
from src.data_prep.db_preparation import sort_db, save_plot

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format
plt.style.use("default")

def plot_value_counts(db, column, printable=True, title=None, colors=None, kind='bar', figsize=(10, 6), save_path=None):
    """Print value counts for a column and plot them (e.g. Line, Maq distributions).

    Args:
        db (pandas DataFrame)
        column (str): Column name to group by
        printable (bool): Printable (True) or not (False)
        title (str or None): Plot title (defaults to 'Distribution of <column>')
        colors (list or None): Colors for the bars
        kind (str): Kind of plot passed to pandas .plot (default 'bar')
        figsize (tuple)
        save_path (str or None): Path to save the figure (defaults to None)
    Returns:
        counts (pandas Series): Row count per category
    """
    counts = db.groupby([column]).size()
    if printable:
        print(f"[LOG] Counts by '{column}':")
        print(counts)

    plt.figure(figsize=figsize)
    db[column].value_counts().plot(kind=kind, title=title or f'Distribution of {column}',
                                    color=colors)
    plt.tight_layout()
    if save_path:
        save_plot(save_path)
    plt.show()

    return counts

def plot_grouped_countplot(db, x, hue='Maq', palette='coolwarm', title=None,
                            figsize=(15, 6), legend_loc='upper right', save_path=None):
    """Plot a countplot of one variable split by another (e.g. Trato x Maq).

    Args:
        db (pandas DataFrame)
        x (str): Column name for the x-axis
        hue (str): Column name to split by (default 'Maq')
        palette (str): Seaborn color palette
        title (str or None): Plot title (defaults to '<x> and <hue> Association')
        figsize (tuple)
        legend_loc (str): Legend location
        save_path (str or None): Path to save the figure (defaults to None)
    """
    plt.figure(figsize=figsize)
    sns.countplot(data=db, x=x, hue=hue, palette=palette)

    plt.title(title or f"{x} and {hue} Association", fontsize=16, fontweight='bold')
    plt.xlabel(x, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title=hue, title_fontsize=10, fontsize=8, loc=legend_loc)

    plt.tight_layout()
    if save_path:
        save_plot(save_path)
    plt.show()

# Seasonal / trend decomposition

def compute_seasonal_decomposition(db, timestamp_col='TIMESTAMP', period=7, model='additive'):
    """Resample row-level data to a daily count and decompose it into
    Observed / Trend / Seasonality / Residual components.

    Args:
        db (pandas DataFrame): Row-level data with a datetime column
        timestamp_col (str)
        period (int): Seasonal period in days (7 for weekly seasonality)
        model (str): 'additive' or 'multiplicative'
    Returns:
        df_decomp (pandas DataFrame): Indexed by date, columns
            ['Observed', 'Trend', 'Seasonality', 'Residual']
    """
    daily = db.set_index(timestamp_col).resample('D').size().asfreq('D', fill_value=0)
    decomposition = seasonal_decompose(daily, model=model, period=period)

    df_decomp = pd.DataFrame({
        'Observed': decomposition.observed,
        'Trend': decomposition.trend,
        'Seasonality': decomposition.seasonal,
        'Residual': decomposition.resid,
    })
    print(f"[LOG] Seasonal decomposition computed: model='{model}', period={period}, "
          f"{len(df_decomp)} days")

    return df_decomp

def plot_decomposition_overview(df_decomp, title='Daily Alarms Decomposition',
                                 figsize=(12, 8), save_path=None):
    """Plot the four decomposition components stacked in a single figure.

    Args:
        df_decomp (pandas DataFrame): Output of compute_seasonal_decomposition
        title (str)
        figsize (tuple)
        save_path (str or None): If given, path to save the figure
    """
    components = ['Observed', 'Trend', 'Seasonality', 'Residual']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    for ax, comp, color in zip(axes, components, colors):
        if comp == 'Residual':
            ax.scatter(df_decomp.index, df_decomp[comp], color=color, s=5, alpha=0.6)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        else:
            ax.plot(df_decomp.index, df_decomp[comp], color=color, linewidth=1)
        ax.set_ylabel(comp)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path:
        save_plot(save_path)
    plt.show()

def plot_decomposition_by_year(df_decomp, exclude_years=None, save_path=None):
    """Plot Observed / Trend / Seasonality / Residual side by side, one row per year.

    Args:
        df_decomp (pandas DataFrame): Output of compute_seasonal_decomposition
        exclude_years (list of int or None): Years to leave out of the plot
        save_path (str or None): If given, path to save the figure
    """
    years = sorted(df_decomp.index.year.unique())
    if exclude_years:
        years = [y for y in years if y not in exclude_years]
        print(f"[LOG] Excluded years: {list(exclude_years)}")

    fig, axes = plt.subplots(nrows=len(years), ncols=4, figsize=(25, 4 * len(years)), sharex=False)
    if len(years) == 1:
        axes = axes.reshape(1, -1)

    for i, year in enumerate(years):
        df_year = df_decomp[df_decomp.index.year == year]

        axes[i, 0].plot(df_year.index, df_year['Observed'], color='tab:blue', linewidth=1)
        axes[i, 0].set_ylabel(f"{year}", fontsize=14, fontweight='bold')
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(df_year.index, df_year['Trend'], color='tab:orange', linewidth=2)
        axes[i, 1].grid(True, alpha=0.3)

        axes[i, 2].plot(df_year.index, df_year['Seasonality'], color='tab:green', linewidth=1)
        axes[i, 2].grid(True, alpha=0.3)

        axes[i, 3].scatter(df_year.index, df_year['Residual'], color='tab:red', s=5, alpha=0.6)
        axes[i, 3].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axes[i, 3].grid(True, alpha=0.3)

        if i == 0:
            axes[i, 0].set_title('Observed (Real)', fontsize=12)
            axes[i, 1].set_title('Trend', fontsize=12)
            axes[i, 2].set_title('Seasonality', fontsize=12)
            axes[i, 3].set_title('Residual (Noise)', fontsize=12)

    title_suffix = f" (Excluding {', '.join(map(str, exclude_years))})" if exclude_years else ""
    plt.suptitle(f'Alarm Decomposition{title_suffix}', fontsize=18, y=1.005)
    plt.tight_layout()
    if save_path:
        save_plot(save_path)
    plt.show()

# Pareto analysis

def pareto_analysis(db, column='Maq', top_n=20, figsize=(12, 6), save_path=None):
    """Plot a Pareto chart (frequency bars + cumulative % line) for a categorical
    column, e.g. which machines account for most alarms.

    Cumulative percentage is computed over ALL categories before slicing to
    top_n, so the line reflects the true cumulative share even when only the
    top categories are shown as bars.
    """
    counts = db[column].value_counts()
    pareto_df = pd.DataFrame({column: counts.index, 'Count': counts.values})
    pareto_df = pareto_df.sort_values('Count', ascending=False).reset_index(drop=True)
    pareto_df['Cumulative_Percentage'] = pareto_df['Count'].cumsum() / pareto_df['Count'].sum() * 100

    pareto_top = pareto_df.head(top_n).reset_index(drop=True)
    print(f"[LOG] Pareto ({column}): top {top_n} of {len(pareto_df)} categories cover "
          f"{pareto_top['Cumulative_Percentage'].iloc[-1]:.1f}% of records")

    x_pos = range(len(pareto_top))
    labels = pareto_top[column].astype(str)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(x_pos, pareto_top['Count'], color='steelblue', label='Frequency')
    ax1.set_ylabel('Count', color='steelblue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    ax2 = ax1.twinx()
    ax2.plot(x_pos, pareto_top['Cumulative_Percentage'], color='orange',
              marker='D', ms=5, label='% Cumulative')
    ax2.axhline(80, color='gray', linestyle='--', linewidth=1)
    ax2.set_ylabel('% Cumulative', color='orange')
    ax2.set_ylim(0, 110)

    plt.title(f'Pareto - {column} with Most Occurrences (Top {top_n})')

    if save_path:
        save_plot(save_path)
    plt.show()

    return pareto_df

# Hour x day-of-week intensity

def plot_hour_weekday_heatmap(db, timestamp_col='TIMESTAMP', figsize=(12, 6),
                               annot=False, save_path=None):
    """Plot a heatmap of record intensity by hour of day and day of week.

    Args:
        db (pandas DataFrame): Row-level data with a datetime column
        timestamp_col (str)
        figsize (tuple)
        annot (bool): Whether to annotate each cell with its count
        save_path (str or None): If given, path to save the figure
    Returns:
        heatmap_data (pandas DataFrame): Counts indexed by day of week, columns by hour
    """
    db = db.copy()
    db['Hour'] = db[timestamp_col].dt.hour
    db['DayOfWeek'] = db[timestamp_col].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    heatmap_data = pd.crosstab(db['DayOfWeek'], db['Hour']).reindex(days_order)

    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_data, cmap='viridis', annot=annot)
    plt.title('Heatmap: Alarm Intensity (Day x Hour)')
    plt.xlabel('Hour')
    plt.ylabel('Day of Week')

    if save_path:
        save_plot(save_path)
    plt.show()

    return heatmap_data

# Burst analysis

def burst_analysis(db, group_col='Maq', timestamp_col='TIMESTAMP', max_minutes=60,
                    bins=60, figsize=(10, 6), save_path=None):
    """Plot the distribution of time between consecutive records within each
    group (e.g. machine), to spot short-interval 'bursts' of alarms.

    Args:
        db (pandas DataFrame): Row-level data with a datetime column
        group_col (str): Column to group by before computing time differences
        timestamp_col (str)
        max_minutes (int): Only intervals shorter than this are plotted/returned
        bins (int): Histogram bin count
        figsize (tuple)
        save_path (str or None): If given, path to save the figure
    Returns:
        df_intervals (pandas DataFrame): Columns [timestamp_col, group_col, 'Time_Diff']
            for all consecutive-record intervals (in minutes), including those
            >= max_minutes
    """
    df_sorted = db.sort_values([group_col, timestamp_col]).copy()
    df_sorted['Time_Diff'] = df_sorted.groupby(group_col)[timestamp_col].diff().dt.total_seconds() / 60

    burst_data = df_sorted[df_sorted['Time_Diff'] < max_minutes]['Time_Diff']
    n_valid = df_sorted['Time_Diff'].notna().sum()
    print(f"[LOG] Burst analysis: {len(burst_data)} of {n_valid} intervals "
          f"under {max_minutes} min ({len(burst_data) / n_valid * 100:.1f}%)")

    plt.figure(figsize=figsize)
    sns.histplot(burst_data, bins=bins, kde=False, color='salmon')
    plt.title(f'Distribution of Time Between Consecutive Alarms (< {max_minutes} min)')
    plt.xlabel('Minutes between Alarms')
    plt.ylabel('Frequency')
    plt.xlim(0, max_minutes)

    if save_path:
        save_plot(save_path)
    plt.show()

    return df_sorted[[timestamp_col, group_col, 'Time_Diff']]
