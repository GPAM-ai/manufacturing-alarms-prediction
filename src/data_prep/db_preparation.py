"""
Functions used in Database pre-process.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format
plt.style.use("default")

def load_db (filepath):
    """Load a CSV file into a pandas DataFrame.

    Args:
        filepath (str)
    Returns:
        db (pandas DataFrame)
    """
    db = pd.read_csv(filepath)

    return db

def sort_db (db, column):
    """Sort the database by a given column.

    Args:
        db (pandas DataFrame)
        column (str): Column name
    Returns:
        db (pandas DataFrame)
    """
    db[column] = db[column].astype('category')
    db[column] = db[column].cat.reorder_categories(
        sorted(db[column].unique())
    )

    return db

def describe_db (db):
    """Summarize the database by printing its head, shape, info, and description.

    Args:
        db (pandas DataFrame)
    """
    print("DATABASE HEADER")
    print(db.head(3))
    print("===================\n\n===================")
    print("DATABASE SHAPE")
    print(db.shape)
    print("===================\n\n===================")
    print("DATABASE INFO")
    db.info()
    print("===================\n\n===================")
    print("DATABASE DESCRIPTION")
    print(db.describe())

def clean_db (db, subset=None, dedup=True, strip_strings=True,
             fillna_strategy=None):
    """Clean the database by:

    - Column names to lowercase
    - Removing duplicate rows (if dedup is True)
    - Treating rows with null values as NaNs
    - Stripping whitespace from string columns (if strip_strings is True)
    - Operation logs

    Args:
        db (pandas DataFrame)
        subset (list of columns to consider for NaN removal)
        dedup (bool)
        strip_strings (bool)
        fillna_strategy (str or None)
    Returns:
        db (pandas DataFrame)
    """
    n_rows_start = db.shape[0]

    # Column names to lower case
    db.columns = (
        db.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    print("[LOG] Columns lowered: ", list(db.columns))

    # Remove duplicate rows
    if dedup:
        n_before = db.shape[0]
        db = db.drop_duplicates()
        print(f"[LOG] Duplicates removed: {n_before - db.shape[0]}")

    # Strip whitespace from string columns
    if strip_strings:
        str_cols = db.select_dtypes(include=object).columns
        db[str_cols] = db[str_cols].apply(lambda col: col.str.strip())
        print(f"[LOG] Columns stripped: {list(db.columns)}")

    # Nulls treatment
    n_nulls = db.isna().sum().sum()
    if fillna_strategy is None:
        db = db.dropna(subset=subset)
        print(f"[LOG] NaNs removed: {n_nulls}")
    else:
        num_cols = db.select_dtypes(include='number').columns
        if fillna_strategy == 'mean':
            db[num_cols] = db[num_cols].fillna(db[num_cols].mean())
        elif fillna_strategy == 'median':
            db[num_cols] = db[num_cols].fillna(db[num_cols].median())
        elif fillna_strategy == 'mode':
            for col in db.columns:
                db[col] = db[col].fillna(db[col].mode().iloc[0] if not db[col].mode().empty else db[col])
        else:
            raise ValueError("fillna_strategy must be None, 'mean', 'median' ou 'mode'")
        n_nulls_after = db.isna().sum().sum()
        print(f"[LOG] Nulls imputed with '{fillna_strategy}': "
              f"{n_nulls - n_nulls_after} filled values")

    # Reset index
    db = db.reset_index(drop=True)

    # Final log
    print(f"[LOG] Total lines: {n_rows_start} -> {db.shape[0]} "
          f"({n_rows_start - db.shape[0]} lines removed)")

    return db

def save_cleaned_dataset(
    df_serie,
    output_path,
    timestamp_col='TIMESTAMP',
    columns=None,
    verbose=True,
):
    """Add temporal features (Year, Month, Day, Hour, DayOfWeek, WeekOfYear) to the
    dataset and save it to CSV.

    Args:
        df_serie (pandas DataFrame): Dataframe with a datetime column
            (e.g. output of prepare_time_series)
        output_path (str): Path to save the resulting CSV
        timestamp_col (str)
        columns (list of str or None): Columns to keep in the saved dataset.
            If None, keeps all columns (original columns + temporal features).
    Returns:
        df_cleaned (pandas DataFrame)
    """
    df_cleaned = df_serie.copy()

    df_cleaned['Year'] = df_cleaned[timestamp_col].dt.year
    df_cleaned['Month'] = df_cleaned[timestamp_col].dt.month
    df_cleaned['Day'] = df_cleaned[timestamp_col].dt.day
    df_cleaned['Hour'] = df_cleaned[timestamp_col].dt.hour
    df_cleaned['DayOfWeek'] = df_cleaned[timestamp_col].dt.day_name()
    df_cleaned['WeekOfYear'] = df_cleaned[timestamp_col].dt.isocalendar().week

    if columns is not None:
        df_cleaned = df_cleaned[[col for col in columns if col in df_cleaned.columns]]

    df_cleaned.to_csv(output_path, index=False)
    if not verbose:
        print(f"[LOG] Cleaned database saved to: {output_path}")
        print(f"[LOG] Total records: {len(df_cleaned)}")

    return df_cleaned

def save_plot(save_path, dpi=150, bbox_inches='tight', verbose=True):
    """Save the current matplotlib plot to a file.
    Args:
        save_path (str): Path to save the figure
        dpi (int): Dpi of the figure
        bbox_inches (str): Bbox of the figure
    """
    if save_path is None:
        return
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    if not verbose:
        print(f"[LOG] Plot saved to: {save_path}")
