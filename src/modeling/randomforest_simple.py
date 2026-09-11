"""
Random Forest Training adapted for features_simple.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
FEATURES_FILE = SRC_DIR / 'feat_eng' / 'features_simple.csv'
RAW_DATA_FILE = SRC_DIR / 'data' / 'alarm-logs-clean.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'results'

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def compute_f1_score(y_true, y_pred, threshold=None):
    if threshold is None:
        threshold = np.median(y_true)
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    return f1, precision, recall, threshold

print(f"Loading features_simple.csv...")
try:
    df = pd.read_csv(FEATURES_FILE)
except FileNotFoundError:
    print(f"File not found: {FEATURES_FILE}")
    exit(1)

# Determine if this is a feature list or a dataset
is_feature_list = len(df.columns) == 1

if is_feature_list:
    print("CSV recognized as a feature list. Loading original data...")
    feature_cols = df.iloc[:, 0].tolist()
    
    try:
        df_raw = pd.read_csv(RAW_DATA_FILE)
    except FileNotFoundError:
        print(f"File not found for feature extraction: {RAW_DATA_FILE}")
        exit(1)
        
    df_raw['TIMESTAMP'] = pd.to_datetime(df_raw['TIMESTAMP'], format='ISO8601', errors='coerce')
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw['Year'] = df_raw['TIMESTAMP'].dt.year
    df_raw = df_raw.sort_values('Date').reset_index(drop=True)

    # Basic groupby as original
    df_daily = df_raw.groupby('Date').agg({
        'Alarms_Rolling_Avg': 'first',
        'Year': 'first',
        'Month': 'first',
        'Day': 'first',
        'Hour': 'mean',
        'DayOfWeek': 'first',
        'WeekOfYear': 'first',
        'Line': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'Maq': lambda x: x.nunique(),
        'Fault_Code1': lambda x: x.nunique(),
        'Bilancella': 'mean',
        'Trato': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()

    df_daily['Target'] = df_daily['Alarms_Rolling_Avg'].shift(-1)
    df_daily = df_daily[:-1].reset_index(drop=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in ['DayOfWeek', 'Line', 'Trato']:
        if col in df_daily.columns:
            df_daily[f'{col}_Encoded'] = le.fit_transform(df_daily[col])

    # Excluir os primeiros 7 valores devido a features nulas (rolling average)
    df_daily = df_daily.iloc[7:].reset_index(drop=True)

    # Filter feature columns to those present
    available_features = [f for f in feature_cols if f in df_daily.columns]
    X = df_daily[available_features].copy()
    y = df_daily['Target'].copy()
    df_to_split = df_daily
    feature_cols = available_features

else:
    print("CSV recognized as a dataset table.")
    if 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='ISO8601', errors='coerce')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    if 'Year' not in df.columns:
        if 'TIMESTAMP' in df.columns:
            df['Year'] = df['TIMESTAMP'].dt.year
        elif 'Date' in df.columns:
            df['Year'] = df['Date'].dt.year
        else:
            df['Year'] = 2020 

    if 'Target' not in df.columns:
        if 'Alarms_Rolling_Avg' in df.columns:
            df['Target'] = df['Alarms_Rolling_Avg'].shift(-1)
            df = df[:-1].reset_index(drop=True)
        else:
            df['Target'] = df.iloc[:, -1]

    # Excluir os primeiros 7 valores devido a features nulas
    df = df.iloc[7:].reset_index(drop=True)

    exclude_cols = ['Target', 'Date', 'TIMESTAMP', 'Year']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    for col in feature_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])

    X = df[feature_cols].copy()
    y = df['Target'].copy()
    df_to_split = df

if 'Year' in df_to_split.columns:
    train_mask = df_to_split['Year'] < 2024
    test_mask = df_to_split['Year'] == 2024
else:
    split_idx = int(len(X) * 0.8)
    train_mask = np.arange(len(X)) < split_idx
    test_mask = np.arange(len(X)) >= split_idx

X_train = X[train_mask].reset_index(drop=True)
y_train = y[train_mask].reset_index(drop=True)
X_test = X[test_mask].reset_index(drop=True)
y_test = y[test_mask].reset_index(drop=True)

print(f"Training data: {len(X_train)} samples")
print(f"Test data: {len(X_test)} samples")
print(f"\nFeatures used: {feature_cols}")

print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_test = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

mape = mean_absolute_percentage_error(y_test, y_pred_test)
smape = symmetric_mean_absolute_percentage_error(y_test, y_pred_test)
f1, precision, recall, threshold = compute_f1_score(y_test, y_pred_test)

print(f"\n{'='*50}")
print(f"Random Forest Model Performance")
print(f"{'='*50}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"\n{'='*50}")
print(f"Additional Validation Metrics")
print(f"{'='*50}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")
print(f"\n{'='*50}")
print(f"F1-Score Metrics (Binary Classification at threshold={threshold:.2f})")
print(f"{'='*50}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n{'='*50}")
print("Feature Importance (Top 10)")
print(f"{'='*50}")
print(feature_importance.head(10).to_string())

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
suffix_name = 'simple'

plt.figure(figsize=(16, 6))
plt.plot(range(len(y_test)), y_test.values, label='Actual', marker='o', alpha=0.7)
plt.plot(range(len(y_pred_test)), y_pred_test, label='Predicted', marker='s', alpha=0.7)
plt.title(f'Random Forest ({suffix_name}): 1-Day Ahead Alarm Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Number of Alarms')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / f'rf_{suffix_name}_predictions.png', bbox_inches='tight')
plt.close()

residuals = y_test.values - y_pred_test
plt.figure(figsize=(16, 6))
plt.scatter(y_pred_test, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title(f'Residual Plot ({suffix_name})', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / f'rf_{suffix_name}_residuals.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'].head(10), feature_importance['Importance'].head(10))
plt.title(f'Top 10 Feature Importance ({suffix_name})', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(RESULTS_DIR / f'rf_{suffix_name}_feature_importance.png', bbox_inches='tight')
plt.close()
