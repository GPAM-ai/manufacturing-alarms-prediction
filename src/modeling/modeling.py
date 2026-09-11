"""Model training and evaluation for daily alarm-recurrence forecasting.

The module compares Random Forest and XGBoost regressors using the simple and
complete feature tables produced by ``src.feat_eng.feature_eng``. It provides helpers to
load the datasets, reserve a final calendar holdout, tune compact parameter
sets with expanding windows, evaluate count predictions, inspect results, and
save reproducible experiment artifacts.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
)

DATE_COL = 'Date'
TARGET_COL = 'recurrences_lt_30m'
TARGET_AVAILABLE_COL = 'target_available'
NON_FEATURE_COLUMNS = {DATE_COL, TARGET_COL, TARGET_AVAILABLE_COL}
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_DAYS = 365

MODELING_ARTIFACT_SUBDIRECTORIES = {
    'models': 'models',
    'metadata': 'metadata',
    'metrics': 'metrics',
    'predictions': 'predictions',
    'residuals': 'residuals',
    'feature_importance': 'feature_importance',
    'validation': 'validation',
    'comparisons': 'comparisons',
}

RANDOM_FOREST_PARAM_CANDIDATES = (
    {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_leaf': 1,
        'max_features': 1.0,
    },
    {
        'n_estimators': 400,
        'max_depth': 12,
        'min_samples_leaf': 1,
        'max_features': 1.0,
    },
    {
        'n_estimators': 400,
        'max_depth': 12,
        'min_samples_leaf': 3,
        'max_features': 1.0,
    },
    {
        'n_estimators': 400,
        'max_depth': 20,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
    },
    {
        'n_estimators': 600,
        'max_depth': None,
        'min_samples_leaf': 2,
        'max_features': 0.7,
    },
    {
        'n_estimators': 600,
        'max_depth': 20,
        'min_samples_leaf': 5,
        'max_features': 0.7,
    },
)

XGBOOST_PARAM_CANDIDATES = (
    {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'max_depth': 3,
        'min_child_weight': 1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_lambda': 1.0,
    },
    {
        'n_estimators': 500,
        'learning_rate': 0.03,
        'max_depth': 3,
        'min_child_weight': 5,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_lambda': 5.0,
    },
    {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_child_weight': 1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_lambda': 1.0,
    },
    {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_child_weight': 5,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_lambda': 5.0,
    },
    {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
    },
    {
        'n_estimators': 500,
        'learning_rate': 0.03,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 5.0,
    },
)


def get_feature_columns(dataset):
    """Return predictor names, excluding the date, target, and target-quality flag."""
    return [
        column for column in dataset.columns
        if column not in NON_FEATURE_COLUMNS
    ]


def load_feature_dataset(path, timezone='America/Recife'):
    """Load and validate a model-ready feature table in chronological order."""
    dataset_path = Path(path)
    dataset = pd.read_csv(dataset_path)

    required = {DATE_COL, TARGET_COL, TARGET_AVAILABLE_COL}
    missing = required.difference(dataset.columns)
    if missing:
        raise KeyError(f'Missing required model column(s): {sorted(missing)}')

    dataset[DATE_COL] = (
        pd.to_datetime(dataset[DATE_COL], errors='raise', utc=True)
        .dt.tz_convert(timezone)
    )
    dataset = dataset.sort_values(DATE_COL).reset_index(drop=True)

    if dataset[DATE_COL].duplicated().any():
        raise ValueError('The feature table must contain at most one row per date')
    if not dataset[TARGET_COL].ge(0).all():
        raise ValueError(f'{TARGET_COL} must contain non-negative counts')
    if not dataset[TARGET_AVAILABLE_COL].isin([0, 1]).all():
        raise ValueError(f'{TARGET_AVAILABLE_COL} must contain only 0 or 1')

    feature_columns = get_feature_columns(dataset)
    non_numeric = [
        column for column in feature_columns
        if not pd.api.types.is_numeric_dtype(dataset[column])
    ]
    if non_numeric:
        raise TypeError(f'Non-numeric predictor column(s): {non_numeric}')
    if dataset[feature_columns].isna().any().any():
        incomplete = dataset[feature_columns].columns[
            dataset[feature_columns].isna().any()
        ].tolist()
        raise ValueError(f'Incomplete predictor column(s): {incomplete}')

    unavailable = int(dataset[TARGET_AVAILABLE_COL].eq(0).sum())
    if unavailable:
        dataset = dataset.loc[dataset[TARGET_AVAILABLE_COL].eq(1)].reset_index(drop=True)
        print(f'[LOG] Excluded {unavailable:,} target-unavailable row(s) from {dataset_path}')

    print(
        f'[LOG] Loaded {dataset_path}: {len(dataset):,} rows x '
        f'{len(feature_columns)} predictors'
    )
    return dataset


def validate_aligned_datasets(simple_dataset, complete_dataset):
    """Confirm that simple and complete tables use identical dates and targets."""
    simple_keys = simple_dataset[[DATE_COL, TARGET_COL]].reset_index(drop=True)
    complete_keys = complete_dataset[[DATE_COL, TARGET_COL]].reset_index(drop=True)
    if not simple_keys.equals(complete_keys):
        raise ValueError('Simple and complete datasets are not aligned by date and target')
    return True


def split_final_holdout(dataset, test_days=DEFAULT_TEST_DAYS, min_test_rows=30):
    """Reserve the last ``test_days`` calendar days as an untouched final test."""
    if test_days < 1:
        raise ValueError('test_days must be positive')
    if min_test_rows < 1:
        raise ValueError('min_test_rows must be positive')
    if dataset.empty:
        raise ValueError('dataset cannot be empty')

    ordered = dataset.sort_values(DATE_COL).reset_index(drop=True)
    test_end = ordered[DATE_COL].max().normalize()
    test_start = test_end - pd.Timedelta(days=test_days - 1)
    development = ordered.loc[ordered[DATE_COL].lt(test_start)].reset_index(drop=True)
    test = ordered.loc[ordered[DATE_COL].ge(test_start)].reset_index(drop=True)

    if development.empty:
        raise ValueError('The final holdout leaves no development observations')
    if len(test) < min_test_rows:
        raise ValueError(
            f'The final holdout has only {len(test)} rows; '
            f'min_test_rows={min_test_rows}'
        )

    summary = pd.DataFrame([{
        'Development_start': development[DATE_COL].min(),
        'Development_end': development[DATE_COL].max(),
        'Development_rows': len(development),
        'Test_start': test[DATE_COL].min(),
        'Test_end': test[DATE_COL].max(),
        'Test_rows': len(test),
        'Test_calendar_days': test_days,
    }])
    print(
        f'[LOG] Final test isolated: {test[DATE_COL].min().date()} to '
        f'{test[DATE_COL].max().date()} ({len(test):,} available rows)'
    )
    return development, test, summary


def build_expanding_validation(
    development_dataset,
    initial_train_days=730,
    validation_days=180,
    step_days=180,
    min_validation_rows=30,
):
    """Create expanding-window folds inside the development period."""
    if initial_train_days < 1 or validation_days < 1 or step_days < 1:
        raise ValueError('Window sizes must be positive')
    if min_validation_rows < 1:
        raise ValueError('min_validation_rows must be positive')
    if DATE_COL not in development_dataset.columns:
        raise KeyError(f'Missing required column: {DATE_COL}')

    dates = pd.to_datetime(development_dataset[DATE_COL], errors='raise')
    if not dates.is_monotonic_increasing:
        raise ValueError('development_dataset must be sorted chronologically')
    if dates.duplicated().any():
        raise ValueError('development_dataset must contain at most one row per date')

    first_date = dates.min().normalize()
    last_date = dates.max().normalize()
    validation_start = first_date + pd.Timedelta(days=initial_train_days)
    splits = []
    summary_rows = []
    fold_number = 1

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
            summary_rows.append({
                'Fold': fold_number,
                'Train_start': dates.iloc[train_index[0]],
                'Train_end': dates.iloc[train_index[-1]],
                'Validation_start': dates.iloc[validation_index[0]],
                'Validation_end': dates.iloc[validation_index[-1]],
                'Train_rows': len(train_index),
                'Validation_rows': len(validation_index),
            })
            fold_number += 1

        validation_start += pd.Timedelta(days=step_days)

    if not splits:
        raise ValueError('No valid expanding-window split could be created')
    summary = pd.DataFrame(summary_rows)
    print(f'[LOG] Expanding-window validation created: {len(splits)} fold(s)')
    return splits, summary


def get_default_param_candidates(model_name):
    """Return the compact parameter set used for a supported estimator."""
    normalized = model_name.lower()
    if normalized in {'random_forest', 'rf'}:
        return [dict(params) for params in RANDOM_FOREST_PARAM_CANDIDATES]
    if normalized in {'xgboost', 'xgb'}:
        return [dict(params) for params in XGBOOST_PARAM_CANDIDATES]
    raise ValueError("model_name must be 'random_forest' or 'xgboost'")


def build_estimator(
    model_name,
    params=None,
    random_state=DEFAULT_RANDOM_STATE,
    n_jobs=-1,
):
    """Instantiate a configured Random Forest or Poisson XGBoost regressor."""
    normalized = model_name.lower()
    params = dict(params or {})

    if normalized in {'random_forest', 'rf'}:
        defaults = {
            'n_estimators': 400,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 1.0,
            'random_state': random_state,
            'n_jobs': n_jobs,
        }
        defaults.update(params)
        return RandomForestRegressor(**defaults)

    if normalized in {'xgboost', 'xgb'}:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                'XGBoost is required for model_name="xgboost". '
                'Install the dependencies listed in requirements.txt.'
            ) from exc

        defaults = {
            'objective': 'count:poisson',
            'eval_metric': 'poisson-nloglik',
            'tree_method': 'hist',
            'random_state': random_state,
            'n_jobs': n_jobs,
            'verbosity': 0,
        }
        defaults.update(params)
        return XGBRegressor(**defaults)

    raise ValueError("model_name must be 'random_forest' or 'xgboost'")


def predict_nonnegative(estimator, features):
    """Predict daily counts and enforce their non-negative domain."""
    prediction = np.asarray(estimator.predict(features), dtype=float)
    return np.clip(prediction, a_min=0.0, a_max=None)


def calculate_regression_metrics(y_true, y_pred):
    """Calculate count-regression metrics without percentage denominators."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    if actual.size == 0:
        raise ValueError('At least one observation is required')
    if (actual < 0).any():
        raise ValueError('y_true must contain non-negative counts')

    predicted = np.clip(predicted, a_min=0.0, a_max=None)
    poisson_prediction = np.clip(predicted, a_min=np.finfo(float).eps, a_max=None)
    return {
        'Rows': int(actual.size),
        'MAE': float(mean_absolute_error(actual, predicted)),
        'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
        'R2': float(r2_score(actual, predicted)),
        'PoissonDeviance': float(mean_poisson_deviance(actual, poisson_prediction)),
        'MeanActual': float(actual.mean()),
        'MeanPredicted': float(predicted.mean()),
    }


def tune_model(
    model_name,
    features,
    target,
    splits,
    param_candidates=None,
    random_state=DEFAULT_RANDOM_STATE,
    n_jobs=-1,
    verbose=True,
):
    """Select hyperparameters by mean expanding-window validation MAE."""
    if len(features) != len(target):
        raise ValueError('features and target must contain the same number of rows')
    if not splits:
        raise ValueError('At least one validation split is required')

    candidates = (
        get_default_param_candidates(model_name)
        if param_candidates is None
        else list(param_candidates)
    )
    if not candidates:
        raise ValueError('At least one parameter candidate is required')

    fold_rows = []
    for candidate_number, params in enumerate(candidates, start=1):
        if verbose:
            print(
                f'[LOG] {model_name}: candidate {candidate_number}/{len(candidates)} '
                f'across {len(splits)} fold(s)'
            )
        for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
            estimator = build_estimator(
                model_name,
                params=params,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            estimator.fit(features.iloc[train_index], target.iloc[train_index])
            prediction = predict_nonnegative(
                estimator,
                features.iloc[validation_index],
            )
            metrics = calculate_regression_metrics(
                target.iloc[validation_index],
                prediction,
            )
            fold_rows.append({
                'Candidate': candidate_number,
                'Fold': fold_number,
                'Parameters': json.dumps(params, sort_keys=True),
                **metrics,
            })

    fold_metrics = pd.DataFrame(fold_rows)
    search_summary = (
        fold_metrics.groupby(['Candidate', 'Parameters'], as_index=False)
        .agg(
            CV_MAE_mean=('MAE', 'mean'),
            CV_MAE_std=('MAE', 'std'),
            CV_RMSE_mean=('RMSE', 'mean'),
            CV_RMSE_std=('RMSE', 'std'),
            CV_R2_mean=('R2', 'mean'),
            CV_PoissonDeviance_mean=('PoissonDeviance', 'mean'),
            Folds=('Fold', 'nunique'),
        )
        .sort_values(['CV_MAE_mean', 'CV_RMSE_mean', 'Candidate'])
        .reset_index(drop=True)
    )
    best_candidate = int(search_summary.loc[0, 'Candidate'])
    best_params = dict(candidates[best_candidate - 1])
    best_fold_metrics = (
        fold_metrics.loc[fold_metrics['Candidate'].eq(best_candidate)]
        .reset_index(drop=True)
    )
    print(
        f"[LOG] {model_name}: selected candidate {best_candidate} "
        f"(CV MAE={search_summary.loc[0, 'CV_MAE_mean']:.3f})"
    )
    return {
        'best_params': best_params,
        'best_candidate': best_candidate,
        'search_summary': search_summary,
        'fold_metrics': fold_metrics,
        'best_fold_metrics': best_fold_metrics,
    }


def extract_feature_importance(estimator, feature_columns):
    """Return impurity/gain importance exposed by the fitted tree estimator."""
    if not hasattr(estimator, 'feature_importances_'):
        raise TypeError('The fitted estimator does not expose feature_importances_')
    importance = np.asarray(estimator.feature_importances_, dtype=float)
    if len(importance) != len(feature_columns):
        raise ValueError('Feature names and importance values have different lengths')
    return (
        pd.DataFrame({'Feature': feature_columns, 'Importance': importance})
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )


def run_experiment(
    model_name,
    dataset_name,
    development_dataset,
    test_dataset,
    splits,
    param_candidates=None,
    random_state=DEFAULT_RANDOM_STATE,
    n_jobs=-1,
    verbose=True,
):
    """Tune on development folds, refit on all development data, and test once."""
    feature_columns = get_feature_columns(development_dataset)
    if feature_columns != get_feature_columns(test_dataset):
        raise ValueError('Development and test predictor columns do not match')

    development_features = development_dataset[feature_columns]
    development_target = development_dataset[TARGET_COL]
    test_features = test_dataset[feature_columns]
    test_target = test_dataset[TARGET_COL]

    tuning = tune_model(
        model_name=model_name,
        features=development_features,
        target=development_target,
        splits=splits,
        param_candidates=param_candidates,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    estimator = build_estimator(
        model_name,
        params=tuning['best_params'],
        random_state=random_state,
        n_jobs=n_jobs,
    )
    estimator.fit(development_features, development_target)
    test_prediction = predict_nonnegative(estimator, test_features)
    test_metrics = calculate_regression_metrics(test_target, test_prediction)
    test_metrics.update({
        'Model': model_name.lower(),
        'Dataset': dataset_name.lower(),
        'Test_start': test_dataset[DATE_COL].min(),
        'Test_end': test_dataset[DATE_COL].max(),
    })

    predictions = pd.DataFrame({
        DATE_COL: test_dataset[DATE_COL].to_numpy(),
        'Actual': test_target.to_numpy(),
        'Predicted': test_prediction,
    })
    predictions['Residual'] = predictions['Actual'] - predictions['Predicted']
    importance = extract_feature_importance(estimator, feature_columns)

    print(
        f"[LOG] {model_name}/{dataset_name}: test MAE={test_metrics['MAE']:.3f}, "
        f"RMSE={test_metrics['RMSE']:.3f}, R2={test_metrics['R2']:.3f}"
    )
    return {
        'model_name': model_name.lower(),
        'dataset_name': dataset_name.lower(),
        'estimator': estimator,
        'best_params': tuning['best_params'],
        'best_candidate': tuning['best_candidate'],
        'search_summary': tuning['search_summary'],
        'fold_metrics': tuning['fold_metrics'],
        'best_fold_metrics': tuning['best_fold_metrics'],
        'feature_columns': feature_columns,
        'feature_importance': importance,
        'test_metrics': test_metrics,
        'predictions': predictions,
        'development_start': development_dataset[DATE_COL].min(),
        'development_end': development_dataset[DATE_COL].max(),
    }


def evaluate_lag1_baseline(full_dataset, test_dataset):
    """Evaluate previous-calendar-day recurrence persistence when observable."""
    ordered = full_dataset.sort_values(DATE_COL).reset_index(drop=True)
    if ordered[DATE_COL].duplicated().any():
        raise ValueError('full_dataset must contain at most one row per date')

    target_by_date = ordered.set_index(DATE_COL)[TARGET_COL]
    previous_dates = test_dataset[DATE_COL] - pd.Timedelta(days=1)
    prediction = previous_dates.map(target_by_date)
    predictions = pd.DataFrame({
        DATE_COL: test_dataset[DATE_COL].to_numpy(),
        'Actual': test_dataset[TARGET_COL].to_numpy(),
        'Predicted': prediction.to_numpy(),
    })
    predictions['Baseline_available'] = predictions['Predicted'].notna().astype(int)
    predictions['Residual'] = predictions['Actual'] - predictions['Predicted']

    comparable = predictions.loc[predictions['Baseline_available'].eq(1)]
    if comparable.empty:
        raise ValueError('No final-test day has an available previous-day target')
    metrics = calculate_regression_metrics(
        comparable['Actual'],
        comparable['Predicted'],
    )
    metrics.update({
        'Model': 'lag1_baseline',
        'Dataset': 'target_only',
        'Coverage': float(len(comparable) / len(predictions)),
        'Test_start': test_dataset[DATE_COL].min(),
        'Test_end': test_dataset[DATE_COL].max(),
    })
    print(
        f"[LOG] Lag-1 baseline: MAE={metrics['MAE']:.3f}, "
        f"coverage={metrics['Coverage']:.1%} ({len(comparable)}/{len(predictions)} rows)"
    )
    return {'metrics': metrics, 'predictions': predictions}


def build_model_comparison(experiments, baseline_result=None):
    """Combine final-test metrics from all experiments and the optional baseline."""
    rows = [dict(experiment['test_metrics']) for experiment in experiments]
    for row in rows:
        row.setdefault('Coverage', 1.0)
    if baseline_result is not None:
        rows.append(dict(baseline_result['metrics']))
    return (
        pd.DataFrame(rows)
        .sort_values(['MAE', 'RMSE'])
        .reset_index(drop=True)
    )


def prepare_modeling_artifact_directories(output_dir):
    """Create and return the organized modeling-artifact directories."""
    root = Path(output_dir)
    directories = {
        section: root / directory
        for section, directory in MODELING_ARTIFACT_SUBDIRECTORIES.items()
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def save_experiment_artifacts(experiment, output_dir):
    """Persist a fitted model, parameters, metrics, predictions, and CV tables."""
    destination = Path(output_dir)
    directories = prepare_modeling_artifact_directories(destination)
    stem = f"{experiment['model_name']}_{experiment['dataset_name']}"

    paths = {
        'model': directories['models'] / f'{stem}.joblib',
        'metadata': directories['metadata'] / f'{stem}_metadata.json',
        'predictions': directories['predictions'] / f'{stem}_predictions.csv',
        'feature_importance': (
            directories['feature_importance'] / f'{stem}_feature_importance.csv'
        ),
        'cv_search': directories['validation'] / f'{stem}_cv_search.csv',
        'cv_folds': directories['validation'] / f'{stem}_cv_folds.csv',
    }
    joblib.dump(experiment['estimator'], paths['model'])
    experiment['predictions'].to_csv(paths['predictions'], index=False)
    experiment['feature_importance'].to_csv(paths['feature_importance'], index=False)
    experiment['search_summary'].to_csv(paths['cv_search'], index=False)
    experiment['best_fold_metrics'].to_csv(paths['cv_folds'], index=False)

    metadata = {
        'model': experiment['model_name'],
        'dataset': experiment['dataset_name'],
        'target': TARGET_COL,
        'best_candidate': experiment['best_candidate'],
        'best_params': experiment['best_params'],
        'feature_columns': experiment['feature_columns'],
        'development_start': experiment['development_start'].isoformat(),
        'development_end': experiment['development_end'].isoformat(),
        'test_metrics': {
            key: value.isoformat() if isinstance(value, pd.Timestamp) else value
            for key, value in experiment['test_metrics'].items()
        },
    }
    paths['metadata'].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(f'[LOG] Saved experiment artifacts under {destination}')
    return paths


def save_baseline_artifacts(baseline_result, output_dir):
    """Persist lag-1 baseline predictions and metrics."""
    destination = Path(output_dir)
    directories = prepare_modeling_artifact_directories(destination)
    predictions_path = directories['predictions'] / 'lag1_baseline_predictions.csv'
    metrics_path = directories['metrics'] / 'lag1_baseline_metrics.json'
    baseline_result['predictions'].to_csv(predictions_path, index=False)
    serializable_metrics = {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value
        for key, value in baseline_result['metrics'].items()
    }
    metrics_path.write_text(
        json.dumps(serializable_metrics, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return {'predictions': predictions_path, 'metrics': metrics_path}


def plot_validation_windows(split_summary, figsize=(12, 4)):
    """Plot the training and validation calendar ranges of each expanding fold."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=figsize)
    for position, row in split_summary.reset_index(drop=True).iterrows():
        axis.plot(
            [row['Train_start'], row['Train_end']],
            [position, position],
            color='#2878B5',
            linewidth=7,
            solid_capstyle='butt',
            label='Training' if position == 0 else None,
        )
        axis.plot(
            [row['Validation_start'], row['Validation_end']],
            [position, position],
            color='#F28E2B',
            linewidth=7,
            solid_capstyle='butt',
            label='Validation' if position == 0 else None,
        )
    axis.set_yticks(range(len(split_summary)))
    axis.set_yticklabels([f'Fold {fold}' for fold in split_summary['Fold']])
    axis.set_xlabel('Date')
    axis.set_title('Expanding-window validation')
    axis.grid(axis='x', alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_predictions(predictions, title=None, figsize=(14, 5)):
    """Plot observed and predicted recurrence counts through the final test."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(predictions[DATE_COL], predictions['Actual'], label='Observed', linewidth=1.5)
    axis.plot(predictions[DATE_COL], predictions['Predicted'], label='Predicted', linewidth=1.5)
    axis.set_title(title or 'Observed and predicted daily recurrences')
    axis.set_xlabel('Date')
    axis.set_ylabel('Recurrences with IAT < 30 min')
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_residuals(predictions, title=None, figsize=(12, 5)):
    """Plot final-test residuals against fitted values."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=figsize)
    axis.scatter(predictions['Predicted'], predictions['Residual'], alpha=0.6)
    axis.axhline(0, color='black', linestyle='--', linewidth=1)
    axis.set_title(title or 'Final-test residuals')
    axis.set_xlabel('Predicted recurrences')
    axis.set_ylabel('Observed - predicted')
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure, axis


def plot_feature_importance(feature_importance, top_n=15, title=None, figsize=(10, 6)):
    """Plot the most influential features reported by a fitted tree model."""
    import matplotlib.pyplot as plt

    selected = feature_importance.head(top_n).sort_values('Importance')
    figure, axis = plt.subplots(figsize=figsize)
    axis.barh(selected['Feature'], selected['Importance'], color='#2878B5')
    axis.set_title(title or f'Top {top_n} feature importances')
    axis.set_xlabel('Importance')
    figure.tight_layout()
    return figure, axis


def plot_model_comparison(comparison, metric='MAE', figsize=(10, 5)):
    """Plot a selected final-test metric for all model/dataset combinations."""
    import matplotlib.pyplot as plt

    if metric not in comparison.columns:
        raise KeyError(f'Metric not found in comparison table: {metric}')
    plot_data = comparison.copy()
    plot_data['Experiment'] = plot_data['Model'] + ' / ' + plot_data['Dataset']
    plot_data = plot_data.sort_values(metric, ascending=True)
    figure, axis = plt.subplots(figsize=figsize)
    axis.barh(plot_data['Experiment'], plot_data[metric], color='#59A14F')
    axis.set_title(f'Final-test model comparison — {metric}')
    axis.set_xlabel(metric)
    figure.tight_layout()
    return figure, axis
