"""Shared integration of neural models with both engineered feature sets."""

from pathlib import Path

from src.modeling.lstm import LSTMRegressor
from src.modeling.lstm_grid_search import grid_search_lstm
from src.modeling.lstm_training import train_lstm_experiment
from src.modeling.mlp import MLPRegressor
from src.modeling.mlp_optuna import optimize_mlp
from src.modeling.mlp_training import train_mlp_experiment
from src.modeling.modeling import (
    TARGET_COL,
    build_expanding_validation,
    get_feature_columns,
    load_feature_dataset,
    split_final_holdout,
    validate_aligned_datasets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_NEURAL_FEATURE_PATHS = {
    'simple': PROJECT_ROOT / 'src' / 'feat_eng' / 'features_simple.csv',
    'complete': PROJECT_ROOT / 'src' / 'feat_eng' / 'features_final.csv',
}

EXPECTED_FEATURE_COUNTS = {
    'simple': 16,
    'complete': 36,
}

NEURAL_MODEL_REGISTRY = {
    'mlp': {
        'model_class': MLPRegressor,
        'search_function': optimize_mlp,
        'training_function': train_mlp_experiment,
        'search_method': 'optuna',
    },
    'lstm': {
        'model_class': LSTMRegressor,
        'search_function': grid_search_lstm,
        'training_function': train_lstm_experiment,
        'search_method': 'grid_search',
    },
}


def _resolve_feature_paths(feature_paths):
    """Validate feature-set names and normalize their paths."""
    paths = (
        DEFAULT_NEURAL_FEATURE_PATHS
        if feature_paths is None
        else feature_paths
    )
    if not isinstance(paths, dict):
        raise TypeError('feature_paths must be a dictionary')

    required = set(DEFAULT_NEURAL_FEATURE_PATHS)
    missing = required.difference(paths)
    unknown = set(paths).difference(required)
    if missing:
        raise KeyError(f'Missing neural feature set(s): {sorted(missing)}')
    if unknown:
        raise KeyError(f'Unknown neural feature set(s): {sorted(unknown)}')
    return {name: Path(paths[name]) for name in DEFAULT_NEURAL_FEATURE_PATHS}


def prepare_neural_feature_sets(
    feature_paths=None,
    test_days=365,
    min_test_rows=30,
    initial_train_days=730,
    validation_days=180,
    step_days=180,
    min_validation_rows=30,
):
    """Prepare aligned simple and complete data for neural experiments.

    The final holdout and expanding-window folds are defined from the simple
    feature set and reused by position for the complete set after their dates
    and targets have been verified as identical. No model is trained here.

    Returns:
        dict: Prepared data for the ``simple`` and ``complete`` experiments.
    """
    paths = _resolve_feature_paths(feature_paths)
    datasets = {
        name: load_feature_dataset(path)
        for name, path in paths.items()
    }
    validate_aligned_datasets(datasets['simple'], datasets['complete'])

    development_simple, test_simple, holdout_summary = split_final_holdout(
        datasets['simple'],
        test_days=test_days,
        min_test_rows=min_test_rows,
    )
    development_complete, test_complete, _ = split_final_holdout(
        datasets['complete'],
        test_days=test_days,
        min_test_rows=min_test_rows,
    )
    validate_aligned_datasets(development_simple, development_complete)
    validate_aligned_datasets(test_simple, test_complete)

    splits, validation_summary = build_expanding_validation(
        development_simple,
        initial_train_days=initial_train_days,
        validation_days=validation_days,
        step_days=step_days,
        min_validation_rows=min_validation_rows,
    )
    split_copies = tuple(
        (train_index.copy(), validation_index.copy())
        for train_index, validation_index in splits
    )

    partitions = {
        'simple': (development_simple, test_simple),
        'complete': (development_complete, test_complete),
    }
    prepared = {}
    for name, (development, test) in partitions.items():
        feature_columns = tuple(get_feature_columns(development))
        expected_count = EXPECTED_FEATURE_COUNTS[name]
        if len(feature_columns) != expected_count:
            raise ValueError(
                f'{name} feature set must contain {expected_count} predictors; '
                f'found {len(feature_columns)}'
            )

        prepared[name] = {
            'feature_set': name,
            'dataset_path': paths[name],
            'feature_columns': feature_columns,
            'input_size': len(feature_columns),
            'full_dataset': datasets[name],
            'development_dataset': development,
            'test_dataset': test,
            'development_features': development.loc[:, feature_columns],
            'development_target': development[TARGET_COL],
            'test_features': test.loc[:, feature_columns],
            'test_target': test[TARGET_COL],
            'splits': split_copies,
            'validation_summary': validation_summary.copy(),
            'holdout_summary': holdout_summary.copy(),
        }
    return prepared


def build_neural_experiment_matrix(feature_sets=None, **preparation_options):
    """Connect each neural pipeline to the simple and complete feature sets.

    The returned definitions contain model and search callables but do not
    instantiate, optimize, or train a neural network.
    """
    if feature_sets is not None and preparation_options:
        raise ValueError(
            'preparation_options cannot be used with prepared feature_sets'
        )
    prepared = (
        prepare_neural_feature_sets(**preparation_options)
        if feature_sets is None
        else feature_sets
    )

    required_feature_sets = set(EXPECTED_FEATURE_COUNTS)
    missing = required_feature_sets.difference(prepared)
    if missing:
        raise KeyError(f'Missing prepared feature set(s): {sorted(missing)}')

    experiments = {}
    for model_name, model_definition in NEURAL_MODEL_REGISTRY.items():
        for feature_set_name in DEFAULT_NEURAL_FEATURE_PATHS:
            feature_set = prepared[feature_set_name]
            experiment_name = f'{model_name}_{feature_set_name}'
            experiments[experiment_name] = {
                'experiment_name': experiment_name,
                'artifact_prefix': experiment_name,
                'model_name': model_name,
                'feature_set': feature_set_name,
                'model_class': model_definition['model_class'],
                'search_function': model_definition['search_function'],
                'training_function': model_definition['training_function'],
                'search_method': model_definition['search_method'],
                **feature_set,
            }
    return experiments


def instantiate_neural_model(experiment, **architecture_params):
    """Instantiate, but do not train, the model defined by an experiment."""
    required = {'model_name', 'model_class', 'input_size'}
    missing = required.difference(experiment)
    if missing:
        raise KeyError(f'Incomplete neural experiment definition: {sorted(missing)}')
    if experiment['model_name'] not in NEURAL_MODEL_REGISTRY:
        raise ValueError(f"Unknown neural model: {experiment['model_name']}")

    return experiment['model_class'](
        input_size=experiment['input_size'],
        **architecture_params,
    )


__all__ = [
    'DEFAULT_NEURAL_FEATURE_PATHS',
    'EXPECTED_FEATURE_COUNTS',
    'NEURAL_MODEL_REGISTRY',
    'build_neural_experiment_matrix',
    'instantiate_neural_model',
    'prepare_neural_feature_sets',
]
