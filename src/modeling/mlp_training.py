"""Final training and evaluation pipeline for the Optuna-selected MLP."""

import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.mlp import MLPRegressor
from src.modeling.mlp_optuna import (
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MLP_SEEDS,
    DEFAULT_OPTUNA_TRIALS,
    DEFAULT_PRUNING_INTERVAL,
    DEFAULT_SAMPLER_SEED,
    optimize_mlp,
)
from src.modeling.modeling import (
    DATE_COL,
    calculate_regression_metrics,
    plot_predictions,
    plot_residuals,
    prepare_modeling_artifact_directories,
)


def _resolve_device(device):
    """Resolve the requested PyTorch device."""
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resolved = torch.device(device if device is not None else default_device)
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A CUDA device was requested, but CUDA is not available')
    return resolved


def _set_seed(seed):
    """Seed all random generators used during MLP fitting."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _validate_training_params(params):
    """Validate the parameters required to refit the selected MLP."""
    required = {
        'hidden_sizes',
        'dropout',
        'learning_rate',
        'weight_decay',
        'batch_size',
        'max_epochs',
    }
    missing = required.difference(params)
    if missing:
        raise KeyError(f'Missing final MLP parameter(s): {sorted(missing)}')

    hidden_sizes = tuple(params['hidden_sizes'])
    if not hidden_sizes or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1
        for size in hidden_sizes
    ):
        raise ValueError('hidden_sizes must contain positive integers')
    if not 0.0 <= params['dropout'] < 1.0:
        raise ValueError('dropout must be in the interval [0, 1)')
    if params['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    if params['weight_decay'] < 0:
        raise ValueError('weight_decay cannot be negative')
    for name in ('batch_size', 'max_epochs'):
        value = params[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name} must be a positive integer')

    normalized = dict(params)
    normalized['hidden_sizes'] = hidden_sizes
    return normalized


def _validate_seeds(seeds):
    """Validate and normalize final-training random seeds."""
    normalized = tuple(seeds)
    if not normalized:
        raise ValueError('At least one final-training seed is required')
    if any(
        not isinstance(seed, int) or isinstance(seed, bool)
        for seed in normalized
    ):
        raise TypeError('seeds must contain only integers')
    if len(set(normalized)) != len(normalized):
        raise ValueError('seeds must not contain duplicates')
    return normalized


def _fit_mlp(features, target, params, seed, device):
    """Fit one final MLP replica on the complete development partition."""
    _set_seed(seed)
    model = MLPRegressor(
        input_size=features.shape[1],
        hidden_sizes=params['hidden_sizes'],
        dropout=params['dropout'],
    ).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(target),
        ),
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
    )
    loss_function = nn.L1Loss()

    final_epoch_loss = np.nan
    for _ in range(params['max_epochs']):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_features, batch_target in loader:
            batch_features = batch_features.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_features)
            loss = loss_function(prediction, batch_target)
            loss.backward()
            optimizer.step()

            batch_size = len(batch_target)
            loss_sum += loss.detach().item() * batch_size
            sample_count += batch_size
        final_epoch_loss = loss_sum / sample_count
    return model, float(final_epoch_loss)


def _predict_mlp(model, features, batch_size, device):
    """Predict non-negative counts in their original row order."""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for (batch_features,) in loader:
            prediction = model(batch_features.to(device))
            predictions.append(prediction.cpu().numpy())
    raw_prediction = np.concatenate(predictions).astype(float, copy=False)
    return np.clip(raw_prediction, a_min=0.0, a_max=None)


def train_mlp_experiment(
    experiment,
    n_trials=DEFAULT_OPTUNA_TRIALS,
    seeds=DEFAULT_MLP_SEEDS,
    max_epochs=DEFAULT_MAX_EPOCHS,
    pruning_interval=DEFAULT_PRUNING_INTERVAL,
    timeout=None,
    sampler_seed=DEFAULT_SAMPLER_SEED,
    suggest_params=None,
    sampler=None,
    pruner=None,
    device=None,
    verbose=True,
):
    """Optimize, refit, and evaluate one prepared MLP experiment.

    Hyperparameters are selected exclusively with expanding-window folds from
    the development partition. The best architecture is then fitted once per
    seed on all development rows. The final prediction is the mean of those
    replicas and the untouched holdout is evaluated only after refitting.
    """
    if experiment.get('model_name') != 'mlp':
        raise ValueError('train_mlp_experiment requires an MLP experiment')
    required = {
        'feature_set',
        'feature_columns',
        'development_features',
        'development_target',
        'test_features',
        'test_target',
        'test_dataset',
        'splits',
    }
    missing = required.difference(experiment)
    if missing:
        raise KeyError(f'Incomplete MLP experiment definition: {sorted(missing)}')

    seeds = _validate_seeds(seeds)
    resolved_device = _resolve_device(device)
    search_function = experiment.get('search_function', optimize_mlp)
    if not callable(search_function):
        raise TypeError('The MLP search function must be callable')

    optimization = search_function(
        experiment['development_features'],
        experiment['development_target'],
        experiment['splits'],
        n_trials=n_trials,
        seeds=seeds,
        max_epochs=max_epochs,
        pruning_interval=pruning_interval,
        timeout=timeout,
        sampler_seed=sampler_seed,
        suggest_params=suggest_params,
        sampler=sampler,
        pruner=pruner,
        device=resolved_device,
        study_name=f"mlp_{experiment['feature_set']}",
        verbose=verbose,
    )
    best_params = _validate_training_params(optimization['best_params'])

    scaler = StandardScaler()
    development_features = scaler.fit_transform(
        experiment['development_features']
    ).astype(np.float32, copy=False)
    test_features = scaler.transform(experiment['test_features']).astype(
        np.float32,
        copy=False,
    )
    development_target = np.asarray(
        experiment['development_target'],
        dtype=np.float32,
    )
    test_target = np.asarray(experiment['test_target'], dtype=np.float32)

    models = {}
    seed_rows = []
    seed_predictions = {}
    refit_start = perf_counter()
    for seed in seeds:
        training_start = perf_counter()
        model, final_epoch_loss = _fit_mlp(
            development_features,
            development_target,
            best_params,
            seed,
            resolved_device,
        )
        training_seconds = perf_counter() - training_start

        inference_start = perf_counter()
        prediction = _predict_mlp(
            model,
            test_features,
            best_params['batch_size'],
            resolved_device,
        )
        inference_seconds = perf_counter() - inference_start
        metrics = calculate_regression_metrics(test_target, prediction)
        seed_rows.append({
            'Seed': seed,
            'FinalEpochL1Loss': final_epoch_loss,
            'TrainingSeconds': training_seconds,
            'InferenceSeconds': inference_seconds,
            'RunSeconds': training_seconds + inference_seconds,
            **metrics,
        })
        seed_predictions[seed] = prediction
        models[seed] = model.cpu()

    refit_seconds = perf_counter() - refit_start
    ensemble_prediction = np.mean(
        np.stack(list(seed_predictions.values())),
        axis=0,
    )
    ensemble_metrics = calculate_regression_metrics(
        test_target,
        ensemble_prediction,
    )
    ensemble_metrics.update({
        'Model': 'mlp',
        'Dataset': experiment['feature_set'],
        'Seeds': len(seeds),
        'Test_start': experiment['test_dataset'][DATE_COL].min(),
        'Test_end': experiment['test_dataset'][DATE_COL].max(),
        'OptimizationSeconds': optimization['search_seconds'],
        'RefitSeconds': refit_seconds,
        'TotalExperimentSeconds': (
            optimization['search_seconds'] + refit_seconds
        ),
    })

    predictions = pd.DataFrame({
        DATE_COL: experiment['test_dataset'][DATE_COL].to_numpy(),
        'Actual': test_target,
        **{
            f'Predicted_seed_{seed}': seed_predictions[seed]
            for seed in seeds
        },
        'Predicted': ensemble_prediction,
    })
    predictions['Residual'] = predictions['Actual'] - predictions['Predicted']

    if verbose:
        print(
            f"[LOG] MLP/{experiment['feature_set']}: "
            f"test MAE={ensemble_metrics['MAE']:.3f}, "
            f"RMSE={ensemble_metrics['RMSE']:.3f}, "
            f"R2={ensemble_metrics['R2']:.3f}"
        )

    return {
        'model_name': 'mlp',
        'dataset_name': experiment['feature_set'],
        'models': models,
        'scaler': scaler,
        'seeds': seeds,
        'best_params': best_params,
        'best_trial': optimization['best_trial'],
        'trials_summary': optimization['trials_summary'],
        'run_metrics': optimization['run_metrics'],
        'fold_metrics': optimization['fold_metrics'],
        'best_fold_metrics': optimization['best_fold_metrics'],
        'feature_columns': list(experiment['feature_columns']),
        'seed_metrics': pd.DataFrame(seed_rows),
        'test_metrics': ensemble_metrics,
        'predictions': predictions,
        'development_start': experiment['development_dataset'][DATE_COL].min(),
        'development_end': experiment['development_dataset'][DATE_COL].max(),
        'optimization_seconds': optimization['search_seconds'],
        'refit_seconds': refit_seconds,
        'device': str(resolved_device),
        'study': optimization['study'],
    }


def train_mlp_feature_sets(experiments=None, **training_options):
    """Train the MLP pipeline for both prepared feature sets."""
    if experiments is None:
        from src.modeling.neural_experiments import (
            build_neural_experiment_matrix,
        )

        experiments = build_neural_experiment_matrix()

    required = ('mlp_simple', 'mlp_complete')
    missing = [name for name in required if name not in experiments]
    if missing:
        raise KeyError(f'Missing MLP experiment(s): {missing}')
    return {
        name: train_mlp_experiment(experiments[name], **training_options)
        for name in required
    }


def _serializable(value):
    """Convert metadata values to JSON-compatible Python objects."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _build_checkpoint(experiment):
    """Build a portable checkpoint containing replicas and preprocessing."""
    scaler = experiment['scaler']
    return {
        'format_version': 1,
        'model_name': experiment['model_name'],
        'dataset_name': experiment['dataset_name'],
        'feature_columns': experiment['feature_columns'],
        'best_params': experiment['best_params'],
        'seeds': experiment['seeds'],
        'scaler': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'var': scaler.var_.tolist(),
            'n_features_in': int(scaler.n_features_in_),
            'n_samples_seen': int(scaler.n_samples_seen_),
        },
        'state_dicts': {
            str(seed): model.state_dict()
            for seed, model in experiment['models'].items()
        },
    }


def save_mlp_experiment_artifacts(experiment, output_dir):
    """Save an MLP checkpoint, metadata, metrics, validation, and diagnostics."""
    if experiment.get('model_name') != 'mlp':
        raise ValueError('Only MLP experiment results can be saved here')
    directories = prepare_modeling_artifact_directories(Path(output_dir))
    stem = f"mlp_{experiment['dataset_name']}"
    paths = {
        'model': directories['models'] / f'{stem}.pt',
        'metadata': directories['metadata'] / f'{stem}_metadata.json',
        'test_metrics': directories['metrics'] / f'{stem}_test_metrics.json',
        'test_seed_metrics': directories['metrics'] / f'{stem}_test_seeds.csv',
        'predictions': directories['predictions'] / f'{stem}_predictions.csv',
        'prediction_plot': (
            directories['predictions'] / f'{stem}_predictions.png'
        ),
        'residual_plot': directories['residuals'] / f'{stem}_residuals.png',
        'optuna_trials': (
            directories['validation'] / f'{stem}_optuna_trials.csv'
        ),
        'cv_runs': directories['validation'] / f'{stem}_cv_runs.csv',
        'cv_folds': directories['validation'] / f'{stem}_cv_folds.csv',
        'cv_best_folds': (
            directories['validation'] / f'{stem}_cv_best_folds.csv'
        ),
    }

    torch.save(_build_checkpoint(experiment), paths['model'])
    experiment['seed_metrics'].to_csv(paths['test_seed_metrics'], index=False)
    experiment['predictions'].to_csv(paths['predictions'], index=False)
    experiment['trials_summary'].to_csv(paths['optuna_trials'], index=False)
    experiment['run_metrics'].to_csv(paths['cv_runs'], index=False)
    experiment['fold_metrics'].to_csv(paths['cv_folds'], index=False)
    experiment['best_fold_metrics'].to_csv(paths['cv_best_folds'], index=False)

    serializable_test_metrics = {
        key: _serializable(value)
        for key, value in experiment['test_metrics'].items()
    }
    paths['test_metrics'].write_text(
        json.dumps(serializable_test_metrics, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    prediction_figure, _ = plot_predictions(
        experiment['predictions'],
        title=f"MLP ({experiment['dataset_name']}): final holdout",
    )
    prediction_figure.savefig(
        paths['prediction_plot'],
        dpi=150,
        bbox_inches='tight',
    )
    residual_figure, _ = plot_residuals(
        experiment['predictions'],
        title=f"MLP ({experiment['dataset_name']}): residuals",
    )
    residual_figure.savefig(
        paths['residual_plot'],
        dpi=150,
        bbox_inches='tight',
    )
    import matplotlib.pyplot as plt

    plt.close(prediction_figure)
    plt.close(residual_figure)

    metadata = {
        'model': experiment['model_name'],
        'dataset': experiment['dataset_name'],
        'best_trial': experiment['best_trial'],
        'best_params': {
            key: _serializable(value)
            for key, value in experiment['best_params'].items()
        },
        'seeds': list(experiment['seeds']),
        'feature_columns': experiment['feature_columns'],
        'development_start': experiment['development_start'].isoformat(),
        'development_end': experiment['development_end'].isoformat(),
        'device': experiment['device'],
        'optimization_seconds': experiment['optimization_seconds'],
        'refit_seconds': experiment['refit_seconds'],
        'test_metrics': serializable_test_metrics,
    }
    paths['metadata'].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(f'[LOG] Saved MLP artifacts under {Path(output_dir)}')
    return paths


__all__ = [
    'save_mlp_experiment_artifacts',
    'train_mlp_experiment',
    'train_mlp_feature_sets',
]
