"""Optuna hyperparameter optimization for the PyTorch MLP regressor."""

import json
import random
from time import perf_counter

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.mlp import MLPRegressor
from src.modeling.modeling import calculate_regression_metrics


DEFAULT_MLP_SEEDS = (17, 42, 73)
DEFAULT_OPTUNA_TRIALS = 30
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PRUNING_INTERVAL = 5
DEFAULT_SAMPLER_SEED = 42


def suggest_mlp_params(trial):
    """Define the default conditional MLP search space for one Optuna trial."""
    layer_count = trial.suggest_int('n_layers', 1, 3)
    hidden_sizes = tuple(
        trial.suggest_categorical(
            f'hidden_size_{layer_number}',
            (16, 32, 64, 128),
        )
        for layer_number in range(1, layer_count + 1)
    )
    return {
        'hidden_sizes': hidden_sizes,
        'dropout': trial.suggest_float('dropout', 0.0, 0.4, step=0.1),
        'learning_rate': trial.suggest_float(
            'learning_rate',
            1e-4,
            1e-2,
            log=True,
        ),
        'weight_decay': trial.suggest_float(
            'weight_decay',
            1e-6,
            1e-2,
            log=True,
        ),
        'batch_size': trial.suggest_categorical('batch_size', (16, 32, 64)),
    }


def _validate_expanding_splits(splits, row_count):
    """Validate and normalize chronological expanding-window index pairs."""
    if not splits:
        raise ValueError('At least one validation split is required')

    normalized = []
    for fold, (train_index, validation_index) in enumerate(splits, start=1):
        train_index = np.asarray(train_index)
        validation_index = np.asarray(validation_index)
        for name, index in (
            ('train_index', train_index),
            ('validation_index', validation_index),
        ):
            if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
                raise TypeError(
                    f'Fold {fold} {name} must be a one-dimensional integer array'
                )
            if index.size == 0:
                raise ValueError(f'Fold {fold} {name} cannot be empty')
            if np.any(index < 0) or np.any(index >= row_count):
                raise IndexError(
                    f'Fold {fold} {name} contains positions outside the dataset'
                )
            if np.any(np.diff(index) <= 0):
                raise ValueError(f'Fold {fold} {name} must be strictly increasing')

        expected_train = np.arange(train_index[-1] + 1)
        expected_validation = np.arange(validation_index[0], validation_index[-1] + 1)
        if not np.array_equal(train_index, expected_train):
            raise ValueError(
                f'Fold {fold} training indices must form an expanding prefix'
            )
        if not np.array_equal(validation_index, expected_validation):
            raise ValueError(f'Fold {fold} validation indices must be contiguous')
        if validation_index[0] != train_index[-1] + 1:
            raise ValueError(
                f'Fold {fold} validation must begin immediately after its '
                'training prefix'
            )
        normalized.append((train_index.astype(int), validation_index.astype(int)))
    return normalized


def _validate_inputs(features, target, splits, seeds, max_epochs, pruning_interval):
    """Validate optimization inputs and return normalized arrays and indices."""
    feature_values = np.asarray(features, dtype=float)
    target_values = np.asarray(target, dtype=float)
    if feature_values.ndim != 2:
        raise ValueError('features must be a two-dimensional table')
    if target_values.ndim != 1:
        raise ValueError('target must be one-dimensional')
    if len(feature_values) != len(target_values):
        raise ValueError('features and target must contain the same number of rows')
    if len(feature_values) == 0:
        raise ValueError('features and target cannot be empty')
    if not np.isfinite(feature_values).all():
        raise ValueError('features must contain only finite numeric values')
    if not np.isfinite(target_values).all():
        raise ValueError('target must contain only finite numeric values')
    if np.any(target_values < 0):
        raise ValueError('target must contain non-negative counts')

    seeds = tuple(seeds)
    if not seeds:
        raise ValueError('At least one random seed is required')
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise TypeError('seeds must contain only integers')
    if len(set(seeds)) != len(seeds):
        raise ValueError('seeds must not contain duplicates')
    for name, value in (
        ('max_epochs', max_epochs),
        ('pruning_interval', pruning_interval),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name} must be a positive integer')

    normalized_splits = _validate_expanding_splits(splits, len(feature_values))
    return feature_values, target_values, normalized_splits, seeds


def _resolve_device(device):
    """Resolve an explicit device or select CUDA when it is available."""
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resolved = torch.device(device if device is not None else default_device)
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A CUDA device was requested, but CUDA is not available')
    return resolved


def _set_seed(seed):
    """Seed Python, NumPy, and PyTorch for a repeatable training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _validate_suggested_params(params):
    """Validate and normalize parameters returned by the suggestion function."""
    required = {
        'hidden_sizes',
        'dropout',
        'learning_rate',
        'weight_decay',
        'batch_size',
    }
    missing = required.difference(params)
    unknown = set(params).difference(required)
    if missing:
        raise KeyError(f'Missing suggested MLP parameter(s): {sorted(missing)}')
    if unknown:
        raise KeyError(f'Unknown suggested MLP parameter(s): {sorted(unknown)}')
    hidden_sizes = params['hidden_sizes']
    if isinstance(hidden_sizes, (str, bytes)):
        raise TypeError('hidden_sizes must be a sequence of positive integers')
    hidden_sizes = tuple(hidden_sizes)
    if not hidden_sizes:
        raise ValueError('hidden_sizes must contain at least one hidden layer')
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in hidden_sizes
    ):
        raise ValueError('hidden_sizes must contain only positive integers')
    if not 0.0 <= params['dropout'] < 1.0:
        raise ValueError('dropout must be in the interval [0, 1)')
    if params['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    if params['weight_decay'] < 0:
        raise ValueError('weight_decay cannot be negative')
    batch_size = params['batch_size']
    invalid_batch_size = (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    )
    if invalid_batch_size:
        raise ValueError('batch_size must be a positive integer')
    normalized = dict(params)
    normalized['hidden_sizes'] = hidden_sizes
    return normalized


def _predict_mlp(model, features, batch_size, device):
    """Generate ordered MLP predictions without retaining gradients."""
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
            batch_prediction = model(batch_features.to(device))
            predictions.append(batch_prediction.cpu().numpy())
    return np.concatenate(predictions).astype(float, copy=False)


def _fit_mlp_trial_run(
    trial,
    train_features,
    train_targets,
    validation_features,
    validation_targets,
    params,
    seed,
    device,
    max_epochs,
    pruning_interval,
    run_position,
    completed_validation_maes,
):
    """Fit one fold/seed run while reporting intermediate MAE for pruning."""
    _set_seed(seed)
    model = MLPRegressor(
        input_size=train_features.shape[1],
        hidden_sizes=params['hidden_sizes'],
        dropout=params['dropout'],
    ).to(device)
    inputs = torch.from_numpy(train_features)
    targets = torch.from_numpy(train_targets)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
    )

    final_epoch_loss = np.nan
    for epoch in range(max_epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            current_batch_size = len(batch_targets)
            loss_sum += loss.detach().item() * current_batch_size
            sample_count += current_batch_size
        final_epoch_loss = loss_sum / sample_count

        should_report = (
            (epoch + 1) % pruning_interval == 0
            or epoch == max_epochs - 1
        )
        if should_report:
            validation_prediction = _predict_mlp(
                model,
                validation_features,
                params['batch_size'],
                device,
            )
            current_mae = float(np.mean(np.abs(
                validation_targets - np.clip(validation_prediction, 0.0, None)
            )))
            running_mae = float(np.mean([
                *completed_validation_maes,
                current_mae,
            ]))
            global_step = run_position * max_epochs + epoch
            trial.report(running_mae, step=global_step)
            if trial.should_prune():
                trial.set_user_attr('pruned_at_step', global_step)
                trial.set_user_attr('partial_mean_mae', running_mae)
                raise optuna.TrialPruned(
                    f'Pruned at fold/seed run {run_position + 1}, epoch {epoch + 1}'
                )

    return model, float(final_epoch_loss)


def _aggregate_fold_metrics(run_metrics):
    """Average seed-level metrics within each completed trial and fold."""
    if run_metrics.empty:
        return pd.DataFrame()
    return (
        run_metrics.groupby(['Trial', 'Fold', 'Parameters'], as_index=False)
        .agg(
            Seeds=('Seed', 'nunique'),
            TrainRows=('TrainRows', 'first'),
            ValidationRows=('ValidationRows', 'first'),
            MAE=('MAE', 'mean'),
            MAE_seed_std=('MAE', 'std'),
            RMSE=('RMSE', 'mean'),
            R2=('R2', 'mean'),
            PoissonDeviance=('PoissonDeviance', 'mean'),
            FinalEpochL1Loss=('FinalEpochL1Loss', 'mean'),
            TrainingSeconds=('TrainingSeconds', 'sum'),
            InferenceSeconds=('InferenceSeconds', 'sum'),
            RunSeconds=('RunSeconds', 'sum'),
        )
        .fillna({'MAE_seed_std': 0.0})
    )


def _build_trials_summary(study):
    """Build a stable, compact table for completed, pruned, and failed trials."""
    rows = []
    for trial in study.trials:
        duration = (
            trial.duration.total_seconds()
            if trial.duration is not None
            else np.nan
        )
        rows.append({
            'Trial': trial.number,
            'State': trial.state.name,
            'Value': trial.value,
            'DurationSeconds': duration,
            'Parameters': json.dumps(trial.params, sort_keys=True),
            'ResolvedParameters': json.dumps(
                trial.user_attrs.get('resolved_params', {}),
                sort_keys=True,
            ),
            'CompletedRuns': trial.user_attrs.get('completed_runs', 0),
            'CV_MAE_mean': trial.user_attrs.get('CV_MAE_mean', np.nan),
            'CV_MAE_std': trial.user_attrs.get('CV_MAE_std', np.nan),
            'CV_RMSE_mean': trial.user_attrs.get('CV_RMSE_mean', np.nan),
            'CV_R2_mean': trial.user_attrs.get('CV_R2_mean', np.nan),
            'CV_PoissonDeviance_mean': trial.user_attrs.get(
                'CV_PoissonDeviance_mean',
                np.nan,
            ),
            'TrainingSeconds': trial.user_attrs.get('TrainingSeconds', np.nan),
            'InferenceSeconds': trial.user_attrs.get('InferenceSeconds', np.nan),
            'TotalRunSeconds': trial.user_attrs.get('TotalRunSeconds', np.nan),
            'PrunedAtStep': trial.user_attrs.get('pruned_at_step', np.nan),
        })
    return pd.DataFrame(rows).sort_values('Trial').reset_index(drop=True)


def optimize_mlp(
    features,
    target,
    splits,
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
    study_name=None,
    verbose=True,
):
    """Optimize the MLP by mean expanding-window validation MAE.

    A separate ``StandardScaler`` is fitted only on the training rows of each
    fold. Each trial is evaluated across every fold and seed. The default TPE
    sampler explores a conditional architecture, while the median pruner can
    stop trials based on intermediate validation MAE.

    Returns:
        dict: Optuna study, best parameters, trial/run/fold metrics, elapsed
        search time, and the selected PyTorch device.
    """
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError('n_trials must be a positive integer')
    if timeout is not None and timeout <= 0:
        raise ValueError('timeout must be positive when provided')
    if not isinstance(sampler_seed, int) or isinstance(sampler_seed, bool):
        raise TypeError('sampler_seed must be an integer')

    feature_values, target_values, normalized_splits, seeds = _validate_inputs(
        features,
        target,
        splits,
        seeds,
        max_epochs,
        pruning_interval,
    )
    resolved_device = _resolve_device(device)
    suggestion_function = (
        suggest_mlp_params if suggest_params is None else suggest_params
    )
    if not callable(suggestion_function):
        raise TypeError('suggest_params must be callable')

    sampler = sampler or optuna.samplers.TPESampler(
        seed=sampler_seed,
        n_startup_trials=5,
    )
    pruner = pruner or optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=pruning_interval,
        n_min_trials=2,
    )
    run_rows = []
    search_start = perf_counter()

    def objective(trial):
        trial_start = perf_counter()
        params = _validate_suggested_params(dict(suggestion_function(trial)))
        parameter_record = {
            **params,
            'hidden_sizes': list(params['hidden_sizes']),
            'max_epochs': max_epochs,
        }
        trial.set_user_attr('resolved_params', parameter_record)
        parameter_text = json.dumps(parameter_record, sort_keys=True)
        completed_validation_maes = []
        current_trial_rows = []
        run_position = 0

        try:
            for fold_number, (train_index, validation_index) in enumerate(
                normalized_splits,
                start=1,
            ):
                scaler = StandardScaler()
                train_features = scaler.fit_transform(
                    feature_values[train_index]
                ).astype(np.float32, copy=False)
                validation_features = scaler.transform(
                    feature_values[validation_index]
                ).astype(np.float32, copy=False)
                train_targets = target_values[train_index].astype(
                    np.float32,
                    copy=False,
                )
                validation_targets = target_values[validation_index].astype(
                    np.float32,
                    copy=False,
                )

                for seed in seeds:
                    training_start = perf_counter()
                    model, final_epoch_loss = _fit_mlp_trial_run(
                        trial,
                        train_features,
                        train_targets,
                        validation_features,
                        validation_targets,
                        params,
                        seed,
                        resolved_device,
                        max_epochs,
                        pruning_interval,
                        run_position,
                        completed_validation_maes,
                    )
                    training_seconds = perf_counter() - training_start

                    inference_start = perf_counter()
                    prediction = _predict_mlp(
                        model,
                        validation_features,
                        params['batch_size'],
                        resolved_device,
                    )
                    inference_seconds = perf_counter() - inference_start
                    metrics = calculate_regression_metrics(
                        validation_targets,
                        prediction,
                    )
                    completed_validation_maes.append(metrics['MAE'])
                    current_trial_rows.append({
                        'Trial': trial.number,
                        'Fold': fold_number,
                        'Seed': seed,
                        'Parameters': parameter_text,
                        'TrainRows': len(train_index),
                        'ValidationRows': len(validation_index),
                        'FinalEpochL1Loss': final_epoch_loss,
                        'TrainingSeconds': training_seconds,
                        'InferenceSeconds': inference_seconds,
                        'RunSeconds': training_seconds + inference_seconds,
                        **metrics,
                    })
                    run_position += 1
        except optuna.TrialPruned:
            run_rows.extend(current_trial_rows)
            trial.set_user_attr('completed_runs', len(current_trial_rows))
            trial.set_user_attr('TrialWallSeconds', perf_counter() - trial_start)
            raise

        run_rows.extend(current_trial_rows)
        trial_frame = pd.DataFrame(current_trial_rows)
        fold_frame = _aggregate_fold_metrics(trial_frame)
        cv_mae_mean = float(fold_frame['MAE'].mean())
        trial.set_user_attr('completed_runs', len(current_trial_rows))
        trial.set_user_attr('CV_MAE_mean', cv_mae_mean)
        cv_mae_std = fold_frame['MAE'].std(ddof=1)
        trial.set_user_attr(
            'CV_MAE_std',
            0.0 if pd.isna(cv_mae_std) else float(cv_mae_std),
        )
        trial.set_user_attr('CV_RMSE_mean', float(fold_frame['RMSE'].mean()))
        trial.set_user_attr('CV_R2_mean', float(fold_frame['R2'].mean()))
        trial.set_user_attr(
            'CV_PoissonDeviance_mean',
            float(fold_frame['PoissonDeviance'].mean()),
        )
        trial.set_user_attr(
            'TrainingSeconds',
            float(trial_frame['TrainingSeconds'].sum()),
        )
        trial.set_user_attr(
            'InferenceSeconds',
            float(trial_frame['InferenceSeconds'].sum()),
        )
        trial.set_user_attr(
            'TotalRunSeconds',
            float(trial_frame['RunSeconds'].sum()),
        )
        trial.set_user_attr('TrialWallSeconds', perf_counter() - trial_start)
        return cv_mae_mean

    previous_verbosity = optuna.logging.get_verbosity()
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    finally:
        optuna.logging.set_verbosity(previous_verbosity)

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError('Optuna finished without a completed MLP trial')

    search_seconds = perf_counter() - search_start
    run_metrics = pd.DataFrame(run_rows)
    fold_metrics = _aggregate_fold_metrics(run_metrics)
    trials_summary = _build_trials_summary(study)
    best_trial = study.best_trial
    best_run_metrics = (
        run_metrics.loc[run_metrics['Trial'].eq(best_trial.number)]
        .reset_index(drop=True)
    )
    best_fold_metrics = (
        fold_metrics.loc[fold_metrics['Trial'].eq(best_trial.number)]
        .reset_index(drop=True)
    )
    best_params = dict(best_trial.user_attrs['resolved_params'])
    best_params['hidden_sizes'] = tuple(best_params['hidden_sizes'])
    if verbose:
        print(
            f'[LOG] Optuna selected MLP trial {best_trial.number} '
            f'(CV MAE={best_trial.value:.3f}, time={search_seconds:.1f}s, '
            f'device={resolved_device})'
        )

    return {
        'study': study,
        'best_trial': best_trial.number,
        'best_params': best_params,
        'trials_summary': trials_summary,
        'run_metrics': run_metrics,
        'fold_metrics': fold_metrics,
        'best_run_metrics': best_run_metrics,
        'best_fold_metrics': best_fold_metrics,
        'search_seconds': search_seconds,
        'device': str(resolved_device),
    }


__all__ = [
    'DEFAULT_MAX_EPOCHS',
    'DEFAULT_MLP_SEEDS',
    'DEFAULT_OPTUNA_TRIALS',
    'DEFAULT_PRUNING_INTERVAL',
    'DEFAULT_SAMPLER_SEED',
    'optimize_mlp',
    'suggest_mlp_params',
]
