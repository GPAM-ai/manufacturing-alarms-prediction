"""Expanding-window Grid Search for the PyTorch LSTM regressor."""

import json
import random
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.lstm import LSTMRegressor
from src.modeling.modeling import calculate_regression_metrics


DEFAULT_LSTM_SEEDS = (17, 42, 73)

DEFAULT_LSTM_PARAMS = {
    'window_size': 7,
    'hidden_size': 32,
    'num_layers': 1,
    'dense_size': 32,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'weight_decay': 0.0,
    'batch_size': 32,
    'epochs': 100,
}

# Eight deliberately compact combinations. Training parameters not listed here
# retain the fixed values above so that architecture is the primary comparison.
DEFAULT_LSTM_PARAM_GRID = {
    'window_size': (7, 14),
    'hidden_size': (32, 64),
    'dropout': (0.0, 0.2),
}


def build_lstm_sequences(features, target, target_indices, window_size):
    """Build temporal windows ending at the requested target-row indices.

    The feature row at each target index is included as the final time step.
    Earlier validation feature rows may therefore become historical context for
    later validation predictions, but target values are never part of an input
    sequence.

    Returns:
        tuple: ``(sequences, targets, retained_target_indices)`` as NumPy arrays.
    """
    feature_values = np.asarray(features, dtype=np.float32)
    target_values = np.asarray(target, dtype=np.float32)
    indices = np.asarray(target_indices)

    if feature_values.ndim != 2:
        raise ValueError('features must be a two-dimensional array')
    if target_values.ndim != 1:
        raise ValueError('target must be a one-dimensional array')
    if len(feature_values) != len(target_values):
        raise ValueError('features and target must contain the same number of rows')
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
        raise ValueError('window_size must be a positive integer')
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise TypeError('target_indices must be a one-dimensional integer array')
    if indices.size == 0:
        raise ValueError('target_indices cannot be empty')
    if np.any(indices < 0) or np.any(indices >= len(feature_values)):
        raise IndexError('target_indices contain positions outside the dataset')
    if np.any(np.diff(indices) <= 0):
        raise ValueError('target_indices must be strictly increasing')

    retained_indices = indices[indices >= window_size - 1].astype(int, copy=False)
    if retained_indices.size == 0:
        raise ValueError(
            f'No target row has the {window_size} observations required by the window'
        )

    sequences = np.stack([
        feature_values[index - window_size + 1:index + 1]
        for index in retained_indices
    ])
    sequence_targets = target_values[retained_indices]
    return sequences, sequence_targets, retained_indices


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
                raise TypeError(f'Fold {fold} {name} must be a one-dimensional integer array')
            if index.size == 0:
                raise ValueError(f'Fold {fold} {name} cannot be empty')
            if np.any(index < 0) or np.any(index >= row_count):
                raise IndexError(f'Fold {fold} {name} contains positions outside the dataset')
            if np.any(np.diff(index) <= 0):
                raise ValueError(f'Fold {fold} {name} must be strictly increasing')

        expected_train = np.arange(train_index[-1] + 1)
        expected_validation = np.arange(validation_index[0], validation_index[-1] + 1)
        if not np.array_equal(train_index, expected_train):
            raise ValueError(f'Fold {fold} training indices must form an expanding prefix')
        if not np.array_equal(validation_index, expected_validation):
            raise ValueError(f'Fold {fold} validation indices must be contiguous')
        if validation_index[0] != train_index[-1] + 1:
            raise ValueError(
                f'Fold {fold} validation must begin immediately after its training prefix'
            )
        normalized.append((train_index.astype(int), validation_index.astype(int)))
    return normalized


def _resolve_device(device):
    """Resolve an explicit device or select CUDA when it is available."""
    resolved = torch.device(
        device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
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


def _validate_params(params):
    """Validate training parameters not already checked by LSTMRegressor."""
    for name in ('window_size', 'batch_size', 'epochs'):
        value = params[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if params['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    if params['weight_decay'] < 0:
        raise ValueError('weight_decay cannot be negative')


def _fit_lstm(train_sequences, train_targets, params, seed, device):
    """Fit one LSTM configuration and return its final-epoch mean L1 loss."""
    _set_seed(seed)
    model = LSTMRegressor(
        input_size=train_sequences.shape[2],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dense_size=params['dense_size'],
        dropout=params['dropout'],
    ).to(device)

    inputs = torch.from_numpy(train_sequences)
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

    final_loss = np.nan
    for _ in range(params['epochs']):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            batch_size = len(batch_targets)
            loss_sum += loss.detach().item() * batch_size
            sample_count += batch_size
        final_loss = loss_sum / sample_count

    return model, float(final_loss)


def _predict_lstm(model, sequences, batch_size, device):
    """Generate ordered predictions without retaining gradients."""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(sequences)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for (batch_inputs,) in loader:
            batch_prediction = model(batch_inputs.to(device))
            predictions.append(batch_prediction.cpu().numpy())
    return np.concatenate(predictions).astype(float, copy=False)


def grid_search_lstm(
    features,
    target,
    splits,
    param_grid=None,
    seeds=DEFAULT_LSTM_SEEDS,
    device=None,
    verbose=True,
):
    """Select an LSTM configuration by mean expanding-window validation MAE.

    ``StandardScaler`` is fitted independently on each fold's training rows.
    Every candidate is repeated for all requested seeds. Seed-level metrics are
    first averaged within each fold, and candidate ranking then uses mean MAE
    across folds.

    Args:
        features: Numeric feature table ordered chronologically.
        target: Non-negative daily recurrence counts in the same row order.
        splits: Expanding ``(train_indices, validation_indices)`` pairs.
        param_grid: Mapping or list of mappings accepted by ``ParameterGrid``.
        seeds: Random seeds used to repeat each candidate and fold.
        device: PyTorch device name; defaults to CUDA when available, else CPU.
        verbose: Suppress logs when True; print progress when False.

    Returns:
        dict: Best parameters, seed-level and fold-level metrics, the ranked
        search summary, elapsed search time, and the device used.
    """
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

    normalized_splits = _validate_expanding_splits(splits, len(feature_values))
    grid_definition = DEFAULT_LSTM_PARAM_GRID if param_grid is None else param_grid
    candidates = list(ParameterGrid(grid_definition))
    if not candidates:
        raise ValueError('param_grid must produce at least one candidate')

    allowed_params = set(DEFAULT_LSTM_PARAMS)
    unknown_params = set().union(*(candidate.keys() for candidate in candidates)) - allowed_params
    if unknown_params:
        raise KeyError(f'Unknown LSTM parameter(s): {sorted(unknown_params)}')

    resolved_device = _resolve_device(device)
    run_rows = []
    candidate_wall_seconds = {}
    search_start = perf_counter()

    for candidate_number, candidate in enumerate(candidates, start=1):
        candidate_start = perf_counter()
        params = {**DEFAULT_LSTM_PARAMS, **candidate}
        _validate_params(params)
        parameter_text = json.dumps(params, sort_keys=True)
        if not verbose:
            print(
                f'[LOG] LSTM candidate {candidate_number}/{len(candidates)} '
                f'across {len(normalized_splits)} fold(s) and {len(seeds)} seed(s)'
            )

        for fold_number, (train_index, validation_index) in enumerate(
            normalized_splits,
            start=1,
        ):
            scaler = StandardScaler()
            scaler.fit(feature_values[train_index])
            context_end = validation_index[-1] + 1
            scaled_features = scaler.transform(feature_values[:context_end]).astype(
                np.float32,
                copy=False,
            )
            train_sequences, train_targets, _ = build_lstm_sequences(
                scaled_features,
                target_values[:context_end],
                train_index,
                params['window_size'],
            )
            validation_sequences, validation_targets, retained_validation_index = (
                build_lstm_sequences(
                    scaled_features,
                    target_values[:context_end],
                    validation_index,
                    params['window_size'],
                )
            )

            for seed in seeds:
                training_start = perf_counter()
                model, final_epoch_loss = _fit_lstm(
                    train_sequences,
                    train_targets,
                    params,
                    seed,
                    resolved_device,
                )
                training_seconds = perf_counter() - training_start

                inference_start = perf_counter()
                prediction = _predict_lstm(
                    model,
                    validation_sequences,
                    params['batch_size'],
                    resolved_device,
                )
                inference_seconds = perf_counter() - inference_start
                metrics = calculate_regression_metrics(validation_targets, prediction)
                run_rows.append({
                    'Candidate': candidate_number,
                    'Fold': fold_number,
                    'Seed': seed,
                    'Parameters': parameter_text,
                    'TrainSequences': len(train_sequences),
                    'ValidationSequences': len(validation_sequences),
                    'ValidationFirstIndex': int(retained_validation_index[0]),
                    'ValidationLastIndex': int(retained_validation_index[-1]),
                    'FinalEpochL1Loss': final_epoch_loss,
                    'TrainingSeconds': training_seconds,
                    'InferenceSeconds': inference_seconds,
                    'RunSeconds': training_seconds + inference_seconds,
                    **metrics,
                })
        candidate_wall_seconds[candidate_number] = perf_counter() - candidate_start

    run_metrics = pd.DataFrame(run_rows)
    fold_metrics = (
        run_metrics.groupby(['Candidate', 'Fold', 'Parameters'], as_index=False)
        .agg(
            Seeds=('Seed', 'nunique'),
            TrainSequences=('TrainSequences', 'first'),
            ValidationSequences=('ValidationSequences', 'first'),
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
            Seeds=('Seeds', 'first'),
            TrainingSeconds=('TrainingSeconds', 'sum'),
            InferenceSeconds=('InferenceSeconds', 'sum'),
            TotalRunSeconds=('RunSeconds', 'sum'),
        )
        .fillna({'CV_MAE_std': 0.0, 'CV_RMSE_std': 0.0})
    )
    search_summary['CandidateWallSeconds'] = search_summary['Candidate'].map(
        candidate_wall_seconds
    )
    search_summary = search_summary.sort_values(
        ['CV_MAE_mean', 'CandidateWallSeconds', 'Candidate']
    ).reset_index(drop=True)

    best_candidate = int(search_summary.loc[0, 'Candidate'])
    best_params = json.loads(search_summary.loc[0, 'Parameters'])
    best_fold_metrics = (
        fold_metrics.loc[fold_metrics['Candidate'].eq(best_candidate)]
        .reset_index(drop=True)
    )
    search_seconds = perf_counter() - search_start
    if not verbose:
        print(
            f'[LOG] LSTM selected candidate {best_candidate} '
            f"(CV MAE={search_summary.loc[0, 'CV_MAE_mean']:.3f}, "
            f'time={search_seconds:.1f}s, device={resolved_device})'
        )

    return {
        'best_params': best_params,
        'best_candidate': best_candidate,
        'search_summary': search_summary,
        'run_metrics': run_metrics,
        'fold_metrics': fold_metrics,
        'best_fold_metrics': best_fold_metrics,
        'search_seconds': search_seconds,
        'device': str(resolved_device),
    }


__all__ = [
    'DEFAULT_LSTM_PARAM_GRID',
    'DEFAULT_LSTM_PARAMS',
    'DEFAULT_LSTM_SEEDS',
    'build_lstm_sequences',
    'grid_search_lstm',
]
