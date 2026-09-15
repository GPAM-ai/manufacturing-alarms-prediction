# Industrial Alarm Forecasting Framework

This repository contains a research framework for forecasting rapidly recurring industrial
alarms. The work investigates whether historical alarm behavior can support one-day-ahead
predictions of periods with increased recurrence, which may help industrial teams identify
conditions that deserve attention.

The research does not assume that temporally close alarms form a verified physical cascade.
Instead, rapid same-source recurrence is used as an observable proxy for a cascade regime.
The framework is designed to make this definition, the temporal assumptions, the model
comparison, and the resulting limitations explicit and auditable.

The industrial dataset is confidential and is not distributed with this repository. This
README intentionally describes the workflow without documenting company-specific data,
locations, operational identifiers, or internal process details.

## Framework objective

The framework turns event-level alarm history into a reproducible forecasting experiment.
Its main objectives are to:

- inspect data quality and temporal discontinuities before modeling;
- define a measurable daily recurrence target;
- transform temporal patterns and interruptions into leakage-safe predictors;
- compare tabular and sequential models under the same information constraints;
- evaluate future generalization through chronological validation;
- preserve the models, predictions, metrics, and experimental metadata required for review.

The framework is intended for research and decision-support development. It does not infer
fault causality, alarm severity, or maintenance actions, and its outputs should not be used
as autonomous industrial decisions without external and prospective validation.

## Workflow

The project follows this sequence:

```text
Event records
    -> preparation and daily time-series construction
    -> exploratory temporal analysis
    -> recurrence target and feature engineering
    -> chronological development/holdout partition
    -> expanding-window model selection
    -> final training and evaluation
    -> saved evidence and research reporting
```

Two aligned feature representations are used. The **simple** representation focuses on
calendar, interruption context, and recent global history. The **complete** representation
adds further operational and external context. Their dates, targets, and evaluation windows
remain aligned, so differences can be attributed to information scope and model behavior
rather than different samples.

## Business logic in `src/`

`src/` contains the reusable business and experimental logic. The notebooks call these
modules to assemble complete pipelines.

### `src/data_prep/`

Responsible for loading and standardizing event records, cleaning invalid or duplicated
entries, ordering observations, constructing daily time series, and producing descriptive
versions used to study outliers. It establishes the consistent temporal base used by later
stages.

### `src/eda/`

Implements the exploratory analyses that guide modeling decisions. The modules examine
inter-alarm intervals, rapid recurrence, temporal gaps, calendar behavior, seasonality,
spectral components, decomposition, autocorrelation, categorical concentration, and
candidate external associations. Plotting functions save the corresponding evidence under
`results/`.

### `src/feat_eng/`

Converts the prepared history into the forecasting problem. It builds the daily recurrence
proxy, marks whether a target is available, creates calendar and interruption-context
features, adds lagged histories, and constructs the aligned simple and complete datasets.
It also prepares model-ready rows and expanding-window splits while preserving temporal
causality.

### `src/modeling/`

Provides the common evaluation logic and the model-specific implementations:

- persistence, Random Forest, and Extreme Gradient Boosting for baseline and tabular
  experiments;
- multilayer perceptron training and Optuna-based hyperparameter optimization;
- long short-term memory training and bounded Grid Search;
- shared feature-set preparation, chronological partitions, metrics, plots, checkpoint
  generation, and artifact persistence.

Predictions are evaluated through the same temporal protocol. Mean absolute error is the
primary selection criterion, while complementary metrics, residual behavior, peak errors,
and neural computational cost support interpretation.

## Executable pipelines

The notebooks at the repository root are the main workflow entry points and should be run
from the project root in the following order:

1. `data_analysis_pipeline.ipynb` prepares the records and performs the exploratory study.
2. `build_feats_pipeline.ipynb` creates the target, feature representations, model-ready
   datasets, and temporal validation windows.
3. `modeling_pipeline.ipynb` runs the persistence, Random Forest, and XGBoost experiments.
4. `neural_modeling_pipeline.ipynb` runs MLP with Optuna and LSTM with Grid Search, then
   updates the consolidated comparison.

The notebooks expose the sequence of decisions and provide an inspectable research record;
the implementation itself remains in `src/` so that individual stages can be reused or
tested independently.

## Environment

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Input paths must be configured for an authorized local copy of the industrial data. Because
the source data are confidential, a fresh clone cannot reproduce the numerical results
without equivalent authorized inputs. The saved artifacts still document how the reported
experiments were configured and evaluated.

## Results and documentation

Generated evidence is organized by purpose:

- `results/eda/`, `results/time_series/`, and `results/seasonality/` contain exploratory
  tables and figures;
- `results/modeling/validation/` records search candidates, temporal folds, seeded runs, and
  holdout definitions;
- `results/modeling/metrics/`, `predictions/`, and `residuals/` contain final evaluation
  evidence;
- `results/modeling/feature_importance/` documents the use of engineered attributes by tree
  models;
- `results/modeling/models/` and `metadata/` preserve fitted estimators, neural checkpoints,
  feature order, parameters, and protocol information;
- `results/modeling/comparisons/` consolidates model rankings and artifact manifests.

`results/modeling/README.md` describes the modeling artifacts in more detail.
`docs/experimental_results/` records complete experimental executions, while `docs/paper/`
contains the manuscript sources used to report the research, methodology, findings,
limitations, and proposed framework.

## Research perspective

The main contribution is not a new learning algorithm. It is an end-to-end approach for
learning from imperfect industrial alarm histories while preserving temporal validity. By
combining investigative exploratory analysis, robust feature engineering, aligned model
comparisons, and traceable artifacts, the framework shows both the predictive information
available in the records and the limits that must be understood before deployment.
