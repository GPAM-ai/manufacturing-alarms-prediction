# Modeling results

Artifacts are grouped by their role in the modeling workflow:

- `models/`: serialized fitted estimators.
- `metadata/`: model, dataset, feature, and configuration descriptions.
- `metrics/`: standalone evaluation metrics.
- `predictions/`: prediction tables and observed-versus-predicted figures.
- `residuals/`: residual diagnostic figures.
- `feature_importance/`: feature-importance tables and figures [REDACTED].
- `validation/`: expanding-window searches, fold metrics, and holdout definitions.
- `comparisons/`: consolidated comparisons across models and baselines.

Neural experiments use portable `.pt` checkpoints containing every seeded
replica together with the fitted scaler and feature order. Their validation
tables distinguish Optuna trials or Grid Search candidates, fold/seed runs,
fold aggregates, and the folds of the selected configuration. The notebook
`neural_modeling_pipeline.ipynb` also writes the experimental protocol and an
artifact manifest so results can be audited after separate notebook sessions.
