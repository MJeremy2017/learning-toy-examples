---
name: auto-ml-modeling
description: >-
  Train an ML model for an auto-ML project using the prepared features and
  splits. Start simple (logistic regression / ridge / naive baseline) and
  escalate the model family only when prior evaluation justifies it. Use only
  when invoked by the auto-ml orchestrator. Delegates NN tuning to
  neural-net-tuning, and time-series forecasting to prophet-forecasting.
---

# Auto-ML Modeling Subagent

Inputs you read:
- `inputs/requirements.md` (task type, primary metric, interpretability constraint)
- `features/v<latest>/pipeline.py`, `feature_spec.json`, `train.parquet`, `val.parquet`, (`test.parquet`)
- (optional) `evals/v<K-1>/metrics.json` — for tuning direction from the previous iteration

Outputs you write under `models/v<K>/`:
- `train.py` — runnable script that re-trains the model from scratch
- `model.pkl` — fitted estimator (joblib) OR `model.pt` for NN
- `model_card.md` — algorithm, hyperparams, training time, splits
- `feature_importance.png` and `feature_importance.csv`

Return JSON: standard contract, `summary` should name the algorithm and headline val metric.

## Workflow

```
- [ ] 1. Pick the model family per §2 (ladder) and prior eval (§3).
- [ ] 2. Build the full Pipeline: features pipeline + estimator.
- [ ] 3. Fit on train; for CV, use the splitter recorded in feature_spec.json.split.
- [ ] 4. Tune hyperparameters per §4 (small, principled — no random search > 50 trials in v1).
- [ ] 5. Persist model + model_card + feature importance.
```

## 1. Read the requirement

From `requirements.md`:
- `task.type` ∈ {classification, regression, forecast}
- `interpretability` ∈ {low, medium, high} — caps the family ladder
  - `high` → stop at family **3** (no NN). Prefer linear or single tree.
  - `medium` → up to family 3 (gradient boosting) is fine.
  - `low` → NN allowed.

## 2. Model family ladder

The orchestrator dispatches you with a current "rung". On the first modeling iteration, start at rung 1 unless the dataset is clearly huge (>1M rows AND >100 features), in which case start at rung 3.

### Classification

| Rung | Algorithm | Hyperparam defaults |
|---|---|---|
| 1 | `LogisticRegression(class_weight="balanced" if imbalanced, max_iter=2000, C=1.0, solver="lbfgs")` | tune `C ∈ {0.01, 0.1, 1, 10}` |
| 2 | `RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, class_weight="balanced" if imbalanced)` | tune `max_depth ∈ {None, 6, 12, 24}` |
| 3 | `LGBMClassifier` or `XGBClassifier` | tune learning_rate, num_leaves, min_child_samples, subsample, colsample (§4) |
| 4 | MLP via PyTorch — invoke the `neural-net-tuning` skill | per that skill |

### Regression

| Rung | Algorithm | Defaults |
|---|---|---|
| 1 | `Ridge(alpha=1.0)` | tune `alpha ∈ {0.01, 0.1, 1, 10, 100}` |
| 2 | `RandomForestRegressor(n_estimators=400, n_jobs=-1)` | tune `max_depth` |
| 3 | `LGBMRegressor` / `XGBRegressor` | §4 |
| 4 | MLP via PyTorch + `neural-net-tuning` | — |

### Forecast (univariate, date-indexed)

| Rung | Algorithm | Notes |
|---|---|---|
| 1 | Naive seasonal baseline (last value or seasonal-naive) | always run first |
| 2 | Prophet — invoke `prophet-forecasting` skill | follow that skill |
| 3 | LightGBM on lag/rolling features (treated as regression) | use rung 3 of regression |

## 3. Reading prior eval to choose tuning direction

If `evals/v<K-1>/metrics.json` exists and `decision.json.matched_rule` says:

| Prior matched_rule | Action this iteration |
|---|---|
| 4 — overfit, regularize | stay on same family but increase regularization (LR: ↓C; tree: ↑min_samples_leaf; GBM: ↓num_leaves, ↑min_child_samples, ↓learning_rate + ↑n_estimators with early stopping) |
| 5 — overfit, FE handled it | re-fit same hyperparams on new features |
| 6 — underfit, escalate family | bump to next rung |
| 7 — underfit, FE handled it | stay on same family |
| 8 — instability | ↓ learning_rate, add early stopping, ensure scaling/clipping (NN: grad clip, smaller batch) |
| 9 — class_imbalance | turn on `class_weight="balanced"` if not already; if at rung 3, also tune `scale_pos_weight` |
| 10 — calibrated but below target | leave family, look for new features instead — but still re-train with current features for fairness |

## 4. Hyperparameter tuning policy (v1)

- **No grid search bigger than 50 trials.**
- Prefer `GridSearchCV` over a small explicit grid (≤16 combos) for rungs 1–2.
- For rung 3 (GBM/XGB), use `optuna` (or `sklearn.model_selection.RandomizedSearchCV`) with **20 trials**, search space:
  ```
  learning_rate:      log-uniform [0.01, 0.3]
  num_leaves:         int [15, 127]                  # lightgbm
  max_depth:          int [3, 12]                    # xgboost
  min_child_samples:  int [5, 100]                   # lightgbm
  subsample:          uniform [0.6, 1.0]
  colsample_bytree:   uniform [0.6, 1.0]
  reg_lambda:         log-uniform [1e-3, 10]
  n_estimators:       fixed 5000 with early_stopping_rounds=50
  ```
- Use the CV splitter from `feature_spec.json.split.cv`. For time-based splits, **never shuffle** — use `TimeSeriesSplit`.
- Final model: re-fit on `train + val` with the chosen hyperparams; report `val` metric from the CV run (not the re-fit), and `test` metric on the held-out test set.

## 5. NN escalation (rung 4)

When you escalate to NN:
- Read the `neural-net-tuning` skill in full before writing any code.
- Use PyTorch, fixed `random_state=0` everywhere.
- Normalize inputs **with train-split stats only**.
- Apply gradient clipping (`max_norm=1.0`), early stopping with restore-best-weights.
- Log per-epoch train/val metric to `models/v<K>/training_log.csv` so the eval phase can read the curves.

## 6. `train.py` requirements

- Single‑file, runnable: `python train.py` reproduces `model.pkl` and `model_card.md`.
- Config block at top: paths, hyperparams.
- Use `joblib.dump(pipe, "model.pkl")` (or torch `.pt`).
- Print final val + test metric to stdout.

## 7. `model_card.md`

Sections:
- **Algorithm** — family + version + key hyperparameters.
- **Training data** — n_train, n_val, n_test, split strategy.
- **Hyperparameter search** — what was tried, what was chosen.
- **Training time** — wall clock.
- **Caveats** — known limitations, dependencies on EDA recommendations.

## 8. Feature importance

- Tree models: `model.feature_importances_`, sorted, top 30 plotted as a bar chart.
- Linear: absolute value of standardized coefficients.
- NN: skip or use permutation importance on val (only if cheap).

Always emit both `feature_importance.csv` (full list) and `feature_importance.png` (top 30).

## Anti-patterns

- Don't start at rung 3 "to save time" on a small dataset — you lose the baseline signal.
- Don't tune more than one rung above where the orchestrator dispatched you to.
- Don't refit the feature pipeline on `train+val` for hyperparameter tuning — use the same fit-on-train rule.
- Don't change the split — that's frozen in `feature_spec.json`.
- Don't write your own evaluation here. Compute metrics for `model_card.md` only; the eval phase is the source of truth.
