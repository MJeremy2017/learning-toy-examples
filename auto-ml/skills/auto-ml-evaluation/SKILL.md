---
name: auto-ml-evaluation
description: >-
  Evaluate a trained auto-ML model on the project's val/test splits, compute
  primary + secondary metrics, draw diagnostic plots, and produce a structured
  diagnosis (overfit/underfit/instability/leakage/etc.) that the orchestrator
  uses to decide the next iteration. Use only when invoked by the auto-ml
  orchestrator.
---

# Auto-ML Evaluation Subagent

Inputs you read:
- `inputs/requirements.md` (primary metric, target value)
- `features/v<latest>/{val,test}.parquet`, `feature_spec.json`
- `models/v<latest>/model.pkl` (or `.pt`), `model_card.md`
- (optional) prior `evals/v<K-1>/metrics.json` for trend comparison
- (if it exists) `models/v<latest>/training_log.csv` for NN curves

Outputs you write under `evals/v<K>/`:
- `metrics.json` (schema below)
- `report.md`
- `plots/*.png`
- `decision.json` — your suggested next action + matched routing rule

Return JSON (orchestrator §4 contract): `status`, `artifacts`, `summary`, `metrics` (mirror of metrics.json), `diagnosis` (one of the 9 values), `recommendations`.

## Workflow

```
- [ ] 1. Load model + splits.
- [ ] 2. Compute primary + secondary metrics per task type (§2).
- [ ] 3. Compute gap, stability, calibration (§3).
- [ ] 4. Make the diagnostic plots (§4).
- [ ] 5. Choose a diagnosis using the rule table (§5).
- [ ] 6. Suggest a next action with the matched routing rule.
- [ ] 7. Write all artifacts and return JSON.
```

## 1. Authority

You are the **only** entity allowed to set `diagnosis`. The orchestrator trusts your `decision.json.matched_rule` but reserves the right to override it (e.g., loop guards). Be conservative and explicit.

## 2. Metrics by task type

### Classification

- Primary: whatever `requirements.md` says (e.g. `roc_auc`).
- Always also compute: `f1@0.5`, `pr_auc` (average precision), `logloss`, `brier`, `accuracy`, per-class precision/recall/F1, and for binary: `minority_recall@0.5`.
- If `class_imbalance.present == true` from EDA → also pick the best threshold by maximizing F1 on val, record `best_threshold` and `f1@best_threshold`.

### Regression

- Primary: `rmse`, `mae`, `r2`, or what requirements says.
- Always also compute: `mae`, `rmse`, `r2`, `mape` (guarded for zeros), `median_absolute_error`.

### Forecast

- Primary: `mae` or `smape` per requirements.
- Compute per-cutoff metrics (same as `prophet-forecasting`) when applicable.
- Always also: residual autocorrelation check (ACF up to lag 24 / appropriate).

## 3. Gap, stability, calibration

- `train_metric` — recompute on the train split (re-loading the model and scoring on train, same metric).
- `val_metric` — primary on val.
- `gap = train_metric - val_metric` (use signed difference, with sign matching the metric direction).
- `stability.fold_std`, `stability.fold_std_over_mean` — only if CV was used.
- `calibration.brier` (classification only) + `is_well_calibrated = brier < 0.1 * naive_brier`.

## 4. Required plots

Under `evals/v<K>/plots/`:

| Plot | Task | Filename |
|---|---|---|
| ROC curve | classification | `roc.png` |
| PR curve | classification | `pr.png` |
| Confusion matrix at chosen threshold | classification | `confusion.png` |
| Calibration curve | classification | `calibration.png` |
| Lift / gain chart | classification | `lift.png` |
| Residuals vs prediction (scatter) | regression | `residuals.png` |
| Predicted vs actual | regression | `pred_vs_actual.png` |
| Error histogram | regression | `error_hist.png` |
| Forecast vs actual on val window | forecast | `forecast_vs_actual.png` |
| Training curves (if NN) | any | `training_curves.png` |
| Per-fold metric bar chart | any with CV | `cv_metric.png` |

## 5. Diagnosis rules

Pick the **first** rule that matches, in this order:

```
1. val_metric meets target_value                                       → meets_target
2. EDA flagged suspected_leak AND that column is in the model AND
   removing it drops val_metric by >5%                                  → leakage_suspected
3. EDA warnings about data quality reappeared (missing handling failed,
   class distribution very off, prediction degenerate)                  → data_quality
4. train_metric > 0.9 * theoretical_ceiling AND gap > 0.05              → overfit
5. fold_std_over_mean > 0.15  OR  NN training_log shows loss bouncing  → instability
6. val_metric ≤ baseline (LR or naive) AND train_metric also low        → low_signal (also underfit; pick low_signal if FE.v1 only)
7. train_metric and val_metric both flat AND below target               → underfit
8. minority_recall < 0.3 AND class_imbalance.present                    → class_imbalance
9. calibration.is_well_calibrated AND val_metric < target               → well_calibrated_but_below_target
```

Notes:
- `theoretical_ceiling` defaults to 1.0 for AUC/accuracy. For low-signal domains (finance, healthcare noise) the ceiling may be lower — use the user's `target_value * 1.1` as a soft ceiling if no better estimate exists.
- For rule 2, prove the leakage claim by re-scoring after dropping the column on val; only fire if val drops materially. Otherwise demote to rule 3.

## 6. `metrics.json` schema

```json
{
  "primary": {"name": "roc_auc", "val": 0.873, "test": null, "folds": [0.86, 0.88, 0.87, 0.88, 0.88]},
  "secondary": {
    "f1@0.5": 0.41, "pr_auc": 0.55, "logloss": 0.12, "brier": 0.04,
    "minority_recall@0.5": 0.62, "best_threshold": 0.34, "f1@best_threshold": 0.46
  },
  "train_metric": 0.96,
  "val_metric":   0.873,
  "test_metric":  null,
  "gap":          0.087,
  "stability":    {"fold_std": 0.008, "fold_std_over_mean": 0.0092},
  "calibration":  {"brier": 0.04, "is_well_calibrated": true},
  "baseline_comparison": {"majority_class": 0.5, "logistic_regression": 0.81}
}
```

## 7. `decision.json` schema

```json
{
  "diagnosis": "overfit",
  "matched_rule": 4,                  // index into §5
  "next_action_suggestion": "modeling",
  "reasoning": "Train ROC AUC 0.96 vs val 0.87; gap 0.087 > 0.05 and train is near ceiling.",
  "what_to_try_next": [
    "Regularize: drop num_leaves to 31, raise min_child_samples to 50",
    "Add early stopping with rounds=50 on val"
  ]
}
```

The orchestrator owns the final routing decision; `next_action_suggestion` is advisory.

## 8. `report.md`

Sections:
1. **Headline** — primary metric val + test, vs target, vs baseline.
2. **Per-metric table** — all secondaries.
3. **Stability** — fold scores + std.
4. **Diagnosis** — which rule fired and why, with the exact numbers.
5. **What to try next** — bullets that map to `what_to_try_next`.
6. **Plots** — inline references.

## Anti-patterns

- Don't compute test metric every iteration. Reserve `test_metric` until the orchestrator stops or asks for a final read. Set `"test": null` otherwise.
- Don't claim leakage without the drop-and-rescore confirmation.
- Don't claim `meets_target` based on a single fold — require either `val_metric` ≥ target AND `fold_std_over_mean < 0.10`, or test metric ≥ target.
- Don't pick a new threshold for the primary metric if the primary is threshold-free (AUC). Threshold tuning belongs to secondaries.
- Don't write your own model. You only evaluate the one you were given.
