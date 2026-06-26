---
name: auto-ml-feature-engineering
description: >-
  Build the feature pipeline (cleaning, imputing, encoding, scaling, derived
  features, interactions) for an auto-ML project, materialize train/val/test
  splits, and emit a runnable build_pipeline() in pipeline.py. Use only when
  invoked by the auto-ml orchestrator. Decisions are driven by the EDA
  recommendations.json from the same project.
---

# Auto-ML Feature Engineering Subagent

Inputs you read:
- `inputs/data.<ext>`, `inputs/data_description.md`, `inputs/requirements.md`
- `eda/v<latest>/stats.json` and `eda/v<latest>/recommendations.json` (**authoritative**)
- (optional) `features/v<K-1>/feature_spec.json` — if doing a refinement iteration

Outputs you write under `features/v<K>/`:
- `pipeline.py` — exports `build_pipeline() -> sklearn.pipeline.Pipeline`
- `feature_spec.md` — human-readable feature list + rationale
- `feature_spec.json` — machine-readable manifest (schema below)
- `train.parquet`, `val.parquet`, `test.parquet` — materialized post-FE splits

Return JSON: standard contract, with `recommendations` describing what you'd try next iteration if the modeling phase doesn't reach target.

## Workflow

```
- [ ] 1. Decide split strategy from recommendations.splitting_hint.
- [ ] 2. Apply the recommended drops + leakage exclusions.
- [ ] 3. Build the sklearn ColumnTransformer per the menu in §3.
- [ ] 4. Fit transformers on TRAIN ONLY, apply to val/test (no leakage).
- [ ] 5. Add derived features (interactions, time features, ratios) — light touch on v1.
- [ ] 6. Write pipeline.py, materialize splits, write specs.
```

## 1. Splitting strategy

| Hint from EDA | Implementation |
|---|---|
| `time-based on <col>` | sort by `<col>`; split chronologically by share unless dates are given. Default: 70% train, 15% val, 15% test (last). |
| `stratified-kfold` | classification with no time signal. Hold out 15% test stratified, then `feature_spec.json.split.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`. |
| `kfold` | regression with no time signal. Same but `KFold`. |

Write the split decision into `feature_spec.json.split` so modeling + eval use the exact same splits.

## 2. What to drop / exclude

- All columns in `recommendations.drop`.
- All columns in `recommendations.investigate_leakage` (the orchestrator has already cleared or removed any genuine leaks before this phase is called).
- IDs / timestamps not used as features (a timestamp may still be used as the split key — see §1).

## 3. Transform menu (driven by recommendations.json)

For each surviving column, apply the transform recommended by EDA. If EDA didn't recommend anything, use the defaults below.

| Column type | Default | When to override |
|---|---|---|
| Numeric, well-behaved | `StandardScaler` only for linear/NN models — wrap in `ColumnTransformer` that the modeling phase can swap | tree models don't need scaling |
| Numeric, heavy-tailed (`log_transform` in EDA) | `log1p` then `StandardScaler` | — |
| Numeric, outlier-heavy (`clip_outliers`) | clip to `[p1, p99]` from train then scale | — |
| Numeric, with missing | recommended `handle_missing.strategy` (default `median` + missing indicator) | — |
| Categorical, low-card (≤20) | `OneHotEncoder(handle_unknown="ignore")` | always |
| Categorical, high-card | `TargetEncoder` (with smoothing=10) for classification/regression; for time-series use **out-of-fold** target encoding | never use plain target encoding on test |
| Boolean | cast to int8 | — |
| Datetime | extract: year, month, day, dayofweek, hour, is_weekend, days_since_epoch | only if not used as split key |
| Text (free-text) | TF-IDF top 5000 unigrams + bigrams, or skip if no clear signal in EDA | only if EDA found text columns |

Class imbalance handling lives in `feature_spec.json.class_balance`:
- `none` (default), or
- `class_weight="balanced"` (preferred — pass through to model), or
- `oversample_minority` via `imbalanced-learn` only if minority class < 1% AND model is linear.

## 4. Derived features (light touch in v1)

Add only when EDA recommends it or when they're cheap and almost always helpful:
- Datetime: `is_weekend`, `month_sin/cos`, `hour_sin/cos`.
- Numeric ratios: `amount / median_by_<id>` if EDA recommends interactions involving a numeric and a group key.
- Aggregations: per-entity rolling stats only if the dataset is explicitly time-aware AND splitting is time-based.
- Pairwise interactions: only the pairs listed in `recommendations.candidate_interactions`, polynomial degree 2.

Cap the total feature count at **min(500, 5 × original feature count)** in v1. The modeling phase can prune further.

## 5. `pipeline.py` contract

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def build_pipeline(model_family: str = "tree") -> Pipeline:
    """
    Return an unfitted Pipeline that ends with a `Passthrough` step (or wraps
    only the preprocessing). The modeling subagent appends its estimator.
    model_family ∈ {"linear", "tree", "nn"} controls whether scaling is applied
    to numerics.
    """
    ...
```

The file MUST be importable as `auto_ml.projects.<project>.features.v<K>.pipeline` (or via direct file path). It must:
- be deterministic (`random_state=0` everywhere)
- not reference the data path — it operates on a passed DataFrame
- be fittable on train and reusable on val/test

## 6. `feature_spec.json` schema

```json
{
  "version": <K>,
  "split": {
    "strategy": "time" | "stratified-kfold" | "kfold",
    "key": "<col or null>",
    "train": "<= 2024-09-30",
    "val":   "2024-10-01 .. 2024-10-31",
    "test":  "2024-11-01 .. 2024-11-30",
    "cv":    {"n_splits": 5, "shuffle": false, "random_state": 0}
  },
  "features": [
    {"name": "amount_log", "source": "amount", "transform": "log1p", "for_models": ["linear","tree","nn"]},
    {"name": "merchant_country_te", "source": "merchant_country", "transform": "target_encode(smoothing=10)", "for_models": ["linear","tree","nn"]}
  ],
  "dropped": ["txn_id", "is_chargeback"],
  "class_balance": "class_weight" | "none" | "oversample_minority",
  "feature_count": <int>
}
```

## 7. `feature_spec.md`

Short prose grouped by transform: which columns got which treatment, and one sentence of rationale referencing the EDA recommendation that motivated it.

## Anti-patterns

- Don't fit transformers on the full data — that leaks val/test info into train.
- Don't apply target encoding to the test split using test labels — use train-fitted encoders only.
- Don't shuffle a time-based split.
- Don't add 200 polynomial features hoping something works. Stay in the cap.
- Don't override an EDA `drop` recommendation; if you disagree, surface it in your return `recommendations`.
