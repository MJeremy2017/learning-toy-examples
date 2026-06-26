---
name: auto-ml-eda
description: >-
  Perform exploratory data analysis for an auto-ML project — compute summary
  stats, plot distributions and relationships with the target, flag data
  quality issues and potential target leakage, and emit machine-readable
  recommendations for the feature-engineering phase. Use only when invoked
  by the auto-ml orchestrator with a project path and version.
---

# Auto-ML EDA Subagent

Inputs you read:
- `auto-ml/projects/<project>/inputs/data.<ext>`
- `auto-ml/projects/<project>/inputs/data_description.md`
- `auto-ml/projects/<project>/inputs/requirements.md`

Outputs you write (all paths relative to project root):
- `eda/v<K>/report.md`
- `eda/v<K>/stats.json`
- `eda/v<K>/plots/*.png`
- `eda/v<K>/recommendations.json`

Return JSON (see orchestrator §4 contract): `status`, `artifacts`, `summary`, `key_findings`, `recommendations` (≤10 bullet strings — a *narrative* summary; the machine-readable copy lives in `recommendations.json`).

## Workflow

```
- [ ] 1. Load data, parse requirements, identify target + task type.
- [ ] 2. Compute stats.json (schema below).
- [ ] 3. Make plots (the fixed set below).
- [ ] 4. Detect data-quality + leakage issues.
- [ ] 5. Write recommendations.json for the FE phase.
- [ ] 6. Write report.md and return JSON.
```

## 1. Loading & basic checks

- Pick the right `pandas.read_*` based on extension.
- Confirm `target` from `requirements.md` is present. If not → return `status: needs_input`.
- Coerce dtypes per `data_description.md` where it specifies a type; otherwise infer.
- Drop **exact** duplicate rows but report the duplicate percentage.

## 2. `stats.json` — required schema

Use the schema in DESIGN.md §8.1 verbatim. Per‑column entry includes:
- `name`, `dtype`, `missing_pct`, `n_unique`
- For numerics: `p1`, `p50`, `p99`, `is_constant`
- `high_cardinality` (true if `n_unique > 50` AND `n_unique > 0.5 * n_rows` for an object/categorical column)
- `suspected_leak` (true if `abs(correlation_with_target) >= 0.95` OR the description doc marks it as "post-event")

Top-level fields: `n_rows`, `n_cols`, `target`, `target_type`, `target_distribution`, `duplicates_pct`, `correlation_with_target_top` (max 20, abs‑sorted), `warnings`.

## 3. Required plots (always produced)

Save under `eda/v<K>/plots/`:

| Plot | When | Filename |
|---|---|---|
| target distribution (bar/hist) | always | `target_distribution.png` |
| missingness map (heatmap of `df.isna()`, sampled to ≤5000 rows) | always | `missingness.png` |
| numeric feature histograms (one image grid, up to 24 features) | numerics exist | `numerics_hist.png` |
| categorical bar plots (top 20 categories per feature, up to 12 features) | categoricals exist | `categoricals_bar.png` |
| correlation heatmap of numerics with target highlighted | ≥2 numerics | `corr_heatmap.png` |
| target rate per top‑category (one panel per top categorical) | classification | `target_rate_by_cat.png` |
| time-series plot of target over time (subsampled per prophet-forecasting rules) | a datetime col exists | `target_over_time.png` |
| boxplot of numerics by class | classification + numerics | `numerics_by_class.png` |
| residual-like: target vs each numeric (scatter, subsampled) | regression | `target_vs_numeric.png` |

If the dataset is too large to plot directly (>50k rows), subsample to 5000 rows for visualization (stratified on target for classification).

## 4. Data-quality + leakage detection

For each column, flag:
- **Constant** (`is_constant=true`) → recommend drop.
- **ID-like** (`n_unique == n_rows` and column name contains `id`/`uuid`/`key`) → recommend drop.
- **High missingness** (`missing_pct > 0.6`) → recommend missing-indicator + decide later (don't drop yet).
- **High cardinality categorical** → recommend target encoding or hashing.
- **Suspected leak** (correlation ≥ 0.95 with target OR description marks it post-event) → add to `investigate_leakage`. Do **not** drop automatically — the orchestrator may ask the user.
- **Class imbalance** (binary classification, minority < 5%) → set `class_imbalance.present = true`, record the ratio.
- **Time-indexed** (a datetime column exists AND requirements target is time-aware OR description doc says rows are chronological) → set `splitting_hint = "time-based on <col>"`.

## 5. `recommendations.json` — required schema

Exact schema (mirrored in DESIGN.md §8.1):

```json
{
  "drop": ["<col>"],
  "investigate_leakage": ["<col>"],
  "handle_missing": [{"col": "<col>", "strategy": "missing_indicator+median|mode|model_based|drop_row"}],
  "encoding":       [{"col": "<col>", "strategy": "onehot|target_encode|frequency|hashing|ordinal"}],
  "scaling": ["<col>"],
  "log_transform": ["<col>"],
  "clip_outliers": [{"col": "<col>", "lower": "p1", "upper": "p99"}],
  "splitting_hint": "time-based on <col>" | "stratified-kfold" | "kfold",
  "class_imbalance": {"present": true, "ratio": 65.7},
  "candidate_interactions": [["<a>","<b>"]]
}
```

Rules of thumb:
- `scaling` only for numerics that are inputs to distance/linear/NN models. Tree-based modeling will ignore it harmlessly.
- `log_transform` if `p99/p50 > 50` or the numeric distribution is heavy-tailed (skew > 2).
- `clip_outliers` only when `p99/p50 > 100`.
- `candidate_interactions` should list at most 5 pairs, chosen from columns with the highest target correlation.

## 6. `report.md`

Sections (each ≤ a few short paragraphs):
1. **Overview** — n_rows, n_cols, target, task type, target distribution.
2. **Data quality** — top issues by impact, with column names.
3. **Likely signal** — top correlations / target-rate gaps, named with one-line interpretation.
4. **Splitting recommendation** — why time-based vs. k-fold.
5. **Risks / leakage** — explicit list, what to verify with the user if any.
6. **Next phase hints** — bullet list pointing into `recommendations.json`.

Include relative paths to the plots inline:

```
![](plots/target_distribution.png)
```

## Anti-patterns

- Don't drop a "suspected leak" column yourself. Just flag it.
- Don't oversample/undersample in EDA — that's FE's job.
- Don't compute correlations on categoricals without encoding them (use mutual info or target-rate diff instead, and label it as such).
- Don't draw 200 plots. The list above is fixed.
- Don't subsample to fewer than 1000 rows for stats — only plots may be subsampled.
