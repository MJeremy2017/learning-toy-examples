# Auto‑ML Agent System — Design Doc

A multi‑agent system that builds a supervised ML model end‑to‑end from three inputs (data + dataset description + requirement doc) with minimal human intervention. The system is iterative: an **orchestrator agent** decides which phase to run next and when to stop, dispatching to specialized **phase subagents** that are guided by **skills**.

This doc is the spec for the agent that will implement the system. Everything described below lives under the `auto-ml/` folder. No code is written yet — this is the design.

---

## 1. Goals & non‑goals

**Goals**
- Take `{dataset, dataset description doc, requirement doc}` and produce a trained, evaluated, persisted model with a written report — with the human only optionally answering a small number of high‑value clarifying questions.
- Make the build process **iterative** (EDA → FE → Model → Eval → decide) and **version every iteration** so we can compare and roll back.
- Make the system **deterministic in structure** (always the same folder layout, the same `state.json` schema, the same eval contract) so any agent can pick it up mid‑run.
- Make decisions **explainable** by recording the rubric the orchestrator used at each step.

**Non‑goals (v1)**
- No online inference server, hyperparameter sweep cluster, or model registry. The artifact is a local model file + report.
- No unsupervised / RL / generative tasks. v1 covers tabular **classification, regression, and time‑series forecasting**.
- No automatic deployment.

---

## 2. Inputs (user‑provided)

The user drops these into a project folder under `auto-ml/projects/<project_name>/inputs/`:

| File | Required | Format | Purpose |
|---|---|---|---|
| `data.<ext>` (or `data/` dir) | yes | csv / parquet / json / xlsx — anything `pandas.read_*` handles | the dataset |
| `data_description.md` | yes | markdown | one section per column: name, dtype, semantics, units, known caveats, expected range, target indicator |
| `requirements.md` | yes | markdown | task type (classification/regression/forecast), target column, success metric + target value, constraints (latency / interpretability / class balance / fairness), training budget |

The orchestrator **never** invents a target column or a metric — those come from `requirements.md`. If either is missing, it asks the user once via `AskQuestion`, then writes the answer back into `requirements.md` so the run is reproducible.

`requirements.md` minimum schema (the agent enforces this):

```
# Task
type: classification | regression | forecast
target: <column_name>

# Metric
primary: <e.g. roc_auc | f1 | rmse | mae | smape>
target_value: <number or "beat_baseline">
direction: maximize | minimize

# Constraints (optional)
max_iterations: 6
interpretability: low | medium | high
must_handle_class_imbalance: true | false
```

---

## 3. Output layout (per project)

```
auto-ml/projects/<project_name>/
├── inputs/
│   ├── data.<ext>
│   ├── data_description.md
│   └── requirements.md
├── state.json                  # orchestrator state — single source of truth
├── eda/
│   └── v1/
│       ├── report.md           # written EDA findings
│       ├── stats.json          # machine‑readable summary (dtypes, missingness, cardinalities, target stats)
│       ├── plots/              # histograms, correlations, target vs feature, missingness map, etc.
│       └── recommendations.json # suggestions FE should consider
├── features/
│   └── v1/
│       ├── pipeline.py         # sklearn-style Pipeline / ColumnTransformer
│       ├── feature_spec.md     # human-readable list of features + rationale
│       ├── feature_spec.json   # machine-readable feature manifest
│       └── train.parquet, val.parquet, test.parquet  # materialized splits (or fold indices)
├── models/
│   └── v1/
│       ├── train.py            # runnable training script
│       ├── model.pkl           # fitted estimator (or .pt for nn)
│       ├── model_card.md       # algorithm, hyperparams, training time, splits
│       └── feature_importance.{png,csv}
├── evals/
│   └── v1/
│       ├── metrics.json        # primary + secondary metrics, per split + per fold
│       ├── report.md           # narrative incl. diagnosis (overfit/underfit/etc.)
│       ├── plots/              # ROC, PR, calibration, residuals, lift, confusion matrix…
│       └── decision.json       # what orchestrator chose next + why
└── final/
    ├── model.pkl               # symlink/copy of the chosen iteration
    ├── REPORT.md               # human‑facing summary of the whole run
    └── chosen_iteration.json
```

Versioning rule: a new `vN+1` folder is created **only** when an iteration produces an artifact that downstream phases will consume. EDA, features, models, and evals each have their own independent version counter; the iteration log in `state.json` ties them together.

---

## 4. Architecture

```
                     ┌──────────────────────────────┐
                     │      Orchestrator Agent       │
                     │  (skill: auto-ml-orchestrator)│
                     └──────────────┬───────────────┘
                                    │ reads state.json, picks next phase
        ┌───────────────┬───────────┼─────────────┬─────────────┐
        ▼               ▼           ▼             ▼             ▼
   EDA Subagent   FE Subagent   Model Subagent   Eval Subagent  STOP
   skill:         skill:         skill:           skill:
   auto-ml-eda    auto-ml-       auto-ml-         auto-ml-
                  feature-       modeling         evaluation
                  engineering
        │               │           │             │
        └─── writes ────┴── writes ─┴── writes ───┘
                        eda/, features/, models/, evals/
                                    │
                                    ▼
                            updates state.json
```

### 4.1 Why subagents (vs. a single agent)
- **Context hygiene** — each phase only needs the artifacts it consumes, not the whole history. A modeling subagent doesn't need the raw EDA plots, just `eda/vK/recommendations.json` and `features/vK/feature_spec.json`.
- **Forced structure** — the orchestrator must read a structured return value, which prevents drift.
- **Recoverability** — any phase can be re‑run in isolation.
- **Cost** — phase subagents can use a smaller model (e.g. fast) for boilerplate work; the orchestrator stays on the stronger model because it does the reasoning.

### 4.2 How subagents are invoked
Use the `Task` tool with `subagent_type: generalPurpose`. Each invocation is given a **fixed prompt template** (see §7) that says "use the `<skill-name>` skill, your inputs are these files, write artifacts to this folder, return this JSON". The orchestrator never embeds business logic in the prompt — that lives in the skill files.

### 4.3 Why skills
Skills are stable "how‑to" guides. Putting the actual procedure for EDA / FE / modeling / eval in skills means:
- They are version‑controlled separately from the orchestration code.
- They can be improved without touching the orchestrator.
- A human can read and trust them.

Five skills are introduced (drafts in `auto-ml/skills/`):

| Skill | Where used | Purpose |
|---|---|---|
| `auto-ml-orchestrator` | orchestrator agent | the decision loop & rubrics (this doc, in skill form) |
| `auto-ml-eda` | EDA subagent | what to compute / plot, what to recommend |
| `auto-ml-feature-engineering` | FE subagent | which feature transforms to try based on EDA |
| `auto-ml-modeling` | Modeling subagent | model family ladder, splitting strategy, fit procedure |
| `auto-ml-evaluation` | Evaluation subagent | metric choice, plots, diagnosis output schema |

Reused existing skills (linked, not duplicated):
- `neural-net-tuning` — used by the modeling subagent **only** when it escalates to a neural net.
- `prophet-forecasting` — used by the modeling subagent when task type is `forecast` and the series is date‑indexed + univariate.

When deployed, the five new SKILL.md files move to `.cursor/skills/<name>/SKILL.md`. They are drafted in `auto-ml/skills/<name>/SKILL.md` so they live with the design.

---

## 5. `state.json` — the single source of truth

The orchestrator reads and writes this on every turn. It is the only piece of mutable state outside per‑phase artifact folders.

```json
{
  "project_name": "fraud_demo",
  "created_at": "2026-06-26T10:00:00Z",
  "task": {
    "type": "classification",
    "target": "is_fraud",
    "primary_metric": "roc_auc",
    "metric_direction": "maximize",
    "target_value": 0.90
  },
  "budget": {
    "max_iterations": 6,
    "used_iterations": 2
  },
  "phase_versions": {
    "eda": 1,
    "features": 2,
    "models": 2,
    "evals": 2
  },
  "current_best": {
    "iteration": 2,
    "model_version": 2,
    "val_metric": 0.873,
    "test_metric": null
  },
  "baselines": {
    "majority_class": 0.5,
    "logistic_regression": 0.81
  },
  "history": [
    {
      "iteration": 1,
      "phase": "eda",     "version": 1, "summary": "...", "artifacts": ["eda/v1/report.md"]
    },
    {
      "iteration": 1,
      "phase": "features","version": 1, "summary": "...", "artifacts": ["features/v1/"]
    },
    {
      "iteration": 1,
      "phase": "model",   "version": 1, "summary": "LogReg baseline", "artifacts": ["models/v1/"]
    },
    {
      "iteration": 1,
      "phase": "eval",    "version": 1, "summary": "ROC AUC 0.81, big train‑val gap",
      "diagnosis": "overfit", "next_action": "modeling",
      "artifacts": ["evals/v1/"]
    }
  ],
  "status": "running",
  "stop_reason": null
}
```

Conventions:
- `current_best` only updates when val metric strictly improves.
- `history` entries are append‑only.
- `status ∈ {running, stopped_success, stopped_budget, stopped_error, stopped_user}`.

---

## 6. The decision loop & rubrics

The orchestrator agent runs this loop on each turn:

```
1. Load state.json (create from requirements.md if missing).
2. If status != running → exit.
3. Pick next phase via §6.1.
4. Dispatch the phase subagent (§7). Wait for structured return.
5. Validate the return + artifacts. If invalid, retry once with feedback.
6. Append to history, bump phase_versions, maybe update current_best.
7. If phase == eval → run the stop check (§6.3) + the routing rubric (§6.2).
8. Goto 1.
```

### 6.1 Picking the next phase (cold start & forced sequencing)

```
if eda.version == 0:        → run EDA
elif features.version == 0: → run FE
elif models.version == 0:   → run Modeling
elif evals.version <
     models.version:        → run Evaluation
else:                       → consult §6.2 (routing rubric)
```

This guarantees every project does at least one full pass before any optimization happens.

### 6.2 Routing rubric — "after eval, where do I go?"

Inputs the orchestrator considers from `evals/vK/metrics.json` + `decision.json`:
- `val_metric`, `train_metric`, `test_metric` (if available)
- `gap = train_metric − val_metric` (positive means train > val for a maximize metric)
- `baseline_metric` (trivial / current LR baseline)
- `target_value`
- Subagent‑reported `diagnosis ∈ {meets_target, overfit, underfit, low_signal, data_quality, leakage_suspected, instability, class_imbalance, well_calibrated_but_below_target}`
- Whether each phase has been run more than once already (avoid pathological loops).

Decision table (orchestrator uses the **highest‑applicable** row):

| Condition | Diagnosis | Next phase | Rationale |
|---|---|---|---|
| `val_metric` ≥ `target_value` for ≥1 holdout | `meets_target` | **STOP** | Done. |
| `val_metric` ≤ baseline AND no FE done yet AND no obvious data issue | `low_signal` | **FE** | Model can't fix bad features. |
| Eval flags missing‑value issue / suspicious column / target leakage / weird distribution | `data_quality` or `leakage_suspected` | **EDA** | Need to understand the data more. |
| Large gap (`gap > 0.05` and train near ceiling) AND model not yet regularized/tuned | `overfit` | **Modeling** | Regularize / simpler model / early stop. |
| Large gap AND model already tuned, FE looks thin | `overfit` | **FE** | Add stronger signal & drop noisy features. |
| Train and val both flat and below target, current family is linear | `underfit` | **Modeling** | Escalate model family (LR→tree→GBM→NN). |
| Train and val both flat and below target, model family already strong | `underfit` | **FE** | Need better features. |
| Predictions are unstable across folds / loss bouncing | `instability` | **Modeling** | Lower LR, fix scaling, gradient clip. |
| Minority class predicted poorly despite reasonable AUC | `class_imbalance` | **FE** then **Modeling** | Add class‑weight / resample / threshold tune. |
| Model calibrated but still below target | `well_calibrated_but_below_target` | **FE** | Need more signal, not better fitting. |

Tie‑breaker rules to avoid loops:
- Don't run the **same phase twice in a row** unless its previous run produced a clearly different plan.
- If the same diagnosis appears 2 iterations in a row with the same routing, **escalate model family** (or, if at top of ladder, STOP with a low‑signal warning).

The orchestrator writes the chosen action and the *row of the table it matched* into `evals/vK/decision.json` so a human can audit the path.

### 6.3 Stop conditions

Stop when **any** of these is true:
1. `val_metric` meets `target_value` and is stable across folds (std/mean of fold scores < 10%).
2. `budget.used_iterations >= budget.max_iterations`.
3. Two consecutive iterations show `|Δ val_metric| < 1%` *and* the routing rubric would otherwise repeat the same phase.
4. A subagent reports `status=needs_input` and the user does not respond.
5. Any phase fails twice in a row.

On stop, the orchestrator:
- Copies the best iteration into `final/`.
- Writes `final/REPORT.md` (summary, best metrics, key plots inline, decision trail).
- Sets `state.status` and `state.stop_reason`.

---

## 7. Subagent prompt templates

Each phase subagent receives a prompt with this exact structure:

```
You are the <PHASE> subagent of an auto-ML pipeline.

Project root:   auto-ml/projects/<project_name>/
Iteration:      <N>
Phase version:  <K>     (you will write to <phase>/v<K>/)

Read these inputs (and ONLY these):
  - inputs/data.<ext>
  - inputs/data_description.md
  - inputs/requirements.md
  - <previous phase artifacts the skill says you need>

Follow the skill: <skill-name> (read it first).

Write all artifacts under <phase>/v<K>/ exactly per the skill.

When done, return ONLY this JSON (no prose, no markdown fence):

{
  "status": "success" | "failed" | "needs_input",
  "artifacts": ["<relative path>", ...],
  "summary": "<≤500 char human summary>",
  "key_findings": ["...", "..."],          // EDA only
  "recommendations": ["...", "..."],       // EDA/FE/Eval
  "diagnosis": "<one of the §6.2 diagnoses>",  // Eval only
  "metrics": { ... },                      // Eval only — mirrors metrics.json
  "needs_input_question": "<asked of user>"    // only if needs_input
}
```

The orchestrator validates the JSON against a schema and only accepts the run if all promised artifacts exist on disk.

---

## 8. Phase specs (what each subagent must do)

The full procedure for each phase lives in the corresponding skill. Below are the **contracts** the orchestrator depends on.

### 8.1 EDA subagent
Reads: `inputs/*`.
Writes: `eda/vK/{report.md, stats.json, plots/, recommendations.json}`.

`stats.json` schema (the orchestrator and FE subagent rely on this — do not change without bumping the version):

```json
{
  "n_rows": 100000,
  "n_cols": 42,
  "target": "is_fraud",
  "target_type": "binary",
  "target_distribution": {"0": 0.985, "1": 0.015},
  "columns": [
    {"name": "amount", "dtype": "float64", "missing_pct": 0.0, "n_unique": 9123,
     "p1": 0.5, "p50": 25.0, "p99": 980.0, "is_constant": false,
     "high_cardinality": false, "suspected_leak": false}
    // ...
  ],
  "duplicates_pct": 0.001,
  "correlation_with_target_top": [{"name": "amount_zscore", "value": 0.31}, ...],
  "warnings": ["column `txn_id` looks like an ID — should be excluded",
               "column `is_chargeback` has 0.99 correlation with target — possible leakage"]
}
```

`recommendations.json` is a flat list of suggestions for downstream phases:

```json
{
  "drop": ["txn_id"],
  "investigate_leakage": ["is_chargeback"],
  "handle_missing": [{"col": "device_id", "strategy": "missing_indicator + mode_impute"}],
  "encoding": [{"col": "merchant_country", "strategy": "target_encode"}],
  "scaling": ["amount"],
  "splitting_hint": "time-based on `txn_ts`",
  "class_imbalance": {"present": true, "ratio": 65.7}
}
```

### 8.2 Feature‑engineering subagent
Reads: `inputs/*`, `eda/vK/recommendations.json`, optional prior `features/v(K-1)/feature_spec.json`.
Writes: `features/vK/{pipeline.py, feature_spec.md, feature_spec.json, train.parquet, val.parquet, test.parquet}`.

The `pipeline.py` must export a `build_pipeline()` returning an `sklearn.pipeline.Pipeline` (or a `ColumnTransformer`) so the modeling subagent can `pickle.load` it and call `.fit_transform` deterministically.

`feature_spec.json`:

```json
{
  "features": [
    {"name": "amount_log", "source": "amount", "transform": "log1p"},
    {"name": "merchant_country_te", "source": "merchant_country", "transform": "target_encode(smoothing=10)"}
  ],
  "dropped": ["txn_id", "is_chargeback"],
  "split": {"strategy": "time", "key": "txn_ts", "train": "<= 2024-09-30", "val": "2024-10", "test": "2024-11"}
}
```

### 8.3 Modeling subagent
Reads: `features/vK/*`, `requirements.md`, optional `evals/v(K-1)/*` for tuning hints.
Writes: `models/vK/{train.py, model.pkl, model_card.md, feature_importance.{png,csv}}`.

Model‑family ladder (start at top, escalate only when justified):

```
classification:
  1. LogisticRegression (with class_weight if imbalanced)
  2. RandomForestClassifier
  3. LightGBM / XGBoost classifier
  4. (only if needed) MLP via PyTorch — invoke `neural-net-tuning` skill

regression:
  1. Ridge / Lasso
  2. RandomForestRegressor
  3. LightGBM / XGBoost regressor
  4. (only if needed) MLP via PyTorch

forecast (univariate, date-indexed):
  1. naive seasonal baseline (last value / seasonal lag)
  2. Prophet — invoke `prophet-forecasting` skill
  3. LightGBM with lag/rolling features (treat as regression)
```

Splitting strategy:
- If `splitting_hint == "time-based"` (from EDA) **or** the target / data is time‑indexed → chronological split, never shuffle. For CV, expanding‑window or `TimeSeriesSplit`.
- Otherwise stratified k‑fold for classification, k‑fold for regression. Default k=5.
- The same split must be used by FE and Modeling — `feature_spec.json.split` is the authority.

### 8.4 Evaluation subagent
Reads: `models/vK/model.pkl`, `features/vK/{val,test}.parquet`, `requirements.md`.
Writes: `evals/vK/{metrics.json, report.md, plots/, decision.json}`.

`metrics.json`:

```json
{
  "primary": {"name": "roc_auc", "val": 0.873, "test": null, "folds": [0.86, 0.88, 0.87, 0.88, 0.88]},
  "secondary": {
    "f1@0.5": 0.41, "pr_auc": 0.55, "logloss": 0.12, "brier": 0.04,
    "minority_recall@0.5": 0.62
  },
  "train_metric": 0.96,
  "val_metric": 0.873,
  "gap": 0.087,
  "stability": {"fold_std": 0.008, "fold_std_over_mean": 0.0092},
  "calibration": {"brier": 0.04, "is_well_calibrated": true},
  "baseline_comparison": {"majority_class": 0.5, "logistic_regression": 0.81}
}
```

The eval subagent is the **only** entity allowed to set `diagnosis` in its return JSON. It uses these rules (mirrored in §6.2 wording):

```
if val ≥ target_value           → meets_target
if gap > 0.05 and train_metric > 0.9*ceiling   → overfit
if train_metric < baseline + 0.02              → underfit / low_signal
if fold_std_over_mean > 0.15                   → instability
if minority_recall < 0.3 and class_imbalance   → class_imbalance
if any suspected_leak flag from EDA confirmed  → leakage_suspected
otherwise                                       → well_calibrated_but_below_target
```

---

## 9. Skills — drafts

The five skills are drafted as separate files in `auto-ml/skills/<skill>/SKILL.md`:

1. `auto-ml/skills/auto-ml-orchestrator/SKILL.md` — the loop, rubrics, dispatch templates.
2. `auto-ml/skills/auto-ml-eda/SKILL.md` — what to compute, what to plot, schema of `stats.json` / `recommendations.json`.
3. `auto-ml/skills/auto-ml-feature-engineering/SKILL.md` — transform menu keyed off EDA recommendations, `feature_spec.json` schema, splitting rules.
4. `auto-ml/skills/auto-ml-modeling/SKILL.md` — the model ladder, when to escalate, fit procedure, integration with `neural-net-tuning` and `prophet-forecasting`.
5. `auto-ml/skills/auto-ml-evaluation/SKILL.md` — metric selection per task type, diagnosis rules, decision.json contract.

When deploying for real, move these into `.cursor/skills/<name>/SKILL.md` (skill names match the folder names). They are kept under `auto-ml/skills/` here so the design and its skills ship together.

---

## 10. Implementation plan (for the implementing agent)

Concrete steps to build the system from this doc:

1. **Scaffolding**
   - Create `auto-ml/projects/.gitkeep`.
   - Create `auto-ml/lib/` with three modules:
     - `state.py` — typed dataclasses for `state.json` + load/save + schema validation.
     - `paths.py` — single helper that maps `(project, phase, version) → Path`. All subagents must use it.
     - `contracts.py` — pydantic models for `stats.json`, `recommendations.json`, `feature_spec.json`, `metrics.json`, `decision.json`, and the subagent return JSON.
2. **CLI entrypoint** `auto-ml/run.py`:
   - `python auto-ml/run.py init <project_name>` — scaffolds the project folder, validates `inputs/requirements.md`, writes initial `state.json`.
   - `python auto-ml/run.py step <project_name>` — runs ONE orchestrator turn.
   - `python auto-ml/run.py loop <project_name>` — runs turns until `state.status != running`.
   - `python auto-ml/run.py report <project_name>` — regenerates `final/REPORT.md` from history.
3. **Orchestrator agent**: implemented as a function `orchestrator.step(state)`. It does *not* do any ML — it only:
   - reads/writes state,
   - calls the right subagent via the `Task` tool with the §7 template,
   - validates the returned JSON against `contracts.py`,
   - applies §6.1–§6.3.
4. **Phase subagents**: each is a thin wrapper that loads the relevant skill and follows it. The skill carries all domain logic; the wrapper handles I/O contracts.
5. **Idempotency**: every phase script must be re‑runnable. If `<phase>/vK/` already exists, it gets overwritten only when the orchestrator explicitly asks for "redo version K"; otherwise a new version is created.
6. **Dependencies** (`auto-ml/requirements.txt`):
   ```
   pandas
   numpy
   scikit-learn
   pyarrow
   matplotlib
   seaborn
   lightgbm
   xgboost
   pydantic>=2
   prophet            # only used by forecast tasks
   torch              # only used when NN escalation is triggered
   ```
7. **Testing**:
   - Ship a tiny synthetic toy project under `auto-ml/projects/_smoke/` (e.g. UCI adult, 5k rows) so `python auto-ml/run.py loop _smoke` runs the whole loop in <2 min.
   - Add a unit test that mocks subagent returns and verifies the §6.2 routing table.

Folder tree the implementing agent must end up with:

```
auto-ml/
├── DESIGN.md                    (this file)
├── README.md                    (1‑pager pointing at DESIGN.md + how to run)
├── requirements.txt
├── run.py                       (CLI)
├── lib/
│   ├── __init__.py
│   ├── state.py
│   ├── paths.py
│   └── contracts.py
├── skills/
│   ├── auto-ml-orchestrator/SKILL.md
│   ├── auto-ml-eda/SKILL.md
│   ├── auto-ml-feature-engineering/SKILL.md
│   ├── auto-ml-modeling/SKILL.md
│   └── auto-ml-evaluation/SKILL.md
└── projects/
    └── _smoke/                  (sample project for tests)
```

---

## 11. Open questions / explicit human‑in‑the‑loop points

The system pauses and asks the user **only** in these cases:
1. `requirements.md` is missing the target column or the metric.
2. EDA finds a column with ≥0.95 correlation to the target → asks "is this leakage?".
3. The dataset has a date column but `requirements.md` doesn't say "forecast" → asks "should the split be time‑based?".
4. Budget exhausted but val metric is still trending up → offers to extend budget.

All other decisions are made autonomously per the rubrics above.
