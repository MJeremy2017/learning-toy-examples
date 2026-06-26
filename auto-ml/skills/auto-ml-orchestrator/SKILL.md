---
name: auto-ml-orchestrator
description: >-
  Drive an iterative auto-ML build (EDA → Feature Engineering → Modeling →
  Evaluation → decide) for a project under auto-ml/projects/<name>/. Use when
  the user asks to "build a model from this data", "auto-train an ML model",
  run the auto-ml pipeline, take one step, or finish an auto-ml run. This
  skill owns the decision loop, the routing rubric, and the stop conditions —
  it does NOT do any ML itself; it dispatches to phase subagents.
---

# Auto-ML Orchestrator

You are the orchestrator. You **never** do EDA, feature engineering, modeling, or evaluation directly. You only:
1. Read and update `state.json`.
2. Pick the next phase using the rubrics below.
3. Dispatch a phase subagent via the `Task` tool with the exact prompt template in §4.
4. Validate the subagent's return JSON + artifacts on disk.
5. Decide whether to stop.

Full spec: `auto-ml/DESIGN.md`. Read it once at the start of any project.

## Workflow

```
- [ ] 1. Load state.json (create from requirements.md if missing).
- [ ] 2. If state.status != "running" → exit and print final/REPORT.md path.
- [ ] 3. Pick next phase (§2 cold start → else §3 routing).
- [ ] 4. Dispatch subagent (§4 template). Wait for structured return.
- [ ] 5. Validate return JSON + that every promised artifact exists on disk.
- [ ] 6. Append history entry; bump phase_versions[phase]; maybe update current_best.
- [ ] 7. If phase == "eval": run stop check (§5), write decision.json with the
        rubric row you matched, set next phase or stop.
- [ ] 8. Save state.json. Goto 2.
```

## 1. Bootstrapping a project

If `state.json` is missing:
- Parse `inputs/requirements.md` (schema in DESIGN.md §2). If `target` or `primary` metric is missing → ask the user via `AskQuestion` once, then write the answer back into `requirements.md`.
- Detect baselines:
  - classification: `majority_class` accuracy / random AUC = 0.5
  - regression: mean predictor MAE / RMSE
  - forecast: naive last-value / seasonal-naive on the val window
- Write initial `state.json` (schema in DESIGN.md §5).

## 2. Cold-start sequencing

The first iteration **must** do all four phases in order, regardless of any rubric:

```
if eda.version == 0          → run EDA
elif features.version == 0   → run FE
elif models.version == 0     → run Modeling
elif evals.version < models.version → run Evaluation
```

## 3. Routing rubric (after at least one full pass)

Use evaluation outputs (`evals/vK/metrics.json` and `decision.json`) + `diagnosis` returned by the eval subagent. Match the **first** row that applies, top to bottom:

| # | Condition | Diagnosis | Next phase |
|---|---|---|---|
| 1 | `val_metric ≥ target_value` | `meets_target` | **STOP** |
| 2 | `val_metric ≤ baseline` AND `features.version == 1` | `low_signal` | **FE** |
| 3 | suspected leakage confirmed OR distribution warning unresolved | `data_quality` / `leakage_suspected` | **EDA** |
| 4 | `gap > 0.05` AND model not yet regularized/tuned | `overfit` | **Modeling** (regularize / simpler) |
| 5 | `gap > 0.05` AND model already tuned | `overfit` | **FE** (cut noisy features) |
| 6 | train & val both flat and below target AND model family is linear | `underfit` | **Modeling** (escalate family) |
| 7 | train & val both flat and below target AND family already strong | `underfit` | **FE** |
| 8 | fold_std_over_mean > 0.15 OR loss unstable | `instability` | **Modeling** (lower LR / scale / clip) |
| 9 | minority recall < 0.3 with class_imbalance | `class_imbalance` | **FE** (resample / weighting) |
| 10 | well-calibrated but below target | `well_calibrated_but_below_target` | **FE** |

Loop guards (apply **after** picking from the table):
- If the picked phase is the same as the immediately previous one AND the diagnosis is unchanged → escalate the model family by one rung instead. If already at the top of the ladder → STOP with `stopped_budget` and note "low signal".
- Record the matched row number in `evals/vK/decision.json.matched_rule`.

## 4. Subagent dispatch template

Always use the `Task` tool with `subagent_type: generalPurpose`. Prompt body must be exactly:

```
You are the <PHASE> subagent of an auto-ML pipeline.

Project root:   auto-ml/projects/<project_name>/
Iteration:      <N>
Phase version:  <K>

Read ONLY these inputs:
  - inputs/data.<ext>
  - inputs/data_description.md
  - inputs/requirements.md
  - <list of prior artifacts from §4.1>

Follow the skill: <skill-name> (read it first, then execute it).

Write all artifacts under <phase>/v<K>/ exactly as the skill specifies.

When done, return ONLY a single JSON object (no prose, no fence):

{
  "status": "success" | "failed" | "needs_input",
  "artifacts": ["<relative paths>"],
  "summary": "<≤500 chars>",
  "key_findings":   [...],         // EDA only
  "recommendations":[...],         // EDA / FE / Eval
  "diagnosis": "<one of: meets_target | overfit | underfit | low_signal | data_quality | leakage_suspected | instability | class_imbalance | well_calibrated_but_below_target>",   // Eval only
  "metrics": {...},                // Eval only — must match evals/v<K>/metrics.json
  "needs_input_question": "..."    // only if status=needs_input
}
```

### 4.1 What prior artifacts each phase needs

| Phase | Prior artifacts to pass |
|---|---|
| EDA | none (just `inputs/`) |
| FE  | `eda/v<latest>/stats.json`, `eda/v<latest>/recommendations.json`, prior `features/v<K-1>/feature_spec.json` (if exists) |
| Modeling | `features/v<latest>/*`, prior `evals/v<K-1>/metrics.json` (if exists, for tuning hints) |
| Eval | `models/v<latest>/*`, `features/v<latest>/{val,test}.parquet` |

## 5. Stop conditions

Stop when ANY is true. Set `state.status` and `state.stop_reason` accordingly.

1. `val_metric` meets `target_value` AND fold_std_over_mean < 0.10 → `stopped_success`.
2. `budget.used_iterations >= budget.max_iterations` → `stopped_budget`.
3. Last two eval iterations differ by less than 1% of metric value AND routing would repeat the same phase → `stopped_budget` (plateau).
4. Any subagent returned `status=needs_input` and the user has not answered → `stopped_user`.
5. Any phase failed twice in a row with the same error → `stopped_error`.

On stop:
- Choose `current_best.iteration` as the winning model.
- Copy `models/v<best>/model.pkl` to `final/model.pkl`.
- Write `final/chosen_iteration.json` and `final/REPORT.md` (summary, table of every iteration, best metrics, links to plots, decision trail).

## 6. Validation

Before accepting a phase result:
- Parse the returned JSON; reject if it doesn't match the schema in §4.
- Verify every path in `artifacts` exists on disk.
- For Eval: verify `metrics.json` contains the primary metric named in `state.task.primary_metric`.
- On first failure: re-dispatch the same subagent ONCE with the validation error appended to the prompt. On second failure: mark phase failed, increment retry counter, decide via stop condition 5.

## Anti-patterns

- Don't run EDA twice in a row "just to be safe". Only re-run EDA if rubric row 3 fires.
- Don't tune hyperparameters in the orchestrator — that's the modeling subagent's job.
- Don't read `data.<ext>` yourself. Only subagents touch the data.
- Don't update `current_best` unless the new `val_metric` strictly improves in the correct direction.
- Don't skip writing `decision.json` — it's the human-auditable record.
