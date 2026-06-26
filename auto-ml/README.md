# auto-ml

Agentic, iterative auto-ML pipeline. Drop a dataset, a data description, and a requirement doc into a project folder and let an orchestrator agent run EDA → Feature Engineering → Modeling → Evaluation in a loop until the metric is hit or the budget runs out.

## Status

Design only. No code yet. The implementing agent should read `DESIGN.md` first and follow §10 ("Implementation plan") to build the system.

## Layout

```
auto-ml/
├── DESIGN.md                  # full design doc — start here
├── README.md                  # this file
├── skills/                    # drafted skills (move to .cursor/skills/ on deploy)
│   ├── auto-ml-orchestrator/SKILL.md
│   ├── auto-ml-eda/SKILL.md
│   ├── auto-ml-feature-engineering/SKILL.md
│   ├── auto-ml-modeling/SKILL.md
│   └── auto-ml-evaluation/SKILL.md
└── projects/                  # one folder per ML project; user populates inputs/
    └── <project_name>/
        ├── inputs/{data.<ext>, data_description.md, requirements.md}
        ├── eda/, features/, models/, evals/   # produced by the agents
        ├── state.json                          # orchestrator state
        └── final/                              # winning model + report
```

## How to use (once implemented)

```
# 1. Create a project, drop your three input files into inputs/
mkdir -p auto-ml/projects/my_task/inputs
cp my_data.csv             auto-ml/projects/my_task/inputs/data.csv
cp my_data_description.md  auto-ml/projects/my_task/inputs/data_description.md
cp my_requirements.md      auto-ml/projects/my_task/inputs/requirements.md

# 2. Run the loop
python auto-ml/run.py loop my_task

# 3. Inspect the result
open auto-ml/projects/my_task/final/REPORT.md
```

## Key documents

- **`DESIGN.md`** — architecture, state schema, decision rubric, stop conditions, implementation plan.
- **`skills/auto-ml-orchestrator/SKILL.md`** — decision loop + routing rubric (10-row table).
- **`skills/auto-ml-eda/SKILL.md`** — what to compute / plot, recommendations contract.
- **`skills/auto-ml-feature-engineering/SKILL.md`** — transform menu + split policy.
- **`skills/auto-ml-modeling/SKILL.md`** — model-family ladder + tuning policy.
- **`skills/auto-ml-evaluation/SKILL.md`** — metrics, diagnosis rules, `decision.json` contract.

Reused existing skills:
- `.cursor/skills/neural-net-tuning/` — invoked by modeling when escalating to a neural net.
- `.cursor/skills/prophet-forecasting/` — invoked by modeling for univariate time-series forecasting.
