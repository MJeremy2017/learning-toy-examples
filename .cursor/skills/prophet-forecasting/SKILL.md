---
name: prophet-forecasting
description: >-
  Builds and evaluates time series models with Facebook Prophet, including
  data inspection, seasonality/trend/holiday configuration, cross-validation,
  and MAE/sMAPE evaluation. Use when the user mentions Prophet, time series
  modeling, seasonal data, or wants to cross-validate a date-indexed numeric
  series.
---

# Prerequisites

Install the required packages before running any generated code:

```bash
pip install prophet holidays matplotlib pandas
```

# Input

- The user must specify a data source whose rows contain a datetime column and a numeric target column. Any format that `pandas.read_*` can load is acceptable.
- The user must provide a project name (used in output paths).

If either is missing, ask the user before proceeding.

The model is built from the Facebook Prophet package: `from prophet import Prophet`.

# Output layout

All artifacts go under `{project_root}/prophet/{project_name}/`, where `{project_root}` is the repository root (the directory containing the `prophet/` folder):

```
{project_root}/prophet/{project_name}/
├── code.py                 # End-to-end runnable script
├── plots/
│   ├── overall.png         # Raw data plot (Phase 2)
│   ├── mae.png             # CV MAE plot (Phase 4)
│   └── smape.png           # CV sMAPE plot (Phase 4)
└── metrics/
    └── cutoff_metrics.csv  # Per-cutoff MAE/sMAPE (Phase 4)
```

# Steps

## Phase 1: Inspect the data

- Read the data. If it has more than 2 columns, list all columns and ask the user which is `ds` (datetime) and which is `y` (target).
- Convert the datetime column to `pandas.Timestamp`:

```python
df['<datetime_col>'] = pd.to_datetime(df['<datetime_col>'])
```

- Rename to Prophet's required column names:

```python
df.rename(columns={"<datetime_col>": "ds", "<target_col>": "y"}, inplace=True)
```

- Sort by `ds` and drop rows with missing `ds` or `y`:

```python
df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
```

## Phase 2: Plot the graph

Save the raw data plot to `{project_root}/prophet/{project_name}/plots/overall.png`.

If the dataset is large, subset it before plotting. Pick the largest window that yields ≤ 5000 rows, in this preference order:

1. The whole dataset (if it already has ≤ 5000 rows)
2. Last 2 years
3. Last 2 months
4. Last 2 weeks
5. Last 2 days
6. Last 2 hours

Example:

```python
import matplotlib.pyplot as plt

plot_df = df  # replace with the subset chosen by the rules above

plt.figure(figsize=(10, 6))
plt.plot(plot_df['ds'], plot_df['y'])
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{project_root}/prophet/{project_name}/plots/overall.png")
plt.close()
```

## Phase 3: Build model for cross validation

### Seasonality

Include a seasonality only if the training data spans more than 2 of its cycles:

- ≥ 2 years → `yearly_seasonality=True`
- ≥ 2 weeks → `weekly_seasonality=True`
- ≥ 2 days → `daily_seasonality=True`

Otherwise set the corresponding flag to `False`.

```python
m = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=False,
    daily_seasonality=True,
)
```

### Trend

Inspect the plot from Phase 2 and pick a `growth` value:

- `flat` — the series oscillates around a roughly constant mean.
- `linear` — there is a monotonic upward or downward drift.
- `logistic` — the series saturates near a known cap/floor. Requires the user to also supply `cap` (and optionally `floor`) columns on `df`.

Prophet's docstring:

```
growth: String 'linear', 'logistic' or 'flat' to specify a linear, logistic or
        flat trend.
```

### Holiday and Events

Ask the user whether to include holidays.

- If **no**, omit the `holidays=` argument from `Prophet(...)`.
- If **yes**, ask for the country (and subdivision, if relevant), then build a holidays DataFrame covering the training span:

```python
import holidays

years = list(range(df['ds'].dt.year.min(), df['ds'].dt.year.max() + 1))
vic_holidays = holidays.Australia(years=years, subdiv='VIC')

holiday_list = []
for date, name in sorted(vic_holidays.items()):
    holiday_list.append({
        'ds': date,
        'holiday': name,
        'lower_window': 0,
        'upper_window': 0,
    })

holidays_df = pd.DataFrame(holiday_list)
holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])

m = Prophet(
    growth="flat",
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
    holidays=holidays_df,
)
```

### Training with cross-validation

Choose `initial`, `period`, and `horizon` so that the total number of CV splits is < 10. Approximate the count as:

```
splits ≈ (total_span - initial - horizon) / period + 1
```

```python
from prophet.diagnostics import cross_validation, performance_metrics

m = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
    holidays=holidays_df,
    growth="flat",
)
m.fit(df)

df_cv = cross_validation(m, initial='366 days', period='180 days', horizon='60 days')
df_cv.head()
```

## Phase 4: Evaluate model

### Evaluate by each cutoff point

Compute MAE and sMAPE per cutoff and save to `metrics/cutoff_metrics.csv`:

```python
import numpy as np

df_cv['abs_error'] = (df_cv['y'] - df_cv['yhat']).abs()

# sMAPE = |y - yhat| / ((|y| + |yhat|) / 2); guard against 0/0
denominator = (df_cv['y'].abs() + df_cv['yhat'].abs()) / 2
df_cv['smape_comp'] = np.where(denominator == 0, 0, df_cv['abs_error'] / denominator)

cutoff_metrics = df_cv.groupby('cutoff').agg(
    mae=('abs_error', 'mean'),
    smape=('smape_comp', 'mean'),
).reset_index()

cutoff_metrics.to_csv(
    f"{project_root}/prophet/{project_name}/metrics/cutoff_metrics.csv",
    index=False,
)
```

### Plot MAE and SMAPE

```python
from prophet.plot import plot_cross_validation_metric

fig_mae = plot_cross_validation_metric(df_cv, metric='mae')
fig_mae.savefig(f"{project_root}/prophet/{project_name}/plots/mae.png")

fig_smape = plot_cross_validation_metric(df_cv, metric='smape')
fig_smape.savefig(f"{project_root}/prophet/{project_name}/plots/smape.png")
```

# Output

Save the full end-to-end pipeline as a single runnable script at
`{project_root}/prophet/{project_name}/code.py` with this structure:

1. Imports (`os`, `pandas`, `numpy`, `matplotlib.pyplot`, `prophet`, `holidays` if used)
2. Config block at the top: data path, project name, datetime/target column names, country/subdivision, CV settings (`initial`, `period`, `horizon`)
3. `os.makedirs(...)` calls to create `plots/` and `metrics/` (`exist_ok=True`)
4. Phase 1 — load & clean
5. Phase 2 — overall plot
6. Phase 3 — build `holidays_df` (if any) and run cross-validation
7. Phase 4 — per-cutoff metrics + CV plots

The script should be runnable with `python code.py`.
