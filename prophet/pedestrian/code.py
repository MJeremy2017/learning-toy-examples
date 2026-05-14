"""Prophet forecast for the Melbourne pedestrian COVID dataset.

Run with: python code.py
"""

import os

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/facebook/prophet/main/examples/"
    "example_pedestrians_covid.csv"
)
PROJECT_NAME = "pedestrian"
DATETIME_COL = "ds"
TARGET_COL = "y"
COUNTRY = "Australia"
SUBDIVISION = "VIC"

# CV settings: total span ~1490 days -> ~8 splits
CV_INITIAL = "730 days"
CV_PERIOD = "90 days"
CV_HORIZON = "90 days"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, PROJECT_NAME)
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Phase 1 - load & clean
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_URL)
df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
df.rename(columns={DATETIME_COL: "ds", TARGET_COL: "y"}, inplace=True)
df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
print(f"Loaded {len(df)} rows from {df['ds'].min().date()} to {df['ds'].max().date()}")

# ---------------------------------------------------------------------------
# Phase 2 - overall plot
# ---------------------------------------------------------------------------
plot_df = df  # 1490 rows <= 5000, use the whole dataset
plt.figure(figsize=(10, 6))
plt.plot(plot_df["ds"], plot_df["y"])
plt.xlabel("Date")
plt.ylabel("Pedestrian count")
plt.title("Melbourne pedestrian counts (raw)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "overall.png"))
plt.close()

# ---------------------------------------------------------------------------
# Phase 3 - holidays + cross-validation
# ---------------------------------------------------------------------------
years = list(range(df["ds"].dt.year.min(), df["ds"].dt.year.max() + 1))
vic_holidays = holidays.country_holidays(COUNTRY, subdiv=SUBDIVISION, years=years)

holiday_records = [
    {
        "ds": pd.Timestamp(date),
        "holiday": name,
        "lower_window": 0,
        "upper_window": 0,
    }
    for date, name in sorted(vic_holidays.items())
]
holidays_df = pd.DataFrame(holiday_records)

# The series oscillates around a roughly constant mean with a COVID anomaly,
# so use a flat trend and let weekly/yearly seasonality + holidays explain
# the variation. Daily seasonality is off (one observation per day).
m = Prophet(
    growth="flat",
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
    holidays=holidays_df,
)
m.fit(df)

df_cv = cross_validation(
    m,
    initial=CV_INITIAL,
    period=CV_PERIOD,
    horizon=CV_HORIZON,
)
print(f"Cross-validation produced {df_cv['cutoff'].nunique()} cutoff(s)")

# ---------------------------------------------------------------------------
# Phase 4 - per-cutoff metrics + CV plots
# ---------------------------------------------------------------------------
df_cv["abs_error"] = (df_cv["y"] - df_cv["yhat"]).abs()
denominator = (df_cv["y"].abs() + df_cv["yhat"].abs()) / 2
df_cv["smape_comp"] = np.where(denominator == 0, 0, df_cv["abs_error"] / denominator)

cutoff_metrics = (
    df_cv.groupby("cutoff")
    .agg(mae=("abs_error", "mean"), smape=("smape_comp", "mean"))
    .reset_index()
)
cutoff_metrics.to_csv(os.path.join(METRICS_DIR, "cutoff_metrics.csv"), index=False)
print("Per-cutoff metrics:")
print(cutoff_metrics.to_string(index=False))

fig_mae = plot_cross_validation_metric(df_cv, metric="mae")
fig_mae.savefig(os.path.join(PLOTS_DIR, "mae.png"))
plt.close(fig_mae)

fig_smape = plot_cross_validation_metric(df_cv, metric="smape")
fig_smape.savefig(os.path.join(PLOTS_DIR, "smape.png"))
plt.close(fig_smape)

print(f"Artifacts written to {OUT_DIR}")
