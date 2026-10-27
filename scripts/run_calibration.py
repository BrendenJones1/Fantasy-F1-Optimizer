"""Check how well calibrated the deployed model's prediction intervals are.

Reads the Ridge out-of-sample predictions written by scripts/run_evaluation.py (so run that first),
refits each interval method before every race on earlier races only, and writes coverage tables
and plots to reports/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1opt.calibration import (LEVELS, METHODS, coverage_by_prediction_bin, coverage_table,
                               per_race_interval_metrics, rolling_origin_intervals)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--predictions", default="reports/predictions.csv")
parser.add_argument("--model", default="Ridge")
parser.add_argument("--warm-races", type=int, default=20, help="races used only to fit the first calibrator")
parser.add_argument("--n-boot", type=int, default=5000)
parser.add_argument("--out", default="reports")
args = parser.parse_args()

out = Path(args.out)
preds = pd.read_csv(args.predictions).query("model == @args.model").drop(columns="model")
GRID = tuple(np.round(np.arange(0.1, 0.96, 0.05), 2))
COLORS = {"Gaussian": "#9aa0a6", "Pooled quantiles": "#4c78a8", "Binned quantiles": "#e45756"}

intervals = {m: rolling_origin_intervals(preds, make, GRID, args.warm_races) for m, make in METHODS.items()}
per_race = pd.concat([per_race_interval_metrics(df, m) for m, df in intervals.items()], ignore_index=True)
per_race["level"] = per_race["level"].round(2)
n_races = per_race["race_index"].nunique()
n_rows = int(intervals["Binned quantiles"].query("level == 0.8").shape[0])
print(f"{args.model}: {n_races} evaluated races after {args.warm_races} warm-up races, {n_rows} driver-race rows")

# ---- headline table --------------------------------------------------------------------------
full = coverage_table(per_race, args.n_boot)
full.to_csv(out / "calibration_reliability.csv", index=False)
head = full[full["level"].isin(LEVELS)].copy()
head["coverage"] = head.apply(lambda r: f"{r.coverage:.3f} [{r.coverage_low:.3f}, {r.coverage_high:.3f}]", axis=1)
head = head.assign(width=head["width"].round(1), interval_score=head["interval_score"].round(1))
head = head[["level", "method", "coverage", "width", "interval_score"]]
head.to_csv(out / "calibration_results.csv", index=False)
(out / "calibration_results.md").write_text(
    f"Prediction-interval coverage for {args.model} on {n_races} races ({n_rows} driver-race rows). Each "
    f"method is refit before every race on earlier races only, after {args.warm_races} warm-up races. "
    f"Coverage is the mean over races with a 95% bootstrap CI resampling races; width and interval "
    f"score are in Fantasy points (lower is better).\n\n" + head.to_markdown(index=False) + "\n")
print(head.to_string(index=False))

# ---- coverage by predicted-points quartile ---------------------------------------------------
by_bin = pd.concat([coverage_by_prediction_bin(df, m) for m, df in intervals.items()], ignore_index=True)
by_bin["level"] = by_bin["level"].round(2)
by_bin = by_bin[by_bin["level"].isin(LEVELS)]
by_bin.to_csv(out / "calibration_by_bin.csv", index=False)
print("\nCoverage at the 80% level by quartile of predicted points (0 = lowest):")
print(by_bin.query("level == 0.8").pivot(index="bin", columns="method", values="coverage").round(3).to_string())

# ---- plots -----------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
for m in METHODS:
    g = full[full["method"] == m]
    ax.errorbar(g["level"], g["coverage"], yerr=[g["coverage"] - g["coverage_low"], g["coverage_high"] - g["coverage"]],
                color=COLORS[m], marker="o", ms=3, lw=1, capsize=2, label=m)
ax.set_xlabel("nominal coverage")
ax.set_ylabel("observed coverage")
ax.set_title(f"{args.model}: interval reliability, 95% CI over races", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "calibration_reliability.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 3.8))
width = 0.27
sub = by_bin[by_bin["level"] == 0.8]
for i, m in enumerate(METHODS):
    g = sub[sub["method"] == m]
    ax.bar(g["bin"] + (i - 1) * width, g["coverage"], width, color=COLORS[m], label=m)
ax.axhline(0.8, color="k", ls="--", lw=0.8)
ax.set_xticks(range(4), ["lowest", "2nd", "3rd", "highest"])
ax.set_xlabel("quartile of predicted points")
ax.set_ylabel("observed coverage of the 80% interval")
ax.set_ylim(0.5, 1.02)
ax.legend(fontsize=8, loc="upper center", ncol=3)
fig.tight_layout()
fig.savefig(out / "calibration_by_bin.png", dpi=150)
plt.close(fig)
print(f"\nwrote calibration results to {out}/")
