"""Run the expanding-window backtest and write results tables and plots to reports/."""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1opt.evaluation import backtest, bootstrap_ci, paired_bootstrap_diff
from f1opt.features import FEATURE_COLUMNS, TARGET, build_features
from f1opt.models import get_model_factories
from f1opt.models_heavy import make_lightgbm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", default="data/processed/weekends.csv")
parser.add_argument("--first-test-season", type=int, default=2022)
parser.add_argument("--out", default="reports")
parser.add_argument("--n-boot", type=int, default=5000)
args = parser.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
MODEL_FACTORIES = get_model_factories()
BASELINE = "Naive (rolling-3 mean)"
METRICS = {"mae": "MAE (lower is better)", "mse": "MSE (lower is better)",
           "spearman": "Spearman rank corr. (higher is better)",
           "top5_capture": "Top-5 points captured (higher is better)"}

weekends = pd.read_csv(args.data, parse_dates=["date"])
features = build_features(weekends)
first_test = int(features.loc[features["season"] == args.first_test_season, "race_index"].min())
per_race = backtest(features, MODEL_FACTORIES, first_test_race=first_test)
per_race.to_csv(out / "per_race_metrics.csv", index=False)
per_race.attrs["predictions"].to_csv(out / "predictions.csv", index=False)

n_races = per_race["race_index"].nunique()
n_rows = int(per_race.drop_duplicates("race_index")["n_drivers"].sum())
print(f"Backtest: {n_races} races, {n_rows} driver-race predictions per model")

# ---- results table ---------------------------------------------------------------------------
cis = {m: bootstrap_ci(per_race, m, args.n_boot) for m in METRICS}
table = pd.DataFrame(index=cis["mae"].index)
for m, ci in cis.items():
    table[m] = ci.apply(lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1)
table = table.loc[list(MODEL_FACTORIES)]
table.columns = [METRICS[m].split(" (")[0] for m in METRICS]
table.to_csv(out / "results_table.csv")
(out / "results_table.md").write_text(
    f"Expanding-window backtest, {n_races} races from {args.first_test_season} "
    f"({n_rows} driver-race predictions). Mean of per-race metrics with 95% bootstrap CI "
    f"(resampling races, {args.n_boot} draws).\n\n" + table.to_markdown() + "\n")

diffs = {}
tables = []
for ref in (BASELINE, "Ridge"):
    for m in ("mae", "spearman", "top5_capture"):
        d = paired_bootstrap_diff(per_race, m, ref, args.n_boot)
        d.insert(0, "metric", m)
        d.insert(0, "reference", ref)
        tables.append(d)
        if ref == BASELINE:
            diffs[m] = d
diff_table = pd.concat(tables).rename_axis("model").reset_index()
diff_table.to_csv(out / "paired_differences.csv", index=False)
print(table.to_string())
print("\nPaired differences (model - reference), 95% bootstrap CI over races:")
print(diff_table.round(4).to_string(index=False))

# ---- plots -----------------------------------------------------------------------------------
colors = {"Naive (rolling-3 mean)": "#9aa0a6", "Ridge": "#4c78a8", "LightGBM": "#54a24b", "PyTorch MLP": "#e45756"}
models = list(MODEL_FACTORIES)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, m in zip(axes, ("mae", "spearman", "top5_capture")):
    ci = cis[m].loc[models]
    ax.bar(range(len(models)), ci["mean"], color=[colors[x] for x in models],
           yerr=[ci["mean"] - ci["ci_low"], ci["ci_high"] - ci["mean"]], capsize=4)
    ax.set_xticks(range(len(models)), [x.replace(" (", "\n(") for x in models], fontsize=8)
    ax.set_title(METRICS[m], fontsize=9)
    lo = ci["ci_low"].min()
    ax.set_ylim(max(0, lo - (ci["ci_high"].max() - lo) * 1.5), ci["ci_high"].max() * 1.03)
fig.suptitle("Out-of-sample performance, mean over races with 95% bootstrap CI", fontsize=10)
fig.tight_layout()
fig.savefig(out / "metrics_with_ci.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, m, label in zip(axes, ("mae", "spearman"), ("MAE difference (negative = better)", "Spearman difference (positive = better)")):
    d = diffs[m].loc[[x for x in models if x != BASELINE]]
    y = np.arange(len(d))
    ax.errorbar(d["diff"], y, xerr=[d["diff"] - d["ci_low"], d["ci_high"] - d["diff"]], fmt="o", color="k", capsize=4)
    ax.axvline(0, color="grey", ls="--")
    ax.set_yticks(y, d.index)
    ax.set_title(f"vs. naive baseline: {label}", fontsize=9)
fig.tight_layout()
fig.savefig(out / "paired_diff_vs_naive.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.6))
wide = per_race.pivot(index="race_index", columns="model", values="spearman")[models]
for m in models:
    ax.plot(wide.index, wide[m].rolling(15, min_periods=5).mean(), label=m, color=colors[m])
season_starts = features.groupby("season")["race_index"].min()
for s, r in season_starts.items():
    if r >= wide.index.min():
        ax.axvline(r, color="lightgrey", lw=0.8)
        ax.text(r + 0.3, ax.get_ylim()[0], str(s), fontsize=7, va="bottom", color="grey")
ax.set_ylabel("Spearman (15-race rolling mean)")
ax.set_xlabel("race index")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "rolling_spearman.png", dpi=150)
plt.close(fig)

best = cis["spearman"]["mean"].idxmax()
pred = per_race.attrs["predictions"].query("model == @best")
fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.hexbin(pred["y_pred"], pred["y_true"], gridsize=35, cmap="Blues", mincnt=1)
lim = [min(pred["y_pred"].min(), pred["y_true"].min()), max(pred["y_pred"].max(), pred["y_true"].max())]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlabel("predicted Fantasy points")
ax.set_ylabel("actual Fantasy points")
ax.set_title(f"{best}: predicted vs actual", fontsize=9)
fig.tight_layout()
fig.savefig(out / "predicted_vs_actual.png", dpi=150)
plt.close(fig)

# Feature importance from a model fit on all pre-test data (descriptive only).
train = features["race_index"] < first_test
gbm = make_lightgbm().fit(features.loc[train, FEATURE_COLUMNS], features.loc[train, TARGET])
imp = pd.Series(gbm.booster_.feature_importance("gain"), index=FEATURE_COLUMNS).sort_values()
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.barh(imp.index, imp / imp.sum(), color="#54a24b")
ax.set_xlabel("share of total gain")
ax.set_title("LightGBM feature importance (trained on pre-test seasons)", fontsize=9)
fig.tight_layout()
fig.savefig(out / "lightgbm_feature_importance.png", dpi=150)
plt.close(fig)
print(f"\nwrote results and plots to {out}/")
