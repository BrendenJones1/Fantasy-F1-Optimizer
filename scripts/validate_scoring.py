"""Compare this repo's reconstructed Fantasy points with the game's own season totals."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weekends", default="data/processed/weekends.csv")
parser.add_argument("--official", default="data/validation/official_season_points_2026.csv")
parser.add_argument("--out", default="reports")
args = parser.parse_args()

official = pd.read_csv(args.official)
w = pd.read_csv(args.weekends)
through = int(official["through_round"].max())
mine = (w[(w.season == 2026) & (w["round"] <= through)]
        .groupby("driver_id").agg(mine=("fantasy_points", "sum"), races=("round", "count")))
df = official.set_index("driver_id").join(mine, how="inner")
df["shortfall"] = df["official_season_points"] - df["mine"]
df["shortfall_per_race"] = df["shortfall"] / df["races"]

pearson = pearsonr(df["mine"], df["official_season_points"]).statistic
spearman = spearmanr(df["mine"], df["official_season_points"]).statistic
mean_gap = df["shortfall_per_race"].mean()
resid_sd = df["shortfall_per_race"].std()

out = Path(args.out)
out.mkdir(exist_ok=True)
df.round(2).to_csv(out / "scoring_validation.csv")
summary = (f"Reconstructed vs official 2026 season totals through round {through}, {len(df)} drivers.\n"
           f"Pearson r = {pearson:.3f}, Spearman = {spearman:.3f}.\n"
           f"Official minus reconstructed: mean {mean_gap:.1f} points per race (sd {resid_sd:.1f} across drivers); "
           f"consistent with overtakes (+1 each) and Driver of the Day (+10), which are not in result tables.\n")
(out / "scoring_validation.txt").write_text(summary)
print(summary)
print(df.sort_values("official_season_points", ascending=False).round(1).to_string())

fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.scatter(df["mine"], df["official_season_points"], color="#4c78a8")
lim = [min(df["mine"].min(), df["official_season_points"].min()) - 20,
       max(df["mine"].max(), df["official_season_points"].max()) + 20]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlabel("reconstructed season points")
ax.set_ylabel("official season points")
ax.set_title(f"2026 scoring check (r = {pearson:.3f})", fontsize=9)
fig.tight_layout()
fig.savefig(out / "scoring_validation.png", dpi=150)
