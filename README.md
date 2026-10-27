# Fantasy F1 Optimizer

Predicts each driver's Fantasy F1 points for the next race and picks the best team under the $100M budget cap. The focus is on evaluating the predictions honestly (time-aware validation, baselines, bootstrap confidence intervals, calibrated prediction intervals) and on running the pipeline as a service (scheduled ingestion, MySQL, FastAPI, Streamlit dashboard, Docker Compose).

## Problem

A Fantasy F1 team is 5 drivers and 2 constructors under a $100M cap, with a DRS booster that doubles one driver's points. Choosing a team takes two steps:

1. Predict the points each driver will score at the next race, using only information available before the team lock (before qualifying).
2. Solve the budget-constrained selection exactly, not greedily.

## Architecture

```mermaid
flowchart LR
    J[Jolpica-F1 API<br/>results, qualifying, sprints] --> S
    P[Prices<br/>CSV / experimental API / demo] --> S
    subgraph compose [Docker Compose]
        S[scheduler<br/>ingest, retrain, log] --> DB[(MySQL)]
        DB --> A[FastAPI]
        S -- model.joblib --> A
        M[migrate<br/>Alembic] --> DB
        D[dashboard<br/>Streamlit] --> A
    end
    A --> E["/predictions  /team  /metrics"]
```

- **Ingestion** (`f1opt/ingestion.py`): upserts the calendar, drivers, constructors and results. Re-running changes nothing. Each attempt is recorded in `ingestion_runs`, and failures are retried with exponential backoff. The scheduler runs it daily (`INGEST_CRON`), retrains the model when new races arrive, and writes the next race's predictions and recommended team to `prediction_log` / `team_recommendation_log`.
- **Features** (`f1opt/features.py`): rolling driver form, constructor form, track history, sprint flag. Every feature comes from strictly earlier races.
- **Optimizer** (`f1opt/optimizer.py`): a mixed-integer program (HiGHS via SciPy) over the roster, the budget and the DRS pick. Tested against brute force.
- **Schema**: MySQL, managed with Alembic migrations (`migrations/`).

## Run it

```bash
docker compose up --build
```

This starts MySQL, applies the migrations, serves the API on `http://localhost:8010`, starts the scheduler and serves the dashboard on `http://localhost:8501`. The first cycle downloads 2018 onward (about 3 minutes), trains the model and stores prices. Then:

```bash
curl localhost:8010/predictions        # predicted points per driver, each with an 80% interval
curl "localhost:8010/team?budget=100"  # optimal team, with the greedy baseline for comparison
curl localhost:8010/metrics            # backtest results and deployed-model metadata
```

Ports are configurable with `API_PORT`, `DB_PORT` and `DASHBOARD_PORT` (see `.env.example`).

**Dashboard.** The Streamlit app has three tabs: next-race predictions with a calibrated interval chart, the optimal team for an adjustable budget, and the backtest metrics. It reads everything through the API and holds no data of its own. To run it outside Docker, with the API up:

```bash
API_URL=http://localhost:8010 python -m streamlit run dashboard/app.py   # from the repo root
```

**Prices.** There is no public source of Fantasy prices, so they are loaded by hand or collected going forward. Without real prices the scheduler stores clearly labelled `demo` prices (derived from recent form) so the pipeline runs end to end; real prices override them.

```bash
# CSV columns: entity (driver|constructor), id, price
export DATABASE_URL=mysql+pymysql://f1:f1-dev-password@127.0.0.1:3307/f1opt
python scripts/load_prices.py --csv data/prices/2026_r19_mexico_city.csv --source fantasy-manual
```

IDs are Jolpica IDs (`max_verstappen`, `red_bull`, `rb`, ...); see the sample file. `FantasyApiPriceSource` can read prices with a session cookie in `F1_FANTASY_COOKIE`, but it has not been verified against the live service.

### Development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt     # on macOS LightGBM also needs: brew install libomp
python -m pytest tests -q
python scripts/fetch_data.py            # rebuilds data/processed/weekends.csv from Jolpica
python scripts/run_evaluation.py        # backtest, about 5 minutes; writes reports/
python scripts/validate_scoring.py      # scoring check against official season totals
python scripts/run_calibration.py       # interval coverage check, a few seconds; needs run_evaluation first
```

## Evaluation

**Target.** Fantasy points per driver per race weekend (qualifying, sprint and race), reconstructed from results using the official 2026 rules.

**Protocol.** Expanding-window backtest, one fold per race: train on every earlier race (2018 onward), predict the next. 110 test races from 2022 through round 18 of 2026, 2,234 driver-race predictions per model. Hyperparameters were fixed beforehand and not tuned on test races.

**Leakage controls.**
- No qualifying, grid or price information from the target race is used, because teams lock before qualifying.
- A test scrambles every outcome at or after a race and asserts that the features for that race do not change. It fails if the time shift is removed.
- Scalers and imputers are fit on each fold's training data only.
- Serving builds features for an unplayed race with the same code as the backtest (tested for equality).

**Metrics.** MAE and MSE in points, per-race Spearman rank correlation between predicted and actual points (team selection depends on ranking), and top-5 capture (actual points of the predicted top five as a share of the best possible top five). Values are means over races. Confidence intervals are 95% bootstrap intervals that resample races, since drivers within a race are correlated.

### Results

| Model | MAE | MSE | Spearman | Top-5 capture |
|:--|:--|:--|:--|:--|
| Naive (rolling mean of last 3) | 10.30 [9.80, 10.81] | 186.2 [170.2, 202.5] | 0.491 [0.456, 0.528] | 0.722 [0.688, 0.754] |
| Ridge | **8.86** [8.36, 9.36] | **147.5** [133.6, 161.3] | **0.571** [0.539, 0.603] | 0.749 [0.713, 0.784] |
| LightGBM | 8.93 [8.41, 9.45] | 150.0 [135.9, 164.2] | 0.562 [0.530, 0.593] | 0.736 [0.703, 0.770] |
| PyTorch MLP | 8.95 [8.45, 9.45] | 148.7 [135.0, 162.2] | 0.561 [0.525, 0.596] | **0.751** [0.717, 0.787] |

Plots and per-race data are in `reports/` (`metrics_with_ci.png`, `rolling_spearman.png`, `paired_differences.csv`).

- All three learned models beat the naive baseline. Paired over the same races, MAE improves by about 1.4 points and Spearman by 0.07 to 0.08, with intervals that exclude zero.
- **The neural network does not beat the simpler models.** Ridge, LightGBM and the MLP are statistically close. Ridge has slightly better MAE and Spearman than the MLP (paired MAE difference 0.09, p = 0.03, among several comparisons, so weak evidence).
- **Deployed model: Ridge.** It has the best point estimates on MAE, MSE and Spearman, is tied on the rest, trains in milliseconds, and can be inspected. There is no accuracy reason to pay for a heavier model.

How strong is that? Pooled R² is 0.38 and MAE is 8.9 points against 11.8 for a constant prediction. The median per-race Spearman is 0.60 (93% of races above 0.3). A random five-driver pick captures about 24% of the best five drivers' points, and the model captures 75%. Most of what it learns is persistent car and driver pace. It cannot predict retirements, safety cars or first-lap incidents, which dominate the remaining error.

### Scoring validation

The reconstructed points are checked against the game's own 2026 season totals for all 22 drivers (`data/validation/`, `reports/scoring_validation.*`): Pearson r = 0.991, Spearman 0.984. The game's totals are higher by 5.9 points per race on average (sd 1.7 across drivers), which matches the two scored items that are not in result tables: overtakes (+1 each) and Driver of the Day (+10). The 2026 rules are applied to every season, except that the sprint Not Classified penalty is -20 before 2026.

### Calibration

A point forecast says nothing about how far off it can be, and the error here is wide and skewed: a DNF costs a driver about 20 points. `/predictions` therefore returns an interval for each driver (`interval_low`, `interval_high`, level set with `?level=`, 0.5 to 0.95, default 0.8). An interval is the forecast plus empirical quantiles of the model's own out-of-sample residuals, taken separately within quartiles of the predicted points (`f1opt/calibration.py`). The scheduler fits it with the model; `/predictions` returns null intervals when the history is too short to calibrate.

**Check.** `scripts/run_calibration.py` refits each method before every race using only earlier races, after 20 warm-up races, and measures how often the actual points fall inside. 90 races, 1,834 driver-race rows, coverage with 95% bootstrap CIs resampling races. Gaussian and pooled-quantile intervals are the baselines. The interval score is the width plus a penalty for every point outside the interval (lower is better), so it rewards intervals that are both narrow and right.

| Nominal | Method | Observed coverage | Width (points) | Interval score |
|:--|:--|:--|:--|:--|
| 0.5 | Gaussian | 0.600 [0.566, 0.633] | 16.0 | 30.4 |
| 0.5 | Pooled quantiles | 0.485 [0.452, 0.519] | 10.7 | 29.9 |
| 0.5 | **Binned quantiles** | 0.478 [0.444, 0.511] | 10.8 | **29.5** |
| 0.8 | Gaussian | 0.790 [0.764, 0.815] | 30.5 | 45.6 |
| 0.8 | Pooled quantiles | 0.794 [0.771, 0.815] | 30.9 | 44.0 |
| 0.8 | **Binned quantiles** | 0.782 [0.756, 0.807] | 32.6 | **43.3** |
| 0.9 | Gaussian | 0.873 [0.854, 0.890] | 39.1 | 54.7 |
| 0.9 | Pooled quantiles | 0.895 [0.877, 0.911] | 39.7 | 52.9 |
| 0.9 | **Binned quantiles** | 0.876 [0.858, 0.894] | 38.8 | **49.3** |

Plots are in `reports/` (`calibration_reliability.png`, `calibration_by_bin.png`).

- **Skew breaks the Gaussian assumption.** Gaussian intervals cover 60% when they claim 50%, because most errors are small and a few are very large. Empirical quantiles fix that: at the 50% level they are a third narrower, and at 80% and 90% the widths are within about 2 points of the Gaussian.
- **Pooled quantiles are right on average and wrong where it matters.** At the 80% level they cover 90% of the lowest-predicted quartile and only 70% of the highest, the drivers team selection cares about. Binning by predicted points gives 77% to 80% in every quartile. It also has the best interval score at every level.
- **Not perfect.** The 90% intervals cover 87.6%, slightly under nominal (the CI excludes 0.9). Per-quartile values are pooled over driver-races without CIs, so differences of a few points between quartiles are within noise.
- **The intervals are wide.** An 80% interval is about 33 points across, against a point-estimate MAE of 8.9. A forecast of 15 points comes with a range of roughly -12 to 33. That is the real uncertainty about one driver in one race, not a flaw in the method.

## Limitations

- **No price history.** Historical Fantasy prices are not public, so prices are not a model feature and the points-per-cost baseline and a team-selection backtest are not possible yet. The optimizer and `/team` use whatever prices are stored for the next race; check `price_sources` in the response.
- **Overtakes and Driver of the Day are missing from the target.** They are mostly a constant offset; the driver-specific part (sd about 1.7 points per race) is small next to the model's 8.9 point MAE.
- **Intervals are per driver.** Constructors have no interval, and the correlation between drivers (teammates, safety cars, shared retirements) is not modelled, so there is no distribution for a whole team's score. Calibration assumes future errors look like past ones; a rule change or a new team could break that.
- **Constructor points** are the sum of the two drivers' predicted points. Qualifying-progress and pit-stop bonuses are not modelled.
- **Entry list** for the next race is taken from the last completed race.
- **Prospective check.** Pre-race predictions and teams are logged, and `python scripts/prospective_report.py` scores them once the race is ingested. One race (Mexico City 2026) has been logged and is not yet scored.

## Possible extension

Risk-aware team selection: choose riskier or safer teams depending on the gap to a league rival. It needs a distribution for the whole team (driver correlations) and could only be evaluated in simulation, because historical prices and league data are not available. Not implemented.

## Data

Results, qualifying and sprint classifications come from the [Jolpica-F1 API](https://github.com/jolpica/jolpica-f1) (Ergast-compatible), 2018 to the latest completed round. Official scoring rules and prices were copied from the Fantasy F1 site by the repository owner.

## Layout

```
f1opt/            package: jolpica, scoring, dataset, features, models, evaluation, calibration,
                  db, ingestion, prices, serving, optimizer, recommend, prospective, api, scheduler
dashboard/        Streamlit app (app, api_client, views) and its Dockerfile
migrations/       Alembic revisions
scripts/          fetch_data, run_evaluation, run_calibration, validate_scoring, load_prices, ingest, prospective_report
tests/            pytest suite (ingestion, features and leakage, calibration, optimizer, API, scheduler, dashboard, migrations)
data/processed/   weekends.csv, the reconstructed per-driver-per-race dataset
reports/          results table, plots, per-race metrics, scoring validation
```
