"""Expanding-window, race-by-race backtest with per-race metrics and bootstrap CIs."""
import logging
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from f1opt.features import FEATURE_COLUMNS, TARGET

logger = logging.getLogger(__name__)

TOP_K = 5  # a fantasy team holds 5 drivers


def expanding_window_splits(race_index: pd.Series, first_test_race: int,
                            min_train_races: int = 10) -> Iterable[tuple]:
    """Yield (train_mask, test_mask) per race: train on every earlier race, test on the next.

    Training never contains the test race or anything after it.
    """
    for r in sorted(race_index.unique()):
        if r < first_test_race or r < min_train_races:
            continue
        yield (race_index < r).to_numpy(), (race_index == r).to_numpy()


def race_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = TOP_K) -> Dict[str, float]:
    err = y_pred - y_true
    rho = spearmanr(y_pred, y_true).statistic if len(y_true) > 2 and np.ptp(y_pred) > 0 else 0.0
    top_pred = np.argsort(-y_pred)[:k]
    top_true = np.sort(y_true)[::-1][:k]
    return {
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "spearman": float(rho),
        # Points captured by the predicted top-k as a share of the best possible top-k.
        "top5_capture": float(y_true[top_pred].sum() / top_true.sum()) if top_true.sum() > 0 else np.nan,
    }


def backtest(features: pd.DataFrame, factories: Dict[str, Callable], first_test_race: int,
             min_train_races: int = 10) -> pd.DataFrame:
    """Return one row per (model, test race) with metrics, plus the raw predictions as ``attrs``."""
    X, y = features[FEATURE_COLUMNS], features[TARGET].to_numpy()
    rows: List[dict] = []
    preds = []
    splits = list(expanding_window_splits(features["race_index"], first_test_race, min_train_races))
    for i, (tr, te) in enumerate(splits, 1):
        race = features.loc[te, ["season", "round", "race_index"]].iloc[0]
        for name, factory in factories.items():
            model = factory().fit(X[tr], y[tr])
            p = np.asarray(model.predict(X[te]), dtype=float)
            rows.append({"model": name, "season": int(race["season"]), "round": int(race["round"]),
                         "race_index": int(race["race_index"]), "n_drivers": int(te.sum()),
                         **race_metrics(y[te], p)})
            preds.append(pd.DataFrame({"model": name, "race_index": race["race_index"],
                                       "driver_id": features.loc[te, "driver_id"].to_numpy(),
                                       "y_true": y[te], "y_pred": p}))
        if i % 20 == 0:
            logger.info("backtest %d/%d races", i, len(splits))
    out = pd.DataFrame(rows)
    out.attrs["predictions"] = pd.concat(preds, ignore_index=True)
    return out


def bootstrap_ci(per_race: pd.DataFrame, metric: str, n_boot: int = 5000, seed: int = 0,
                 alpha: float = 0.05) -> pd.DataFrame:
    """Mean of ``metric`` per model with a percentile CI from resampling *races* (not drivers).

    Drivers within a race are correlated, so races are the independent unit.
    """
    wide = per_race.pivot(index="race_index", columns="model", values=metric)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wide), size=(n_boot, len(wide)))
    boot = wide.to_numpy()[idx].mean(axis=1)  # n_boot x models
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)
    return pd.DataFrame({"mean": wide.mean().to_numpy(), "ci_low": lo, "ci_high": hi}, index=wide.columns)


def paired_bootstrap_diff(per_race: pd.DataFrame, metric: str, baseline: str, n_boot: int = 5000,
                          seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """CI for (model - baseline) on the same resampled races; accounts for shared race difficulty."""
    wide = per_race.pivot(index="race_index", columns="model", values=metric)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wide), size=(n_boot, len(wide)))
    rows = {}
    for m in wide.columns:
        if m == baseline:
            continue
        d = (wide[m] - wide[baseline]).to_numpy()
        boot = d[idx].mean(axis=1)
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        rows[m] = {"diff": d.mean(), "ci_low": lo, "ci_high": hi,
                   "p_boot": float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))}
    return pd.DataFrame(rows).T
