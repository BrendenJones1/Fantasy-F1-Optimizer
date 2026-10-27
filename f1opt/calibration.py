"""Predictive intervals for the point forecasts, and a leakage-free check of their coverage.

The point model says where a driver's Fantasy points are centred, not how far off it can be. The
error is wide, skewed (a DNF costs a driver 20 points) and larger for the drivers predicted to
score most. An interval is the forecast plus empirical quantiles of the model's own *out-of-sample*
residuals, taken separately within quartiles of the predicted points. The evaluation refits the
calibrator before every race using only earlier races, so the reported coverage is what a user
would have got.
"""
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from f1opt.evaluation import backtest, bootstrap_ci

DEFAULT_LEVEL = 0.8
LEVELS = (0.5, 0.8, 0.9)
N_BINS = 4
MIN_RESIDUALS = 200  # fewer residuals than this and the tail quantiles are too noisy to publish
MIN_PER_BIN = 50     # a bin with fewer residuals falls back to the pooled residuals


class NotEnoughResiduals(ValueError):
    pass


class ResidualCalibrator:
    """Interval = forecast + quantiles of past residuals within the forecast's own bin.

    With ``n_bins=1`` this is the plain pooled version, kept as a baseline for the evaluation.
    """

    def __init__(self, n_bins: int = N_BINS, min_residuals: int = MIN_RESIDUALS,
                 min_per_bin: int = MIN_PER_BIN):
        self.n_bins, self.min_residuals, self.min_per_bin = n_bins, min_residuals, min_per_bin

    def fit(self, y_pred, y_true) -> "ResidualCalibrator":
        y_pred, y_true = np.asarray(y_pred, float), np.asarray(y_true, float)
        if len(y_pred) < self.min_residuals:
            raise NotEnoughResiduals(f"{len(y_pred)} residuals, need {self.min_residuals}")
        resid = y_true - y_pred
        self.edges_ = np.unique(np.quantile(y_pred, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        bins = np.digitize(y_pred, self.edges_)
        pooled = np.sort(resid)
        self.residuals_ = [np.sort(resid[bins == b]) if (bins == b).sum() >= self.min_per_bin else pooled
                           for b in range(len(self.edges_) + 1)]
        self.n_residuals_ = len(resid)
        return self

    def interval(self, y_pred, level: float = DEFAULT_LEVEL) -> tuple:
        y_pred = np.asarray(y_pred, float)
        a = (1 - level) / 2
        lo = np.array([np.quantile(r, a) for r in self.residuals_])
        hi = np.array([np.quantile(r, 1 - a) for r in self.residuals_])
        bins = np.digitize(y_pred, self.edges_)
        return y_pred + lo[bins], y_pred + hi[bins]


class GaussianCalibrator:
    """Baseline: forecast +/- z * the standard deviation of past residuals."""

    def __init__(self, min_residuals: int = MIN_RESIDUALS):
        self.min_residuals = min_residuals

    def fit(self, y_pred, y_true) -> "GaussianCalibrator":
        resid = np.asarray(y_true, float) - np.asarray(y_pred, float)
        if len(resid) < self.min_residuals:
            raise NotEnoughResiduals(f"{len(resid)} residuals, need {self.min_residuals}")
        self.sigma_ = float(resid.std())
        return self

    def interval(self, y_pred, level: float = DEFAULT_LEVEL) -> tuple:
        y_pred = np.asarray(y_pred, float)
        half = norm.ppf(1 - (1 - level) / 2) * self.sigma_
        return y_pred - half, y_pred + half


METHODS: Dict[str, Callable] = {
    "Gaussian": GaussianCalibrator,
    "Pooled quantiles": lambda: ResidualCalibrator(n_bins=1),
    "Binned quantiles": ResidualCalibrator,
}


def out_of_sample_predictions(features: pd.DataFrame, make_model: Callable,
                              min_train_races: int) -> pd.DataFrame:
    """Expanding-window predictions (race_index, driver_id, y_true, y_pred) from races with at
    least ``min_train_races`` earlier races to train on."""
    if features["race_index"].nunique() <= min_train_races:
        return pd.DataFrame(columns=["race_index", "driver_id", "y_true", "y_pred"])
    per_race = backtest(features, {"model": make_model}, first_test_race=min_train_races,
                        min_train_races=min_train_races)
    return per_race.attrs["predictions"].drop(columns="model")


def rolling_origin_intervals(preds: pd.DataFrame, make_calibrator: Callable,
                             levels: Sequence[float] = LEVELS, warm_races: int = 20) -> pd.DataFrame:
    """Interval for every driver-race after the warm-up, calibrated on strictly earlier races only."""
    races = np.sort(preds["race_index"].unique())
    out = []
    for r in races[warm_races:]:
        prior, test = preds[preds["race_index"] < r],preds[preds["race_index"] == r]
        calibrator = make_calibrator().fit(prior["y_pred"], prior["y_true"])
        for level in levels:
            low, high = calibrator.interval(test["y_pred"], level)
            out.append(test.assign(level=level, low=low, high=high))
    return pd.concat(out, ignore_index=True)


def interval_score(y, low, high, level: float) -> np.ndarray:
    """Winkler interval score (lower is better): width plus a penalty of 2/alpha per point outside."""
    y, low, high = (np.asarray(v, float) for v in (y, low, high))
    alpha = 1 - level
    return (high - low) + (2 / alpha) * np.maximum(low - y, 0) + (2 / alpha) * np.maximum(y - high, 0)


def per_race_interval_metrics(intervals: pd.DataFrame, method: str) -> pd.DataFrame:
    """One row per (level, race): share of drivers inside the interval, mean width and score."""
    df = intervals.assign(
        covered=(intervals["y_true"] >= intervals["low"]) & (intervals["y_true"] <= intervals["high"]),
        width=intervals["high"] - intervals["low"])
    df["interval_score"] = [interval_score(y, lo, hi, lv) for y, lo, hi, lv
                            in zip(df["y_true"], df["low"], df["high"], df["level"])]
    out = (df.groupby(["level", "race_index"])
           .agg(coverage=("covered", "mean"), width=("width", "mean"), interval_score=("interval_score", "mean"))
           .reset_index())
    out.insert(0, "model", method)
    return out


def coverage_table(per_race: pd.DataFrame, n_boot: int = 5000) -> pd.DataFrame:
    """Mean coverage, width and score per (level, method); coverage has a bootstrap CI over races."""
    rows = []
    for level, g in per_race.groupby("level"):
        cov = bootstrap_ci(g, "coverage", n_boot)
        for method in cov.index:
            m = g[g["model"] == method]
            rows.append({"level": level, "method": method, "coverage": cov.loc[method, "mean"],
                         "coverage_low": cov.loc[method, "ci_low"], "coverage_high": cov.loc[method, "ci_high"],
                         "width": m["width"].mean(), "interval_score": m["interval_score"].mean()})
    return pd.DataFrame(rows)


def coverage_by_prediction_bin(intervals: pd.DataFrame, method: str, n_bins: int = N_BINS) -> pd.DataFrame:
    """Coverage within quartiles of the predicted points (does it hold for the favourites too?)."""
    df = intervals.copy()
    df["bin"] = pd.qcut(df["y_pred"], n_bins, labels=False, duplicates="drop")
    df["covered"] = (df["y_true"] >= df["low"]) & (df["y_true"] <= df["high"])
    out = (df.groupby(["level", "bin"])
           .agg(coverage=("covered", "mean"), n=("covered", "size"), mean_pred=("y_pred", "mean"))
           .reset_index())
    out.insert(0, "method", method)
    return out
