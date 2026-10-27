import numpy as np
import pandas as pd
import pytest

from f1opt.calibration import (GaussianCalibrator, NotEnoughResiduals, ResidualCalibrator,
                               coverage_by_prediction_bin, interval_score, out_of_sample_predictions,
                               per_race_interval_metrics, rolling_origin_intervals)
from f1opt.features import FEATURE_COLUMNS
from f1opt.models import make_ridge


def _skewed(n, seed):
    """Forecasts with a left-skewed error whose spread grows with the forecast, like Fantasy points."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0, 30, n)
    noise = (rng.gamma(2.0, 1.0, n) - 2.0) * (1 + pred / 8)
    return pred, pred + noise


def _bin_coverage(cal, pred, y, level):
    low, high = cal.interval(pred, level)
    inside = (y >= low) & (y <= high)
    bins = np.digitize(pred, np.quantile(pred, [.25, .5, .75]))
    return np.array([inside[bins == b].mean() for b in range(4)]), inside.mean()


def test_binned_quantiles_cover_every_prediction_bin_but_pooled_do_not():
    pred, y = _skewed(20000, 0)
    new_pred, new_y = _skewed(20000, 1)
    binned_bins, binned_all = _bin_coverage(ResidualCalibrator().fit(pred, y), new_pred, new_y, 0.8)
    pooled_bins, pooled_all = _bin_coverage(ResidualCalibrator(n_bins=1).fit(pred, y), new_pred, new_y, 0.8)
    assert abs(binned_all - 0.8) < 0.02 and np.abs(binned_bins - 0.8).max() < 0.03
    assert abs(pooled_all - 0.8) < 0.02          # right on average ...
    assert np.abs(pooled_bins - 0.8).max() > 0.08  # ... but wrong for small and large forecasts


def test_higher_level_gives_wider_interval_and_gaussian_is_symmetric():
    pred, y = _skewed(2000, 0)
    cal = ResidualCalibrator().fit(pred, y)
    narrow, wide = cal.interval([10.0], 0.5), cal.interval([10.0], 0.9)
    assert wide[1] - wide[0] > narrow[1] - narrow[0]
    low, high = GaussianCalibrator().fit(pred, y).interval([10.0], 0.8)
    assert (low + high) / 2 == pytest.approx(10.0)


def test_too_few_residuals_is_refused():
    with pytest.raises(NotEnoughResiduals):
        ResidualCalibrator().fit(np.arange(50.0), np.arange(50.0))
    with pytest.raises(NotEnoughResiduals):
        GaussianCalibrator().fit(np.arange(50.0), np.arange(50.0))


def test_interval_score_known_values():
    # Inside: just the width. Below the interval by 2 at level 0.8: width + (2 / 0.2) * 2.
    assert interval_score([5.0], [0.0], [10.0], 0.8)[0] == 10.0
    assert interval_score([-2.0], [0.0], [10.0], 0.8)[0] == pytest.approx(10.0 + 10 * 2)
    assert interval_score([13.0], [0.0], [10.0], 0.8)[0] == pytest.approx(10.0 + 10 * 3)


def _race_preds(n_races=40, n_drivers=20, seed=0):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0, 30, n_races * n_drivers)
    return pd.DataFrame({"race_index": np.repeat(np.arange(n_races), n_drivers),
                         "driver_id": np.tile([f"d{i}" for i in range(n_drivers)], n_races),
                         "y_pred": pred, "y_true": pred + rng.normal(0, 5, pred.size)})


def test_rolling_intervals_ignore_the_race_itself_and_everything_after():
    """Scrambling outcomes at or after race r must not move the intervals for race r."""
    preds = _race_preds()
    r = 30
    base = rolling_origin_intervals(preds, ResidualCalibrator, levels=(0.8,), warm_races=20)
    scrambled = preds.copy()
    later = scrambled["race_index"] >= r
    scrambled.loc[later, "y_true"] = scrambled.loc[later, "y_true"] + 1e6
    other = rolling_origin_intervals(scrambled, ResidualCalibrator, levels=(0.8,), warm_races=20)
    base, other = (f.sort_values(["race_index", "driver_id"], ignore_index=True) for f in (base, other))
    before = base["race_index"] <= r
    pd.testing.assert_frame_equal(base.loc[before, ["low", "high"]], other.loc[before, ["low", "high"]])
    assert base["race_index"].min() == 20  # the warm-up races are calibration data only


def test_per_race_metrics_and_bin_coverage_shapes():
    intervals = rolling_origin_intervals(_race_preds(), ResidualCalibrator, levels=(0.5, 0.8), warm_races=20)
    per_race = per_race_interval_metrics(intervals, "Binned quantiles")
    assert set(per_race["level"]) == {0.5, 0.8} and per_race["race_index"].nunique() == 20
    assert per_race["coverage"].between(0, 1).all()
    by_bin = coverage_by_prediction_bin(intervals, "Binned quantiles")
    assert set(by_bin["bin"]) == {0, 1, 2, 3} and by_bin["n"].sum() == len(intervals)


def test_out_of_sample_predictions_start_after_the_minimum_history():
    rng = np.random.default_rng(0)
    n_races, n_drivers = 15, 4
    feats = pd.DataFrame(rng.normal(size=(n_races * n_drivers, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    feats["race_index"] = np.repeat(np.arange(n_races), n_drivers)
    feats["season"], feats["round"] = 2024, feats["race_index"] + 1
    feats["driver_id"] = np.tile(list("abcd"), n_races)
    feats["fantasy_points"] = rng.normal(size=len(feats))
    preds = out_of_sample_predictions(feats, make_ridge, min_train_races=10)
    assert sorted(preds["race_index"].unique()) == [10, 11, 12, 13, 14]
    assert out_of_sample_predictions(feats, make_ridge, min_train_races=15).empty
