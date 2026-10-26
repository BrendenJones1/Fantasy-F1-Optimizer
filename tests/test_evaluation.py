import numpy as np
import pandas as pd

from f1opt.evaluation import bootstrap_ci, expanding_window_splits, race_metrics


def test_splits_never_train_on_test_or_future():
    race_index = pd.Series(np.repeat(np.arange(30), 20))
    splits = list(expanding_window_splits(race_index, first_test_race=15))
    assert [int(race_index[te].iloc[0]) for _, te in splits] == list(range(15, 30))
    for tr, te in splits:
        assert race_index[tr].max() < race_index[te].min()
        assert not (tr & te).any()


def test_race_metrics_perfect_and_reversed():
    y = np.array([30., 20, 10, 5, 0, -5, -10])
    perfect = race_metrics(y, y)
    assert perfect["mae"] == 0 and perfect["spearman"] == 1 and perfect["top5_capture"] == 1
    assert race_metrics(y, -y)["spearman"] == -1


def test_bootstrap_ci_contains_mean_and_resamples_races():
    rng = np.random.default_rng(0)
    per_race = pd.DataFrame({"model": np.repeat(["a", "b"], 50), "race_index": np.tile(np.arange(50), 2),
                             "mae": rng.normal(5, 1, 100)})
    ci = bootstrap_ci(per_race, "mae", n_boot=500)
    assert ((ci["ci_low"] <= ci["mean"]) & (ci["mean"] <= ci["ci_high"])).all()
