import numpy as np
import pandas as pd
import pytest

from f1opt.features import FEATURE_COLUMNS, OUTCOME_COLUMNS, build_features


@pytest.fixture(scope="module")
def weekends():
    rng = np.random.default_rng(0)
    drivers = [f"d{i}" for i in range(8)]
    teams = {d: f"t{i // 2}" for i, d in enumerate(drivers)}
    circuits = ["a", "b", "c"]
    rows = []
    for season in (2022, 2023):
        for rnd in range(1, 9):
            order = rng.permutation(len(drivers)) + 1
            for d, pos in zip(drivers, order):
                dnf = rng.random() < 0.1
                rows.append(dict(
                    season=season, round=rnd, date=pd.Timestamp(season, 1, 1) + pd.Timedelta(days=rnd * 14),
                    circuit_id=circuits[rnd % 3], driver_id=d, constructor_id=teams[d],
                    grid=int(rng.integers(1, 9)), finish_position=int(pos),
                    position_text="R" if dnf else str(pos), quali_position=float(rng.integers(1, 9)),
                    is_sprint_weekend=int(rnd % 4 == 0), quali_pts=float(rng.integers(0, 10)),
                    sprint_pts=0.0, race_pts=float(rng.normal(8, 10)),
                    fantasy_points=float(rng.normal(8, 12))))
    return pd.DataFrame(rows)


def test_features_do_not_depend_on_current_or_future_outcomes(weekends):
    """Scramble outcomes of every race >= k; features of races <= k must not change."""
    rng = np.random.default_rng(1)
    base = build_features(weekends)
    ordered = weekends.sort_values(["season", "round"])
    race_keys = ordered[["season", "round"]].drop_duplicates().to_records(index=False).tolist()

    for k in (0, 3, 8, 12):
        season_k, round_k = race_keys[k]
        future = (weekends["season"] > season_k) | ((weekends["season"] == season_k) &
                                                    (weekends["round"] >= round_k))
        scrambled = weekends.copy()
        for col in OUTCOME_COLUMNS:
            if pd.api.types.is_numeric_dtype(scrambled[col]):
                scrambled[col] = scrambled[col].astype(float)
                scrambled.loc[future, col] = rng.normal(100, 50, size=future.sum())
            else:
                scrambled.loc[future, col] = rng.choice(["1", "R", "9"], size=future.sum())
        out = build_features(scrambled)

        upto = (base["race_index"] <= k)
        pd.testing.assert_frame_equal(
            base.loc[upto, FEATURE_COLUMNS].reset_index(drop=True),
            out.loc[upto, FEATURE_COLUMNS].reset_index(drop=True))


def test_first_race_has_no_history(weekends):
    feats = build_features(weekends)
    first = feats[feats["race_index"] == 0]
    for col in ["drv_pts_ma3", "drv_pts_ma10", "drv_season_mean", "team_pts_ma3", "circuit_drv_mean"]:
        assert first[col].isna().all(), col


def test_rolling_form_matches_manual_computation(weekends):
    feats = build_features(weekends)
    d = feats[feats["driver_id"] == "d0"].sort_values("race_index")
    expected = d["fantasy_points"].shift(1).rolling(3, min_periods=1).mean()
    pd.testing.assert_series_equal(d["drv_pts_ma3"], expected, check_names=False)


def test_columns_are_unique(weekends):
    assert build_features(weekends).columns.is_unique
