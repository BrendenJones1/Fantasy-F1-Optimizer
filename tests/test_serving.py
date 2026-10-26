import numpy as np
import pandas as pd
import pytest

from f1opt import ingestion, serving
from f1opt.features import FEATURE_COLUMNS, build_features
from tests.fakes import FakeClient, make_season


@pytest.fixture
def populated(session):
    client = FakeClient({2024: make_season(2024, 6, 6), 2025: make_season(2025, 3, 6)})
    for season in (2024, 2025):
        ingestion.ingest_season(session, client, season, refresh=True)
    session.commit()
    return session


def test_upcoming_features_equal_backtest_features(populated):
    """Serving builds features for an unplayed race exactly as the evaluation does for a played one."""
    weekends = serving.load_weekends(populated)
    full = build_features(weekends)
    last = weekends[(weekends.season == 2025) & (weekends["round"] == 3)]
    history = weekends.drop(last.index)
    entries = last[["driver_id", "constructor_id"]]
    built = serving.features_for_upcoming(history, 2025, 3, last["date"].iloc[0], last["circuit_id"].iloc[0],
                                          bool(last["is_sprint_weekend"].iloc[0]), entries)
    expected = full[(full.season == 2025) & (full["round"] == 3)].reset_index(drop=True)
    pd.testing.assert_frame_equal(built[FEATURE_COLUMNS], expected[FEATURE_COLUMNS])


def test_train_save_load_predict(populated, tmp_path):
    meta = serving.train_and_save(populated, tmp_path)
    assert meta["trained_through"] == {"season": 2025, "round": 3}
    store = serving.ModelStore(tmp_path)
    race, preds = serving.predict_next_race(populated, store)
    assert (race.season, race.round) == (2025, 4)
    assert len(preds) == 4 and preds["predicted_points"].notna().all()
    assert preds["predicted_points"].is_monotonic_decreasing


def test_predict_without_model_raises(populated, tmp_path):
    with pytest.raises(serving.ModelNotTrained):
        serving.predict_next_race(populated, serving.ModelStore(tmp_path))


def test_no_upcoming_race_raises(session):
    client = FakeClient({2025: make_season(2025, 3, 3)})
    ingestion.ingest_season(session, client, 2025, refresh=True)
    session.commit()
    with pytest.raises(serving.NoUpcomingRace):
        serving.next_race(session)
