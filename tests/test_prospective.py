import json

import pytest
from sqlalchemy import func, select

from f1opt import ingestion, prices, prospective, serving
from f1opt.db import PredictionLog, Race, TeamRecommendationLog
from tests.fakes import FakeClient, make_season


@pytest.fixture
def setup(session, tmp_path):
    ingestion.ingest_season(session, FakeClient({2025: make_season(2025, 6, 8, n_drivers=12)}), 2025, True)
    session.commit()
    serving.train_and_save(session, tmp_path)
    return session, serving.ModelStore(tmp_path)


def test_logging_is_idempotent_and_records_team_only_with_prices(setup):
    session, store = setup
    out = prospective.log_next_race(session, store, 200.0)
    assert out["predictions"] == 12 and out["team_logged"] is False
    race = serving.next_race(session)
    prices.store_prices(session, race, prices.demo_sheet(session, race), prices.DEMO_SOURCE)
    for _ in range(2):
        assert prospective.log_next_race(session, store, 200.0)["team_logged"] is True
    assert session.scalar(select(func.count()).select_from(PredictionLog)) == 12
    assert session.scalar(select(func.count()).select_from(TeamRecommendationLog)) == 1


def test_scoring_waits_for_the_race_then_compares_to_realised_points(setup):
    session, store = setup
    race = serving.next_race(session)
    prices.store_prices(session, race, prices.demo_sheet(session, race), prices.DEMO_SOURCE)
    prospective.log_next_race(session, store, 200.0)
    assert prospective.score_logged_races(session).empty  # race not run yet

    # The race happens: ingest it (7 of 8 rounds complete now).
    ingestion.ingest_season(session, FakeClient({2025: make_season(2025, 7, 8, n_drivers=12)}), 2025, True)
    session.commit()
    scored = prospective.score_logged_races(session)
    assert len(scored) == 1 and scored.loc[0, "round"] == 7
    assert scored.loc[0, "mae"] >= 0 and -1 <= scored.loc[0, "spearman"] <= 1
    assert scored.loc[0, "price_source"] == "demo"
    # Realised team points recomputed independently from the stored payload
    team = session.scalars(select(TeamRecommendationLog)).one()
    payload = json.loads(team.payload)
    assert len(payload["drivers"]) == 5 and payload["drs_driver"] in {d["id"] for d in payload["drivers"]}
    assert scored.loc[0, "team_realised"] == pytest.approx(
        prospective.realised_team_points(payload, *_actuals(session, race)))


def _actuals(session, race):
    from f1opt.db import Result
    rows = session.execute(select(Result.driver_id, Result.constructor_id, Result.fantasy_points)
                           .where(Result.race_id == race.id)).all()
    return {d: p for d, _, p in rows}, {d: c for d, c, _ in rows}
