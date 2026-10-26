from datetime import date

import pytest
from sqlalchemy import func, select

from f1opt import ingestion
from f1opt.db import (Constructor, Driver, IngestionRun, Race, Result, make_session_factory)
from tests.fakes import FakeClient, make_season


def counts(session):
    return {m.__name__: session.scalar(select(func.count()).select_from(m))
            for m in (Driver, Constructor, Race, Result)}


def test_ingest_is_idempotent(session):
    client = FakeClient({2025: make_season(2025, completed_rounds=2)})
    ingestion.ingest_season(session, client, 2025, refresh=True)
    session.commit()
    first = counts(session)
    assert first == {"Driver": 4, "Constructor": 2, "Race": 3, "Result": 8}

    ingestion.ingest_season(session, client, 2025, refresh=True)
    session.commit()
    assert counts(session) == first


def test_new_results_update_in_place_and_mark_race_completed(session):
    ingestion.ingest_season(session, FakeClient({2025: make_season(2025, 1)}), 2025, True)
    session.commit()
    assert session.scalar(select(func.count()).select_from(Race).where(Race.completed.is_(True))) == 1

    ingestion.ingest_season(session, FakeClient({2025: make_season(2025, 3)}), 2025, True)
    session.commit()
    assert session.scalar(select(func.count()).select_from(Race).where(Race.completed.is_(True))) == 3
    assert counts(session)["Result"] == 12
    assert session.scalar(select(func.count()).select_from(Race)) == 3


def test_fantasy_points_stored_match_scoring(session):
    ingestion.ingest_season(session, FakeClient({2025: make_season(2025, 1)}), 2025, True)
    session.commit()
    # Round 1: order rotates by 1 -> bravo wins from grid 4 (+3 gained, 25 race pts, fastest lap +10), quali P2 = 9.
    row = session.scalar(select(Result).where(Result.driver_id == "bravo"))
    assert row.finish_position == 1 and row.quali_position == 2
    assert row.fantasy_points == 9 + 25 + 3 + 10


def test_run_ingestion_records_success_and_backfills(engine):
    factory = make_session_factory(engine)
    client = FakeClient({2023: make_season(2023, 3), 2024: make_season(2024, 3), 2025: make_season(2025, 1)})
    run = ingestion.run_ingestion(factory, client, first_season=2023, today=date(2025, 6, 1), sleep=lambda s: None)
    assert run.status == "success" and run.new_races == 7
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(Race).where(Race.completed.is_(True))) == 7

    # Second run: 2023 already has data so it is not re-fetched; nothing new.
    client.calls = 0
    run2 = ingestion.run_ingestion(factory, client, first_season=2023, today=date(2025, 6, 1), sleep=lambda s: None)
    assert run2.new_races == 0
    with factory() as s:
        assert [r.status for r in s.scalars(select(IngestionRun))] == ["success", "success"]


def test_run_ingestion_retries_then_succeeds(engine):
    factory = make_session_factory(engine)

    class Flaky(FakeClient):
        failures = 2

        def fetch_schedule(self, season, refresh=False):
            if self.failures:
                self.failures -= 1
                raise ConnectionError("upstream down")
            return super().fetch_schedule(season, refresh)

    sleeps = []
    run = ingestion.run_ingestion(factory, Flaky({2025: make_season(2025, 2)}), first_season=2025,
                                  max_attempts=3, backoff=1.0, today=date(2025, 6, 1), sleep=sleeps.append)
    assert run.status == "success" and run.attempt == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff
    with factory() as s:
        statuses = [r.status for r in s.scalars(select(IngestionRun).order_by(IngestionRun.id))]
        assert statuses == ["failed", "failed", "success"]
        assert "upstream down" in s.scalars(select(IngestionRun)).first().error


def test_run_ingestion_raises_after_exhausting_retries(engine):
    factory = make_session_factory(engine)

    class Down(FakeClient):
        def fetch_schedule(self, season, refresh=False):
            raise ConnectionError("down")

    with pytest.raises(RuntimeError):
        ingestion.run_ingestion(factory, Down({}), first_season=2025, max_attempts=2, backoff=0,
                                today=date(2025, 6, 1), sleep=lambda s: None)
    with factory() as s:
        assert [r.status for r in s.scalars(select(IngestionRun))] == ["failed", "failed"]
