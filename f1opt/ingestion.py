"""Idempotent ingestion of race calendars and results into MySQL.

Safe to run repeatedly: every write is an upsert on a unique key, so re-running
the same data produces identical rows. Each attempt is recorded in
``ingestion_runs`` and failures are retried with exponential backoff.
"""
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Iterable, List, Optional

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from f1opt.dataset import build_weekend_table
from f1opt.db import Constructor, Driver, IngestionRun, Race, Result, upsert, utcnow
from f1opt.jolpica import JolpicaClient

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    races_completed: int = 0
    results_upserted: int = 0


def ingest_season(session: Session, client: JolpicaClient, season: int, refresh: bool) -> IngestStats:
    schedule = client.fetch_schedule(season, refresh=refresh)
    weekends = build_weekend_table(client, [season], refresh_seasons=[season] if refresh else [])
    completed_rounds = set(weekends["round"].unique().tolist())

    # Names are not part of the weekend table; read them from the (now cached) results payload.
    drivers, constructors = {}, {}
    for race in client.fetch_races(f"{season}/results"):
        for res in race["Results"]:
            d, c = res["Driver"], res["Constructor"]
            drivers[d["driverId"]] = {"driver_id": d["driverId"], "code": d.get("code"),
                                      "name": f"{d['givenName']} {d['familyName']}"}
            constructors[c["constructorId"]] = {"constructor_id": c["constructorId"], "name": c["name"]}
    upsert(session, Driver, drivers.values(), ["driver_id"], ["code", "name"])
    upsert(session, Constructor, constructors.values(), ["constructor_id"], ["name"])

    upsert(session, Race, [{
        "season": season, "round": int(r["round"]), "name": r["raceName"],
        "date": date.fromisoformat(r["date"]), "circuit_id": r["Circuit"]["circuitId"],
        "is_sprint_weekend": "Sprint" in r, "completed": int(r["round"]) in completed_rounds,
    } for r in schedule], ["season", "round"],
        ["name", "date", "circuit_id", "is_sprint_weekend", "completed"])
    race_ids = {rnd: rid for rid, rnd in session.execute(
        select(Race.id, Race.round).where(Race.season == season))}

    rows = [{
        "race_id": race_ids[int(r.round)], "driver_id": r.driver_id, "constructor_id": r.constructor_id,
        "grid": int(r.grid), "finish_position": int(r.finish_position), "position_text": r.position_text,
        "quali_position": None if pd.isna(r.quali_position) else int(r.quali_position),
        "quali_pts": float(r.quali_pts), "sprint_pts": float(r.sprint_pts),
        "race_pts": float(r.race_pts), "fantasy_points": float(r.fantasy_points),
    } for r in weekends.itertuples()]
    upserted = upsert(session, Result, rows, ["race_id", "driver_id"],
                      ["constructor_id", "grid", "finish_position", "position_text", "quali_position",
                       "quali_pts", "sprint_pts", "race_pts", "fantasy_points"])
    return IngestStats(races_completed=len(completed_rounds), results_upserted=upserted)


def seasons_to_ingest(session: Session, first_season: int, today: Optional[date] = None) -> List[tuple]:
    """(season, refresh) pairs: always refresh the current and previous season; backfill empty ones."""
    year = (today or date.today()).year
    plan = []
    for season in range(first_season, year + 1):
        has_data = session.scalar(select(func.count()).select_from(Race)
                                  .where(Race.season == season, Race.completed.is_(True))) > 0
        if season >= year - 1:
            plan.append((season, True))
        elif not has_data:
            plan.append((season, False))
    return plan


def completed_race_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Race).where(Race.completed.is_(True))) or 0


def run_ingestion(session_factory: sessionmaker, client: JolpicaClient, first_season: int,
                  max_attempts: int = 3, backoff: float = 5.0, today: Optional[date] = None,
                  sleep: Callable[[float], None] = time.sleep) -> IngestionRun:
    """Run one ingestion with logging and retry. Raises after the final failed attempt."""
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        with session_factory() as session:
            run = IngestionRun(status="running", attempt=attempt)
            session.add(run)
            session.commit()
            try:
                plan = seasons_to_ingest(session, first_season, today)
                run.seasons = ",".join(str(s) for s, _ in plan)[:64]
                logger.info("ingestion attempt %d: seasons %s", attempt, run.seasons)
                before = completed_race_count(session)
                total = IngestStats()
                for season, refresh in plan:
                    stats = ingest_season(session, client, season, refresh)
                    total.results_upserted += stats.results_upserted
                    logger.info("season %d: %d completed races, %d results upserted",
                                season, stats.races_completed, stats.results_upserted)
                session.commit()
                run.new_races = completed_race_count(session) - before
                run.results_upserted = total.results_upserted
                run.status, run.finished_at = "success", utcnow()
                session.commit()
                logger.info("ingestion succeeded: %d new races, %d results upserted",
                            run.new_races, run.results_upserted)
                return run
            except Exception as exc:  # noqa: BLE001 - record and retry any failure
                session.rollback()
                last_error = exc
                run.status, run.error, run.finished_at = "failed", repr(exc)[:2000], utcnow()
                session.add(run)
                session.commit()
                logger.exception("ingestion attempt %d/%d failed", attempt, max_attempts)
        if attempt < max_attempts:
            sleep(backoff * 2 ** (attempt - 1))
    raise RuntimeError(f"ingestion failed after {max_attempts} attempts") from last_error
