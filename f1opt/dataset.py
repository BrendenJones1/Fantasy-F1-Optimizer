"""Build one row per (driver, race weekend) with realised Fantasy points.

The columns produced here describe what *happened* at a weekend (including
qualifying and race outcomes), so they must never be used directly as model
inputs for that same weekend. ``f1opt.features`` derives the pre-race features.
"""
import logging
from typing import Iterable, List

import pandas as pd

from f1opt import scoring
from f1opt.jolpica import JolpicaClient

logger = logging.getLogger(__name__)

WEEKEND_COLUMNS = [
    "season", "round", "date", "circuit_id", "driver_id", "constructor_id",
    "grid", "finish_position", "position_text", "quali_position", "is_sprint_weekend",
    "quali_pts", "sprint_pts", "race_pts", "fantasy_points",
]


def _int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_weekend_table(client: JolpicaClient, seasons: Iterable[int],
                        refresh_seasons: Iterable[int] = ()) -> pd.DataFrame:
    refresh_seasons = set(refresh_seasons)
    rows: List[dict] = []
    for season in seasons:
        refresh = season in refresh_seasons
        races = client.fetch_races(f"{season}/results", refresh=refresh)
        qualis = {r["round"]: r for r in client.fetch_races(f"{season}/qualifying", refresh=refresh)}
        sprints = {r["round"]: r for r in client.fetch_races(f"{season}/sprint", refresh=refresh)}

        for race in races:
            rnd = race["round"]
            results = race["Results"]
            field_size = len(results)
            quali_pos = {q["Driver"]["driverId"]: _int(q["position"], None)
                         for q in qualis.get(rnd, {}).get("QualifyingResults", [])}
            sprint_rows = {s["Driver"]["driverId"]: s
                           for s in sprints.get(rnd, {}).get("SprintResults", [])}

            for res in results:
                driver = res["Driver"]["driverId"]
                position = _int(res["position"])
                grid = _int(res["grid"])
                fastest = res.get("FastestLap", {}).get("rank") == "1"

                race_pts = scoring.session_points(
                    position, res["positionText"], grid, field_size,
                    scoring.RACE_POINTS, fastest, scoring.RACE_FASTEST_LAP_POINTS)
                q_pts = scoring.qualifying_points(quali_pos.get(driver))
                sprint_pts = 0.0
                if driver in sprint_rows:
                    s = sprint_rows[driver]
                    sprint_pts = scoring.session_points(
                        _int(s["position"]), s["positionText"], _int(s["grid"]),
                        len(sprint_rows), scoring.SPRINT_POINTS, False,
                        scoring.SPRINT_FASTEST_LAP_POINTS)

                rows.append({
                    "season": season,
                    "round": int(rnd),
                    "date": race["date"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "driver_id": driver,
                    "constructor_id": res["Constructor"]["constructorId"],
                    "grid": grid,
                    "finish_position": position,
                    "position_text": res["positionText"],
                    "quali_position": quali_pos.get(driver),
                    "is_sprint_weekend": int(rnd in sprints),
                    "quali_pts": q_pts,
                    "sprint_pts": sprint_pts,
                    "race_pts": race_pts,
                    "fantasy_points": q_pts + sprint_pts + race_pts,
                })

    df = pd.DataFrame(rows, columns=WEEKEND_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    logger.info("Weekend table: %d rows, %d races, seasons %s",
                len(df), df.groupby(["season", "round"]).ngroups, sorted(df["season"].unique()))
    return df
