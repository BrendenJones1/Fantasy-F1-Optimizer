"""Prospective validation: log predictions before a race, score them once the result is known.

A backtest can still be fooled by accidental peeking; predictions written to the database before the
race cannot be. ``log_next_race`` stores the deployed model's predictions and recommended team for
the upcoming race, and ``score_logged_races`` compares the latest logged snapshot with what happened.
"""
import json
import logging
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from f1opt import serving
from f1opt.db import (PredictionLog, Race, Result, TeamRecommendationLog, run_with_retry, upsert, utcnow)
from f1opt.evaluation import race_metrics
from f1opt.optimizer import InfeasibleTeam
from f1opt.recommend import NotEnoughPrices, recommend_team

logger = logging.getLogger(__name__)


def log_next_race(session: Session, store: serving.ModelStore, budget: float) -> dict:
    """Store predictions (always) and the recommended team (when prices exist) for the next race."""
    race, preds = serving.predict_next_race(session, store)
    loaded = store.get()
    version = "{season}-{round}".format(**loaded.metadata["trained_through"])
    now = utcnow()
    rows = [{"race_id": race.id, "driver_id": r.driver_id, "constructor_id": r.constructor_id,
             "model": loaded.metadata["model"], "trained_through": version,
             "predicted_points": float(r.predicted_points), "created_at": now} for r in preds.itertuples()]
    run_with_retry(session, lambda: upsert(
        session, PredictionLog, rows, ["race_id", "driver_id", "model", "trained_through"],
        ["constructor_id", "predicted_points", "created_at"]))

    team_logged = False
    try:
        rec = recommend_team(session, store, budget)
        sel = rec.selection
        payload = {
            "drs_driver": sel.drs_driver, "total_price": sel.total_price,
            "drivers": [{"id": a.id, "price": a.price, "predicted_points": a.points} for a in sel.drivers],
            "constructors": [{"id": a.id, "price": a.price, "predicted_points": a.points}
                             for a in sel.constructors]}
        team_row = {
            "race_id": race.id, "model": loaded.metadata["model"], "trained_through": version,
            "budget": budget, "price_source": ",".join(rec.price_sources),
            "expected_points": sel.expected_points, "payload": json.dumps(payload), "created_at": now}
        run_with_retry(session, lambda: upsert(
            session, TeamRecommendationLog, [team_row],
            ["race_id", "model", "trained_through", "budget", "price_source"],
            ["expected_points", "payload", "created_at"]))
        team_logged = True
    except (NotEnoughPrices, InfeasibleTeam) as exc:
        logger.info("no team logged for %s: %s", race.name, exc)
    logger.info("logged %d predictions for %s %d (team logged: %s)", len(rows), race.name, race.season, team_logged)
    return {"race_id": race.id, "predictions": len(rows), "team_logged": team_logged}


def realised_team_points(payload: dict, actual: dict, constructor_of: dict) -> float:
    """Realised points of a logged team on the reconstructed scoring (DRS doubles one driver).

    A constructor scores the sum of its two drivers' points; qualifying-progress and pit-stop bonuses
    are not available, so this is the repo's scoring basis rather than the official game's total.
    """
    drivers = sum(actual.get(d["id"], 0.0) for d in payload["drivers"])
    drs = actual.get(payload["drs_driver"], 0.0)
    cons = sum(sum(v for k, v in actual.items() if constructor_of.get(k) == c["id"])
               for c in payload["constructors"])
    return drivers + drs + cons


def score_logged_races(session: Session) -> pd.DataFrame:
    """One row per completed race that has a logged prediction: ranking metrics and team outcome."""
    rows = []
    race_ids = session.scalars(select(PredictionLog.race_id).distinct()).all()
    for race_id in race_ids:
        race = session.execute(select(Race).where(Race.id == race_id)
                               .execution_options(populate_existing=True)).scalar_one()
        if not race.completed:
            continue
        logged = pd.read_sql(select(PredictionLog).where(PredictionLog.race_id == race_id),
                             session.get_bind())
        logged = logged.sort_values("created_at").groupby("driver_id").tail(1)  # latest snapshot per driver
        actual = pd.read_sql(select(Result.driver_id, Result.constructor_id, Result.fantasy_points)
                             .where(Result.race_id == race_id), session.get_bind())
        merged = logged.merge(actual, on="driver_id", suffixes=("", "_actual"))
        if len(merged) < 5:
            continue
        m = race_metrics(merged["fantasy_points"].to_numpy(), merged["predicted_points"].to_numpy())
        row = {"season": race.season, "round": race.round, "race": race.name, "n_drivers": len(merged),
               "mae": m["mae"], "spearman": m["spearman"], "top5_capture": m["top5_capture"]}
        team = session.scalars(select(TeamRecommendationLog).where(TeamRecommendationLog.race_id == race_id)
                               .order_by(TeamRecommendationLog.created_at.desc())).first()
        if team:
            payload = json.loads(team.payload)
            act = dict(zip(actual["driver_id"], actual["fantasy_points"]))
            con = dict(zip(actual["driver_id"], actual["constructor_id"]))
            row.update(team_expected=team.expected_points, price_source=team.price_source,
                       team_realised=realised_team_points(payload, act, con))
        rows.append(row)
    return pd.DataFrame(rows)
