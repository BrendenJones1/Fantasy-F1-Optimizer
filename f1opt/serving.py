"""Train the deployed model from the database and predict the next race."""
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from f1opt.db import Constructor, Driver, Race, Result
from f1opt.features import FEATURE_COLUMNS, OUTCOME_COLUMNS, TARGET, build_features
from f1opt.models import make_ridge

logger = logging.getLogger(__name__)

# Chosen from the Goal 1 backtest: ridge had the best MAE and Spearman point estimates,
# statistically tied with LightGBM and the MLP, and is the simplest and fastest to retrain.
DEPLOYED_MODEL_NAME = "Ridge"
MODEL_FILE, METADATA_FILE = "model.joblib", "metadata.json"


class NoUpcomingRace(RuntimeError):
    pass


class ModelNotTrained(RuntimeError):
    pass


def load_weekends(session: Session) -> pd.DataFrame:
    """Completed weekends in the same layout ``f1opt.features.build_features`` expects."""
    stmt = (select(Race.season, Race.round, Race.date, Race.circuit_id, Race.is_sprint_weekend,
                   Result.driver_id, Result.constructor_id, Result.grid, Result.finish_position,
                   Result.position_text, Result.quali_position, Result.quali_pts, Result.sprint_pts,
                   Result.race_pts, Result.fantasy_points)
            .join(Result, Result.race_id == Race.id).where(Race.completed.is_(True)))
    df = pd.read_sql(stmt, session.get_bind())
    df["date"] = pd.to_datetime(df["date"])
    df["is_sprint_weekend"] = df["is_sprint_weekend"].astype(int)
    return df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)


def features_for_upcoming(weekends: pd.DataFrame, season: int, rnd: int, date, circuit_id: str,
                          is_sprint: bool, entries: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for a race that has not happened, built from history only.

    The race's outcome columns are NaN; ``build_features`` never reads them for the race itself
    (enforced by the leakage test), so this is the same computation used in the backtest.
    """
    future = pd.DataFrame({
        "season": season, "round": rnd, "date": pd.Timestamp(date), "circuit_id": circuit_id,
        "driver_id": entries["driver_id"].to_numpy(), "constructor_id": entries["constructor_id"].to_numpy(),
        "is_sprint_weekend": int(is_sprint)})
    for col in OUTCOME_COLUMNS:
        future[col] = np.nan if col != "position_text" else None
    feats = build_features(pd.concat([weekends, future], ignore_index=True))
    return feats[(feats["season"] == season) & (feats["round"] == rnd)].reset_index(drop=True)


def next_race(session: Session) -> Race:
    race = session.scalars(select(Race).where(Race.completed.is_(False))
                           .order_by(Race.season, Race.round)).first()
    if race is None:
        raise NoUpcomingRace("no upcoming race in the calendar; run ingestion")
    return race


def latest_entries(session: Session) -> pd.DataFrame:
    """Drivers and teams from the most recent completed race (proxy for the next entry list)."""
    last = session.scalars(select(Race).where(Race.completed.is_(True))
                           .order_by(Race.season.desc(), Race.round.desc())).first()
    if last is None:
        raise NoUpcomingRace("no completed races yet")
    rows = session.execute(select(Result.driver_id, Result.constructor_id)
                           .where(Result.race_id == last.id)).all()
    return pd.DataFrame(rows, columns=["driver_id", "constructor_id"])


def train_and_save(session: Session, model_dir: Path) -> dict:
    weekends = load_weekends(session)
    if weekends.empty:
        raise ModelNotTrained("no results in the database")
    feats = build_features(weekends)
    model = make_ridge().fit(feats[FEATURE_COLUMNS], feats[TARGET])
    last = feats.sort_values("race_index").iloc[-1]
    meta = {
        "model": DEPLOYED_MODEL_NAME,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_through": {"season": int(last["season"]), "round": int(last["round"])},
        "n_training_rows": int(len(feats)),
        "n_races": int(feats["race_index"].nunique()),
        "features": FEATURE_COLUMNS,
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = model_dir / (MODEL_FILE + ".tmp")
    joblib.dump(model, tmp)
    tmp.replace(model_dir / MODEL_FILE)  # atomic swap so the API never reads a half-written file
    (model_dir / METADATA_FILE).write_text(json.dumps(meta, indent=2))
    logger.info("trained %s on %d rows through %s", DEPLOYED_MODEL_NAME, meta["n_training_rows"],
                meta["trained_through"])
    return meta


@dataclass
class LoadedModel:
    model: object
    metadata: dict
    mtime: float


class ModelStore:
    """Loads the model from disk and reloads when the training job replaces the file."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._loaded: Optional[LoadedModel] = None

    def get(self) -> LoadedModel:
        path = self.model_dir / MODEL_FILE
        if not path.exists():
            raise ModelNotTrained(f"{path} not found; run the training job")
        mtime = path.stat().st_mtime
        if self._loaded is None or self._loaded.mtime != mtime:
            meta = json.loads((self.model_dir / METADATA_FILE).read_text())
            self._loaded = LoadedModel(joblib.load(path), meta, mtime)
        return self._loaded


def predict_next_race(session: Session, store: ModelStore) -> tuple:
    """Return (Race, DataFrame[driver_id, name, constructor_id, constructor_name, predicted_points])."""
    race = next_race(session)
    loaded = store.get()
    feats = features_for_upcoming(load_weekends(session), race.season, race.round, race.date,
                                  race.circuit_id, race.is_sprint_weekend, latest_entries(session))
    out = feats[["driver_id", "constructor_id"]].copy()
    out["predicted_points"] = loaded.model.predict(feats[FEATURE_COLUMNS])
    names = dict(session.execute(select(Driver.driver_id, Driver.name)).all())
    cnames = dict(session.execute(select(Constructor.constructor_id, Constructor.name)).all())
    out["name"] = out["driver_id"].map(names)
    out["constructor_name"] = out["constructor_id"].map(cnames)
    return race, out.sort_values("predicted_points", ascending=False).reset_index(drop=True)
