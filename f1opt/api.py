"""FastAPI service: predictions, the optimal team under the budget cap, and model metrics."""
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from f1opt import prices as price_mod
from f1opt import serving
from f1opt.config import Settings, get_settings
from f1opt.db import make_engine, make_session_factory
from f1opt.evaluation import bootstrap_ci
from f1opt.optimizer import InfeasibleTeam
from f1opt.recommend import NotEnoughPrices, recommend_team

logger = logging.getLogger(__name__)


class RaceOut(BaseModel):
    season: int
    round: int
    name: str
    date: str
    circuit_id: str
    is_sprint_weekend: bool


class DriverPrediction(BaseModel):
    driver_id: str
    name: str
    constructor_id: str
    constructor_name: str
    predicted_points: float
    price: Optional[float] = None
    price_source: Optional[str] = None
    points_per_million: Optional[float] = None


class PredictionsOut(BaseModel):
    race: RaceOut
    model: str
    trained_through: dict
    drivers: List[DriverPrediction]


class PickedAsset(BaseModel):
    id: str
    name: str
    price: float
    predicted_points: float
    drs: bool = False


class TeamOut(BaseModel):
    race: RaceOut
    budget: float
    total_price: float
    expected_points: float
    drs_driver: str
    drivers: List[PickedAsset]
    constructors: List[PickedAsset]
    price_sources: List[str]
    greedy_baseline_points: Optional[float] = None
    note: str


class ModelMetric(BaseModel):
    model: str
    mae: float
    mae_ci: List[float]
    mse: float
    mse_ci: List[float]
    spearman: float
    spearman_ci: List[float]
    top5_capture: float
    top5_capture_ci: List[float]


class MetricsOut(BaseModel):
    deployed_model: str
    deployed_model_metadata: dict
    backtest_races: int
    backtest_description: str
    models: List[ModelMetric]


def _race_out(race) -> RaceOut:
    return RaceOut(season=race.season, round=race.round, name=race.name, date=race.date.isoformat(),
                   circuit_id=race.circuit_id, is_sprint_weekend=bool(race.is_sprint_weekend))


@lru_cache(maxsize=4)
def _backtest_summary(path: str, mtime: float) -> tuple:
    per_race = pd.read_csv(path)
    cis = {m: bootstrap_ci(per_race, m, n_boot=2000) for m in ("mae", "mse", "spearman", "top5_capture")}
    rows = []
    for model in per_race["model"].unique():
        row = {"model": model}
        for m, ci in cis.items():
            row[m] = float(ci.loc[model, "mean"])
            row[f"{m}_ci"] = [float(ci.loc[model, "ci_low"]), float(ci.loc[model, "ci_high"])]
        rows.append(ModelMetric(**row))
    return per_race["race_index"].nunique(), rows


def create_app(settings: Optional[Settings] = None, session_factory: Optional[sessionmaker] = None,
               store: Optional[serving.ModelStore] = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Fantasy F1 Optimizer", version="1.0")
    app.state.settings = settings
    app.state.session_factory = session_factory or make_session_factory(make_engine(settings.database_url))
    app.state.store = store or serving.ModelStore(settings.model_dir)

    def get_session(request: Request):
        with request.app.state.session_factory() as session:
            yield session

    @app.exception_handler(serving.NoUpcomingRace)
    async def _no_race(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(serving.ModelNotTrained)
    async def _no_model(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.get("/health")
    def health(session: Session = Depends(get_session), request: Request = None):
        session.execute(text("SELECT 1"))
        try:
            request.app.state.store.get()
            model_ready = True
        except serving.ModelNotTrained:
            model_ready = False
        return {"status": "ok", "model_ready": model_ready}

    @app.get("/predictions", response_model=PredictionsOut)
    def predictions(session: Session = Depends(get_session), request: Request = None):
        race, preds = serving.predict_next_race(session, request.app.state.store)
        drv_prices, _ = price_mod.latest_prices(session, race)
        loaded = request.app.state.store.get()
        out = []
        for row in preds.itertuples():
            price, source = drv_prices.get(row.driver_id, (None, None))
            out.append(DriverPrediction(
                driver_id=row.driver_id, name=row.name, constructor_id=row.constructor_id,
                constructor_name=row.constructor_name, predicted_points=round(float(row.predicted_points), 2),
                price=price, price_source=source,
                points_per_million=round(float(row.predicted_points) / price, 3) if price else None))
        return PredictionsOut(race=_race_out(race), model=loaded.metadata["model"],
                              trained_through=loaded.metadata["trained_through"], drivers=out)

    @app.get("/team", response_model=TeamOut)
    def team(budget: Optional[float] = Query(None, gt=0, le=1000, description="budget cap in $M"),
             session: Session = Depends(get_session), request: Request = None):
        budget = budget or request.app.state.settings.budget
        try:
            rec = recommend_team(session, request.app.state.store, budget)
        except NotEnoughPrices as exc:
            raise HTTPException(409, str(exc))
        except InfeasibleTeam as exc:
            raise HTTPException(422, str(exc))
        sel, names, cnames = rec.selection, rec.driver_names, rec.constructor_names
        note = ("Prices are DEMO prices derived from recent form, not the official game's."
                if price_mod.DEMO_SOURCE in rec.price_sources else "Prices from the listed source(s).")
        return TeamOut(
            race=_race_out(rec.race), budget=budget, total_price=round(sel.total_price, 2),
            expected_points=round(sel.expected_points, 2), drs_driver=sel.drs_driver,
            drivers=[PickedAsset(id=a.id, name=names[a.id], price=a.price,
                                 predicted_points=round(a.points, 2), drs=a.id == sel.drs_driver)
                     for a in sel.drivers],
            constructors=[PickedAsset(id=a.id, name=cnames[a.id], price=a.price,
                                      predicted_points=round(a.points, 2)) for a in sel.constructors],
            price_sources=rec.price_sources,
            greedy_baseline_points=round(rec.greedy.expected_points, 2) if rec.greedy else None, note=note)

    @app.get("/metrics", response_model=MetricsOut)
    def metrics(request: Request):
        loaded = request.app.state.store.get()
        path = Path(request.app.state.settings.reports_dir) / "per_race_metrics.csv"
        if not path.exists():
            raise HTTPException(503, f"{path} not found; run scripts/run_evaluation.py")
        n_races, rows = _backtest_summary(str(path), path.stat().st_mtime)
        return MetricsOut(
            deployed_model=loaded.metadata["model"], deployed_model_metadata=loaded.metadata,
            backtest_races=n_races,
            backtest_description=("Expanding-window backtest, one fold per race; per-race metrics averaged, "
                                  "95% bootstrap CI resampling races. See reports/results_table.md."),
            models=rows)

    return app


def app_factory() -> FastAPI:  # uvicorn --factory f1opt.api:app_factory
    return create_app()
