"""Turn predictions and prices into a recommended team (shared by the API and the prediction log)."""
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from f1opt import prices as price_mod
from f1opt import serving
from f1opt.db import Race
from f1opt.optimizer import Asset, TeamSelection, greedy_team, optimize_team


class NotEnoughPrices(RuntimeError):
    pass


@dataclass
class Recommendation:
    race: Race
    selection: TeamSelection
    greedy: Optional[TeamSelection]
    driver_names: dict
    constructor_names: dict
    price_sources: List[str]
    predictions: pd.DataFrame


def recommend_team(session: Session, store: serving.ModelStore, budget: float) -> Recommendation:
    race, preds = serving.predict_next_race(session, store)
    drv_prices, con_prices = price_mod.latest_prices(session, race)

    drivers = [Asset(r.driver_id, drv_prices[r.driver_id][0], float(r.predicted_points))
               for r in preds.itertuples() if r.driver_id in drv_prices]
    # Constructor points: both cars' predicted points. Qualifying-progress and pit-stop bonuses are not modelled.
    con_points = preds.groupby("constructor_id")["predicted_points"].sum()
    constructors = [Asset(cid, con_prices[cid][0], float(con_points[cid]))
                    for cid in con_points.index if cid in con_prices]
    if len(drivers) < 5 or len(constructors) < 2:
        raise NotEnoughPrices(f"Not enough prices loaded for {race.name} {race.season} "
                              f"({len(drivers)} drivers, {len(constructors)} constructors). "
                              "Load prices with scripts/load_prices.py.")
    selection = optimize_team(drivers, constructors, budget)  # may raise InfeasibleTeam
    sources = sorted({s for _, s in drv_prices.values()} | {s for _, s in con_prices.values()})
    return Recommendation(
        race=race, selection=selection, greedy=greedy_team(drivers, constructors, budget),
        driver_names=dict(zip(preds["driver_id"], preds["name"])),
        constructor_names=dict(zip(preds["constructor_id"], preds["constructor_name"])),
        price_sources=sources, predictions=preds)
