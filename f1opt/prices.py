"""Fantasy prices: a source interface, CSV/API implementations, demo prices, and storage.

Historical Fantasy prices are not available from any public source, so prices are *collected going
forward* into ``driver_prices`` / ``constructor_prices`` and used only by the optimizer, never as a
model feature. When no real prices exist for the next race, ``demo`` prices (derived from recent form)
let the pipeline run end to end; every response states which source was used.
"""
import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from f1opt import serving
from f1opt.db import Constructor, ConstructorPrice, Driver, DriverPrice, Race, upsert, utcnow

logger = logging.getLogger(__name__)

DEMO_SOURCE = "demo"


@dataclass
class PriceSheet:
    drivers: Dict[str, float] = field(default_factory=dict)       # driver_id -> $M
    constructors: Dict[str, float] = field(default_factory=dict)  # constructor_id -> $M


class PriceSource(Protocol):
    name: str

    def fetch(self) -> PriceSheet: ...


class CsvPriceSource:
    """Prices from a CSV with columns ``entity`` (driver|constructor), ``id``, ``price``."""

    def __init__(self, path: Path, name: str = "csv"):
        self.path, self.name = Path(path), name

    def fetch(self) -> PriceSheet:
        sheet = PriceSheet()
        with open(self.path, newline="") as fh:
            for row in csv.DictReader(fh):
                target = sheet.drivers if row["entity"].strip() == "driver" else sheet.constructors
                target[row["id"].strip()] = float(row["price"])
        return sheet


class FantasyApiPriceSource:
    """EXPERIMENTAL and unverified against the live service.

    Reads prices from the Fantasy F1 partner API using a session cookie *you* provide in
    ``F1_FANTASY_COOKIE`` (no anti-bot token spoofing). The payload shape (``drivers``/``constructors``
    lists with ``name`` and ``price``) is taken from the legacy client in this repository and from
    the public API docs, and entries are matched to drivers by family name. Treat as a starting point.
    """
    name = "fantasy-api"

    def __init__(self, season: int, driver_names: Dict[str, str], constructor_names: Dict[str, str],
                 base_url: str = "https://fantasy-api.formula1.com/partner_games/f1",
                 session: Optional[requests.Session] = None):
        self.season, self.base_url = season, base_url
        self.driver_names, self.constructor_names = driver_names, constructor_names
        self.http = session or requests.Session()

    def _get(self, kind: str) -> list:
        cookie = os.environ.get("F1_FANTASY_COOKIE")
        if not cookie:
            raise RuntimeError("F1_FANTASY_COOKIE is not set")
        resp = self.http.get(f"{self.base_url}/game/{self.season}/{kind}",
                             headers={"X-F1-Cookie-Data": cookie}, timeout=20)
        resp.raise_for_status()
        return resp.json().get(kind, [])

    @staticmethod
    def _match(name: str, candidates: Dict[str, str]) -> Optional[str]:
        name = name.lower()
        for ident, full in candidates.items():
            if full.lower() == name or full.lower().split()[-1] in name.split():
                return ident
        return None

    def fetch(self) -> PriceSheet:
        sheet = PriceSheet()
        for item in self._get("drivers"):
            ident = self._match(item.get("name", ""), self.driver_names)
            if ident and item.get("price"):
                sheet.drivers[ident] = float(item["price"])
        for item in self._get("constructors"):
            ident = self._match(item.get("name", ""), self.constructor_names)
            if ident and item.get("price"):
                sheet.constructors[ident] = float(item["price"])
        return sheet


def demo_sheet(session: Session, race: Race) -> PriceSheet:
    """Illustrative prices from trailing form (history only), scaled so the $100M cap binds."""
    weekends = serving.load_weekends(session)
    feats = serving.features_for_upcoming(weekends, race.season, race.round, race.date, race.circuit_id,
                                          race.is_sprint_weekend, serving.latest_entries(session))
    sheet = PriceSheet()
    for row in feats.itertuples():
        form = 0.0 if row.drv_pts_ma10 != row.drv_pts_ma10 else row.drv_pts_ma10
        sheet.drivers[row.driver_id] = round(min(32.0, max(5.0, 12.0 + 0.6 * form)), 1)
        team = 0.0 if row.team_pts_ma10 != row.team_pts_ma10 else row.team_pts_ma10
        sheet.constructors[row.constructor_id] = round(min(30.0, max(5.0, 14.0 + 0.7 * team)), 1)
    return sheet


def store_prices(session: Session, race: Race, sheet: PriceSheet, source: str) -> int:
    known_d = set(session.scalars(select(Driver.driver_id)))
    known_c = set(session.scalars(select(Constructor.constructor_id)))
    now = utcnow()
    drivers = [{"race_id": race.id, "driver_id": k, "source": source, "price": v, "captured_at": now}
               for k, v in sheet.drivers.items() if k in known_d]
    constructors = [{"race_id": race.id, "constructor_id": k, "source": source, "price": v, "captured_at": now}
                    for k, v in sheet.constructors.items() if k in known_c]
    skipped = len(sheet.drivers) + len(sheet.constructors) - len(drivers) - len(constructors)
    if skipped:
        logger.warning("skipped %d prices with unknown ids", skipped)
    n = upsert(session, DriverPrice, drivers, ["race_id", "driver_id", "source"], ["price", "captured_at"])
    n += upsert(session, ConstructorPrice, constructors, ["race_id", "constructor_id", "source"],
                ["price", "captured_at"])
    session.commit()
    return n


def _rank(source: str, captured_at) -> tuple:
    return (source != DEMO_SOURCE, captured_at)  # real sources beat demo; newer beats older


def latest_prices(session: Session, race: Race) -> tuple:
    """({driver_id: (price, source)}, {constructor_id: (price, source)}) preferring real over demo."""
    out = []
    for model, key in ((DriverPrice, "driver_id"), (ConstructorPrice, "constructor_id")):
        best: Dict[str, tuple] = {}
        for row in session.scalars(select(model).where(model.race_id == race.id)):
            ident = getattr(row, key)
            cand = (_rank(row.source, row.captured_at), float(row.price), row.source)
            if ident not in best or cand[0] > best[ident][0]:
                best[ident] = cand
        out.append({k: (v[1], v[2]) for k, v in best.items()})
    return tuple(out)
