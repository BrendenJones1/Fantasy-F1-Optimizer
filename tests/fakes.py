"""A stand-in for JolpicaClient that serves small hand-built Ergast-style payloads."""
import copy
from typing import Dict, List


def _result(driver, constructor, pos, grid, text=None, fastest=False):
    return {"position": str(pos), "positionText": text or str(pos), "grid": str(grid),
            "Driver": {"driverId": driver, "code": driver[:3].upper(), "givenName": driver.title(),
                       "familyName": "X"},
            "Constructor": {"constructorId": constructor, "name": constructor.title()},
            "FastestLap": {"rank": "1" if fastest else "2"}}


def make_season(season: int, completed_rounds: int, total_rounds: int = 3,
                n_drivers: int = 4) -> Dict[str, List[dict]]:
    if n_drivers == 4:
        drivers = [("alpha", "red"), ("bravo", "red"), ("charlie", "blue"), ("delta", "blue")]
    else:
        drivers = [(f"drv{i}", f"team{i // 2}") for i in range(n_drivers)]
    schedule, results, qualis = [], [], []
    for rnd in range(1, total_rounds + 1):
        race = {"season": str(season), "round": str(rnd), "raceName": f"GP {rnd}",
                "date": f"{season}-0{rnd}-15", "Circuit": {"circuitId": f"circuit{rnd}"}}
        schedule.append(dict(race))
        if rnd <= completed_rounds:
            order = drivers[rnd % n_drivers:] + drivers[:rnd % n_drivers]
            results.append({**race, "Results": [
                _result(d, c, i + 1, n_drivers - i, fastest=(i == 0)) for i, (d, c) in enumerate(order)]})
            qualis.append({**race, "QualifyingResults": [
                {"position": str(i + 1), "Driver": {"driverId": d}} for i, (d, _) in enumerate(drivers)]})
    return {"schedule": schedule, "results": results, "qualifying": qualis, "sprint": []}


class FakeClient:
    def __init__(self, seasons: Dict[int, dict]):
        self.seasons = seasons
        self.calls = 0

    def fetch_schedule(self, season, refresh=False):
        self.calls += 1
        return copy.deepcopy(self.seasons[season]["schedule"])

    def fetch_races(self, path, refresh=False):
        self.calls += 1
        season, kind = path.split("/")
        return copy.deepcopy(self.seasons[int(season)][kind])
