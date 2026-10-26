"""Client for the Jolpica-F1 API (Ergast-compatible), with on-disk caching and retry.

Completed seasons are cached forever; the in-progress season is re-fetched when
``refresh=True`` so new races are picked up.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100
DEFAULT_CACHE_DIR = Path("data/cache/jolpica")


class JolpicaClient:
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, delay: float = 0.4,
                 max_retries: int = 5, session: requests.Session = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.max_retries = max_retries
        self.session = session or requests.Session()

    def _get(self, path: str, offset: int) -> Dict[str, Any]:
        url = f"{BASE_URL}/{path}.json"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params={"limit": PAGE_SIZE, "offset": offset}, timeout=30)
                if resp.status_code == 429:
                    raise requests.HTTPError("rate limited", response=resp)
                resp.raise_for_status()
                time.sleep(self.delay)
                return resp.json()["MRData"]
            except (requests.RequestException, ValueError, KeyError) as exc:
                wait = min(2 ** attempt, 60)
                logger.warning("GET %s offset=%s failed (%s); retry %s/%s in %ss",
                               url, offset, exc, attempt, self.max_retries, wait)
                time.sleep(wait)
        raise RuntimeError(f"Giving up on {url} after {self.max_retries} attempts")

    def fetch_races(self, path: str, refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch every page of a ``.../results|qualifying|sprint`` endpoint, merged by race.

        The API paginates over result rows, so a race can be split across pages;
        rows are merged back into one entry per round.
        """
        cache_file = self.cache_dir / (path.replace("/", "_") + ".json")
        if cache_file.exists() and not refresh:
            return json.loads(cache_file.read_text())

        races: Dict[str, Dict[str, Any]] = {}
        offset = 0
        while True:
            data = self._get(path, offset)
            total = int(data["total"])
            for race in data["RaceTable"]["Races"]:
                key = race["round"]
                rows_key = next(k for k in race if k.endswith("Results"))
                if key not in races:
                    races[key] = {**race, rows_key: []}
                races[key][rows_key].extend(race[rows_key])
            offset += PAGE_SIZE
            if offset >= total:
                break

        out = [races[k] for k in sorted(races, key=int)]
        cache_file.write_text(json.dumps(out))
        logger.info("Fetched %s (%d races)", path, len(out))
        return out

    def fetch_schedule(self, season: int, refresh: bool = False) -> List[Dict[str, Any]]:
        """Race calendar for a season (includes races that have not happened yet)."""
        cache_file = self.cache_dir / f"{season}.json"
        if cache_file.exists() and not refresh:
            return json.loads(cache_file.read_text())
        races = self._get(str(season), 0)["RaceTable"]["Races"]
        cache_file.write_text(json.dumps(races))
        return races
