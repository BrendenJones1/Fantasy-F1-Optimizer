"""Thin client for the f1opt FastAPI service. The dashboard holds no data or model of its own."""
import os
from typing import Optional

import requests

DEFAULT_API_URL = "http://localhost:8010"


class ApiError(RuntimeError):
    """The service answered with an error, or could not be reached (``status`` is then None)."""

    def __init__(self, status: Optional[int], detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class ApiClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or os.environ.get("API_URL", DEFAULT_API_URL)).rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        try:
            resp = requests.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ApiError(None, f"Cannot reach the API at {self.base_url} ({type(exc).__name__}).") from exc
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            if isinstance(detail, list):  # FastAPI validation errors are a list of dicts
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
            raise ApiError(resp.status_code, str(detail))
        return resp.json()

    def health(self) -> dict:
        return self._get("/health")

    def predictions(self, level: float) -> dict:
        return self._get("/predictions", {"level": level})

    def team(self, budget: float) -> dict:
        return self._get("/team", {"budget": budget})

    def metrics(self) -> dict:
        return self._get("/metrics")
