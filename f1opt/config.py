"""Runtime configuration, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: _env(
        "DATABASE_URL", "mysql+pymysql://f1:f1@127.0.0.1:3307/f1opt"))
    model_dir: Path = field(default_factory=lambda: Path(_env("MODEL_DIR", "models")))
    reports_dir: Path = field(default_factory=lambda: Path(_env("REPORTS_DIR", "reports")))
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", "data/cache/jolpica")))
    first_season: int = field(default_factory=lambda: int(_env("FIRST_SEASON", "2018")))
    budget: float = field(default_factory=lambda: float(_env("BUDGET_CAP", "100.0")))
    demo_prices: bool = field(default_factory=lambda: _env("DEMO_PRICES", "0") == "1")
    ingest_cron: str = field(default_factory=lambda: _env("INGEST_CRON", "0 6 * * *"))  # daily 06:00 UTC


def get_settings() -> Settings:
    return Settings()
