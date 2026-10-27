import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def engine(tmp_path):
    """A fresh SQLite database with the schema applied by running the real migrations."""
    from alembic import command
    from alembic.config import Config

    from f1opt.db import make_engine

    url = f"sqlite:///{tmp_path / 'test.db'}"
    cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    cfg.set_main_option("script_location", str(Path(__file__).resolve().parents[1] / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    engine = make_engine(url)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    from f1opt.db import make_session_factory

    with make_session_factory(engine)() as s:
        yield s


@pytest.fixture
def long_history(session):
    from f1opt import ingestion
    from tests.fakes import FakeClient, make_season

    """54 races of 12 drivers: enough for the calibrator (30 training races, 200+ residuals)."""
    seasons = {y: make_season(y, 9, 9, n_drivers=12) for y in range(2018, 2024)}
    seasons[2024] = make_season(2024, 0, 3, n_drivers=12)  # an upcoming race to predict
    client = FakeClient(seasons)
    for season in seasons:
        ingestion.ingest_season(session, client, season, refresh=True)
    session.commit()
    return session
