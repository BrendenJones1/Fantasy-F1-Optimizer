from datetime import date

from f1opt import scheduler, serving
from f1opt.config import Settings
from f1opt.db import make_session_factory
from f1opt import prices
from tests.fakes import FakeClient, make_season


def settings(tmp_path, demo=False):
    return Settings(database_url="unused", model_dir=tmp_path / "models", demo_prices=demo,
                    first_season=date.today().year)


def test_cycle_trains_model_and_stores_demo_prices(engine, tmp_path):
    year = date.today().year
    factory = make_session_factory(engine)
    client = FakeClient({year: make_season(year, 5, 8, n_drivers=12)})
    scheduler.run_cycle(settings(tmp_path, demo=True), factory, client)
    assert (tmp_path / "models" / serving.MODEL_FILE).exists()
    with factory() as s:
        drv, con = prices.latest_prices(s, serving.next_race(s))
        assert len(drv) == 12 and len(con) == 6


def test_cycle_does_not_retrain_without_new_races(engine, tmp_path):
    year = date.today().year
    factory = make_session_factory(engine)
    client = FakeClient({year: make_season(year, 5, 8, n_drivers=12)})
    cfg = settings(tmp_path)
    scheduler.run_cycle(cfg, factory, client)
    model = tmp_path / "models" / serving.MODEL_FILE
    first = model.stat().st_mtime_ns
    scheduler.run_cycle(cfg, factory, client)
    assert model.stat().st_mtime_ns == first


def test_price_failure_does_not_break_cycle(engine, tmp_path, monkeypatch):
    year = date.today().year
    factory = make_session_factory(engine)
    monkeypatch.setenv("F1_FANTASY_COOKIE", "x")
    monkeypatch.setattr(prices.FantasyApiPriceSource, "fetch", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    scheduler.run_cycle(settings(tmp_path), factory, FakeClient({year: make_season(year, 3, 5, n_drivers=12)}))
    assert (tmp_path / "models" / serving.MODEL_FILE).exists()
