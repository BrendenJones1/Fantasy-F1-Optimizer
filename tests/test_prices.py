import pytest
from sqlalchemy import func, select

from f1opt import ingestion, prices, serving
from f1opt.db import DriverPrice
from tests.fakes import FakeClient, make_season


@pytest.fixture
def populated(session):
    ingestion.ingest_season(session, FakeClient({2025: make_season(2025, 3, 5)}), 2025, refresh=True)
    session.commit()
    return session


def test_csv_source_and_idempotent_storage(populated, tmp_path):
    csv_file = tmp_path / "p.csv"
    csv_file.write_text("entity,id,price\ndriver,alpha,25.5\ndriver,bravo,20\nconstructor,red,30\ndriver,ghost,9\n")
    race = serving.next_race(populated)
    sheet = prices.CsvPriceSource(csv_file).fetch()
    assert prices.store_prices(populated, race, sheet, "csv") == 3  # unknown 'ghost' skipped
    prices.store_prices(populated, race, sheet, "csv")
    assert populated.scalar(select(func.count()).select_from(DriverPrice)) == 2
    drv, con = prices.latest_prices(populated, race)
    assert drv["alpha"] == (25.5, "csv") and con["red"] == (30.0, "csv")


def test_real_prices_override_demo(populated):
    race = serving.next_race(populated)
    prices.store_prices(populated, race, prices.demo_sheet(populated, race), prices.DEMO_SOURCE)
    prices.store_prices(populated, race, prices.PriceSheet(drivers={"alpha": 11.0}), "csv")
    drv, _ = prices.latest_prices(populated, race)
    assert drv["alpha"] == (11.0, "csv")
    assert drv["bravo"][1] == "demo"


def test_demo_prices_use_history_only_and_stay_in_range(populated):
    race = serving.next_race(populated)
    sheet = prices.demo_sheet(populated, race)
    assert set(sheet.drivers) == {"alpha", "bravo", "charlie", "delta"}
    assert all(5.0 <= p <= 32.0 for p in sheet.drivers.values())


def test_fantasy_api_source_maps_names(monkeypatch):
    class Resp:
        def __init__(self, payload): self.payload = payload
        def raise_for_status(self): pass
        def json(self): return self.payload

    class Http:
        def get(self, url, headers=None, timeout=None):
            assert headers["X-F1-Cookie-Data"] == "abc"
            if url.endswith("/drivers"):
                return Resp({"drivers": [{"name": "Max Verstappen", "price": 30.5},
                                         {"name": "Unknown", "price": 1}]})
            return Resp({"constructors": [{"name": "Red Bull Racing", "price": 28}]})

    monkeypatch.setenv("F1_FANTASY_COOKIE", "abc")
    src = prices.FantasyApiPriceSource(2025, {"max_verstappen": "Max Verstappen"},
                                       {"red_bull": "Red Bull"}, session=Http())
    sheet = src.fetch()
    assert sheet.drivers == {"max_verstappen": 30.5}
    assert sheet.constructors == {"red_bull": 28.0}


def test_fantasy_api_source_requires_cookie(monkeypatch):
    monkeypatch.delenv("F1_FANTASY_COOKIE", raising=False)
    with pytest.raises(RuntimeError):
        prices.FantasyApiPriceSource(2025, {}, {}).fetch()


def test_store_prices_retries_on_mysql_deadlock(populated, monkeypatch):
    from sqlalchemy.exc import OperationalError

    race = serving.next_race(populated)
    real_upsert, calls = prices.upsert, {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OperationalError("INSERT", {}, Exception(1213, "Deadlock found"))
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(prices, "upsert", flaky)
    n = prices.store_prices(populated, race, prices.PriceSheet(drivers={"alpha": 12.0}), "csv")
    assert n == 1 and calls["n"] == 3  # failed driver upsert, retried driver + constructor upserts


def test_non_retryable_database_errors_propagate(populated):
    from sqlalchemy.exc import OperationalError
    from f1opt.db import run_with_retry

    def boom():
        raise OperationalError("INSERT", {}, Exception(1045, "Access denied"))

    with pytest.raises(OperationalError):
        run_with_retry(populated, boom, sleep=lambda s: None)
