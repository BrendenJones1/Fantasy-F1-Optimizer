import pandas as pd
import pytest
from fastapi.testclient import TestClient

from f1opt import ingestion, prices, serving
from f1opt.api import create_app
from f1opt.config import Settings
from f1opt.db import make_session_factory
from tests.fakes import FakeClient, make_season


@pytest.fixture
def client(engine, session, tmp_path):
    ingestion.ingest_season(session, FakeClient({2024: make_season(2024, 6, 6),
                                                 2025: make_season(2025, 3, 6)}), 2025, True)
    ingestion.ingest_season(session, FakeClient({2024: make_season(2024, 6, 6)}), 2024, True)
    session.commit()
    serving.train_and_save(session, tmp_path / "models")
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame({"model": ["Ridge"] * 20 + ["Naive (rolling-3 mean)"] * 20,
                  "race_index": list(range(20)) * 2, "mae": [8.0] * 20 + [10.0] * 20,
                  "mse": [100.0] * 40, "spearman": [0.5] * 20 + [0.4] * 20,
                  "top5_capture": [0.7] * 40}).to_csv(reports / "per_race_metrics.csv", index=False)
    settings = Settings(database_url="unused", model_dir=tmp_path / "models", reports_dir=reports)
    app = create_app(settings, make_session_factory(engine), serving.ModelStore(settings.model_dir))
    return TestClient(app), session


def test_health(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok", "model_ready": True}


def test_predictions_without_prices(client):
    c, _ = client
    body = c.get("/predictions").json()
    assert (body["race"]["season"], body["race"]["round"]) == (2025, 4)
    assert len(body["drivers"]) == 4 and body["model"] == "Ridge"
    assert all(d["price"] is None for d in body["drivers"])
    # The fixture's history is too short to calibrate, so no intervals are claimed.
    assert body["interval_level"] is None
    assert all(d["interval_low"] is None and d["interval_high"] is None for d in body["drivers"])


def test_team_requires_prices(client):
    c, _ = client
    r = c.get("/team")
    assert r.status_code == 409 and "prices" in r.json()["detail"]


def test_team_needs_enough_priced_candidates(client):
    c, session = client
    race = serving.next_race(session)
    prices.store_prices(session, race, prices.demo_sheet(session, race), prices.DEMO_SOURCE)
    assert c.get("/team").status_code == 409  # fixture has only 4 drivers; a roster needs 5


def test_team_with_prices_respects_budget_and_rules(engine, session, tmp_path):
    client = FakeClient({2025: make_season(2025, 6, 8, n_drivers=12)})
    ingestion.ingest_season(session, client, 2025, True)
    session.commit()
    serving.train_and_save(session, tmp_path / "models")
    race = serving.next_race(session)
    prices.store_prices(session, race, prices.demo_sheet(session, race), prices.DEMO_SOURCE)
    settings = Settings(database_url="unused", model_dir=tmp_path / "models", reports_dir=tmp_path)
    c = TestClient(create_app(settings, make_session_factory(engine), serving.ModelStore(settings.model_dir)))

    resp = c.get("/team", params={"budget": 200})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["drivers"]) == 5 and len(body["constructors"]) == 2
    assert body["total_price"] <= 200
    assert sum(d["drs"] for d in body["drivers"]) == 1
    assert body["price_sources"] == ["demo"] and "DEMO" in body["note"]
    assert body["greedy_baseline_points"] is None or body["expected_points"] >= body["greedy_baseline_points"]
    assert c.get("/team", params={"budget": 1}).status_code == 422
    assert c.get("/team", params={"budget": -5}).status_code == 422


def test_metrics(client):
    c, _ = client
    body = c.get("/metrics").json()
    assert body["deployed_model"] == "Ridge" and body["backtest_races"] == 20
    ridge = next(m for m in body["models"] if m["model"] == "Ridge")
    assert ridge["mae"] == pytest.approx(8.0) and len(ridge["mae_ci"]) == 2


def test_model_not_trained_gives_503(engine, session, tmp_path):
    settings = Settings(database_url="unused", model_dir=tmp_path / "none", reports_dir=tmp_path)
    c = TestClient(create_app(settings, make_session_factory(engine), serving.ModelStore(settings.model_dir)))
    assert c.get("/predictions").status_code in (404, 503)
    assert c.get("/health").json()["model_ready"] is False


def test_metrics_available_before_first_training(engine, tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame({"model": ["Ridge"] * 20, "race_index": list(range(20)), "mae": [8.0] * 20, "mse": [1.0] * 20,
                  "spearman": [0.5] * 20, "top5_capture": [0.7] * 20}).to_csv(reports / "per_race_metrics.csv", index=False)
    settings = Settings(database_url="unused", model_dir=tmp_path / "none", reports_dir=reports)
    c = TestClient(create_app(settings, make_session_factory(engine), serving.ModelStore(settings.model_dir)))
    body = c.get("/metrics").json()
    assert body["deployed_model"] is None and body["models"][0]["model"] == "Ridge"


def test_predictions_include_calibrated_intervals(long_history, engine, tmp_path):
    serving.train_and_save(long_history, tmp_path / "models")
    settings = Settings(database_url="unused", model_dir=tmp_path / "models", reports_dir=tmp_path)
    c = TestClient(create_app(settings, make_session_factory(engine), serving.ModelStore(settings.model_dir)))

    body = c.get("/predictions").json()
    assert body["interval_level"] == 0.8 and len(body["drivers"]) == 12
    assert all(d["interval_low"] < d["interval_high"] for d in body["drivers"])

    def mean_width(level):
        drivers = c.get("/predictions", params={"level": level}).json()["drivers"]
        return sum(d["interval_high"] - d["interval_low"] for d in drivers) / len(drivers)

    assert mean_width(0.5) < mean_width(0.8) < mean_width(0.95)
    assert c.get("/predictions", params={"level": 0.2}).status_code == 422
    assert c.get("/predictions", params={"level": 0.99}).status_code == 422
