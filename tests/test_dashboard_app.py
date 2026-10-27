from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from dashboard.api_client import ApiClient, ApiError  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")

RACE = {"season": 2026, "round": 19, "name": "Mexico City Grand Prix", "date": "2026-10-25",
        "circuit_id": "rodriguez", "is_sprint_weekend": False}


def _driver(i, price_source="fantasy-manual", with_interval=True):
    return {"driver_id": f"d{i}", "name": f"Driver {i}", "constructor_id": f"t{i // 2}",
            "constructor_name": f"Team {i // 2}", "predicted_points": 30.0 - 2 * i,
            "price": 25.0 - i, "price_source": price_source, "points_per_million": 1.2,
            "interval_low": 30.0 - 2 * i - 20 if with_interval else None,
            "interval_high": 30.0 - 2 * i + 15 if with_interval else None}


def predictions(**kw):
    return {"race": RACE, "model": "Ridge", "trained_through": {"season": 2026, "round": 18},
            "interval_level": 0.8 if kw.get("with_interval", True) else None,
            "drivers": [_driver(i, **kw) for i in range(6)]}


TEAM = {"race": RACE, "budget": 100.0, "total_price": 98.5, "expected_points": 140.0, "drs_driver": "d0",
        "drivers": [{"id": f"d{i}", "name": f"Driver {i}", "price": 20.0, "predicted_points": 25.0, "drs": i == 0}
                    for i in range(5)],
        "constructors": [{"id": f"t{i}", "name": f"Team {i}", "price": 15.0, "predicted_points": 40.0, "drs": False}
                         for i in range(2)],
        "price_sources": ["fantasy-manual"], "greedy_baseline_points": 132.0, "note": "Prices from the listed source(s)."}

METRICS = {"deployed_model": "Ridge", "backtest_races": 110, "backtest_description": "Expanding-window backtest.",
           "deployed_model_metadata": {"calibration": {"method": "empirical quantiles", "n_residuals": 3254}},
           "models": [{"model": "Ridge", "mae": 8.86, "mae_ci": [8.36, 9.36], "mse": 147.5, "mse_ci": [133.6, 161.3],
                       "spearman": 0.571, "spearman_ci": [0.539, 0.603], "top5_capture": 0.749,
                       "top5_capture_ci": [0.713, 0.784]}]}


@pytest.fixture
def api(monkeypatch):
    """Stub the HTTP layer: path -> payload, or an ApiError to raise."""
    responses = {"/predictions": predictions(), "/team": TEAM, "/metrics": METRICS}

    def fake_get(self, path, params=None):
        value = responses[path]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(ApiClient, "_get", fake_get)
    st.cache_data.clear()
    return responses


def run():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_renders_predictions_team_and_metrics(api):
    at = run()
    assert not at.exception
    assert "Mexico City Grand Prix 2026" in at.subheader[0].value
    assert [t.label for t in at.tabs] == ["Predictions", "Team", "Model quality"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Expected points"] == "140.0" and metrics["DRS driver"] == "Driver 0"
    assert len(at.dataframe) == 3 and not at.warning  # predictions, team, model metrics


def test_demo_prices_are_flagged(api):
    api["/predictions"] = predictions(price_source="demo")
    at = run()
    assert not at.exception
    assert len(at.warning) == 1 and "demo" in at.warning[0].value.lower()  # once, not per tab


def test_missing_greedy_baseline_is_explained(api):
    api["/team"] = {**TEAM, "greedy_baseline_points": None}
    at = run()
    assert not at.exception
    assert any("greedy baseline" in c.value for c in at.caption)
    assert {m.label: m.delta for m in at.metric}["Expected points"] == ""


def test_missing_intervals_are_explained_not_faked(api):
    api["/predictions"] = predictions(with_interval=False)
    at = run()
    assert not at.exception and any("No prediction intervals" in i.value for i in at.info)


def test_unreachable_api_shows_an_error_and_stops(api):
    api["/predictions"] = ApiError(None, "Cannot reach the API at http://localhost:8010 (ConnectionError).")
    at = run()
    assert not at.exception and "Cannot reach the API" in at.error[0].value
    assert not at.tabs  # nothing else rendered


def test_team_needing_prices_is_an_info_message(api):
    api["/team"] = ApiError(409, "Not enough prices loaded for Mexico City Grand Prix 2026.")
    at = run()
    assert not at.exception and any("Not enough prices" in i.value for i in at.info)
    assert not at.error


def test_interval_slider_reloads_predictions(api, monkeypatch):
    seen = []
    original = ApiClient.predictions

    def spy(self, level):
        seen.append(level)
        return original(self, level)

    monkeypatch.setattr(ApiClient, "predictions", spy)
    at = run()
    at.slider[0].set_value(0.5).run()
    assert not at.exception and seen == [0.8, 0.5]
