import pytest
import requests

from dashboard import api_client
from dashboard.api_client import ApiClient, ApiError
from dashboard.views import metrics_frame, predictions_frame, team_frame


class FakeResponse:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _patch_get(monkeypatch, response=None, raises=None):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        if raises:
            raise raises
        return response

    monkeypatch.setattr(api_client.requests, "get", fake_get)
    return calls


def test_client_builds_urls_and_passes_params(monkeypatch):
    calls = _patch_get(monkeypatch, FakeResponse(body={"drivers": []}))
    client = ApiClient("http://api:8000/")
    assert client.predictions(0.8) == {"drivers": []}
    assert client.team(95.0) == {"drivers": []}
    assert calls[0][:2] == ("http://api:8000/predictions", {"level": 0.8})
    assert calls[1][:2] == ("http://api:8000/team", {"budget": 95.0})
    assert calls[0][2] == client.timeout


def test_api_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("API_URL", "http://example:9")
    assert ApiClient().base_url == "http://example:9"
    monkeypatch.delenv("API_URL")
    assert ApiClient().base_url == api_client.DEFAULT_API_URL


def test_http_errors_carry_the_status_and_detail(monkeypatch):
    _patch_get(monkeypatch, FakeResponse(409, {"detail": "Load prices first"}))
    with pytest.raises(ApiError) as err:
        ApiClient().team(100)
    assert err.value.status == 409 and err.value.detail == "Load prices first"


def test_validation_errors_and_non_json_bodies_are_readable(monkeypatch):
    _patch_get(monkeypatch, FakeResponse(422, {"detail": [{"msg": "ensure this value is greater than 0"}]}))
    with pytest.raises(ApiError, match="greater than 0"):
        ApiClient().team(-1)
    _patch_get(monkeypatch, FakeResponse(502, text="Bad Gateway"))
    with pytest.raises(ApiError, match="Bad Gateway"):
        ApiClient().metrics()


def test_unreachable_api_is_an_api_error_without_status(monkeypatch):
    _patch_get(monkeypatch, raises=requests.ConnectionError("refused"))
    with pytest.raises(ApiError) as err:
        ApiClient("http://nowhere").health()
    assert err.value.status is None and "http://nowhere" in err.value.detail


PREDICTIONS = {"interval_level": 0.8, "drivers": [
    {"driver_id": "a", "name": "A", "constructor_id": "x", "constructor_name": "X", "predicted_points": 20.0,
     "interval_low": -5.0, "interval_high": 40.0, "price": 30.0, "price_source": "demo", "points_per_million": 0.67}]}


def test_predictions_frame_columns():
    df = predictions_frame(PREDICTIONS)
    assert list(df.columns)[:5] == ["Driver", "Team", "Predicted", "Low", "High"]
    assert df.loc[0, "Pts per $M"] == 0.67


def test_team_frame_marks_drs_and_lists_both_asset_types():
    payload = {"drivers": [{"name": "A", "price": 30.0, "predicted_points": 20.0, "drs": True},
                           {"name": "B", "price": 20.0, "predicted_points": 10.0, "drs": False}],
               "constructors": [{"name": "X", "price": 25.0, "predicted_points": 30.0}]}
    df = team_frame(payload)
    assert df["Type"].tolist() == ["Driver", "Driver", "Constructor"]
    assert df["DRS"].tolist() == ["x2", "", ""]


def test_metrics_frame_formats_confidence_intervals():
    m = {"model": "Ridge", "mae": 8.86, "mae_ci": [8.36, 9.36], "mse": 147.5, "mse_ci": [133.6, 161.3],
         "spearman": 0.571, "spearman_ci": [0.539, 0.603], "top5_capture": 0.749, "top5_capture_ci": [0.713, 0.784]}
    df = metrics_frame({"models": [m]})
    assert df.loc[0, "MAE"] == "8.86 [8.36, 9.36]" and df.loc[0, "Spearman"] == "0.571 [0.539, 0.603]"
