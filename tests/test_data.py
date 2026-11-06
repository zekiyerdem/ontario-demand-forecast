"""Unit tests for the weather download helpers in src/data.py (no network: requests.get is faked)."""
import pandas as pd
import pytest
import requests

from src import data


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def fake_hourly(params, suffix=""):
    idx = pd.date_range(params["start_date"], f"{params['end_date']} 23:00", freq="h")
    names = params["hourly"].split(",")
    return {"hourly": {"time": idx.strftime("%Y-%m-%dT%H:%M").tolist(),
                       **{n: [float(i) for i in range(len(idx))] for n in names}}}


def test_year_chunks_split_a_multi_year_range():
    chunks = list(data._year_chunks("2023-06-15", "2025-02-01"))
    assert [(str(s), str(e)) for s, e in chunks] == [
        ("2023-06-15", "2023-12-31"), ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-02-01")]


def test_observed_weather_is_downloaded_one_year_at_a_time(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        return FakeResponse(fake_hourly(params))

    monkeypatch.setattr(data.requests, "get", fake_get)
    df = data.load_weather("2023-12-30", "2024-01-02")
    assert len(calls) == 2
    assert list(df.columns) == ["temp_c", "rh_pct", "cloud_pct"]
    assert len(df) == 4 * 24 and df.index.is_monotonic_increasing and df.index.is_unique
    assert all(p["timezone"] == "Etc/GMT+5" for p in calls)


def test_download_retries_after_a_failure(monkeypatch):
    attempts = []

    def flaky_get(url, params, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.ConnectionError("cut off")
        return FakeResponse(fake_hourly(params))

    monkeypatch.setattr(data.requests, "get", flaky_get)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    df = data.load_weather("2024-01-01", "2024-01-01")
    assert len(attempts) == 3 and len(df) == 24


def test_download_gives_up_after_the_last_retry(monkeypatch):
    def always_fails(url, params, timeout):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(data.requests, "get", always_fails)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    with pytest.raises(requests.ConnectionError):
        data.load_weather("2024-01-01", "2024-01-01")


@pytest.mark.parametrize("days_before", [1, 2])
def test_forecast_weather_asks_for_the_right_run_and_renames(monkeypatch, days_before):
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params))
        return FakeResponse(fake_hourly(params))

    monkeypatch.setattr(data.requests, "get", fake_get)
    df = data.load_weather_forecast("2025-01-01", "2025-01-02", days_before=days_before)
    url, params = calls[0]
    assert url == data.FORECAST_URL
    assert params["hourly"].split(",") == [f"{v}_previous_day{days_before}" for v in data.WEATHER_VARS]
    assert list(df.columns) == ["temp_c", "rh_pct", "cloud_pct"]
    assert len(df) == 48


def test_forecast_days_before_out_of_range():
    with pytest.raises(ValueError):
        data.load_weather_forecast("2025-01-01", "2025-01-02", days_before=0)
