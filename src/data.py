"""Load Ontario hourly demand (IESO) and Toronto weather (Open-Meteo) into one hourly table.

IESO reports demand in Eastern Standard Time all year (no daylight saving),
with Hour 1 = 00:00-01:00. We index each row by the start of its hour in EST
and pull weather on the same fixed UTC-5 clock so the two line up.

Two kinds of weather:
- observed weather (Open-Meteo historical archive), used for training and,
  in notebooks 03-08, as a stand-in for the forecast of day D;
- archived weather *forecasts* (Open-Meteo Previous Runs API): what the
  weather model predicted 1 or 2 days ahead, available from 2024 on.
Weather is downloaded one year at a time with a retry, because one
2019-2026 request can be cut off by some proxies.
"""
import time
from pathlib import Path

import pandas as pd
import requests

IESO_URL = "https://reports-public.ieso.ca/public/Demand/PUB_Demand_{year}.csv"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
TORONTO = dict(latitude=43.65, longitude=-79.38)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_ieso_year(year: int) -> pd.DataFrame:
    df = pd.read_csv(IESO_URL.format(year=year), skiprows=3)
    df["datetime"] = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hour"] - 1, unit="h")
    df = df.rename(columns={"Ontario Demand": "demand_mw", "Market Demand": "market_demand_mw"})
    return df.set_index("datetime")[["demand_mw", "market_demand_mw"]]


def load_ieso(years) -> pd.DataFrame:
    df = pd.concat([load_ieso_year(y) for y in years]).sort_index()
    return df[~df.index.duplicated(keep="last")]


WEATHER_VARS = {"temperature_2m": "temp_c", "relative_humidity_2m": "rh_pct", "cloud_cover": "cloud_pct"}


def _get_hourly(url: str, params: dict, retries: int = 3, wait_s: float = 5.0) -> pd.DataFrame:
    """One Open-Meteo request -> hourly DataFrame indexed by EST datetime. Retries on failure."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            h = r.json()["hourly"]
            break
        except (requests.RequestException, ValueError, KeyError):
            if attempt == retries:
                raise
            time.sleep(wait_s * attempt)
    df = pd.DataFrame(h)
    df["datetime"] = pd.to_datetime(df.pop("time"))
    return df.set_index("datetime")


def _year_chunks(start: str, end: str):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for year in range(start.year, end.year + 1):
        yield max(start, pd.Timestamp(f"{year}-01-01")).date(), min(end, pd.Timestamp(f"{year}-12-31")).date()


def load_weather(start: str, end: str) -> pd.DataFrame:
    """Observed hourly Toronto weather, one request per calendar year."""
    parts = []
    for s, e in _year_chunks(start, end):
        params = dict(
            **TORONTO,
            start_date=str(s),
            end_date=str(e),
            hourly=",".join(WEATHER_VARS),
            timezone="Etc/GMT+5",  # fixed UTC-5 = EST, matches IESO's clock
        )
        parts.append(_get_hourly(WEATHER_URL, params))
    return pd.concat(parts).rename(columns=WEATHER_VARS)


def load_weather_forecast(start: str, end: str, days_before: int = 1) -> pd.DataFrame:
    """Archived day-ahead weather forecasts for Toronto (Open-Meteo Previous Runs API).

    For each hour, the value predicted by the weather-model run issued
    `days_before` days earlier (1 = about a day ahead, 2 = about two days
    ahead). Same column names as `load_weather`, so it can replace the
    observed weather of day D. Available from 2024 on.
    """
    if days_before not in (1, 2, 3, 4, 5, 6, 7):
        raise ValueError("days_before must be between 1 and 7")
    suffix = f"_previous_day{days_before}"
    parts = []
    for s, e in _year_chunks(start, end):
        params = dict(
            **TORONTO,
            start_date=str(s),
            end_date=str(e),
            hourly=",".join(v + suffix for v in WEATHER_VARS),
            timezone="Etc/GMT+5",
        )
        parts.append(_get_hourly(FORECAST_URL, params))
    df = pd.concat(parts)
    return df.rename(columns={v + suffix: name for v, name in WEATHER_VARS.items()})


def build_forecast_weather(start="2024-01-01", end="2025-12-31", days_before=1, refresh=False) -> pd.DataFrame:
    """Archived weather forecasts, cached to data/weather_forecast_d{days_before}.csv."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"weather_forecast_d{days_before}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col="datetime", parse_dates=True)
    df = load_weather_forecast(start, end, days_before)
    df.to_csv(cache)
    return df


def build_dataset(years=range(2019, 2027), refresh=False) -> pd.DataFrame:
    """Hourly demand + weather, cached to data/ontario_hourly.csv."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "ontario_hourly.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col="datetime", parse_dates=True)

    demand = load_ieso(years)
    start, end = demand.index.min().date(), demand.index.max().date()
    weather = load_weather(str(start), str(end))
    df = demand.join(weather, how="left")
    df.to_csv(cache)
    return df


if __name__ == "__main__":
    d = build_dataset(refresh=True)
    print(d.shape, d.index.min(), d.index.max())
    print(d.isna().sum())
    for n in (1, 2):
        f = build_forecast_weather(days_before=n, refresh=True)
        print(f"weather forecast, {n} day(s) ahead:", f.shape, f.index.min(), f.index.max())
