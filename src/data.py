"""Load Ontario hourly demand (IESO) and Toronto weather (Open-Meteo) into one hourly table.

IESO reports demand in Eastern Standard Time all year (no daylight saving),
with Hour 1 = 00:00-01:00. We index each row by the start of its hour in EST
and pull weather on the same fixed UTC-5 clock so the two line up.
"""
from pathlib import Path

import pandas as pd
import requests

IESO_URL = "https://reports-public.ieso.ca/public/Demand/PUB_Demand_{year}.csv"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
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


def load_weather(start: str, end: str) -> pd.DataFrame:
    params = dict(
        **TORONTO,
        start_date=start,
        end_date=end,
        hourly="temperature_2m,relative_humidity_2m,cloud_cover",
        timezone="Etc/GMT+5",  # fixed UTC-5 = EST, matches IESO's clock
    )
    r = requests.get(WEATHER_URL, params=params, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["datetime"] = pd.to_datetime(df.pop("time"))
    return df.set_index("datetime").rename(
        columns={"temperature_2m": "temp_c", "relative_humidity_2m": "rh_pct", "cloud_cover": "cloud_pct"}
    )


def build_dataset(years=range(2019, 2027), refresh=False) -> pd.DataFrame:
    """Hourly demand + weather, cached to data/ontario_hourly.parquet."""
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
