"""Feature building for day-ahead hourly demand forecasting.

Setup: at the end of day D-1 we forecast all 24 hours of day D.
So every demand-based feature must come from day D-1 or earlier
(same hour yesterday = lag 24h, same hour last week = lag 168h).
Weather features use the actual observed weather for day D as a
stand-in for a day-ahead weather forecast (an optimistic assumption,
noted in the README).
"""
import holidays
import numpy as np
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    out = df.reindex(full)
    out.index.name = "datetime"
    return out.interpolate(limit=3)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    on_hols = holidays.Canada(prov="ON", years=range(idx.year.min(), idx.year.max() + 1))
    out["hour"] = idx.hour
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    out["doy"] = idx.dayofyear
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_holiday"] = pd.Series(idx.date, index=idx).map(lambda d: d in on_hols).astype(int)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out


def add_weather(df: pd.DataFrame, base_c: float = 18.0) -> pd.DataFrame:
    out = df.copy()
    out["hdd"] = (base_c - out["temp_c"]).clip(lower=0)  # heating degrees
    out["cdd"] = (out["temp_c"] - base_c).clip(lower=0)  # cooling degrees
    daily_temp = out["temp_c"].resample("D").mean()
    out["temp_day_mean"] = daily_temp.reindex(out.index, method="ffill")
    out["temp_prev_day_mean"] = daily_temp.shift(1).reindex(out.index, method="ffill")
    return out


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    y = out["demand_mw"]
    out["lag_24"] = y.shift(24)
    out["lag_48"] = y.shift(48)
    out["lag_168"] = y.shift(168)
    daily = y.resample("D")
    out["prev_day_mean"] = daily.mean().shift(1).reindex(out.index, method="ffill")
    out["prev_day_max"] = daily.max().shift(1).reindex(out.index, method="ffill")
    out["temp_lag_24"] = out["temp_c"].shift(24)
    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_lags(add_weather(add_calendar(clean(df)))).dropna()


FEATURES = [
    "hour", "dow", "month", "is_weekend", "is_holiday", "doy_sin", "doy_cos",
    "temp_c", "hdd", "cdd", "rh_pct", "cloud_pct", "temp_day_mean", "temp_prev_day_mean", "temp_lag_24",
    "lag_24", "lag_48", "lag_168", "prev_day_mean", "prev_day_max",
]
