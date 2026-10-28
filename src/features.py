"""Feature building for day-ahead hourly demand forecasting.

Setup: on day D-1 we forecast all 24 hours of day D. `cutoff_hour` sets
when the forecast is made, i.e. which demand data is already known:

- cutoff_hour=0 (default): at midnight, the end of D-1. All of D-1 is known,
  so same hour yesterday (lag 24h) and yesterday's mean/max can be used.
- cutoff_hour=9: at 09:00 EST on D-1, a realistic day-ahead deadline. Only
  D-1 hours 00:00-08:00 are known. lag_24 is kept only for those hours (NaN
  for the rest), yesterday's mean/max are replaced by the day-before-yesterday
  versions, and two morning features are added: the last known demand at the
  cutoff and how D-1 morning compares with the same hours a week earlier.

In both cases no demand feature uses data from after the cutoff.
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
    on_hols = holidays.Canada(subdiv="ON", years=range(idx.year.min(), idx.year.max() + 1))
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


def add_lags(df: pd.DataFrame, cutoff_hour: int = 0) -> pd.DataFrame:
    if not 0 <= cutoff_hour <= 23:
        raise ValueError("cutoff_hour must be between 0 and 23")
    out = df.copy()
    y = out["demand_mw"]
    daily = y.resample("D")
    out["lag_24"] = y.shift(24)
    out["lag_48"] = y.shift(48)
    out["lag_168"] = y.shift(168)

    def per_day(series_daily):
        return series_daily.reindex(out.index, method="ffill")

    if cutoff_hour == 0:
        out["prev_day_mean"] = per_day(daily.mean().shift(1))
        out["prev_day_max"] = per_day(daily.max().shift(1))
        out["temp_lag_24"] = out["temp_c"].shift(24)
        return out

    # forecast made at D-1 cutoff_hour:00 -> only D-1 hours < cutoff_hour are known
    out["lag_24"] = out["lag_24"].where(out.index.hour < cutoff_hour)
    out["prev2_day_mean"] = per_day(daily.mean().shift(2))
    out["prev2_day_max"] = per_day(daily.max().shift(2))
    morning = y[y.index.hour < cutoff_hour].resample("D")
    out["last_known_demand"] = per_day(morning.last().shift(1))           # D-1, hour cutoff-1
    out["morning_vs_last_week"] = per_day(morning.mean().shift(1) - morning.mean().shift(8))
    out["temp_lag_24"] = out["temp_c"].shift(24)  # weather: observed, used as a forecast proxy
    return out


def make_features(df: pd.DataFrame, cutoff_hour: int = 0) -> pd.DataFrame:
    out = add_lags(add_weather(add_calendar(clean(df))), cutoff_hour)
    # lag_24 is intentionally missing after the cutoff; LightGBM handles NaN
    required = [c for c in get_features(cutoff_hour) if c != "lag_24" or cutoff_hour == 0]
    return out.dropna(subset=required + ["demand_mw"])


FEATURES = [
    "hour", "dow", "month", "is_weekend", "is_holiday", "doy_sin", "doy_cos",
    "temp_c", "hdd", "cdd", "rh_pct", "cloud_pct", "temp_day_mean", "temp_prev_day_mean", "temp_lag_24",
    "lag_24", "lag_48", "lag_168", "prev_day_mean", "prev_day_max",
]

CUTOFF_FEATURES = [
    "hour", "dow", "month", "is_weekend", "is_holiday", "doy_sin", "doy_cos",
    "temp_c", "hdd", "cdd", "rh_pct", "cloud_pct", "temp_day_mean", "temp_prev_day_mean", "temp_lag_24",
    "lag_24", "lag_48", "lag_168", "prev2_day_mean", "prev2_day_max",
    "last_known_demand", "morning_vs_last_week",
]


def get_features(cutoff_hour: int = 0) -> list:
    """Model feature columns for a given forecast cutoff."""
    return list(FEATURES) if cutoff_hour == 0 else list(CUTOFF_FEATURES)
