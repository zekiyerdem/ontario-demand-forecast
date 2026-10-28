"""Unit tests for src/features.py on small synthetic hourly series.

The key property for day-ahead forecasting: every feature for day D must be
computable at the end of day D-1, so demand on day D must never leak into
the features of day D.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURES,
    add_calendar,
    add_lags,
    add_weather,
    clean,
    get_features,
    make_features,
)


def synthetic(start="2024-06-01", days=14, seed=0):
    """Hourly demand + weather frame shaped like the output of src/data.py."""
    idx = pd.date_range(start, periods=24 * days, freq="h", name="datetime")
    rng = np.random.default_rng(seed)
    hour = idx.hour.to_numpy()
    demand = 15000 + 2000 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 100, len(idx))
    temp = 20 + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.5, len(idx))
    return pd.DataFrame(
        {
            "demand_mw": demand,
            "temp_c": temp,
            "rh_pct": rng.uniform(40, 90, len(idx)),
            "cloud_pct": rng.uniform(0, 100, len(idx)),
        },
        index=idx,
    )


# ---------- clean ----------

def test_clean_fills_short_gap_and_restores_hourly_index():
    df = synthetic(days=2)
    gappy = df.drop(df.index[[10, 11]])          # 2-hour hole
    out = clean(gappy)
    assert out.index.equals(df.index)
    assert out.index.name == "datetime"
    assert not out.isna().any().any()
    # linear interpolation between hour 9 and hour 12
    expected = np.interp([10, 11], [9, 12], df["demand_mw"].iloc[[9, 12]])
    np.testing.assert_allclose(out["demand_mw"].iloc[[10, 11]], expected)


def test_clean_does_not_invent_long_gaps():
    df = synthetic(days=2)
    gappy = df.drop(df.index[10:16])             # 6-hour hole, limit is 3
    out = clean(gappy)
    assert out["demand_mw"].iloc[10:16].isna().sum() == 3


# ---------- calendar ----------

def test_calendar_flags():
    idx = pd.date_range("2025-06-27", "2025-07-02 23:00", freq="h")
    out = add_calendar(pd.DataFrame({"demand_mw": 1.0}, index=idx))
    day = out.resample("D").first()
    # Fri 27 Jun, Sat 28, Sun 29, Mon 30, Tue 1 Jul (Canada Day), Wed 2
    assert day["is_weekend"].tolist() == [0, 1, 1, 0, 0, 0]
    assert day["is_holiday"].tolist() == [0, 0, 0, 0, 1, 0]
    assert out["hour"].min() == 0 and out["hour"].max() == 23
    assert set(out["dow"].unique()) <= set(range(7))


def test_calendar_uses_ontario_holidays():
    # Family Day (3rd Monday of Feb) is an Ontario statutory holiday;
    # Saint-Jean-Baptiste (24 Jun) is a Quebec one and must not be flagged.
    # The August Civic Holiday is not statutory in Ontario, so it is not flagged.
    idx = pd.DatetimeIndex(["2025-02-17 12:00", "2025-06-24 12:00", "2025-08-04 12:00", "2025-12-25 12:00"])
    out = add_calendar(pd.DataFrame({"demand_mw": 1.0}, index=idx))
    assert out.loc["2025-02-17 12:00", "is_holiday"] == 1   # Family Day
    assert out.loc["2025-06-24 12:00", "is_holiday"] == 0   # Quebec only
    assert out.loc["2025-08-04 12:00", "is_holiday"] == 0   # Civic Holiday (not statutory)
    assert out.loc["2025-12-25 12:00", "is_holiday"] == 1   # Christmas


def test_seasonal_encoding_is_on_unit_circle():
    out = add_calendar(synthetic(days=3))
    np.testing.assert_allclose(out["doy_sin"] ** 2 + out["doy_cos"] ** 2, 1.0)


# ---------- weather / degree-days ----------

def test_degree_days():
    idx = pd.date_range("2025-01-01", periods=4, freq="h")
    df = pd.DataFrame({"temp_c": [10.0, 18.0, 25.0, -2.0]}, index=idx)
    out = add_weather(df, base_c=18.0)
    assert out["hdd"].tolist() == [8.0, 0.0, 0.0, 20.0]
    assert out["cdd"].tolist() == [0.0, 0.0, 7.0, 0.0]
    # never both heating and cooling in the same hour
    assert ((out["hdd"] > 0) & (out["cdd"] > 0)).sum() == 0


def test_daily_temperature_means():
    df = synthetic(days=3)
    out = add_weather(df)
    day1 = df.loc["2024-06-01", "temp_c"].mean()
    day2 = df.loc["2024-06-02", "temp_c"].mean()
    assert np.allclose(out.loc["2024-06-02", "temp_day_mean"], day2)
    assert np.allclose(out.loc["2024-06-02", "temp_prev_day_mean"], day1)
    assert out.loc["2024-06-01", "temp_prev_day_mean"].isna().all()


# ---------- lags ----------

def test_lag_values():
    df = synthetic(days=10)
    out = add_lags(df)
    t = pd.Timestamp("2024-06-09 17:00")
    y = df["demand_mw"]
    assert out.loc[t, "lag_24"] == y.loc[t - pd.Timedelta(hours=24)]
    assert out.loc[t, "lag_48"] == y.loc[t - pd.Timedelta(hours=48)]
    assert out.loc[t, "lag_168"] == y.loc[t - pd.Timedelta(hours=168)]
    assert out.loc[t, "temp_lag_24"] == df["temp_c"].loc[t - pd.Timedelta(hours=24)]
    assert np.isclose(out.loc[t, "prev_day_mean"], y.loc["2024-06-08"].mean())
    assert np.isclose(out.loc[t, "prev_day_max"], y.loc["2024-06-08"].max())


def test_prev_day_stats_are_constant_within_a_day():
    out = add_lags(synthetic(days=4))
    per_day = out.loc["2024-06-02":].groupby(out.loc["2024-06-02":].index.date)
    assert (per_day["prev_day_mean"].nunique() == 1).all()
    assert (per_day["prev_day_max"].nunique() == 1).all()


@pytest.mark.parametrize("target_day", ["2024-06-09", "2024-06-12"])
def test_no_leakage_from_forecast_day_demand(target_day):
    """Changing demand on day D (and after) must not change day D's features."""
    df = synthetic(days=14)
    base = make_features(df)

    tampered = df.copy()
    tampered.loc[target_day:, "demand_mw"] = tampered.loc[target_day:, "demand_mw"] * 3 + 5000
    after = make_features(tampered)

    day_base = base.loc[target_day, FEATURES]
    day_after = after.loc[target_day, FEATURES]
    pd.testing.assert_frame_equal(day_base, day_after)


def test_lag_features_reach_back_at_least_24_hours():
    """Each demand lag at hour h of day D must come from day D-1 or earlier."""
    df = synthetic(days=10)
    out = add_lags(df)
    day = out.loc["2024-06-09"]
    y = df["demand_mw"]
    for col in ["lag_24", "lag_48", "lag_168"]:
        src_times = day.index - pd.Timedelta(hours=int(col.split("_")[1]))
        assert (src_times.normalize() < pd.Timestamp("2024-06-09")).all()
        np.testing.assert_array_equal(day[col].to_numpy(), y.loc[src_times].to_numpy())


# ---------- full pipeline ----------

def test_make_features_output():
    df = synthetic(days=14)
    out = make_features(df)
    assert set(FEATURES) <= set(out.columns)
    assert not out[FEATURES].isna().any().any()
    # first 7 days are dropped because lag_168 needs a full week of history
    assert out.index.min() == df.index.min() + pd.Timedelta(hours=168)
    assert out.index.max() == df.index.max()
    assert out["demand_mw"].equals(df.loc[out.index, "demand_mw"])


# ---------- forecast cutoff (cutoff_hour) ----------

def test_cutoff_zero_is_the_default_setup():
    df = synthetic(days=14)
    pd.testing.assert_frame_equal(make_features(df), make_features(df, cutoff_hour=0))
    assert get_features(0) == FEATURES


def test_cutoff_hour_out_of_range():
    with pytest.raises(ValueError):
        add_lags(synthetic(days=3), cutoff_hour=24)


def test_cutoff_9_feature_set():
    feats = get_features(9)
    assert "prev_day_mean" not in feats and "prev_day_max" not in feats
    for col in ["lag_48", "lag_168", "prev2_day_mean", "prev2_day_max",
                "last_known_demand", "morning_vs_last_week"]:
        assert col in feats


def test_cutoff_9_feature_values():
    df = synthetic(days=14)
    out = make_features(df, cutoff_hour=9)
    y = df["demand_mw"]
    day = out.loc["2024-06-12"]
    # lag_24 only for hours already observed at 09:00 on D-1
    assert day.loc[day.index.hour < 9, "lag_24"].notna().all()
    assert day.loc[day.index.hour >= 9, "lag_24"].isna().all()
    np.testing.assert_array_equal(day["lag_24"].iloc[:9].to_numpy(), y.loc["2024-06-11 00:00":"2024-06-11 08:00"].to_numpy())
    # day-before-yesterday stats
    assert np.allclose(day["prev2_day_mean"], y.loc["2024-06-10"].mean())
    assert np.allclose(day["prev2_day_max"], y.loc["2024-06-10"].max())
    # last demand known at the cutoff: D-1 08:00
    assert (day["last_known_demand"] == y.loc["2024-06-11 08:00"]).all()
    # D-1 morning (00-08) vs the same hours a week earlier (D-8)
    expected = y.loc["2024-06-11 00:00":"2024-06-11 08:00"].mean() - y.loc["2024-06-04 00:00":"2024-06-04 08:00"].mean()
    assert np.allclose(day["morning_vs_last_week"], expected)


@pytest.mark.parametrize("cutoff_hour", [0, 6, 9, 15])
@pytest.mark.parametrize("target_day", ["2024-06-10", "2024-06-13"])
def test_no_feature_uses_demand_after_the_cutoff(cutoff_hour, target_day):
    """Changing demand from the cutoff time onwards must not change day D's features."""
    df = synthetic(days=16)
    day = pd.Timestamp(target_day)
    cutoff = day if cutoff_hour == 0 else day - pd.Timedelta(days=1) + pd.Timedelta(hours=cutoff_hour)
    feats = get_features(cutoff_hour)
    base = make_features(df, cutoff_hour)

    tampered = df.copy()
    tampered.loc[cutoff:, "demand_mw"] = tampered.loc[cutoff:, "demand_mw"] * 3 + 5000
    after = make_features(tampered, cutoff_hour)
    pd.testing.assert_frame_equal(base.loc[target_day, feats], after.loc[target_day, feats])


def test_midnight_features_would_leak_at_a_9am_cutoff():
    """Sanity check of the test above: the default features do use D-1 data after 09:00."""
    df = synthetic(days=14)
    base = make_features(df)
    tampered = df.copy()
    tampered.loc["2024-06-11 09:00":, "demand_mw"] += 5000
    after = make_features(tampered)
    changed = (base.loc["2024-06-12", FEATURES] != after.loc["2024-06-12", FEATURES]).any()
    assert set(changed[changed].index) == {"lag_24", "prev_day_mean", "prev_day_max"}
