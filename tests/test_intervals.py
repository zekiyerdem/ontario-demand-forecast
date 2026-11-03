"""Unit tests for src/intervals.py (quantile forecast scoring and calibration)."""
import numpy as np
import pytest

from src.intervals import (
    apply_offsets,
    conformal_offsets,
    fix_crossing,
    interval_report,
    pinball_loss,
)


def test_pinball_loss_hand_computed():
    # y - q = +10 (under-forecast) and -10 (over-forecast)
    y, q = np.array([110.0, 90.0]), np.array([100.0, 100.0])
    # alpha=0.9: under-forecast costs 0.9*10, over-forecast costs 0.1*10
    assert pinball_loss(y, q, 0.9) == pytest.approx((9 + 1) / 2)
    assert pinball_loss(y, q, 0.1) == pytest.approx((1 + 9) / 2)
    # alpha=0.5 is half the absolute error
    assert pinball_loss(y, q, 0.5) == pytest.approx(5.0)


def test_pinball_loss_is_minimised_by_the_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20000)
    grid = np.linspace(-2, 2, 81)
    losses = [pinball_loss(y, np.full_like(y, g), 0.9) for g in grid]
    best = grid[int(np.argmin(losses))]
    assert best == pytest.approx(np.quantile(y, 0.9), abs=0.06)


def test_pinball_loss_rejects_bad_alpha():
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 1.0)


def test_fix_crossing_sorts_each_row():
    p10, p50, p90 = np.array([5.0, 1.0]), np.array([4.0, 2.0]), np.array([3.0, 3.0])
    a, b, c = fix_crossing(p10, p50, p90)
    np.testing.assert_array_equal(a, [3.0, 1.0])
    np.testing.assert_array_equal(b, [4.0, 2.0])
    np.testing.assert_array_equal(c, [5.0, 3.0])
    assert np.all(a <= b) and np.all(b <= c)


def test_interval_report_counts():
    y = np.array([0.0, 5.0, 10.0, 20.0])
    r = interval_report(y, lower=np.full(4, 1.0), upper=np.full(4, 12.0))
    assert r["Coverage (%)"] == pytest.approx(50.0)
    assert r["Below P10 (%)"] == pytest.approx(25.0)
    assert r["Above P90 (%)"] == pytest.approx(25.0)
    assert r["Mean width (MW)"] == pytest.approx(11.0)


def test_conformal_offsets_fix_a_biased_band():
    """A band that is too low (like a model missing load growth) gets shifted
    so each side misses about 10% on new data from the same distribution."""
    rng = np.random.default_rng(1)
    def draw(n):
        return rng.normal(100, 10, n)
    # true P10/P90 of N(100,10) are about 87.2 and 112.8; give a band 8 units too low
    lo, hi = 87.2 - 8, 112.8 - 8
    y_cal = draw(5000)
    offsets = conformal_offsets(y_cal, np.full(5000, lo), np.full(5000, hi))
    assert offsets[0] < 0 < offsets[1]  # raise the lower edge, raise the upper edge
    y_new = draw(20000)
    L, H = apply_offsets(np.full(20000, lo), np.full(20000, hi), offsets)
    r = interval_report(y_new, L, H)
    assert r["Below P10 (%)"] == pytest.approx(10, abs=1.0)
    assert r["Above P90 (%)"] == pytest.approx(10, abs=1.0)
    assert r["Coverage (%)"] == pytest.approx(80, abs=1.5)


def test_conformal_offsets_zero_for_a_calibrated_band():
    y = np.arange(1, 101, dtype=float)  # 1..100
    # 10 points below 11 and 10 points above 90
    c_lo, c_hi = conformal_offsets(y, np.full(100, 11.0), np.full(100, 90.0))
    L, H = apply_offsets(np.full(100, 11.0), np.full(100, 90.0), (c_lo, c_hi))
    r = interval_report(y, L, H)
    assert r["Below P10 (%)"] <= 10 and r["Above P90 (%)"] <= 10
    # the finite-sample correction makes the band slightly wider, never narrower
    assert 0 <= c_lo <= 2 and 0 <= c_hi <= 2
