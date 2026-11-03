"""Scoring and calibration helpers for quantile (P10/P50/P90) forecasts.

- `pinball_loss`: the standard score for a single quantile forecast.
- `fix_crossing`: separately trained quantile models can cross (P10 above P90
  for some hours); sorting the quantiles per row removes that.
- `interval_report`: coverage, misses on each side and average width.
- `conformal_offsets` / `apply_offsets`: split-conformal calibration (CQR).
  On a calibration year the model did not train on, measure how far actual
  demand falls outside each edge of the band, and widen (or narrow) each edge
  by the amount that makes the miss rate on that side equal the target.
"""
import numpy as np
import pandas as pd


def pinball_loss(y, q_pred, alpha: float) -> float:
    """Mean pinball (quantile) loss of forecast `q_pred` for quantile `alpha`."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    e = np.asarray(y, dtype=float) - np.asarray(q_pred, dtype=float)
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))


def fix_crossing(*quantiles):
    """Sort quantile forecasts row by row so that P10 <= P50 <= P90."""
    stacked = np.sort(np.vstack([np.asarray(q, dtype=float) for q in quantiles]), axis=0)
    return tuple(stacked)


def interval_report(y, lower, upper) -> pd.Series:
    """Coverage of the band [lower, upper], misses below/above and mean width."""
    y, lower, upper = (np.asarray(a, dtype=float) for a in (y, lower, upper))
    below, above = y < lower, y > upper
    return pd.Series({
        "Coverage (%)": 100 * np.mean(~below & ~above),
        "Below P10 (%)": 100 * np.mean(below),
        "Above P90 (%)": 100 * np.mean(above),
        "Mean width (MW)": float(np.mean(upper - lower)),
    })


def conformal_offsets(y_cal, lower_cal, upper_cal, tail: float = 0.10):
    """Per-edge offsets so that a share `tail` of calibration points falls
    below the lower edge and `tail` above the upper edge.

    Returns (c_lower, c_upper): the band becomes [lower - c_lower, upper + c_upper].
    A negative offset narrows that side.
    """
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y_cal, lower_cal, upper_cal))
    n = len(y)
    # finite-sample correction used in split conformal prediction
    level = min(1.0, np.ceil((n + 1) * (1 - tail)) / n)
    c_lower = float(np.quantile(lo - y, level, method="higher"))
    c_upper = float(np.quantile(y - hi, level, method="higher"))
    return c_lower, c_upper


def apply_offsets(lower, upper, offsets):
    c_lower, c_upper = offsets
    return np.asarray(lower, dtype=float) - c_lower, np.asarray(upper, dtype=float) + c_upper
