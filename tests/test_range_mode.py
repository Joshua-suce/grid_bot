"""Sizing the ladder from realised range instead of ATR. AUDIT #111.

The range decides where the rungs go, so an over-wide range puts them where price never
travels. 2026-08-18 ran a 2.83% ladder while DOGE moved 0.43% in four hours and filled
once -- the two innermost rungs were 0.474% apart and price never spanned the gap.

ATR(14) x 3.5 is not a bad estimator, just a worse one than the obvious alternative.
Walk-forward on DOGEUSDT 1h, predicting the NEXT 24 hours' actual span, three folds of
363 test bars:

    fold   ATRx3.5 MAE   realised MAE   better by
       1       1.4004%        1.3535%        3.3%
       2       1.3363%        1.2473%        6.7%
       3       1.0183%        0.8310%       18.4%

Better in every fold, and the fitted multiplier is stable across them (4.94-5.07), so
it is a constant rather than a knob.

THE UNITS. That multiplier was fitted against the FULL span; calculate_grid_range
returns a HALF-width. The first version of this applied it directly and produced a
3.91% ladder where ATR gave 2.83% -- twice as wide as intended, and wider than the
thing it was meant to fix. It would have been silently wrong: the range still looks
plausible, it just makes the dormancy worse. Hence test_the_multiplier_is_halved.
"""

import numpy as np
import pandas as pd
import pytest

from grid import calculate_grid_range


def candles(n=200, price=0.07, hourly_range=0.004, seed=0):
    """Bars whose (high-low)/close is exactly `hourly_range`, so the estimator's
    arithmetic is checkable by hand."""
    rng = np.random.default_rng(seed)
    close = np.full(n, price)
    half = price * hourly_range / 2
    return pd.DataFrame({
        "open": close, "high": close + half, "low": close - half, "close": close,
        "volume": np.ones(n),
    })


def width(lo, hi, price):
    return (hi - lo) / price


PRICE = 0.07


# --- the units, which is where this went wrong ----------------------------------------

def test_the_multiplier_is_halved():
    """k was fitted against the FULL next-24h span. This function returns a HALF-width.
    Applying k directly makes the ladder twice as wide as the estimator says."""
    df = candles(hourly_range=0.004)
    lo, hi = calculate_grid_range(df, PRICE, lookback_days=14, timeframe="1h",
                                  mode="realised", realised_window=24,
                                  realised_multiplier=5.0)

    # mean range 0.4% x 5 = 2.0% FULL span -> half-width 1.0% -> full width 2.0%
    assert width(lo, hi, PRICE) == pytest.approx(0.020, rel=0.02)


def test_the_range_is_symmetric_about_price():
    df = candles()
    lo, hi = calculate_grid_range(df, PRICE, lookback_days=14, timeframe="1h",
                                  mode="realised")

    assert PRICE - lo == pytest.approx(hi - PRICE, rel=1e-9)


def test_it_scales_with_measured_volatility():
    quiet = calculate_grid_range(candles(hourly_range=0.002), PRICE, lookback_days=14,
                                 timeframe="1h", mode="realised")
    busy = calculate_grid_range(candles(hourly_range=0.008), PRICE, lookback_days=14,
                                timeframe="1h", mode="realised")

    assert width(*busy, PRICE) == pytest.approx(4 * width(*quiet, PRICE), rel=0.02)


# --- the default must not move --------------------------------------------------------

def test_atr_is_still_the_default():
    from config import Settings

    assert Settings.model_fields["range_mode"].default == "atr"


def test_the_atr_path_is_untouched():
    """An existing config must size its ladder exactly as before."""
    df = candles()
    explicit = calculate_grid_range(df, PRICE, lookback_days=14, atr_multiplier=3.5,
                                    timeframe="1h", mode="atr")
    default = calculate_grid_range(df, PRICE, lookback_days=14, atr_multiplier=3.5,
                                   timeframe="1h")

    assert explicit == default


def test_an_unknown_mode_falls_back_to_atr():
    """A typo in .env must not silently produce a different ladder."""
    df = candles()
    assert (calculate_grid_range(df, PRICE, lookback_days=14, atr_multiplier=3.5,
                                 timeframe="1h", mode="nonsense")
            == calculate_grid_range(df, PRICE, lookback_days=14, atr_multiplier=3.5,
                                    timeframe="1h", mode="atr"))


# --- degenerate input -------------------------------------------------------------------

def test_too_few_candles_still_uses_the_5pct_fallback():
    df = candles(n=10)
    lo, hi = calculate_grid_range(df, PRICE, lookback_days=14, timeframe="1h",
                                  mode="realised")

    assert width(lo, hi, PRICE) == pytest.approx(0.10)


def test_a_dead_flat_window_falls_back_rather_than_collapsing():
    """Zero measured range would give a zero-width ladder -- every rung on one line."""
    flat = candles()
    flat["high"] = flat["close"]
    flat["low"] = flat["close"]
    lo, hi = calculate_grid_range(flat, PRICE, lookback_days=14, atr_multiplier=3.5,
                                  timeframe="1h", mode="realised")

    assert hi > lo, "the ladder collapsed to a single price"


def test_a_short_window_does_not_size_off_two_bars():
    """realised_window=24 with only 13 usable bars should not produce a confident
    number from a third of the evidence."""
    df = candles(n=30)
    lo, hi = calculate_grid_range(df, PRICE, lookback_days=1, atr_multiplier=3.5,
                                  timeframe="1h", mode="realised", realised_window=24)

    assert hi > lo


# --- and it is actually wired ------------------------------------------------------------

def test_main_passes_the_mode_at_both_call_sites():
    from pathlib import Path

    import main

    src = Path(main.__file__).read_text(encoding="utf-8")
    assert src.count("mode=settings.range_mode,") == 2, (
        "the range is computed in two places -- startup and recenter")
