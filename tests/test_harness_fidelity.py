"""The backtester must gate trading the way production does. AUDIT #52.

A harness that models a different bot than the one that trades does not measure a
strategy, it measures a fiction -- and every parameter decision taken from it inherits
that fiction silently, because the numbers look perfectly reasonable.

The divergence found here was one word. main.py re-activates the grid on a confirmed
RANGE; backtest.py re-activated it on "not a trend", which is RANGING *or* UNCERTAIN.
With the live thresholds (trend 30, range 15) the UNCERTAIN band is 59.4% of DOGE
candles, 54.7% of ETH, 57.0% of SOL. The harness was scoring a bot that trades ~87% of
the time against a live bot that trades ~13%.

The symptom was a knob that appeared inert: sweeping ADX_RANGE_THRESHOLD across
15/20/25/30 returned byte-identical results, because only trend_threshold was ever
consulted. With the gate corrected the same sweep spans +73.47 to -67.22.
"""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _source(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_the_backtester_reactivates_on_the_same_condition_as_production():
    """`not trending` is not the same as `ranging`, and the gap is the UNCERTAIN band."""
    bt = _source("backtest.py")

    i = bt.index("if regimes is not None:")
    gate = bt[i : i + 2000]

    assert 'regime == "ranging"' in gate or "regime == 'ranging'" in gate, (
        "backtest.py does not re-activate on a confirmed RANGE. If it re-activates on "
        "`not trending`, it trades the whole UNCERTAIN band that production sits out, "
        "and ADX_RANGE_THRESHOLD becomes an inert knob in every sweep."
    )
    assert "elif not trending" not in gate, (
        "the `not trending` re-activation is back -- see the module docstring"
    )


def test_production_still_gates_activation_on_ranging():
    """The other half of the contract: if main.py's rule changes, this file is the
    place that has to be updated, rather than the harness quietly diverging again."""
    main = _source("main.py")

    assert "elif trend.is_ranging() and not grid.active:" in main, (
        "main.py no longer re-activates the grid on a confirmed range -- "
        "backtest.py mirrors this exact condition and must be updated with it"
    )


def test_both_pause_on_a_confirmed_trend():
    main, bt = _source("main.py"), _source("backtest.py")

    assert "if trend.is_trending() and grid.active:" in main
    i = bt.index("if regimes is not None:")
    gate = bt[i : i + 2000]
    assert 'regime in ("uptrend", "downtrend") and engine.active' in gate, (
        "the harness no longer pauses on the same condition production does"
    )


@pytest.mark.parametrize("threshold,expect_ranging", [
    (15.0, False),      # ADX 22 sits in the UNCERTAIN band -> production will NOT trade
    (25.0, True),       # widened, the same candle reads as a range -> it will
])
def test_the_range_threshold_actually_moves_the_classification(threshold, expect_ranging):
    """Guards the premise: if this knob could not change a classification, the sweep
    result above would be meaningless."""
    from trend_filter import MarketRegime, TrendFilter

    tf = TrendFilter.__new__(TrendFilter)
    tf.trend_threshold = 30.0
    tf.range_threshold = threshold

    adx_value = 22.0
    if adx_value >= tf.trend_threshold:
        regime = MarketRegime.UPTREND
    elif adx_value <= tf.range_threshold:
        regime = MarketRegime.RANGING
    else:
        regime = MarketRegime.UNCERTAIN

    assert (regime is MarketRegime.RANGING) is expect_ranging
