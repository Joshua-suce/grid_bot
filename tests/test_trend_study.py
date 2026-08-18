"""The measurement that decides whether the router is worth switching on. AUDIT #105.

Two claims rest on this code, so it has to be as checkable as the thing it measures:

  * the regime fires often enough to matter -- 28 trend episodes in 62 days
  * the realised reward:risk is ~1.5:1, not the 3:1 the target nominally asks for,
    because the trail closes the trade first

The bug worth pinning is the one the first version had. Walking hourly bars, it
ratcheted the trail up to the bar's HIGH and then tested that same bar's LOW against
the ratcheted value -- an ordering that cannot happen. It manufactured a stop-out on
every bar that ran far enough to reach the target, so every TREND_TAKE_PROFIT_R came
back byte-identical with `target hit 0/66`, and the honest reading of that output would
have been "the target never fires, drop the feature".
"""

import pandas as pd
import pytest

from regime_study import merge, merge_vectorised, verdicts, verify_merge
from trend_filter import MarketRegime
from trend_study import Trade, simulate, stop_distance


def candles(rows, start="2026-06-01", freq="5min"):
    """rows: (open, high, low, close) tuples."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["ts"] = (idx.astype("int64") // 10**6)
    df["time"] = idx
    df["volume"] = 1.0
    return df[["ts", "open", "high", "low", "close", "volume", "time"]]


# --- the vote, fast path against the authoritative one --------------------------------

def test_the_vectorised_merge_matches_trend_filter_everywhere():
    """64 combinations, no sampling. The fast path exists only for speed and must not
    become a second opinion."""
    assert verify_merge() == 64


def test_two_agreeing_trends_are_needed_for_a_trend_verdict():
    """One timeframe is deliberately not enough: a trend verdict pauses the grid."""
    up, rng, unc = MarketRegime.UPTREND, MarketRegime.RANGING, MarketRegime.UNCERTAIN

    assert merge([up, unc, unc]) is not MarketRegime.UPTREND
    assert merge([up, up, unc]) is MarketRegime.UPTREND
    assert merge([up, rng, unc]) is rng


def test_opposing_trends_cancel():
    assert merge([MarketRegime.UPTREND, MarketRegime.DOWNTREND,
                  MarketRegime.UNCERTAIN]) is MarketRegime.UNCERTAIN


def test_the_frame_and_the_list_form_agree_on_a_real_shape():
    frame = pd.DataFrame([[MarketRegime.DOWNTREND, MarketRegime.DOWNTREND,
                           MarketRegime.RANGING]], columns=list("abc"))

    assert merge_vectorised(frame).iloc[0] is MarketRegime.DOWNTREND


# --- per-timeframe verdicts -----------------------------------------------------------

def test_the_warm_up_returns_uncertain_not_a_verdict():
    """_evaluate_timeframe abstains until ema_slow + adx_period candles exist. A study
    that skipped that would score trades the live filter could never have taken."""
    df = candles([(1.0, 1.01, 0.99, 1.0)] * 30)
    v = verdicts(df, ema_fast=5, ema_slow=20, adx_period=14,
                 trend_threshold=25, range_threshold=20)

    assert (v["regime"].iloc[:34] == MarketRegime.UNCERTAIN).all()


def test_a_quiet_oscillating_market_reads_ranging_once_warm():
    """Not perfectly flat -- a zero true range leaves ADX undefined (0/0), which the
    filter reports as UNCERTAIN rather than as a calm market."""
    rows = [(1.0, 1.002, 0.998, 1.001), (1.001, 1.003, 0.999, 1.0)] * 60
    v = verdicts(candles(rows), ema_fast=5, ema_slow=20, adx_period=14,
                 trend_threshold=25, range_threshold=20)

    assert v["regime"].iloc[-1] is MarketRegime.RANGING


# --- trade arithmetic -----------------------------------------------------------------

def test_a_short_that_falls_is_a_win():
    t = Trade("short", pd.Timestamp("2026-06-01", tz="UTC"), 0.10, 0.102, None,
              exit=0.09, exit_time=pd.Timestamp("2026-06-02", tz="UTC"))

    assert t.pnl_pct == pytest.approx(10.0)
    assert t.r_multiple == pytest.approx(5.0)


def test_r_is_measured_against_the_opening_stop():
    t = Trade("long", pd.Timestamp("2026-06-01", tz="UTC"), 1.00, 0.98, None,
              exit=1.06, exit_time=pd.Timestamp("2026-06-02", tz="UTC"))

    assert t.risk == pytest.approx(0.02)
    assert t.r_multiple == pytest.approx(3.0)


def test_the_stop_distance_floor_binds_in_a_quiet_market():
    assert stop_distance(1.0, atr_pct=0.0, mult=2.0, floor_pct=0.005) == pytest.approx(0.005)
    assert stop_distance(1.0, atr_pct=0.01, mult=2.0, floor_pct=0.005) == pytest.approx(0.02)


# --- the causality bug ----------------------------------------------------------------

def _trending_verdicts(index, regime=MarketRegime.UPTREND):
    return pd.Series(regime, index=index, dtype=object)


# A flat warm-up leaves ATR at zero, which makes the stop_loss_pct floor bind for EVERY
# multiplier -- so a 2x and an 8x trail come out identical and R shrinks to the floor,
# putting a 3R target inside the first move. These bars carry a real 1% range so ATR
# settles near 0.01 and the multipliers actually separate.
WARM = 30


def with_warmup(rows):
    """rows are appended after WARM bars of genuine 1%-range chop, plus ONE quiet entry
    bar so the position opens at 1.00 before the bars under test move.

    Entries are taken at the close, so without that bar the trade opens at the close of
    the very move it is supposed to be measuring -- and a trade that never exits is
    never scored at all, which reads as "no trades opened".

    ATR settles at 0.01 (1%), so the 2x opening stop is 0.98, 1R = 0.02 and a 3R target
    sits at 1.06. Returns the walk and a verdict series that is RANGING through the
    warm-up and UPTREND from the entry bar on.
    """
    warm = [(1.00, 1.005, 0.995, 1.00)] * WARM
    entry_bar = [(1.00, 1.001, 0.999, 1.00)]
    walk = candles(warm + entry_bar + list(rows))
    vs = pd.Series(MarketRegime.RANGING, index=walk["time"], dtype=object)
    vs.iloc[WARM:] = MarketRegime.UPTREND
    return walk, vs


def test_a_bar_that_reaches_the_target_is_not_stopped_by_its_own_high():
    """The bug. Ratcheting the trail to this bar's high and then testing this bar's low
    against the ratcheted value invents a stop-out that could not have happened in that
    order -- and it pre-empted every take-profit.

    ATR ~1%, 2x stop -> opening stop 0.98, 1R = 0.02, 3R target = 1.06. One bar runs to
    1.07 (through the target) and pulls back to 1.045. Entering that bar the trail is
    still 0.98, so the target is what gets hit.
    """
    walk, vs = with_warmup([(1.00, 1.07, 1.045, 1.05)] + [(1.05, 1.051, 1.049, 1.05)] * 3)
    frames = {"1h": walk, "30m": walk, "1d": walk}

    trades = simulate("X", take_profit_r=3.0, atr_mult=2.0, floor_pct=0.005,
                      trigger_pct=0.50, ema_fast=5, ema_slow=20, adx_period=14,
                      trend_threshold=25, range_threshold=20, round_trip_fee_pct=0.0,
                      frames=frames, walk=walk, verdict_series=vs)

    assert trades, "no trade opened"
    assert trades[0].reason == "take_profit", (
        f"exited by {trades[0].reason} -- the trail was ratcheted with the same bar's "
        "high before that bar's low was tested")
    assert trades[0].r_multiple == pytest.approx(3.0, abs=0.05)


def test_a_bar_that_only_falls_still_stops_out():
    """The other half: the fix must not make the stop unreachable."""
    walk, vs = with_warmup([(1.00, 1.001, 0.90, 0.91)] + [(0.91, 0.911, 0.909, 0.91)] * 3)
    frames = {"1h": walk, "30m": walk, "1d": walk}

    trades = simulate("X", take_profit_r=3.0, atr_mult=2.0, floor_pct=0.005,
                      trigger_pct=0.50, ema_fast=5, ema_slow=20, adx_period=14,
                      trend_threshold=25, range_threshold=20, round_trip_fee_pct=0.0,
                      frames=frames, walk=walk, verdict_series=vs)

    assert trades[0].reason == "trailing_stop"
    assert trades[0].r_multiple == pytest.approx(-1.0, abs=0.05)


def test_the_stop_wins_a_tie():
    """trend_follower.check_fills resolves a simultaneous hit in favour of the stop, so
    the study must not score that bar as a winner."""
    walk, vs = with_warmup([(1.00, 1.07, 0.97, 1.00)] + [(1.00, 1.001, 0.999, 1.00)] * 3)
    frames = {"1h": walk, "30m": walk, "1d": walk}

    trades = simulate("X", take_profit_r=3.0, atr_mult=2.0, floor_pct=0.005,
                      trigger_pct=0.50, ema_fast=5, ema_slow=20, adx_period=14,
                      trend_threshold=25, range_threshold=20, round_trip_fee_pct=0.0,
                      frames=frames, walk=walk, verdict_series=vs)

    assert trades[0].reason == "trailing_stop"


# --- the knob the study exists to justify ---------------------------------------------

def test_a_wider_trail_lets_a_pullback_survive():
    """Same path, two trail widths. ATR ~1%, so the opening stop is 2% (R = 0.02) and
    the 3R target is 1.06. Price runs to 1.04, gives back to 1.015, then runs to 1.07.

    The 2x trail sits at 1.04 - 0.02 = 1.02 and the pullback takes it out. The 8x trail
    computes 1.04 - 0.08 = 0.96, keeps the ratcheted 0.98, survives, and reaches the
    target. This is the effect measured as 1.46:1 against 2.44:1.
    """
    walk, vs = with_warmup([
        (1.00, 1.040, 0.999, 1.040),
        (1.040, 1.041, 1.015, 1.016),
        (1.016, 1.070, 1.015, 1.070),
        (1.070, 1.071, 1.069, 1.070),
    ])
    frames = {"1h": walk, "30m": walk, "1d": walk}
    common = dict(take_profit_r=3.0, atr_mult=2.0, floor_pct=0.005, trigger_pct=0.50,
                  ema_fast=5, ema_slow=20, adx_period=14, trend_threshold=25,
                  range_threshold=20, round_trip_fee_pct=0.0, frames=frames,
                  walk=walk, verdict_series=vs)

    narrow = simulate("X", trail_mult=2.0, **common)
    wide = simulate("X", trail_mult=8.0, **common)

    assert narrow[0].reason == "trailing_stop"
    assert wide[0].reason == "take_profit"
    assert wide[0].r_multiple > narrow[0].r_multiple


def test_widening_the_trail_does_not_widen_r():
    """If it did, the target would move out by the same factor and the ratio would be
    unchanged while appearing fixed."""
    walk, vs = with_warmup([(1.00, 1.20, 0.999, 1.20)] + [(1.20, 1.201, 1.199, 1.20)] * 2)
    frames = {"1h": walk, "30m": walk, "1d": walk}
    common = dict(take_profit_r=3.0, atr_mult=2.0, floor_pct=0.005, trigger_pct=0.50,
                  ema_fast=5, ema_slow=20, adx_period=14, trend_threshold=25,
                  range_threshold=20, round_trip_fee_pct=0.0, frames=frames,
                  walk=walk, verdict_series=vs)

    narrow = simulate("X", trail_mult=2.0, **common)
    wide = simulate("X", trail_mult=8.0, **common)

    assert narrow[0].stop0 == pytest.approx(wide[0].stop0)
    assert narrow[0].target == pytest.approx(wide[0].target)
    assert narrow[0].risk > 0.01, "ATR collapsed to the floor; the widths cannot separate"


def test_a_flat_market_opens_nothing():
    walk = candles([(1.0, 1.0005, 0.9995, 1.0)] * 40)
    frames = {"1h": walk, "30m": walk, "1d": walk}

    trades = simulate("X", take_profit_r=3.0, atr_mult=2.0, floor_pct=0.005,
                      trigger_pct=0.05, ema_fast=5, ema_slow=20, adx_period=14,
                      trend_threshold=25, range_threshold=20, round_trip_fee_pct=0.0,
                      frames=frames, walk=walk,
                      verdict_series=pd.Series(MarketRegime.RANGING,
                                               index=walk["time"], dtype=object))

    assert trades == []
