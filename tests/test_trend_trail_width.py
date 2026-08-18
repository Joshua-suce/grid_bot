"""The trail may ride wider than the stop the trade opened with. AUDIT #105.

trend_follower used ONE distance for both, and that coupling is what put the take-profit
out of reach. The target sits at take_profit_r x the stop the trade OPENED with, while
the trail follows one stop-width behind the extreme -- so with both equal, price has to
run R widths without ever giving back a single one.

Measured over 62 days of DOGEUSDT on 5m bars, regime on 30m/1h/1d, with a 3R target:

    trail    n   win%   hit 3R   avgWin  avgLoss    RATIO
       2x   70    34%    3/70    +1.28%   -0.88%    1.46:1
       3x   45    24%    5/45    +2.62%   -1.07%    2.44:1
       4x   35    31%    5/35    +2.48%   -1.20%    2.07:1

The point of the knob is the middle column: at a trail equal to the stop, the target the
user asked for is reached three times in seventy. What it is NOT is free money -- no
width in that sweep made the average trade profitable, and none of those means was
statistically distinguishable from zero at n=70.

1R must keep coming from the OPENING stop even though the same method sets both, or
widening the trail silently widens R and moves the target out with it -- which would
leave the ratio exactly where it started while looking like it had changed.
"""

import pytest

from trend_follower import TrendFollower


class FakeExchange:
    def get_positions(self, symbol):
        return []

    def get_open_orders(self, symbol):
        return []


def follower(**kw):
    tf = TrendFollower(
        FakeExchange(), "DOGEUSDT",
        stop_loss_pct=0.005, trailing_sl_trigger_pct=0.50,
        atr_stop_multiplier=kw.pop("atr_stop_multiplier", 2.0), **kw)
    tf._atr_pct = 0.01                       # 1% ATR -> 2x stop = 2% of price
    return tf


# --- the default must not move -----------------------------------------------------

def test_zero_means_same_as_the_opening_stop():
    """The behaviour this replaces. A config that does not set the knob must trade
    exactly as it did before the knob existed."""
    tf = follower(trail_atr_multiplier=0.0)

    assert tf.trail_atr_multiplier == tf.atr_stop_multiplier == 2.0


def test_an_explicit_width_is_kept():
    tf = follower(trail_atr_multiplier=5.0)

    assert tf.trail_atr_multiplier == 5.0
    assert tf.atr_stop_multiplier == 2.0, "the opening stop must not move with it"


# --- the opening stop still defines 1R ---------------------------------------------

def test_the_opening_stop_uses_the_entry_multiplier_not_the_trail():
    """The whole point. If the first stop came from the wider trail, 1R would widen
    with it and the target would move out by the same factor -- the ratio would be
    unchanged while appearing to have been fixed."""
    narrow = follower(trail_atr_multiplier=2.0)
    wide = follower(trail_atr_multiplier=8.0)
    for tf in (narrow, wide):
        tf._peak_price = 0.0
        tf.update_trailing_sl(0.10)

    assert narrow.get_stop_loss_price() == pytest.approx(wide.get_stop_loss_price()), (
        "widening the trail moved the OPENING stop, so it moved 1R too")
    # 2x ATR of 1% = 2% below 0.10
    assert narrow.get_stop_loss_price() == pytest.approx(0.098)


def test_the_opening_short_stop_uses_the_entry_multiplier():
    narrow, wide = follower(trail_atr_multiplier=2.0), follower(trail_atr_multiplier=8.0)
    for tf in (narrow, wide):
        tf._trough_price = 0.10
        tf.update_trailing_sl_short(0.10)

    assert narrow.get_short_stop_loss_price() == pytest.approx(
        wide.get_short_stop_loss_price())
    assert narrow.get_short_stop_loss_price() == pytest.approx(0.102)


# --- and every move after it uses the trail ----------------------------------------

def test_a_wider_trail_gives_a_running_long_more_room():
    narrow, wide = follower(trail_atr_multiplier=2.0), follower(trail_atr_multiplier=6.0)
    for tf in (narrow, wide):
        tf._peak_price = 0.0
        tf.update_trailing_sl(0.10)          # opening stop, both at 0.098
        tf.update_trailing_sl(0.12)          # price runs; now the trail decides

    assert wide.get_stop_loss_price() < narrow.get_stop_loss_price(), (
        "the wider trail must sit further below the peak")
    assert narrow.get_stop_loss_price() == pytest.approx(0.12 * (1 - 0.02))
    assert wide.get_stop_loss_price() == pytest.approx(0.12 * (1 - 0.06))


def test_a_wider_trail_gives_a_running_short_more_room():
    narrow, wide = follower(trail_atr_multiplier=2.0), follower(trail_atr_multiplier=6.0)
    for tf in (narrow, wide):
        tf._trough_price = 0.10
        tf.update_trailing_sl_short(0.10)
        tf.update_trailing_sl_short(0.08)

    assert wide.get_short_stop_loss_price() > narrow.get_short_stop_loss_price()
    assert narrow.get_short_stop_loss_price() == pytest.approx(0.08 * 1.02)
    assert wide.get_short_stop_loss_price() == pytest.approx(0.08 * 1.06)


# --- the ratchet is still the ratchet ----------------------------------------------

def test_a_wider_trail_still_never_moves_against_a_long():
    """AUDIT #14 restated. Widening the trail must not reintroduce a stop that follows
    the market down -- monotonicity is what makes it a stop at all."""
    tf = follower(trail_atr_multiplier=6.0)
    tf._peak_price = 0.0
    tf.update_trailing_sl(0.10)
    tf.update_trailing_sl(0.14)
    high_water = tf.get_stop_loss_price()

    for price in (0.13, 0.11, 0.09, 0.05):
        tf.update_trailing_sl(price)
        assert tf.get_stop_loss_price() == high_water


def test_a_wider_trail_still_never_moves_against_a_short():
    tf = follower(trail_atr_multiplier=6.0)
    tf._trough_price = 0.10
    tf.update_trailing_sl_short(0.10)
    tf.update_trailing_sl_short(0.06)
    low_water = tf.get_short_stop_loss_price()

    for price in (0.07, 0.09, 0.13):
        tf.update_trailing_sl_short(price)
        assert tf.get_short_stop_loss_price() == low_water


def test_the_percentage_cap_still_binds():
    """trailing_sl_trigger_pct caps how far the trail can sit from the extreme, so an
    enormous multiplier cannot produce a stop that protects nothing."""
    tf = TrendFollower(FakeExchange(), "DOGEUSDT", stop_loss_pct=0.005,
                       trailing_sl_trigger_pct=0.05, atr_stop_multiplier=2.0,
                       trail_atr_multiplier=50.0)
    tf._atr_pct = 0.01
    tf._peak_price = 0.0
    tf.update_trailing_sl(0.10)
    tf.update_trailing_sl(0.12)

    assert tf.get_stop_loss_price() == pytest.approx(0.12 * 0.95)


def test_the_stop_loss_floor_still_binds():
    """A dead-quiet market must not produce a trail so tight that noise closes the
    position -- the floor applies to the trail as much as to the opening stop."""
    tf = follower(trail_atr_multiplier=6.0)
    tf._atr_pct = 0.0
    tf._peak_price = 0.0
    tf.update_trailing_sl(0.10)
    tf.update_trailing_sl(0.20)

    assert tf.get_stop_loss_price() == pytest.approx(0.20 * (1 - 0.005))


# --- config ------------------------------------------------------------------------

def test_the_config_default_changes_nothing():
    from config import settings

    assert settings.trend_trail_atr_multiplier == 0.0


def test_main_passes_the_knob_to_the_follower():
    """A knob that is never wired is worse than no knob: it reads as configured."""
    from pathlib import Path

    import main

    src = Path(main.__file__).read_text(encoding="utf-8")
    at = src.index("trend = TrendFollower(")
    block = src[at:at + 700]

    assert "trail_atr_multiplier=settings.trend_trail_atr_multiplier," in block
    assert "take_profit_r=settings.trend_take_profit_r," in block
