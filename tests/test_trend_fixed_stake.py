"""The trend follower can be sized in USDT, like a grid rung. AUDIT #110.

The grid has committed a fixed amount of own capital per trade since AUDIT #63:
CAPITAL_PER_GRID_USDT=5 at 25x is 125 USDT of notional, and the log says so on every
start. The trend follower had only a percent-of-equity knob, so on a 4939 balance one
trend trade was 988 USDT of notional against a rung's 125 -- roughly eight rungs' worth
of risk in a single position, which is not what "same stake" means.

TREND_CAPITAL_USDT mirrors the grid's knob exactly, including the part AUDIT #63 was
written about: when set it is AUTHORITATIVE, not a floor. The grid's original sizing was
`max(fixed, percent)`, which made the fixed setting silently inert whenever the percent
path was larger -- and on a 4930 balance it always was, so every order ran 3.5x the
configured size while the log called it an override. That mistake is deliberately not
repeated here, and this module pins it.

The caps still sit on top. A fixed stake is a request, not a licence.
"""

import pytest

from trend_follower import TrendFollower


class Inner:
    def amount_to_precision(self, symbol, amount):
        return str(int(float(amount)))


class FakeExchange:
    exchange = Inner()

    def get_positions(self, symbol):
        return []


def follower(**kw):
    kw.setdefault("capital_pct", 0.10)
    kw.setdefault("leverage", 25)
    kw.setdefault("max_exposure_pct", 0.50)
    tf = TrendFollower(FakeExchange(), "DOGEUSDT", stop_loss_pct=0.005,
                       trailing_sl_trigger_pct=0.05, atr_stop_multiplier=2.0, **kw)
    tf._atr_pct = 0.004
    return tf


BALANCE, PRICE = 4939.0, 0.0699


def notional(tf, balance=BALANCE, price=PRICE):
    return tf._entry_qty(balance, price) * price


# --- the fixed stake ------------------------------------------------------------------

def test_a_fixed_stake_is_stake_times_leverage():
    """5 USDT of own capital at 25x is 125 of notional -- one grid rung."""
    tf = follower(capital_usdt=5.0)

    assert notional(tf) == pytest.approx(125.0, rel=0.01)


def test_the_stake_is_what_the_user_actually_commits():
    tf = follower(capital_usdt=5.0)

    assert notional(tf) / tf.leverage == pytest.approx(5.0, rel=0.01)


def test_it_scales_with_the_stake_not_the_balance():
    """The point of a fixed stake: a bigger account does not silently trade bigger."""
    tf = follower(capital_usdt=5.0)

    small = notional(tf, balance=1000.0)
    large = notional(tf, balance=4939.0)

    assert small == pytest.approx(large, rel=0.01)


def test_zero_keeps_the_percent_behaviour():
    """The default must not change how an existing config trades."""
    fixed = follower(capital_usdt=0.0)
    legacy = follower()

    assert notional(fixed) == pytest.approx(notional(legacy))
    assert notional(fixed) > 500, "the percent path should still be the large one here"


# --- authoritative, not a floor (the AUDIT #63 mistake) --------------------------------

def test_the_fixed_stake_wins_even_when_the_percent_path_is_larger():
    """The whole point. 10% of 4939 at 25x is 12,347 before caps; the fixed path asks
    for 125. `max(fixed, percent)` would hand back the larger one and log an override,
    which is how the grid ran 3.5x its configured size for weeks."""
    tf = follower(capital_usdt=5.0, capital_pct=0.10)

    assert notional(tf) == pytest.approx(125.0, rel=0.01)


def test_the_fixed_stake_wins_when_the_percent_path_is_smaller_too():
    """Authoritative in both directions, or it is a floor by another name."""
    tf = follower(capital_usdt=20.0, capital_pct=0.0001)

    assert notional(tf) == pytest.approx(500.0, rel=0.01)


# --- the caps still apply -------------------------------------------------------------

def test_the_exposure_cap_still_clamps_a_large_fixed_stake():
    tf = follower(capital_usdt=1000.0, max_exposure_pct=0.50)

    assert notional(tf) == pytest.approx(BALANCE * 0.50, rel=0.01)


def test_the_position_cap_still_clamps_it():
    tf = follower(capital_usdt=1000.0)
    tf.set_position_limit(0.0, 0.0, 987.0 / PRICE)

    assert notional(tf) <= 987.0 + 1.0


def test_a_stake_under_the_exchange_minimum_opens_nothing():
    """0.1 USDT at 25x is 2.50 of notional, under the 5 USDT floor. A rejected entry is
    better than a -4164 the caller has to interpret."""
    tf = follower(capital_usdt=0.1)

    assert tf._entry_qty(BALANCE, PRICE) == 0.0


# --- the reward:risk shape is unchanged by the size -----------------------------------

def test_shrinking_the_stake_does_not_change_the_ratio():
    """Leverage and size appear on both sides and cancel: 1R stays the same PERCENTAGE
    of stake whatever the stake is. Only the absolute amounts move."""
    big, small = follower(capital_usdt=40.0), follower(capital_usdt=5.0)
    ratios = []
    for tf in (big, small):
        n = notional(tf)
        stake = n / tf.leverage
        one_r = n * max(tf._atr_pct * tf.atr_stop_multiplier, tf.stop_loss_pct)
        ratios.append(one_r / stake)

    assert ratios[0] == pytest.approx(ratios[1], rel=1e-6)
    assert notional(big) == pytest.approx(8 * notional(small), rel=0.01)


# --- config ---------------------------------------------------------------------------

def test_the_declared_default_is_zero():
    from config import Settings

    assert Settings.model_fields["trend_capital_usdt"].default == 0.0


def test_main_passes_the_knob_to_the_follower():
    from pathlib import Path

    import main

    src = Path(main.__file__).read_text(encoding="utf-8")
    at = src.index("trend = TrendFollower(")
    block = src[at:at + 700]

    assert "capital_usdt=settings.trend_capital_usdt," in block


def test_the_configured_stake_matches_a_grid_rung():
    """What .env is actually set to now: the trend trade and one grid rung commit the
    same own capital."""
    from config import settings

    if settings.trend_capital_usdt <= 0:
        pytest.skip("running on the percent path")
    assert settings.trend_capital_usdt == pytest.approx(settings.capital_per_grid_usdt)
