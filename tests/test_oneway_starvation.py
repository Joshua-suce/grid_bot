"""A grid must not run out of rungs because price only went one way. AUDIT #77/#78.

Measured 2026-08-15 00:00-03:23. DOGE ground +1.1% with an 0.80% total range. Three
sells filled, none came back, the book went 8 orders -> 6, and the four survivors were
all buys clustered 1-1.7% BELOW price. The bot spent three hours with nothing near the
market to trade.

Two independent causes, both here:

  #77  a filled SELL could only re-arm if price FELL a full spacing below it -- the move
       that by definition does not happen in the trend that filled it. Its counter (a
       buy below) never freed either, so the rung was parked for good.

  #78  the regime vote required an absolute 2 of 3 timeframes to agree, and UNCERTAIN
       counted as a vote rather than an abstention. 27 evaluations, 27 "uncertain": the
       30m and 1d ADX readings sat inside the dead band 100% of the time, so only one
       timeframe could ever vote and two was unreachable.
"""

import pandas as pd
import pytest

from grid import GridEngine, GridLevel
from trend_filter import MarketRegime, TrendFilter


# --- #77: the rung comes back on the side price allows ------------------------------

class FakeExchange:
    class exchange:
        @staticmethod
        def amount_to_precision(symbol, amount):
            return f"{amount:.6f}"

        @staticmethod
        def price_to_precision(symbol, price):
            return f"{price:.6f}"


def engine(spacing=0.00024):
    g = GridEngine.__new__(GridEngine)
    g.grid_spacing = spacing
    g.levels = []
    return g


def held(price, side, counter_price, counter_side):
    lvl = GridLevel(price=price, side=side)
    lvl.status = "awaiting_counter"
    lvl.awaiting_price = counter_price
    lvl.awaiting_side = counter_side
    lvl.order_id = None
    return lvl


def test_a_sell_parked_by_a_rising_market_is_released():
    """The measured case: SELL 0.07014 held, price climbs to 0.0704, counter BUY
    0.06979 stays live because price never comes back down."""
    g = engine()
    lvl = held(0.07014, "sell", 0.06979, "buy")
    # the counter is live, so option 1 (counter frees) is unavailable
    busy = GridLevel(price=0.06979, side="buy")
    busy.order_id = "live"
    busy.status = "pending"
    g.levels = [lvl, busy]

    g._release_awaiting_levels(0.07040)

    assert lvl.status == "pending", "the rung stayed parked in a market that left it"
    assert lvl.awaiting_price is None


def busy_at(price, side):
    """A live order occupying a slot, so the counter route is closed and option 2 --
    release by price clearance -- is what the test actually exercises."""
    lvl = GridLevel(price=price, side=side)
    lvl.order_id = "live"
    lvl.status = "pending"
    return lvl


def test_a_released_sell_below_the_market_comes_back_as_a_buy():
    """Re-arming it as a SELL would post a crossing order -- this is why the clearance
    test could not simply be widened."""
    g = engine()
    lvl = held(0.07014, "sell", 0.06979, "buy")
    g.levels = [lvl, busy_at(0.06979, "buy")]

    g._release_awaiting_levels(0.07040)

    assert lvl.side == "buy", "a rung below the market came back as an offer"
    assert lvl.price == 0.07014, "it should return to its own line, not the counter's"


def test_a_released_buy_above_the_market_comes_back_as_a_sell():
    g = engine()
    lvl = held(0.06910, "buy", 0.06934, "sell")
    g.levels = [lvl, busy_at(0.06934, "sell")]

    g._release_awaiting_levels(0.06860)

    assert lvl.status == "pending"
    assert lvl.side == "sell"


def test_a_rung_price_has_not_left_stays_held():
    """The #58 protection survives: a rung price is still sitting on must not re-arm,
    or it refills at one price and takes the whole cap."""
    g = engine()
    lvl = held(0.07014, "sell", 0.06979, "buy")
    g.levels = [lvl, busy_at(0.06979, "buy")]

    g._release_awaiting_levels(0.07015)  # one tick away, not a full spacing

    assert lvl.status == "awaiting_counter"


def test_two_rungs_on_one_line_do_not_both_claim_it():
    g = engine()
    a = held(0.07014, "sell", 0.06979, "buy")
    b = held(0.07014, "sell", 0.06979, "buy")
    g.levels = [a, b, busy_at(0.06979, "buy")]

    g._release_awaiting_levels(0.07040)

    slots = [(l.price, l.side) for l in g.levels if l.status == "pending" and l.order_id is None]
    assert len(slots) == len(set(slots)), f"two rungs released onto the same slot: {slots}"


def test_the_counter_route_still_wins_when_the_slot_is_free():
    """Option 1 is what #61 intended and must take precedence."""
    g = engine()
    lvl = held(0.07014, "sell", 0.06979, "buy")
    g.levels = [lvl]

    g._release_awaiting_levels(0.06970)

    assert lvl.status == "pending"
    assert (lvl.price, lvl.side) == (0.06979, "buy")


# --- #78: abstentions must not veto a verdict ---------------------------------------

def filt(**over):
    kw = dict(ema_fast=9, ema_slow=21, adx_period=14,
              trend_threshold=25.0, range_threshold=20.0)
    kw.update(over)
    return TrendFilter(**kw)


def merge(regimes):
    f = filt()
    f._timeframes = {f"tf{i}": r for i, r in enumerate(regimes)}
    return f._merge_timeframes()


U, R = MarketRegime.UNCERTAIN, MarketRegime.RANGING
UP, DOWN = MarketRegime.UPTREND, MarketRegime.DOWNTREND


def test_the_measured_deadlock_now_reaches_a_verdict():
    """1h=ranging, 30m and 1d both inside the dead band. This returned UNCERTAIN 27
    times out of 27 while price ground +1.1% into the grid."""
    assert merge([R, U, U]) is R


def test_one_trending_timeframe_still_cannot_pause_the_grid():
    """A trend verdict PAUSES the grid. That needs two agreeing timeframes, and the
    abstention fix must not quietly lower the bar (test_merge_minority_trend_does_not_block)."""
    assert merge([UP, U, U]) is U
    assert merge([DOWN, U, U]) is U


def test_all_abstaining_is_still_uncertain():
    assert merge([U, U, U]) is U


def test_conflicting_directions_are_uncertain_however_many_vote():
    assert merge([UP, DOWN, U]) is U
    assert merge([UP, DOWN, R]) is U


def test_two_agreeing_trends_still_win_over_a_ranging_vote():
    assert merge([UP, UP, R]) is UP


def test_ranging_wins_when_it_is_the_only_thing_anyone_is_sure_of():
    assert merge([R, R, UP]) is R
    assert merge([R, UP, U]) is R


def test_the_old_agreeing_cases_are_unchanged():
    """The fix must only affect outcomes abstentions were blocking."""
    assert merge([UP, UP, U]) is UP
    assert merge([R, R, U]) is R
    assert merge([R, R, R]) is R


def test_a_single_timeframe_still_speaks_for_itself():
    assert merge([U]) is U
    assert merge([R]) is R


def test_the_explanation_names_the_abstentions():
    f = filt()
    f._timeframes = {"1h": R, "30m": U, "1d": U}
    f._adx_by_timeframe = {"1h": 14.2, "30m": 23.6, "1d": 25.1}

    line = f.explain()

    assert "1 of 3 timeframe(s) voting" in line


def test_the_explanation_flags_a_total_abstention():
    f = filt()
    f._timeframes = {"1h": U, "30m": U, "1d": U}
    f._adx_by_timeframe = {"1h": 22.0, "30m": 23.6, "1d": 24.1}

    assert "ALL ABSTAINED" in f.explain()
