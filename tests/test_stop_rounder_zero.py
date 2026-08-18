"""A zero-sized stop leg must never reach the exchange rounder. AUDIT #109.

2026-08-18 06:26:14, twelve seconds after fill #62 opened SHORT 1785 DOGE:

    ERROR | Failed to place/update stop-loss: binanceusdm amount of DOGE/USDT:USDT
            must be greater than minimum amount precision of 1
    WARNING | SIDE BLOCKED | sell — stop-loss missing (AUDIT #50)

Every ten seconds until 06:46:53. Twenty-one minutes with an open short and no stop,
the sell side blocked the whole time.

The cause was mine, from earlier the same day. SL_SCALE_OUT_PCT was bounded gt=0; I
relaxed it to ge=0 and set it to 0, because a 0.5% stop makes the trailing leg fire
inside the grid. With scale=0 and the trail and hard legs at different prices,
build_scale_out_orders fell past its early return and computed

    trail_qty = _round(qty * scale)      # _round(0.0)

and the live rounder is ccxt's amount_to_precision, which RAISES on a zero amount
rather than returning it.

Why the suite missed it: every existing test passed either no rounder or a plain
lambda, both of which return 0 quite happily. The fixture was more permissive than the
thing it stood for, so the one configuration I had just enabled was untested in the only
way that mattered. These tests use a rounder that raises exactly like ccxt's.
"""

import pytest

from grid import MIN_NOTIONAL_USDT
from main import build_scale_out_orders

TRAIL, HARD = 0.07128, 0.07135          # different levels: the live shape
STEP = 1.0                              # DOGE trades in whole coins


def ccxt_like(amount):
    """amount_to_precision as ccxt actually behaves: anything that does not survive the
    amount step is an error, not a zero."""
    rounded = int(float(amount) // STEP) * STEP
    if rounded <= 0:
        raise ValueError(
            "binanceusdm amount of DOGE/USDT:USDT must be greater than minimum "
            "amount precision of 1")
    return rounded


def kinds(orders):
    return [k for k, _, _ in orders]


def build(side, qty, scale, **kw):
    kw.setdefault("rounder", ccxt_like)
    kw.setdefault("min_notional", MIN_NOTIONAL_USDT)
    return build_scale_out_orders(side, qty, scale, TRAIL, HARD, **kw)


# --- the live failure -----------------------------------------------------------------

def test_a_zero_scale_out_does_not_reach_the_rounder():
    """The exact call from 06:26. It must return a stop, not raise."""
    orders = build("short", 1785.0, 0.0)

    assert kinds(orders) == ["hard"]
    assert orders[0][1] == pytest.approx(1785.0), "the stop must cover the whole position"


def test_a_zero_scale_out_long_is_the_same():
    orders = build("long", 1777.0, 0.0)

    assert kinds(orders) == ["hard"]
    assert orders[0][1] == pytest.approx(1777.0)


def test_the_old_fixture_would_have_passed_anyway():
    """Proof that the previous test proved nothing: a permissive rounder makes the
    broken code look fine. This is the fixture the suite used before."""
    permissive = lambda q: float(int(float(q)))

    orders = build_scale_out_orders("short", 1785.0, 0.0, TRAIL, HARD,
                                    rounder=permissive, min_notional=MIN_NOTIONAL_USDT)

    assert kinds(orders) == ["hard"]     # passes with or without the fix


# --- the general case the guard also covers -------------------------------------------

def test_a_split_that_rounds_the_trail_away_still_returns_a_stop():
    """A position small enough that scale x qty falls under the amount step. One
    full-size stop is strictly better protection than one leg plus a raised exception."""
    orders = build("long", 1.0, 0.10, min_notional=0.0)

    assert kinds(orders) == ["hard"]
    assert orders[0][1] == pytest.approx(1.0)


def test_a_normal_split_is_unchanged():
    orders = build("short", 1785.0, 0.5)

    assert kinds(orders) == ["trail", "hard"]
    assert sum(q for _, q, _ in orders) == pytest.approx(1785.0), "the split must be whole"


def test_the_split_still_covers_the_whole_position():
    for qty in (100.0, 1785.0, 7301.0):
        orders = build("long", qty, 0.3)
        assert sum(q for _, q, _ in orders) == pytest.approx(qty)


# --- and nothing that was already right regressed -------------------------------------

def test_a_position_too_small_to_protect_still_returns_nothing():
    """AUDIT #94. Under the exchange minimum, no stop of any size would be accepted, so
    the honest answer is none -- not a raise, and not a stop that will be rejected."""
    orders = build("long", 7.0, 0.0)     # 7 DOGE ~ 0.50 USDT, under the 5 USDT floor

    assert orders == []


def test_a_zero_position_returns_nothing():
    assert build("long", 0.0, 0.5) == []


def test_a_scale_out_already_fired_gives_one_hard_stop():
    orders = build("long", 1777.0, 0.5, scale_out_done=True)

    assert kinds(orders) == ["hard"]
    assert orders[0][1] == pytest.approx(1777.0)


def test_missing_stop_prices_return_nothing_rather_than_raising():
    assert build_scale_out_orders("short", 1785.0, 0.0, None, HARD,
                                  rounder=ccxt_like) == []
    assert build_scale_out_orders("short", 1785.0, 0.0, TRAIL, None,
                                  rounder=ccxt_like) == []


# --- the config that triggered it -----------------------------------------------------

def test_the_configured_scale_out_survives_the_real_rounder():
    """Whatever .env is set to must not be able to reproduce this."""
    from config import settings

    orders = build("short", 1785.0, settings.sl_scale_out_pct)

    assert orders, "the configured scale-out produces no stop at all"
    assert sum(q for _, q, _ in orders) == pytest.approx(1785.0)
