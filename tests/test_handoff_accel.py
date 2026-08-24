"""A router handoff must give the outgoing ladder a chance to reprice toward the
market instead of idling on a rung the price is not visiting until the grace clock
forces a market close. AUDIT #145.

2026-08-24, ADAUSDT: long 114 @ 0.2179, nearest sell exit resting at 0.2195. Price
held 0.2181-0.2183 for the whole 620s (and counting) of a 1800s handoff grace, and
nothing ever repriced the rung it was waiting on -- it sat 0.6% from a market that was
not moving toward it, heading for the same forced dump AUDIT #29 measured at -46.16
across 18 handoffs.

router.py wires this into _continue_handoff (tests/test_router.py covers the wiring);
this file covers the engine's own decision of what to move, to what price, and when
to leave it alone.
"""
from unittest.mock import MagicMock

import pytest

from grid import GridEngine, GridLevel

FEE_PCT = 0.0004472   # round_trip_fee_pct at the default maker/taker/taker_fill_share


def engine(pos_qty=114.0, entry=0.2179, price_now=0.2182, spacing=0.0017):
    ex = MagicMock()
    ex.exchange.amount_to_precision.side_effect = lambda s, a: str(round(float(a), 1))
    ex.exchange.price_to_precision.side_effect = lambda s, p: f"{float(p):.4f}"
    ex.can_place_order.return_value = True
    ex.get_open_order_ids.return_value = {"x"}   # matches resting()'s default oid
    ex.get_price.return_value = price_now
    ex.cancel_order.return_value = True
    ex.place_limit_order.return_value = {"id": "accel", "amount": 114}
    g = GridEngine(ex, "ADAUSDT", grid_lower=0.21366559, grid_upper=0.22373441,
                   grid_count=6, capital_per_grid_pct=0.018, stop_loss_pct=0.02,
                   capital_per_grid_usdt=5.0, leverage=5)
    g.grid_spacing = spacing
    g._pos_qty = pos_qty
    pos_side = "long" if pos_qty > 0 else "short"
    be = entry * (1 + FEE_PCT) if pos_qty > 0 else entry * (1 - FEE_PCT)
    g._position_break_even = lambda: (pos_side, be)
    g._existing_open_order = lambda price, side: None
    return g, ex, be


def resting(side, price, oid="x", qty=113.0):
    lvl = GridLevel(price=price, side=side, order_id=oid, quantity=qty)
    lvl.status = "pending"
    return lvl


# --------------------------------------------------------------- the live incident
def test_a_stale_exit_is_repriced_and_replaced():
    g, ex, be = engine()
    sell = resting("sell", 0.2195)
    g.levels = [sell]

    result = g.accelerate_handoff_exit(0.2182, balance=4866.53)

    assert result is True
    assert sell.price < 0.2195, "the exit was left exactly where it started"
    assert sell.price >= be - 1e-6, (
        "repricing must never quote below the position's own break-even"
    )
    assert ex.cancel_order.called, "the stale order was never cancelled"
    assert ex.place_limit_order.called, "nothing was placed at the new price"


def test_a_short_is_the_mirror():
    """The exit for a short is a BUY, and 'stale' means too far BELOW market -- moving
    it UP toward break-even is the improvement, not down."""
    g, ex, be = engine(pos_qty=-114.0, entry=0.2179, price_now=0.2176)
    buy = resting("buy", 0.2150)
    g.levels = [buy]

    result = g.accelerate_handoff_exit(0.2176, balance=4866.53)

    assert result is True
    assert buy.price > 0.2150, "a short's stale cover must move UP toward the market"
    assert buy.price <= be + 1e-6


# ------------------------------------------------------------------------ inertness
def test_flat_position_does_nothing():
    """pos_qty=0.0 resolves exit_side to 'buy' (0 is not > 0), so the resting level
    here is a genuine match for that side -- if the flat guard were not the thing
    stopping this, the rest of the method would happily reprice it."""
    g, ex, _ = engine(pos_qty=0.0)
    g.levels = [resting("buy", 0.15)]

    assert g.accelerate_handoff_exit(0.2182, balance=1000) is False
    ex.get_open_order_ids.assert_not_called()


def test_only_the_exit_side_is_considered():
    """A long's BUY side adds to the position; it must never be touched here even if
    it is sitting far from the market."""
    g, ex, _ = engine(pos_qty=114.0)
    g.levels = [resting("buy", 0.15)]

    assert g.accelerate_handoff_exit(0.2182, balance=1000) is False
    assert ex.cancel_order.called is False


def test_a_rung_already_near_the_market_is_left_alone():
    g, ex, be = engine()
    sell = resting("sell", 0.2195)
    g.levels = [sell]
    target = g.accelerate_handoff_exit(0.2182, balance=1000) and sell.price
    ex.cancel_order.reset_mock()
    ex.place_limit_order.reset_mock()

    # A fresh level already sitting at the price the first call moved it to.
    g2, ex2, _ = engine()
    already_close = resting("sell", target)
    g2.levels = [already_close]

    assert g2.accelerate_handoff_exit(0.2182, balance=1000) is False
    assert ex2.cancel_order.called is False


# ------------------------------------------------------------------------- cooldown
def test_a_second_call_inside_the_cooldown_is_a_no_op():
    g, ex, _ = engine()
    sell = resting("sell", 0.2195)
    g.levels = [sell]
    assert g.accelerate_handoff_exit(0.2182, balance=1000) is True

    # Put the rung back exactly where it would be genuinely improvable again -- the
    # cooldown, not "nothing left to improve", must be what blocks the second call.
    sell.price = 0.2195
    sell.order_id = "x"
    ex.cancel_order.reset_mock()

    assert g.accelerate_handoff_exit(0.2182, balance=1000) is False, (
        "repriced again inside HANDOFF_ACCEL_COOLDOWN_SECONDS"
    )
    assert ex.cancel_order.called is False


# ------------------------------------------------------------- unconfirmed cancels
def test_an_unconfirmed_cancel_keeps_the_level_claimed():
    """Same discipline as _cancel_resting_orders (AUDIT #51): if the exchange cannot
    confirm the cancel, the level must stay exactly as it was -- clearing it here
    would let place_initial_orders lay a second order on top of a still-live one."""
    g, ex, _ = engine()
    sell = resting("sell", 0.2195, oid="live-order")
    g.levels = [sell]
    ex.get_open_order_ids.return_value = {"live-order"}
    ex.cancel_order.return_value = False

    result = g.accelerate_handoff_exit(0.2182, balance=1000)

    assert result is False
    assert sell.order_id == "live-order"
    assert sell.price == 0.2195
    assert sell.status == "pending"


def test_an_order_already_gone_is_repriced_without_a_cancel_call():
    """If the exchange no longer shows the order open (already filled or cancelled by
    something else), there is nothing to cancel -- just reclaim the level."""
    g, ex, _ = engine()
    g.levels = [resting("sell", 0.2195, oid="ghost")]
    ex.get_open_order_ids.return_value = set()

    result = g.accelerate_handoff_exit(0.2182, balance=1000)

    assert result is True
    assert ex.cancel_order.called is False
