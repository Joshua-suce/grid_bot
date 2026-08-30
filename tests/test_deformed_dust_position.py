"""Dust must not get a permanent veto over the deformed-ladder rebuild.

recenter()'s "flat" gate (AUDIT #116) refuses to rebuild a deformed ladder while ANY
position is open, because rebuilding cancels resting orders and with a position open
those orders are its exits. Correct -- for a real position. But a position below
MIN_NOTIONAL_USDT can never carry a stop-loss at any size (STOP-LOSS UNPROTECTABLE
already refuses to place one), so there is no real exit order at risk: only a resting
line sized for the dust itself.

On 2026-08-29 a 6 ADA (~1.16 USDT) remainder left over from a sizing change reconciled
into two exit rungs 0.05% apart -- under the fee floor. "flat" stayed False for that
alone, so the ladder sat deformed, DEFORMED LADDER (HOLDING) logged every 5 minutes,
and the bot ran its entire ~5-hour session -- 05:58 to 10:53 -- without a single fill,
even while price moved through the gap the deformation had opened up.

WHAT MUST NOT CHANGE: a real position still blocks the rebuild exactly as before --
test_a_real_position_still_blocks_the_rebuild is the load-bearing test here, the
counterpart to test_holding_never_cancels_anything in test_deformed_while_holding.py.
"""

from unittest.mock import MagicMock

import pytest

from grid import MIN_NOTIONAL_USDT, GridEngine, GridLevel

LOWER, UPPER, COUNT = 0.0689520382, 0.0709279618, 8
PRICE = 0.0703

# A too-close pair, not a hole -- the actual shape of the 2026-08-29 incident. 0.07027
# and 0.070275 sit 0.05% apart; every other gap stays under the 2x-spacing hole bar.
TOO_CLOSE = [0.06895, 0.06928, 0.06945, 0.06961, 0.06994, 0.07027, 0.070275, 0.07060, 0.07093]
HEALTHY = [0.06895, 0.06928, 0.06945, 0.06961, 0.06994, 0.07027, 0.07060, 0.07093]

# Comfortably below and above MIN_NOTIONAL_USDT at PRICE, with margin on both sides so
# the assertions do not ride a floating-point boundary.
DUST_QTY = (MIN_NOTIONAL_USDT / PRICE) * 0.5     # notional ~= half the floor
REAL_QTY = (MIN_NOTIONAL_USDT / PRICE) * 4.0     # notional ~= 4x the floor


def engine(prices, pos_qty=0.0, net_long=None, net_short=None):
    ex = MagicMock()
    ex.exchange.amount_to_precision.side_effect = lambda s, a: str(int(float(a)))
    ex.exchange.price_to_precision.side_effect = lambda s, p: f"{float(p):.5f}"
    ex.get_open_order_ids.return_value = []
    g = GridEngine(ex, "DOGEUSDT", grid_lower=LOWER, grid_upper=UPPER, grid_count=COUNT,
                   capital_per_grid_pct=0.018, stop_loss_pct=0.005,
                   capital_per_grid_usdt=5.0, leverage=25)
    levels = []
    for p in prices:
        lvl = GridLevel(price=p, side="buy" if p < PRICE else "sell")
        lvl.status = "pending"
        lvl.order_id = f"id{p}"
        levels.append(lvl)
    g.levels = levels
    g._pos_qty = pos_qty
    g._net_long_qty = pos_qty if net_long is None else net_long
    g._net_short_qty = 0.0 if net_short is None else net_short
    g._last_recenter_time = 0.0
    g._last_deform_warn_time = 0.0
    return g, ex


# --- the premise --------------------------------------------------------------------

def test_the_too_close_ladder_really_is_deformed():
    """If ladder_defects stops flagging this, every test below passes vacuously."""
    g, _ = engine(TOO_CLOSE)

    defects = g.ladder_defects(PRICE)
    assert any("fee floor" in d for d in defects)
    assert not any("hole around the price" in d for d in defects), (
        "this fixture must isolate the too-close defect, not also trip the hole one"
    )


def test_dust_notional_is_actually_below_the_floor():
    assert DUST_QTY * PRICE < MIN_NOTIONAL_USDT


def test_real_notional_is_actually_above_the_floor():
    assert REAL_QTY * PRICE > MIN_NOTIONAL_USDT


# --- dust does not veto the rebuild ---------------------------------------------------

def test_a_dust_long_does_not_block_the_rebuild():
    """The actual incident: a sub-minimum-notional remainder must not hold a deformed
    ladder open for hours."""
    g, ex = engine(TOO_CLOSE, pos_qty=DUST_QTY)

    result = g.recenter(PRICE, 4937.0)

    assert result is True, "dust position blocked a rebuild it should not be able to"
    assert g.grid_lower < PRICE < g.grid_upper


def test_a_dust_short_does_not_block_the_rebuild():
    g, ex = engine(TOO_CLOSE, pos_qty=-DUST_QTY, net_long=0.0, net_short=DUST_QTY)

    result = g.recenter(PRICE, 4937.0)

    assert result is True


def test_a_dust_position_still_leaves_no_warning_logged():
    """The DEFORMED LADDER (HOLDING) branch is for positions that genuinely block the
    rebuild. Dust takes the rebuild path instead, so the holding-warning clock must
    stay untouched."""
    g, _ = engine(TOO_CLOSE, pos_qty=DUST_QTY)

    g.recenter(PRICE, 4937.0)

    assert g._last_deform_warn_time == 0.0


# --- a real position still blocks it, exactly as before -------------------------------

def test_a_real_position_still_blocks_the_rebuild():
    """THE load-bearing test. A position above MIN_NOTIONAL_USDT has a real exit order
    to strand -- the dust carve-out must not swallow this case."""
    g, ex = engine(TOO_CLOSE, pos_qty=REAL_QTY)

    result = g.recenter(PRICE, 4937.0)

    assert result is False, "rebuilt the ladder with a real position open"
    assert not ex.cancel_everything.called
    assert not ex.cancel_all_open_orders.called
    assert not ex.cancel_order.called
    assert g._last_deform_warn_time > 0, "a real position should still warn, not silently rebuild"


def test_a_real_net_long_qty_blocks_the_rebuild_even_if_pos_qty_reads_dust():
    """_net_long_qty and _pos_qty can disagree for one iteration (AUDIT #116's restart
    race, in reverse): _pos_qty must not be the only signal trusted, or a real
    committed exposure the position-cap tracker still knows about gets waved through
    on a stale/transient pos_qty read."""
    g, ex = engine(TOO_CLOSE, pos_qty=DUST_QTY, net_long=REAL_QTY)

    result = g.recenter(PRICE, 4937.0)

    assert result is False
    assert not ex.cancel_everything.called


def test_a_healthy_ladder_with_dust_open_still_does_nothing():
    """Dust changes what recenter is ALLOWED to do; it must not make recenter fire
    when there is nothing to fix."""
    g, ex = engine(HEALTHY, pos_qty=DUST_QTY)

    assert g.recenter(PRICE, 4937.0) is False
    assert not ex.cancel_everything.called