"""A deformed ladder must be visible even while inventory is open. AUDIT #116.

recenter computed `ladder_defects(current_price) if flat else []`, so the entire
deformation check was switched off for as long as any position was open -- and a grid
holds inventory most of the time. The 2026-08-18 run carried a 1.41% hole around the
price from 17:57, when the 0.07027 sell filled and left SHORT 1778 open, until the 20:25
shutdown. 2h28m, no warning, no fills, nearest sell nine ticks above the high reached.

The stale grid_spacing hid it from the other side too: ladder_defects flags a hole wider
than 2x spacing, and against the saved 0.00065864 that bar sat at 0.001317, which the
0.00099 hole cleared. Against the measured 0.00028286 the bar is 0.000566 and it
registers. Both halves had to be wrong for the ladder to look healthy.

WHAT MUST NOT CHANGE: rebuilding while holding stays forbidden. recenter pauses the grid,
and with a position open the resting orders are its exits -- cancelling them is what made
this fire 89 times in one session and left a position unable to unwind. This adds a
signal, not an action. test_holding_never_cancels_anything is the load-bearing test.
"""

import time
from unittest.mock import MagicMock

import pytest

from grid import GridEngine, GridLevel

LOWER, UPPER, COUNT = 0.0689520382, 0.0709279618, 8
PRICE = 0.0703

HEALTHY = [0.06895, 0.06928, 0.06945, 0.06961, 0.06994, 0.07027, 0.07060, 0.07093]
# The book as it actually stood after 17:57: 0.06994 never placeable, 0.07027 consumed.
HOLED = [0.06895, 0.06928, 0.06945, 0.06961, 0.07060, 0.07093]


def engine(prices, pos_qty=0.0):
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
    g._net_long_qty = 0.0
    g._net_short_qty = 0.0
    g._last_recenter_time = 0.0
    g._last_deform_warn_time = 0.0
    return g, ex


# --- the premise ------------------------------------------------------------------------

def test_the_holed_ladder_really_is_deformed():
    """If ladder_defects stops flagging this, every test below passes vacuously."""
    g, _ = engine(HOLED)

    assert any("hole around the price" in d for d in g.ladder_defects(PRICE))


def test_the_full_ladder_is_not_deformed():
    g, _ = engine(HEALTHY)

    assert g.ladder_defects(PRICE) == []


def test_the_stale_spacing_would_have_masked_it():
    """Documents why c498cf9 was a precondition for this fix, not merely adjacent."""
    g, _ = engine(HOLED)
    g.grid_spacing = 0.0006586412          # the value the state file carried

    assert not any("hole around the price" in d for d in g.ladder_defects(PRICE))


# --- the blind spot -----------------------------------------------------------------------

def test_a_deformed_ladder_is_noticed_while_holding():
    """The 2h28m case: SHORT 1778 open, ladder holed, previously invisible."""
    g, _ = engine(HOLED, pos_qty=-1778.0)

    g.recenter(PRICE, 4937.0)

    assert g._last_deform_warn_time > 0, "deformation went unreported while holding"


def test_a_healthy_ladder_says_nothing_while_holding():
    """Or the warning becomes noise and stops being read."""
    g, _ = engine(HEALTHY, pos_qty=-1778.0)

    g.recenter(PRICE, 4937.0)

    assert g._last_deform_warn_time == 0


def test_a_long_position_counts_as_holding_too():
    g, _ = engine(HOLED, pos_qty=+1778.0)

    g.recenter(PRICE, 4937.0)

    assert g._last_deform_warn_time > 0


# --- what must not change -----------------------------------------------------------------

def test_holding_never_cancels_anything():
    """THE load-bearing test. recenter pauses the grid, which cancels resting orders; with
    a position open those orders are its exits. Cancelling them is the 89-recenter
    failure. Detection must stay detection."""
    g, ex = engine(HOLED, pos_qty=-1778.0)

    result = g.recenter(PRICE, 4937.0)

    assert result is False, "rebuilt the ladder with a position open"
    assert not ex.cancel_everything.called
    assert not ex.cancel_all_open_orders.called
    assert not ex.cancel_order.called


def test_the_bounds_are_untouched_while_holding():
    g, _ = engine(HOLED, pos_qty=-1778.0)

    g.recenter(PRICE, 4937.0)

    assert g.grid_lower == pytest.approx(LOWER)
    assert g.grid_upper == pytest.approx(UPPER)


def test_flat_and_deformed_still_rebuilds():
    """AUDIT #34 must survive: with no position there are no exits to strand, and a
    deformed ladder should be rebuilt rather than traded."""
    g, _ = engine(HOLED, pos_qty=0.0)

    result = g.recenter(PRICE, 4937.0)

    assert result is True
    assert g.grid_lower < PRICE < g.grid_upper


def test_flat_and_healthy_does_nothing():
    g, ex = engine(HEALTHY, pos_qty=0.0)

    assert g.recenter(PRICE, 4937.0) is False
    assert not ex.cancel_everything.called


# --- the warning needs its own clock -------------------------------------------------------

def test_the_warning_is_rate_limited():
    """recenter's cooldown only advances when a recenter actually happens, so its body
    runs every poll when none does. Without a separate clock this prints every 10s for
    hours and buries the log."""
    g, _ = engine(HOLED, pos_qty=-1778.0)
    g._last_deform_warn_time = time.time() - 10

    before = g._last_deform_warn_time
    g.recenter(PRICE, 4937.0)

    assert g._last_deform_warn_time == before, "re-warned 10s after the last one"


def test_the_warning_clock_survives_an_engine_built_without_init():
    """The clock must be a CLASS attribute.

    tests/test_startup_ladder_teardown.py builds the engine with
    GridEngine.__new__(GridEngine) on purpose, to reproduce the 05:00:56 startup shape by
    setting only the fields recenter reads. The first version of this fix stored the clock
    in __init__, which raises AttributeError on that path and broke three existing tests
    -- in the guard that stops a restored position having its exit orders torn down.
    """
    bare = GridEngine.__new__(GridEngine)

    assert bare._last_deform_warn_time == 0.0


def test_it_warns_again_once_the_interval_passes():
    """Rate-limited, not one-shot: a ladder still deformed an hour later is still news."""
    g, _ = engine(HOLED, pos_qty=-1778.0)
    g._last_deform_warn_time = time.time() - 400

    before = g._last_deform_warn_time
    g.recenter(PRICE, 4937.0)

    assert g._last_deform_warn_time > before
