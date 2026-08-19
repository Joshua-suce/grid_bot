"""Pausing must not strip the stops off a position it is not closing. AUDIT #123.

Strategy.pause does NOT flatten -- the position survives the pause and the levels are
expected to unwind it later. Cancelling the stop book at the same moment leaves that
position naked until _refresh_sl_stops notices, and that runs on a 120s verification
timer.

recenter() calls pause(). ADAUSDT, 2026-08-19, holding SHORT 6307:

    14:56:28  STOP-MARKET PLACED | BUY 6307.0 ADAUSDT @ stop=0.1775302773632203
    15:05:39  ONE-SIDED GRID | price 0.1767 outside [...] with no active sell orders
              -- forcing recenter inside margin band
    15:05:39  RECENTERING GRID
    15:05:40  BATCH CANCELLED 7 orders for ADAUSDT
    15:05:42  CANCEL EVERYTHING | 8 total orders confirmed cancelled for ADAUSDT
    15:05:42  GRID PAUSED | 8 orders cancelled via cancel_everything
    15:06:28  KILL SWITCH: price 0.1776 above stop loss 0.1775302773632203

Seven limit orders were open. The eighth was the stop. The tell is a line that is NOT
there: every stop-preserving path logs "leaving the stop/conditional book intact" --
15:06:31 does, 16:30:15 does, 15:05:42 does not.

The stop was gone 46 seconds before price crossed it. The short rode 0.1767 -> 0.1806
with nothing on the exchange to close it, and was still open at shutdown four hours later.

emergency_stop has guarded this since the shutdown fix. pause never did. The gap was
raised earlier the same day and DISMISSED, on the grounds that _refresh_sl_stops rebuilds
the stops -- which is true, and took longer than the 46 seconds available.
"""

from unittest.mock import MagicMock

import pytest

from grid import GridEngine, GridLevel


def engine(position_qty=0.0, positions_raise=False):
    ex = MagicMock()
    ex.exchange.amount_to_precision.side_effect = lambda s, a: str(int(float(a)))
    ex.exchange.price_to_precision.side_effect = lambda s, p: f"{float(p):.4f}"
    if positions_raise:
        ex.get_positions.side_effect = RuntimeError("book unreadable")
    else:
        ex.get_positions.return_value = (
            [] if position_qty == 0 else
            [{"side": "short" if position_qty < 0 else "long",
              "contracts": abs(position_qty)}]
        )
    ex.get_open_order_ids.return_value = set()
    ex.cancel_everything.return_value = 8
    g = GridEngine(ex, "ADAUSDT", grid_lower=0.17405296, grid_upper=0.17934704,
                   grid_count=12, capital_per_grid_pct=0.018, stop_loss_pct=0.005,
                   capital_per_grid_usdt=5.0, leverage=25)
    g.active = True
    return g, ex


def keep_stops_arg(ex):
    _, kwargs = ex.cancel_everything.call_args
    return kwargs.get("keep_stops")


# --- the live failure --------------------------------------------------------------------

def test_pausing_with_a_short_open_keeps_its_stops():
    """The exact case: SHORT 6307 open when recenter pauses the grid."""
    g, ex = engine(position_qty=-6307.0)

    g.pause()

    assert ex.cancel_everything.called
    assert keep_stops_arg(ex) is True, "the stop book went down with the ladder"


def test_pausing_with_a_long_open_keeps_them_too():
    g, ex = engine(position_qty=+6307.0)

    g.pause()

    assert keep_stops_arg(ex) is True


def test_pausing_flat_still_clears_everything():
    """Nothing to protect, so nothing is left behind to go stale."""
    g, ex = engine(position_qty=0.0)

    g.pause()

    assert keep_stops_arg(ex) is False


def test_an_unreadable_book_is_treated_as_holding():
    """Assuming a position costs a few orphan stops; assuming none costs an unhedged
    position. _has_open_position already answers True here -- pause must honour it."""
    g, ex = engine(positions_raise=True)

    g.pause()

    assert keep_stops_arg(ex) is True


# --- it still pauses properly -------------------------------------------------------------

def test_the_ladder_is_still_stood_down():
    """Preserving stops must not turn into preserving the grid."""
    g, ex = engine(position_qty=-6307.0)
    lvl = GridLevel(price=0.1743, side="buy")
    lvl.order_id, lvl.status = "live-1", "pending"
    g.levels = [lvl]

    g.pause()

    assert g.active is False
    assert lvl.order_id is None
    assert lvl.status == "pending"


def test_an_inactive_grid_is_left_alone():
    g, ex = engine(position_qty=-6307.0)
    g.active = False

    g.pause()

    assert not ex.cancel_everything.called


def test_it_says_so_when_it_leaves_them_armed():
    """Silence here is how the 15:05:42 cancellation went unnoticed for four hours."""
    from loguru import logger

    g, ex = engine(position_qty=-6307.0)
    sink = []
    h = logger.add(lambda m: sink.append(str(m)), level="WARNING")
    try:
        g.pause()
    finally:
        logger.remove(h)

    assert any("STOPS LEFT ARMED" in line for line in sink)


# --- both teardown paths, one rule ----------------------------------------------------------

def test_pause_and_emergency_stop_agree():
    """emergency_stop already had this. A rule only one teardown path honours is the
    shape of the original bug -- the trend follower's shutdown had it too (AUDIT #113)."""
    from pathlib import Path
    import grid as grid_mod

    src = Path(grid_mod.__file__).read_text(encoding="utf-8")
    body = src.split("def pause(self)")[1].split("def activate(")[0]

    assert "_has_open_position()" in body, "pause does not ask whether a position is open"
    assert "keep_stops=" in body, "pause does not pass keep_stops"
