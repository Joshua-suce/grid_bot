"""A shutdown must never strip the stops off an open position. AUDIT #113.

GridEngine.emergency_stop has preserved them since #65: an inherited position stays
protected while the bot is down, and the restart reconciler adopts the live legs rather
than re-placing them.

The router calls emergency_stop on EVERY strategy. TrendFollower.emergency_stop called
cancel_everything with keep_stops defaulting to False, so on 2026-08-18 15:41 it ran two
seconds after the grid had deliberately left the stops armed:

    15:41:45  grid:emergency_stop      | STOPS LEFT ARMED | a position is still open...
    15:41:47  exchange:cancel_everything | CANCEL EVERYTHING | 1 total orders cancelled
    15:41:47  trend_follower:emergency_stop | TREND FOLLOWER STOPPED | shutdown

That one order was the stop. Verified after the fact against the exchange: SHORT 5350
DOGE open, fetch_open_orders and the raw fapi endpoint both returning zero orders.

Only reachable with STRATEGY_MODE=router -- with the grid alone nothing runs after it,
which is why it appeared the same day the router was switched on.
"""

from unittest.mock import MagicMock

import pytest

from trend_follower import TrendFollower


class Inner:
    def amount_to_precision(self, symbol, amount):
        return str(int(float(amount)))


def follower(positions, raises=False):
    ex = MagicMock()
    ex.exchange = Inner()
    if raises:
        ex.get_positions.side_effect = RuntimeError("book unreadable")
    else:
        ex.get_positions.return_value = positions
    ex.get_open_orders.return_value = []
    tf = TrendFollower(ex, "DOGEUSDT", stop_loss_pct=0.005,
                       trailing_sl_trigger_pct=0.05, atr_stop_multiplier=2.0)
    return tf, ex


def keep_stops_arg(ex):
    _, kwargs = ex.cancel_everything.call_args
    return kwargs.get("keep_stops")


# --- the live failure -------------------------------------------------------------------

def test_an_open_position_keeps_its_stops():
    """The exact case: SHORT 5350 open when shutdown runs."""
    tf, ex = follower([{"contracts": 5350.0}])

    tf.emergency_stop("shutdown")

    assert keep_stops_arg(ex) is True


def test_a_position_this_strategy_did_not_open_still_counts():
    """One-way mode: the grid and the follower share one net position. The follower has
    no _side here -- it never entered — and must still protect what is there."""
    tf, ex = follower([{"contracts": 5350.0}])
    assert tf._side is None

    tf.emergency_stop("shutdown")

    assert keep_stops_arg(ex) is True


def test_a_flat_symbol_still_clears_everything():
    """The other half. A stray stop on a flat book is noise, and leaving it forever
    would eventually trip the order-count limit."""
    tf, ex = follower([])

    tf.emergency_stop("shutdown")

    assert keep_stops_arg(ex) is False


def test_a_zero_contract_entry_reads_as_flat():
    tf, ex = follower([{"contracts": 0.0}])

    tf.emergency_stop("shutdown")

    assert keep_stops_arg(ex) is False


def test_a_short_counts_even_though_its_size_is_negative():
    tf, ex = follower([{"contracts": -5350.0}])

    tf.emergency_stop("shutdown")

    assert keep_stops_arg(ex) is True


# --- unreadable is not the same as flat ---------------------------------------------------

def test_an_unreadable_book_keeps_the_stops():
    """A few orphan reduce-only stops are recoverable. An unhedged position is not."""
    tf, ex = follower(None, raises=True)

    tf.emergency_stop("shutdown")

    assert keep_stops_arg(ex) is True


# --- and the grid's own behaviour is unchanged --------------------------------------------

def test_the_grid_still_keeps_stops_over_a_position():
    from grid import GridEngine

    ex = MagicMock()
    ex.exchange.amount_to_precision.side_effect = lambda s, a: str(int(float(a)))
    ex.get_positions.return_value = [{"contracts": 5350.0}]
    g = GridEngine(ex, "DOGEUSDT", grid_lower=0.068, grid_upper=0.072, grid_count=8,
                   capital_per_grid_pct=0.018, stop_loss_pct=0.005,
                   capital_per_grid_usdt=5.0, leverage=25)

    g.emergency_stop("shutdown")

    ex.cancel_all_open_orders.assert_called_once()
    ex.cancel_everything.assert_not_called()


def test_no_strategy_cancels_everything_unconditionally():
    """Pinned at source, across EVERY module rather than a hardcoded pair.

    router.py:398 fans emergency_stop out over self.strategies.values(), so a third
    strategy registered there inherits this bug for free. The first version of this test
    scanned only grid.py and trend_follower.py while its docstring claimed it would
    catch exactly that -- a guard that does not cover the case it advertises is worse
    than no guard, because it is read as coverage.

    The two existing strategies reach the guarantee differently and both are fine: the
    grid BRANCHES to cancel_all_open_orders while holding, the follower passes
    keep_stops. What must hold either way is that the call is gated on a position check
    -- an earlier version demanded the keyword specifically and failed the grid for
    being correct in the other style.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    checked = []
    for path in sorted(root.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        at = src.find("def emergency_stop")
        while at != -1:
            nxt = src.find("\n    def ", at + 1)  # -1 when it is the file's last method
            body = src[at:nxt if nxt != -1 else len(src)]
            if "cancel_everything(" in body:
                checked.append(path.name)
                assert "_has_open_position()" in body, (
                    f"{path.name}.emergency_stop calls cancel_everything without "
                    f"first asking whether a position is open")
            at = src.find("def emergency_stop", at + 1)

    assert {"grid.py", "trend_follower.py"} <= set(checked), (
        f"the scan found no emergency_stop to check in one of the known strategies, so "
        f"a pass here means nothing — saw {sorted(set(checked))}")
