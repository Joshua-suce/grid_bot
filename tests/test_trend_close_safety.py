"""A trend exit must not sell a position the exchange has already closed.

2026-08-20 04:49. Long 111 ADA @ 0.2237 with a hard stop-market leg at 0.21678.
Price fell through the stop, the exchange sold 111, and one poll later the trailing
-stop exit market-sold another 111 against a mirror that still read long. One-way
account: the second sell did not close harder, it opened SHORT 111. That short then
had to be bought back at 05:35 for a further -0.16.

The same exit also called notifier.on_position_closed, which did not exist. The
AttributeError landed BETWEEN the market close and the bookkeeping, so _disarm_take
_profit never ran and _side was never cleared -- the follower stayed wedged holding
a position it had already sold, and at 05:35:33 the same raise killed the process
during startup where no handler catches it. AUDIT #131 / #132.
"""
from __future__ import annotations

import inspect
import re

import pytest

import telegram_notifier as tn
from trend_follower import TrendFollower


# ------------------------------------------------- the method that did not exist
def test_the_notifier_has_on_position_closed():
    assert hasattr(tn.TelegramNotifier, "on_position_closed")


def test_on_position_closed_accepts_what_the_caller_passes():
    """Existing is not enough -- it has to take the five positional arguments
    _close_position hands it, or the AttributeError just becomes a TypeError."""
    sig = inspect.signature(tn.TelegramNotifier.on_position_closed)
    params = [p for p in sig.parameters if p != "self"]
    assert len(params) == 5, params
    sig.bind(None, "ADAUSDT", "long", 111.0, 0.2165, -0.7992)


PRODUCTION = ["grid.py", "main.py", "trend_follower.py", "router.py", "risk.py"]


def _notifier_calls() -> set[str]:
    found = set()
    for f in PRODUCTION:
        src = open(f, encoding="utf-8").read()
        found |= set(re.findall(r"(?:_notifier|notifier)\.(on_\w+)\s*\(", src))
    return found


def test_every_notifier_method_production_calls_actually_exists():
    """The class-level fix. on_position_closed was called from two sites and defined
    nowhere, and nothing caught it until it killed the process in production."""
    have = {n for n in dir(tn.TelegramNotifier) if n.startswith("on_")}
    missing = sorted(_notifier_calls() - have)
    assert not missing, f"called but not defined on TelegramNotifier: {missing}"


def test_the_notifier_scan_actually_finds_calls():
    """Guard on the guard: a regex that matched nothing would pass the test above
    vacuously -- the exact way three earlier source scans in this repo failed."""
    calls = _notifier_calls()
    assert len(calls) > 5, f"scan found only {calls}; it is not reading production"
    assert "on_position_closed" in calls


# ------------------------------------------------------- the double-close itself
class _Ex:
    """Minimal exchange double. `positions` is what the account really holds."""

    def __init__(self, positions):
        self.positions = positions
        self.closed: list[tuple] = []
        self.cancelled: list[str] = []

    def get_positions(self, symbol):
        if self.positions == "boom":
            raise RuntimeError("endpoint down")
        return self.positions

    def close_position(self, symbol, side, amount, **kw):
        self.closed.append((symbol, side, amount))
        return {"average": 0.2165}

    def get_price(self, symbol):
        return 0.2165

    def cancel_order(self, order_id, symbol=None, **kw):
        self.cancelled.append(order_id)
        return True

    def get_open_orders(self, symbol):
        return []


def _follower(exchange, qty=111.0):
    tf = TrendFollower.__new__(TrendFollower)
    tf.exchange = exchange
    tf.symbol = "ADAUSDT"
    tf._side = "long"
    tf._entry_price = 0.2237
    tf._qty = qty
    tf._entry_time = 0.0
    tf._initial_risk = 0.0
    tf._take_profit_price = None
    tf._tp_order_id = None
    tf._notifier = None
    tf._event_journal = None
    tf.total_fills = 0
    tf.total_completed_cycles = 0
    tf.total_pnl = 0.0
    return tf


def _flat():
    return []


def _long(qty):
    return [{"contracts": qty, "info": {"positionAmt": str(qty)}}]


def test_it_does_not_sell_again_when_the_stop_already_closed_the_position():
    """The money bug. Selling into a flat one-way account opens the other side."""
    ex = _Ex(_flat())
    tf = _follower(ex)

    tf._close_position("trailing_stop")

    assert ex.closed == [], (
        "sold a position the exchange had already closed -- on a one-way account "
        "that opens the opposite side (AUDIT #132)"
    )


def test_skipping_the_sell_still_clears_the_wedged_state():
    """Declining to sell is only correct if the follower stops believing it holds
    something. Otherwise place_initial_orders returns 0 forever and it never trades
    again -- the AUDIT #38 wedge."""
    tf = _follower(_Ex(_flat()))

    tf._close_position("trailing_stop")

    assert tf._side is None
    assert tf._qty == 0.0
    assert tf._entry_price == 0.0


def test_a_position_that_is_still_open_is_closed_normally():
    """The guard must not disarm ordinary exits."""
    ex = _Ex(_long(111.0))
    tf = _follower(ex)

    result = tf._close_position("trailing_stop")

    assert ex.closed == [("ADAUSDT", "long", 111.0)]
    assert result is not None and result["completed_cycle"] is True
    assert tf._side is None


def test_a_partly_closed_position_sells_only_what_is_left():
    ex = _Ex(_long(40.0))
    tf = _follower(ex, qty=111.0)

    tf._close_position("trailing_stop")

    assert ex.closed == [("ADAUSDT", "long", 40.0)], ex.closed


def test_an_unreadable_account_is_not_treated_as_flat():
    """UNKNOWN must never read as flat here: skipping a close that genuinely needs to
    happen strands the position. Same discipline as AUDIT #128."""
    ex = _Ex("boom")
    tf = _follower(ex)

    tf._close_position("trailing_stop")

    assert ex.closed == [("ADAUSDT", "long", 111.0)], (
        "an unreadable positions endpoint suppressed a real close"
    )


def test_live_qty_reports_none_when_the_read_fails():
    assert _follower(_Ex("boom"))._live_position_qty() is None


def test_live_qty_reads_a_short_as_a_magnitude():
    """Binance encodes a short as a negative positionAmt; the guard compares against
    abs(qty), so the sign must not leak through."""
    ex = _Ex([{"contracts": -111.0, "info": {"positionAmt": "-111"}}])
    assert _follower(ex)._live_position_qty() == pytest.approx(111.0)


def test_a_flat_book_reads_as_zero_not_none():
    """Zero and None mean different things here and the guard branches on it."""
    assert _follower(_Ex(_flat()))._live_position_qty() == 0.0
