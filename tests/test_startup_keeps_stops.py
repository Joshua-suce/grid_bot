"""Startup must not cancel the stops shutdown deliberately left armed. AUDIT #92.

emergency_stop ends a session with

    STOPS LEFT ARMED | a position is still open, so its stop-loss legs stay on the
    exchange — it remains protected while the bot is down

and then startup cancelled them anyway and spent 26 seconds placing an identical pair:

    05:00:22  CANCEL EVERYTHING | 2 total orders confirmed cancelled
    05:00:48  STOP-MARKET PLACED | BUY 2671.0 @ 0.0721309
    05:00:48  STOP-MARKET PLACED | BUY 2671.0 @ 0.07312790065400002

Twenty-six seconds with 5342 DOGE of short and no stop, on the one path that had
already been engineered to avoid exactly that. The legs are handed to
reconcile_stop_orders instead, which matches desired against live and touches only what
differs (AUDIT #54), so the window closes to nothing.
"""

import ast
import pathlib

from exchange import Exchange


class FakeCcxt:
    def __init__(self):
        self.algo_cancels = []
        self.batch_cancels = 0

    def cancel_all_orders(self, symbol):
        self.batch_cancels += 1

    def fapiPrivateDeleteAlgoOrder(self, params):
        self.algo_cancels.append(params["algoId"])


def exchange(open_orders, stop_orders):
    e = Exchange.__new__(Exchange)
    e.exchange = FakeCcxt()
    e._open_books = {"regular": list(open_orders), "stops": list(stop_orders)}

    def get_open_orders(symbol):
        return e._open_books["regular"]

    def get_stop_orders(symbol):
        return e._open_books["stops"]

    def cancel_order(order_id, symbol):
        e._open_books["regular"] = [
            o for o in e._open_books["regular"] if o["id"] != order_id
        ]
        return True

    e.get_open_orders = get_open_orders
    e.get_stop_orders = get_stop_orders
    e.cancel_order = cancel_order
    return e


LIMITS = [{"id": "limit-1"}, {"id": "limit-2"}]
STOPS = [{"id": "trail-1"}, {"id": "hard-1"}]


def test_keep_stops_leaves_the_protection_alone():
    """The live shape: two stop legs guarding an inherited short, plus stray limits."""
    e = exchange(LIMITS, STOPS)

    e.cancel_everything("DOGEUSDT", timeout_seconds=1.0, keep_stops=True)

    assert e.exchange.algo_cancels == [], "cancelled the stops guarding an open position"
    assert e.exchange.batch_cancels == 1, "left the stale limit orders on the book"


def test_the_stop_book_is_not_even_read_when_it_is_being_kept():
    """Not just "does not cancel": does not spend a REST call asking. Startup is the
    slowest part of a restart and every avoidable round trip there is dead time."""
    e = exchange(LIMITS, STOPS)
    reads = []
    inner = e.get_stop_orders
    e.get_stop_orders = lambda symbol: (reads.append(symbol), inner(symbol))[1]

    e.cancel_everything("DOGEUSDT", timeout_seconds=1.0, keep_stops=True)

    assert reads == []


def test_the_default_still_sweeps_everything():
    """Flat startup, or any other caller: an untracked stop from a dead session must
    still go. The new argument is opt-in, and defaults to the old behaviour."""
    e = exchange(LIMITS, STOPS)

    e.cancel_everything("DOGEUSDT", timeout_seconds=1.0)

    assert sorted(e.exchange.algo_cancels) == ["hard-1", "trail-1"]


def test_limits_are_still_verified_clean_while_stops_are_kept():
    """Keeping stops must not weaken the guarantee that matters for the ladder: a fresh
    grid laid on top of uncancelled limit orders is double exposure with no record."""
    e = exchange(LIMITS, STOPS)
    e.exchange.cancel_all_orders = lambda symbol: (_ for _ in ()).throw(RuntimeError("batch down"))

    e.cancel_everything("DOGEUSDT", timeout_seconds=5.0, keep_stops=True)

    assert e._open_books["regular"] == [], "gave up on the limit book"
    assert e.exchange.algo_cancels == []


# --- and the caller actually asks for it ---------------------------------------------

def test_startup_cleanup_asks_to_keep_stops():
    """A behavioural test of cancel_everything proves nothing if run_bot still calls it
    the old way. run_bot is not unit-testable, so check the call site itself: the
    startup cleanup must pass keep_stops, not rely on the default."""
    source = pathlib.Path(__file__).resolve().parent.parent / "main.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cancel_everything"
    ]
    assert calls, "no cancel_everything call found in main.py"
    assert any(kw.arg == "keep_stops" for c in calls for kw in c.keywords), (
        "main.py cancels everything unconditionally at startup — the stops shutdown "
        "left armed are still being thrown away"
    )
