"""Stop orders must be cancelled through the ALGO endpoint. AUDIT #74.

Binance keeps stop/conditional orders in a separate algo order space. Sending an algo id
to the ordinary cancel endpoint returns "Unknown order sent", ccxt raises OrderNotFound,
and cancel_order treats that as "already gone" -- reporting a successful cancel of an
order that is still on the book.

Measured live on 2026-08-14 22:59:37. A partial fill armed stops for 233 DOGE; fourteen
seconds later the full 1790 landed and the reconciler moved to 895-qty legs, "cancelling"
the 233 pair. All four stops were still armed sixteen minutes later.

Nothing went naked -- the legs are reduceOnly, so the position was over-covered, not
under. The damage is slower: stale legs accumulate one pair per ratchet step, and
_detect_trail_fill infers "the trailing leg fired" from its absence, which a leftover at
the same price can mask (see AUDIT #26 for what that costs).
"""

import ccxt
import pytest

from exchange import CircuitBreaker, Exchange
from main import reconcile_stop_orders

ALGO_ID = "1000000167339107"


def make_exchange(fake):
    ex = Exchange.__new__(Exchange)
    ex.exchange = fake
    ex.demo = False
    ex.has_credentials = True
    ex.max_retries = 1
    ex.retry_delay = 0.0
    ex._circuit_breaker = CircuitBreaker(failure_threshold=1000, recovery_time=0)
    return ex


class Backend:
    """Models the real asymmetry: the ordinary endpoint does not know algo ids."""

    def __init__(self, algo_ids=(ALGO_ID,), algo_delete_works=True, book=None):
        self.algo_ids = set(algo_ids)
        self.algo_delete_works = algo_delete_works
        self.book = book
        self.calls = []

    def cancel_order(self, order_id, symbol):
        self.calls.append(("regular", order_id))
        if order_id in self.algo_ids:
            raise ccxt.OrderNotFound("Unknown order sent.")  # the whole bug
        raise ccxt.OrderNotFound("no such order")

    def fapiPrivateDeleteAlgoOrder(self, params):
        self.calls.append(("algo", params.get("algoId")))
        if not self.algo_delete_works:
            raise ccxt.ExchangeError("algo delete unavailable")
        self.algo_ids.discard(params.get("algoId"))

    def fetch_open_orders(self, symbol, params=None):
        if self.book is not None:
            if isinstance(self.book, Exception):
                raise self.book
            return self.book
        return [{"id": i} for i in sorted(self.algo_ids)]


# --- the endpoint ------------------------------------------------------------------

def test_a_stop_is_cancelled_through_the_algo_endpoint():
    backend = Backend()
    assert make_exchange(backend).cancel_stop_order(ALGO_ID, "DOGEUSDT") is True
    assert backend.calls[0] == ("algo", ALGO_ID)
    assert ALGO_ID not in backend.algo_ids


def test_the_ordinary_endpoint_is_not_used_for_stops():
    """It is not merely ineffective -- it answers 'already gone' about a live order."""
    backend = Backend()
    make_exchange(backend).cancel_stop_order(ALGO_ID, "DOGEUSDT")
    assert not any(kind == "regular" for kind, _ in backend.calls)


def test_the_ordinary_cancel_still_lies_about_algo_ids():
    """Pinning the behaviour this fix routes around, so a future refactor that sends a
    stop back through cancel_order fails here rather than in production."""
    backend = Backend()
    assert make_exchange(backend).cancel_order(ALGO_ID, "DOGEUSDT") is True
    assert ALGO_ID in backend.algo_ids, "the order was reported cancelled but survived"


def test_a_failed_algo_delete_is_verified_against_the_book_not_assumed():
    """Inferring 'gone' from an error is exactly what caused this. If the book still
    shows it, the answer is False."""
    backend = Backend(algo_delete_works=False, book=[{"id": ALGO_ID}])
    assert make_exchange(backend).cancel_stop_order(ALGO_ID, "DOGEUSDT") is False


def test_a_failed_algo_delete_on_an_absent_order_reports_success():
    backend = Backend(algo_delete_works=False, book=[])
    assert make_exchange(backend).cancel_stop_order(ALGO_ID, "DOGEUSDT") is True


def test_an_unreadable_book_after_a_failed_delete_reports_failure():
    backend = Backend(algo_delete_works=False, book=ccxt.RequestTimeout("down"))
    assert make_exchange(backend).cancel_stop_order(ALGO_ID, "DOGEUSDT") is False


# --- the reconciler ----------------------------------------------------------------

class RecordingExchange:
    def __init__(self, cancel_ok=True):
        self.cancel_ok = cancel_ok
        self.stop_cancels = []
        self.plain_cancels = []
        self.placed = []

    def cancel_stop_order(self, order_id, symbol):
        self.stop_cancels.append(order_id)
        return self.cancel_ok

    def cancel_order(self, order_id, symbol):
        self.plain_cancels.append(order_id)
        return True

    def place_stop_market(self, symbol, side, qty, price, purpose=None):
        self.placed.append((side, qty, price))
        return {"id": f"new-{len(self.placed)}"}


def test_the_reconciler_retires_stale_legs_through_the_stop_path():
    """The measured sequence: 233-qty legs from a partial fill, then the position grows
    to 1790 and the desired legs become 895."""
    ex = RecordingExchange()
    live = [
        {"id": "1000000167339107", "triggerPrice": 0.0719146, "amount": 233.0},
        {"id": "1000000167339112", "triggerPrice": 0.07277207, "amount": 233.0},
    ]
    desired = [("trail", 895.0, 0.0719146), ("hard", 895.0, 0.07277207)]

    kept, covered, wanted = reconcile_stop_orders(ex, "DOGEUSDT", "buy", desired, live)

    assert ex.stop_cancels == ["1000000167339107", "1000000167339112"]
    assert ex.plain_cancels == [], "a stop went through the ordinary cancel endpoint"
    assert len(ex.placed) == 2
    assert covered == pytest.approx(wanted)


def test_an_unconfirmed_stop_cancel_is_reported(caplog):
    import sys

    from loguru import logger

    ex = RecordingExchange(cancel_ok=False)
    live = [{"id": ALGO_ID, "triggerPrice": 0.0719146, "amount": 233.0}]
    desired = [("trail", 895.0, 0.0719146)]

    sink = []
    handle = logger.add(lambda m: sink.append(str(m)), level="INFO")
    try:
        reconcile_stop_orders(ex, "DOGEUSDT", "buy", desired, live)
    finally:
        logger.remove(handle)
        logger.add(sys.stderr, level="INFO")

    assert "not confirmed cancelled" in "".join(sink)


def test_a_matching_leg_is_left_alone():
    """The whole point of #54: do not tear down protection that is already correct."""
    ex = RecordingExchange()
    live = [{"id": "keep-me", "triggerPrice": 0.0719146, "amount": 895.0}]
    desired = [("trail", 895.0, 0.0719146)]

    kept, _, _ = reconcile_stop_orders(ex, "DOGEUSDT", "buy", desired, live)

    assert ex.stop_cancels == []
    assert ex.placed == []
    assert kept["trail"]["id"] == "keep-me"
