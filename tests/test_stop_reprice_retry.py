"""A stop rejected as already-crossed must be repriced and retried, not just logged
and left for the next poll. AUDIT #167.

Observed live on 2026-08-19 16:30:26: a 6307-unit short's hard stop was rejected by
Binance with -2021 ("Order would immediately trigger") -- the desired trigger had
gone stale (poll latency, a fast move) by the time the request reached the exchange.
That left the position at 0% stop coverage for 77 seconds, caught only by #54's
under-protected check, and only recovered because the NEXT full refresh cycle
happened to compute a fresh, valid price before anything worse occurred. Retrying
the identical request would have failed identically -- the price that made it
invalid does not change between attempts, only between polls.
"""

import main
from main import _is_immediately_triggering_rejection, _reprice_past_current


class _RejectOnceEx:
    """Fails the first placement at a given price with a -2021-flavored error,
    then succeeds on whatever price it is asked to place next."""

    def __init__(self, current_price, reject_prices=(), other_error=None):
        self.placed: list[tuple] = []
        self._current_price = current_price
        self._reject_prices = {round(float(p), 8) for p in reject_prices}
        self._rejected_once: set[float] = set()
        self._other_error = other_error
        self._n = 0
        self.exchange = type("X", (), {
            "price_to_precision": staticmethod(lambda s, p: f"{float(p):.6f}"),
        })()

    def get_price(self, symbol):
        return self._current_price

    def place_stop_market(self, symbol, side, amount, stop_price, purpose="stop_hard"):
        key = round(float(stop_price), 8)
        if key in self._reject_prices and key not in self._rejected_once:
            self._rejected_once.add(key)
            if self._other_error is not None:
                raise self._other_error
            raise RuntimeError('binanceusdm {"code":-2021,"msg":"Order would immediately trigger."}')
        self._n += 1
        self.placed.append((side, float(amount), float(stop_price)))
        return {"id": f"new{self._n}"}

    def cancel_stop_order(self, order_id, symbol):
        return True


# --- the detector itself -----------------------------------------------------------

def test_detects_the_dash_2021_rejection():
    e = RuntimeError('binanceusdm {"code":-2021,"msg":"Order would immediately trigger."}')
    assert _is_immediately_triggering_rejection(e) is True


def test_does_not_misclassify_an_unrelated_failure():
    """Every other rejection is at least plausibly retryable as-is -- misreading one
    as -2021 would send a perfectly fine price through a needless reprice."""
    e = RuntimeError("amount of ADA/USDT:USDT must be greater than minimum amount precision of 1")
    assert _is_immediately_triggering_rejection(e) is False


# --- the reprice direction ----------------------------------------------------------

def test_reprice_lands_below_current_price_when_closing_a_long():
    """close_side='sell' closes a long: the stop fires as price falls, so a fresh
    trigger must sit strictly BELOW current price or it is invalid on arrival too."""
    ex = _RejectOnceEx(current_price=0.2100)
    price = _reprice_past_current(ex, "ADAUSDT", "sell", 0.2100)
    assert price < 0.2100


def test_reprice_lands_above_current_price_when_closing_a_short():
    ex = _RejectOnceEx(current_price=0.2100)
    price = _reprice_past_current(ex, "ADAUSDT", "buy", 0.2100)
    assert price > 0.2100


# --- the end-to-end retry inside reconcile_stop_orders -------------------------------

def test_a_2021_rejection_is_repriced_and_retried_successfully():
    """The exact incident: the hard leg's desired trigger has already been crossed.
    reconcile_stop_orders must not just log and move on -- it must fetch a fresh
    price, reprice past it, and get a real stop onto the book before giving up."""
    ex = _RejectOnceEx(current_price=0.18656, reject_prices=[0.18475982])
    desired = [("hard", 6307.0, 0.18475982)]

    kept, covered_qty, desired_qty = main.reconcile_stop_orders(
        ex, "ADAUSDT", "buy", desired, live=[],
    )

    assert "hard" in kept, "the repriced retry never placed a replacement stop"
    assert covered_qty == desired_qty == 6307.0
    # buy closes a short -- the retried trigger must be above the fresh price, not
    # the original (already-crossed) desired price.
    assert kept["hard"]["price"] > 0.18656
    assert kept["hard"]["qty"] == 6307.0


def test_the_kept_price_is_what_the_exchange_actually_holds_not_the_stale_desire():
    """AUDIT #54's own lesson repeated here: recording the desired price instead of
    the one actually resting would make the drift invisible to _sl_needs_update."""
    ex = _RejectOnceEx(current_price=0.18656, reject_prices=[0.18475982])
    kept, _, _ = main.reconcile_stop_orders(
        ex, "ADAUSDT", "buy", [("hard", 6307.0, 0.18475982)], live=[],
    )
    assert kept["hard"]["price"] != 0.18475982


def test_a_non_2021_failure_is_not_retried_with_a_reprice():
    """A precision/quantity rejection is not fixed by a different price. Retrying it
    anyway would just be a second, equally pointless request -- and it would hide
    the real error (a bad qty) behind a reprice log line that has nothing to do with
    what actually failed."""
    other = RuntimeError(
        "amount of ADA/USDT:USDT must be greater than minimum amount precision of 1"
    )
    ex = _RejectOnceEx(current_price=0.18656, reject_prices=[0.18475982], other_error=other)

    kept, covered_qty, desired_qty = main.reconcile_stop_orders(
        ex, "ADAUSDT", "buy", [("hard", 6307.0, 0.18475982)], live=[],
    )

    assert kept == {}
    assert covered_qty == 0.0 and desired_qty == 6307.0
    assert ex.placed == [], "a non-2021 failure must not be retried at all"


def test_when_the_repriced_retry_also_fails_the_leg_stays_uncovered_not_crashed():
    """The retry is best-effort. If it fails too, the caller must see the same
    honest under-coverage it would have seen without this fix -- never a crash, and
    never a phantom 'covered' leg."""
    class _AlwaysRejects(_RejectOnceEx):
        def place_stop_market(self, symbol, side, amount, stop_price, purpose="stop_hard"):
            raise RuntimeError('binanceusdm {"code":-2021,"msg":"Order would immediately trigger."}')

    ex = _AlwaysRejects(current_price=0.18656, reject_prices=[0.18475982])

    kept, covered_qty, desired_qty = main.reconcile_stop_orders(
        ex, "ADAUSDT", "buy", [("hard", 6307.0, 0.18475982)], live=[],
    )

    assert kept == {}
    assert covered_qty == 0.0 and desired_qty == 6307.0
