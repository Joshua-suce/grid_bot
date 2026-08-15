"""An order that did not execute is not a fill. AUDIT #75.

check_fills asked `status == "canceled"` and treated every other answer as a fill. On
2026-08-15 at 23:58:45 a recenter placed three reduce-only unwind buys; Binance EXPIRED
all three with executedQty=0 (a reduce-only order that can no longer reduce is expired,
not cancelled). The grid booked FILL #2/#3/#4 for +0.375 +0.238 +0.095, three completed
cycles, three rows in the trade journal and three trades against the daily counter.

userTrades for that window is empty. Nothing executed. Confirmed against the exchange:

    2319171161  status=EXPIRED  executedQty=0  reduceOnly=True
    2319171168  status=EXPIRED  executedQty=0  reduceOnly=True
    2319171171  status=EXPIRED  executedQty=0  reduceOnly=True

The risk layer printed the contradiction in the same second -- "verified pnl=+0.00 |
grid estimated +0.71" -- and that +0.71 then sat in every status line for three and a
half hours while the session was actually down.

This is trail_stop_fired's allowlist (AUDIT #26) applied to the path that books PnL.
"""

import pytest

from grid import order_was_filled


def order(status, filled=None, executed=None):
    o = {"status": status}
    if filled is not None:
        o["filled"] = filled
    if executed is not None:
        o["info"] = {"executedQty": executed}
    return o


# --- the measured case --------------------------------------------------------------

def test_an_expired_reduce_only_order_is_not_a_fill():
    """The exact shape Binance returned for all three."""
    assert order_was_filled(order("expired", filled=0.0, executed="0")) is False


@pytest.mark.parametrize("status", [
    "expired", "canceled", "cancelled", "rejected", "open", "new", "", None,
])
def test_nothing_but_a_completed_fill_counts(status):
    assert order_was_filled(order(status)) is False


@pytest.mark.parametrize("status", ["closed", "filled"])
def test_a_completed_fill_counts(status):
    assert order_was_filled(order(status)) is True


def test_a_missing_order_is_not_a_fill():
    assert order_was_filled(None) is False
    assert order_was_filled({}) is False


# --- quantity ------------------------------------------------------------------------

def test_a_closed_order_that_moved_nothing_is_not_a_fill():
    """Whatever the exchange calls it, zero quantity is not a trade."""
    assert order_was_filled(order("closed", filled=0.0)) is False
    assert order_was_filled(order("closed", executed="0")) is False


def test_a_closed_order_that_moved_quantity_is_a_fill():
    assert order_was_filled(order("closed", filled=596.0)) is True
    assert order_was_filled(order("closed", executed="596")) is True


def test_an_unknown_quantity_on_a_closed_order_is_trusted():
    """Some ccxt versions omit it. 'closed' with no contradicting evidence stands."""
    assert order_was_filled(order("closed")) is True


def test_an_unparseable_quantity_does_not_crash():
    assert order_was_filled(order("closed", filled="n/a")) is True


# --- the polarity that caused it -----------------------------------------------------

def test_the_rule_is_an_allowlist_not_a_denylist():
    """The defect was structural: enumerate what IS a fill, never what isn't. Any status
    the exchange invents later must default to 'not a fill'."""
    assert order_was_filled(order("some_status_binance_adds_in_2027")) is False


def test_it_matches_the_stop_rule():
    """trail_stop_fired already got this right for stops (AUDIT #26). The two must not
    drift apart -- they answer the same question about different order types."""
    from main import trail_stop_fired

    for status in ("closed", "filled", "expired", "canceled", "rejected", "open"):
        assert order_was_filled({"status": status}) == trail_stop_fired({"status": status}), (
            f"the fill rule and the stop rule disagree about {status!r}"
        )
