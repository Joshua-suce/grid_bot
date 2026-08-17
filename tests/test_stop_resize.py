"""A stop that outgrew its position must actually get resized. AUDIT #93.

_sl_needs_update has asked for a refresh once a leg exceeds its desired size by more than
SL_OVER_COVERAGE_TOLERANCE since AUDIT #54 -- "stale oversized stop, resize to keep
sizing honest". reconcile_stop_orders then matched any live leg at the right price whose
quantity was `>= desired` and kept it, oversized and all. So the caller asked, the
reconciler refused, and the caller asked again on the next poll.

Live on 2026-08-17. Stops armed at 4445/4446 for an 8891 short at 06:49; the position
came down to 3551 and back to 5331 and the legs never moved:

    06:49:11  STOP-MARKET PLACED | BUY 4445.0 @ 0.07332360065400001   (trail)
    06:49:11  STOP-MARKET PLACED | BUY 4446.0 @ 0.07312790065400002   (hard)
    10:31:49  ...still 4445/4446, position 5331

Two costs. The configured 50% scale-out was really 83/83 of the live position, so the
trailing leg was not doing the job it is sized to do. And because the caller re-asked
every poll, _refresh_sl_stops ran every poll -- an extra get_stop_orders round trip and
an SL STATUS line on every single iteration for four hours, which is exactly what the
log shows from 08:05 onwards.
"""

import pytest

from main import reconcile_stop_orders

TOLERANCE = 0.10          # mirrors SL_OVER_COVERAGE_TOLERANCE in run_bot

TRAIL_PRICE = 0.07332360065400001
HARD_PRICE = 0.07312790065400002


class FakeExchange:
    def __init__(self):
        self.cancelled = []
        self.placed = []

    def cancel_stop_order(self, order_id, symbol):
        self.cancelled.append(order_id)
        return True

    def place_stop_market(self, symbol, side, qty, price, purpose=None):
        self.placed.append({"side": side, "qty": qty, "price": price})
        return {"id": f"new-{len(self.placed)}"}


def stop(order_id, qty, price):
    return {"id": order_id, "amount": qty, "triggerPrice": price}


# The 06:49 book against the 10:31 position.
LIVE_OVERSIZED = [stop("trail-4445", 4445.0, TRAIL_PRICE),
                  stop("hard-4446", 4446.0, HARD_PRICE)]
DESIRED_5331 = [("trail", 2666.0, TRAIL_PRICE), ("hard", 2665.0, HARD_PRICE)]


def test_a_leg_that_outgrew_its_tolerance_is_replaced():
    ex = FakeExchange()

    kept, covered, desired_qty = reconcile_stop_orders(
        ex, "DOGEUSDT", "buy", DESIRED_5331, LIVE_OVERSIZED,
        over_coverage_tolerance=TOLERANCE,
    )

    assert sorted(ex.cancelled) == ["hard-4446", "trail-4445"], "kept the oversized legs"
    assert sorted(o["qty"] for o in ex.placed) == [2665.0, 2666.0]
    assert covered >= desired_qty


def test_the_refresh_converges_instead_of_asking_forever():
    """The livelock, stated as the property the caller depends on: after a reconcile,
    nothing kept is more than the tolerance above what was desired -- so the next
    _sl_needs_update has no reason to fire. Unbounded, this is false on every pass."""
    ex = FakeExchange()

    kept, _, _ = reconcile_stop_orders(
        ex, "DOGEUSDT", "buy", DESIRED_5331, LIVE_OVERSIZED,
        over_coverage_tolerance=TOLERANCE,
    )

    for kind, oqty, _price in DESIRED_5331:
        assert kept[kind]["qty"] <= oqty * (1 + TOLERANCE) + 1e-8, (
            f"{kind} leg still {kept[kind]['qty']} against a desired {oqty} — "
            "the caller will ask for this refresh again next poll, forever"
        )


def test_the_scale_out_ratio_is_restored():
    """The point of resizing. 4445/4446 of a 5331 position is 83/83, not the configured
    50/50 -- the trailing leg covers almost the whole position, so the scale-out that is
    supposed to bank half and let the rest run does neither."""
    ex = FakeExchange()

    reconcile_stop_orders(ex, "DOGEUSDT", "buy", DESIRED_5331, LIVE_OVERSIZED,
                          over_coverage_tolerance=TOLERANCE)

    trail = next(o for o in ex.placed if o["price"] == TRAIL_PRICE)
    assert trail["qty"] / sum(q for _, q, _ in DESIRED_5331) == pytest.approx(0.5, abs=0.01)


# --- and it must not start churning stops that are fine -------------------------------

def test_a_leg_inside_the_tolerance_is_left_alone():
    """Refreshing means cancel-then-place, which opens an unprotected round trip. Float
    noise and rounding must not buy one on every poll."""
    ex = FakeExchange()
    live = [stop("trail-1", 2666.0000001, TRAIL_PRICE), stop("hard-1", 2665.0, HARD_PRICE)]

    kept, _, _ = reconcile_stop_orders(
        ex, "DOGEUSDT", "buy", DESIRED_5331, live,
        over_coverage_tolerance=TOLERANCE,
    )

    assert ex.cancelled == []
    assert ex.placed == []
    assert kept["trail"]["id"] == "trail-1"


def test_a_leg_slightly_over_but_within_tolerance_is_kept():
    """5% over is inside the 10% band _sl_needs_update tolerates. Replacing it here
    would re-open the gap the tolerance exists to avoid."""
    ex = FakeExchange()
    live = [stop("trail-1", 2799.0, TRAIL_PRICE), stop("hard-1", 2665.0, HARD_PRICE)]

    reconcile_stop_orders(ex, "DOGEUSDT", "buy", DESIRED_5331, live,
                          over_coverage_tolerance=TOLERANCE)

    assert ex.cancelled == []
    assert ex.placed == []


def test_an_undersized_leg_is_still_replaced():
    """Under-coverage was never tolerated and must not become tolerated: that is naked
    exposure, not an accounting nicety."""
    ex = FakeExchange()
    live = [stop("trail-small", 100.0, TRAIL_PRICE), stop("hard-1", 2665.0, HARD_PRICE)]

    reconcile_stop_orders(ex, "DOGEUSDT", "buy", DESIRED_5331, live,
                          over_coverage_tolerance=TOLERANCE)

    assert ex.cancelled == ["trail-small"]
    assert [o["qty"] for o in ex.placed] == [2666.0]


def test_the_unbounded_default_is_unchanged():
    """Existing callers that do not opt in keep the old matching exactly."""
    ex = FakeExchange()

    kept, _, _ = reconcile_stop_orders(ex, "DOGEUSDT", "buy", DESIRED_5331, LIVE_OVERSIZED)

    assert ex.cancelled == []
    assert ex.placed == []
    assert kept["trail"]["qty"] == 4445.0
