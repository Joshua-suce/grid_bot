"""A trailing stop must not be re-placed on a timer. AUDIT #95.

_sl_needs_update leaves a leg alone until its trigger drifts 0.1% (`oprice * 0.001`). But
AUDIT #54's trust-but-verify pass calls _refresh_sl_stops every 120 seconds regardless of
belief, and reconcile_stop_orders matched live orders to desired legs at a hardcoded
0.01% -- a hundred times tighter. So every two minutes, any ratchet larger than a
hundredth of a percent failed to match, and a leg the caller was perfectly happy with was
cancelled and re-placed.

That is where the stop churn in the 2026-08-17 12:55 session came from. Eleven trailing
re-placements between 13:07 and 15:51, ~14 stop cancels an hour overall, and the trigger
moves behind them were tiny:

    13:07:53  0.0679     -> 0.06791939999999999   0.029%
    13:12:17             -> 0.06793880000000001   0.057% cumulative
    13:32:30             -> 0.06802609999999999   0.186% cumulative
    13:34:18             -> 0.0681134             0.128%
    13:36:33             -> 0.0681328             0.028%
    13:38:44             -> 0.06814250000000001   0.043% cumulative
    13:42:52             -> 0.0681522             0.057% cumulative
    13:50:21             -> 0.0681619             0.071% cumulative

No fee is charged for a cancelled stop, so this costs nothing directly. It costs the one
thing AUDIT #54 exists to minimise: Binance has no atomic replace, so every one of those
is an unprotected round trip on an open position.

The second half of the defect is quieter. A kept leg recorded the price that was DESIRED
when it matched, not the trigger the exchange actually holds, so _sl_needs_update was
comparing a desire against itself and always got zero. Real drift was invisible to the
caller, which is why the periodic verify was the only thing that ever re-placed a
drifting leg.
"""

from pathlib import Path

import main
from main import reconcile_stop_orders

DRIFT = 0.001          # mirrors SL_PRICE_DRIFT_TOLERANCE in run_bot
OLD_MATCH = 1e-4       # what reconcile_stop_orders hardcoded, and its default
QTY = 1785.0

# The live 13:07 -> 13:50 trail ratchet, to the digit.
RATCHET = [0.0679,
           0.06791939999999999,
           0.06793880000000001,
           0.06802609999999999,
           0.0681134,
           0.0681328,
           0.06814250000000001,
           0.0681522,
           0.0681619]


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


def run_ratchet(tolerance, prices=RATCHET):
    """Walk the live ratchet one poll at a time, feeding each pass the book the previous
    one left behind. Returns the exchange so the churn can be counted."""
    ex = FakeExchange()
    live = [stop("trail-0", QTY, prices[0])]
    for target in prices[1:]:
        kept, _, _ = reconcile_stop_orders(
            ex, "DOGEUSDT", "sell", [("trail", QTY, target)], live,
            price_tolerance_pct=tolerance,
        )
        live = [stop(kept["trail"]["id"], kept["trail"]["qty"], kept["trail"]["price"])]
    return ex


# --- the churn -----------------------------------------------------------------------

def test_the_old_match_re_placed_the_leg_on_every_single_ratchet():
    """The baseline, so the improvement below is measured and not asserted. Every one of
    the eight live moves is bigger than 0.01% of price, so every one missed the match."""
    ex = run_ratchet(OLD_MATCH)

    assert len(ex.placed) == len(RATCHET) - 1 == 8
    assert len(ex.cancelled) == 8


def test_a_trigger_that_has_barely_moved_is_left_alone():
    """The single case, isolated: 0.0679 -> 0.0679194 is 0.029% of price. The caller
    would not have asked for this; the verify pass should not force it."""
    ex = FakeExchange()
    live = [stop("trail-0", QTY, 0.0679)]

    kept, _, _ = reconcile_stop_orders(
        ex, "DOGEUSDT", "sell", [("trail", QTY, 0.06791939999999999)], live,
        price_tolerance_pct=DRIFT,
    )

    assert ex.cancelled == [], "cancelled a stop over a 0.029% move"
    assert ex.placed == []
    assert kept["trail"]["id"] == "trail-0"


def test_a_trigger_that_has_genuinely_gone_stale_is_replaced():
    """0.0679 -> 0.0680261 is 0.186%, past the drift the caller tolerates. Protection
    that has fallen this far behind the peak is worth the round trip."""
    ex = FakeExchange()
    live = [stop("trail-0", QTY, 0.0679)]

    reconcile_stop_orders(ex, "DOGEUSDT", "sell",
                          [("trail", QTY, 0.06802609999999999)], live,
                          price_tolerance_pct=DRIFT)

    assert ex.cancelled == ["trail-0"]
    assert ex.placed[0]["price"] == 0.06802609999999999


def test_the_live_ratchet_costs_two_round_trips_instead_of_eight():
    """The whole sequence, end to end. Two of the eight moves genuinely outran the
    tolerance; the other six were noise the bot paid an unprotected window for."""
    ex = run_ratchet(DRIFT)

    assert len(ex.placed) == 2, f"{len(ex.placed)} re-placements over the live ratchet"
    assert len(ex.cancelled) == len(ex.placed), "cancelled a leg without replacing it"
    assert [o["price"] for o in ex.placed] == [0.06802609999999999, 0.0681134]


def test_protection_never_lags_further_than_the_tolerance():
    """Fewer re-placements must not mean a stop drifting arbitrarily far behind. Whatever
    is on the book at the end of every poll is within the tolerated drift of what was
    wanted at that poll."""
    ex = FakeExchange()
    live = [stop("trail-0", QTY, RATCHET[0])]
    for target in RATCHET[1:]:
        kept, _, _ = reconcile_stop_orders(
            ex, "DOGEUSDT", "sell", [("trail", QTY, target)], live,
            price_tolerance_pct=DRIFT,
        )
        held = kept["trail"]["price"]
        assert abs(held - target) <= max(target * DRIFT, 1e-8), (
            f"book holds {held} against a wanted {target} — outside the tolerance"
        )
        live = [stop(kept["trail"]["id"], kept["trail"]["qty"], held)]


# --- the belief must be true ----------------------------------------------------------

def test_a_kept_leg_records_the_price_the_exchange_actually_holds():
    """Recording the desired price instead makes the caller's drift test compare a desire
    against itself: always zero, never stale, so nothing but the 120-second timer ever
    re-places a drifting leg. Every churn test above still passes with that mutant."""
    ex = FakeExchange()
    live = [stop("trail-0", QTY, 0.0679)]

    kept, _, _ = reconcile_stop_orders(
        ex, "DOGEUSDT", "sell", [("trail", QTY, 0.06791939999999999)], live,
        price_tolerance_pct=DRIFT,
    )

    assert kept["trail"]["price"] == 0.0679, (
        "believes the trigger is where it was wanted, not where it is"
    )


def test_a_newly_placed_leg_records_the_price_it_was_placed_at():
    ex = FakeExchange()

    kept, _, _ = reconcile_stop_orders(
        ex, "DOGEUSDT", "sell", [("trail", QTY, 0.0681134)], [],
        price_tolerance_pct=DRIFT,
    )

    assert kept["trail"]["price"] == 0.0681134


def test_the_caller_and_the_reconciler_cannot_disagree():
    """The invariant behind both this fix and AUDIT #93, stated once: nothing kept may
    fail the caller's own staleness test. Tighter here and the verify churns; looser and
    the caller asks forever."""
    for target in RATCHET[1:]:
        ex = FakeExchange()
        live = [stop("trail-0", QTY, RATCHET[0])]
        kept, _, _ = reconcile_stop_orders(
            ex, "DOGEUSDT", "sell", [("trail", QTY, target)], live,
            price_tolerance_pct=DRIFT,
        )
        if kept["trail"]["id"] != "trail-0":
            continue                                    # re-placed, nothing to agree on
        assert abs(kept["trail"]["price"] - target) <= max(target * DRIFT, 1e-8), (
            f"kept a leg at {kept['trail']['price']} that the caller will ask to "
            f"replace against {target} on the very next poll"
        )


def test_one_constant_feeds_both_halves():
    """The two tolerances live in different functions and only work as a pair. Pinning
    the wiring is the only way a later edit to one of them cannot silently reintroduce
    the churn -- which is exactly how it arrived."""
    src = Path(main.__file__).read_text(encoding="utf-8")

    assert "SL_PRICE_DRIFT_TOLERANCE = 0.001" in src
    assert "price_tolerance_pct=SL_PRICE_DRIFT_TOLERANCE" in src, (
        "_refresh_sl_stops no longer hands the reconciler the caller's own tolerance"
    )
    assert "oprice * SL_PRICE_DRIFT_TOLERANCE" in src, (
        "_sl_needs_update no longer uses the tolerance it hands the reconciler"
    )


# --- and the existing callers are untouched -------------------------------------------

def test_the_default_matching_is_unchanged():
    """reconcile_stop_orders has other callers and its own test module. Opting in is the
    only thing that changes behaviour."""
    ex = FakeExchange()
    live = [stop("trail-0", QTY, 0.0679)]

    reconcile_stop_orders(ex, "DOGEUSDT", "sell",
                          [("trail", QTY, 0.06791939999999999)], live)

    assert ex.cancelled == ["trail-0"], "the default match silently widened"


def test_quantity_still_decides_independently_of_price():
    """A leg at exactly the right price but the wrong size is still replaced. Widening
    the price match must not smuggle in a size tolerance."""
    ex = FakeExchange()
    live = [stop("trail-0", 100.0, 0.0679)]

    reconcile_stop_orders(ex, "DOGEUSDT", "sell", [("trail", QTY, 0.0679)], live,
                          price_tolerance_pct=DRIFT)

    assert ex.cancelled == ["trail-0"]
    assert ex.placed[0]["qty"] == QTY
