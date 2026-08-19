"""A failed cancel is not a cancel. AUDIT #51.

`Exchange.cancel_order` already returns a correct bool -- True only when the order is
confirmed gone (OrderNotFound counts as gone), False for "status unknown". Three
production callers discarded it and cleared their bookkeeping anyway, so the engine
forgot an order that was still live on the exchange.

That is worse than doing nothing. The level goes back to "pending", so the very next
place_initial_orders lays a SECOND order at the same price -- and _cancel_resting_orders
is called precisely when the position is AT its cap, so the failure mode is: at the cap,
cancels fail, the engine believes it shrank exposure, and it grows it instead.

Note these paths mostly do not raise: cancel_order swallows the ccxt exception and
returns False, so an `except` around the call never fires. The bug is the ignored
return value, not an unhandled exception.
"""

import pytest

from grid import GridEngine


class _Ex:
    """Cancels always fail to confirm -- the network-timeout case."""
    cancel_result = False

    class exchange:
        @staticmethod
        def amount_to_precision(s, a): return f"{float(a):.0f}"
        @staticmethod
        def price_to_precision(s, p): return f"{float(p):.5f}"

    def __init__(self): self.cancel_calls = []
    def get_price(self, s): return 0.0700
    def get_balance(self, s="USDT"): return 5000.0
    def get_positions(self, s): return []
    def get_open_orders(self, s): return []
    def get_open_order_ids(self, s): return {"A", "B"}
    def can_place_order(self, s): return True
    def place_limit_order(self, *a, **k): return {"id": "x"}
    def cancel_everything(self, s, timeout_seconds=30, keep_stops=False): return 0
    def cancel_order(self, oid, symbol):
        self.cancel_calls.append(oid)
        return self.cancel_result


def _engine(ex):
    g = GridEngine(exchange=ex, symbol="DOGEUSDT", grid_lower=0.0690, grid_upper=0.0710,
                   grid_count=10, capital_per_grid_pct=0.018, stop_loss_pct=0.03,
                   min_profit_multiplier=3.0, max_exposure_pct=0.5, leverage=5)
    g.initialize(0.0700, balance=5000.0)
    return g


def _claim_two_buys(g):
    claimed = [l for l in g.levels if l.side == "buy"][:2]
    for level, oid in zip(claimed, ("A", "B")):
        level.order_id = oid
        level.status = "open"
    return claimed


# --- the position cap: the worst place to forget a live order ---------------

def test_an_unconfirmed_cancel_does_not_free_the_level():
    ex = _Ex()
    g = _engine(ex)
    claimed = _claim_two_buys(g)

    n = g._cancel_resting_orders("buy", "position cap")

    assert n == 0, f"reported {n} cancels but none were confirmed"
    for level in claimed:
        assert level.order_id is not None, (
            "the level was freed on an unconfirmed cancel -- the order is still live on "
            "the exchange and place_initial_orders will now double it"
        )
        assert level.status == "open"


def test_a_confirmed_cancel_still_frees_the_level():
    """The fix must not strand inventory the way #42 did."""
    ex = _Ex()
    ex.cancel_result = True
    g = _engine(ex)
    claimed = _claim_two_buys(g)

    n = g._cancel_resting_orders("buy", "position cap")

    assert n == 2
    for level in claimed:
        assert level.order_id is None and level.status == "pending"


def test_a_partial_failure_reports_only_what_was_confirmed():
    ex = _Ex()
    g = _engine(ex)
    claimed = _claim_two_buys(g)
    ex.cancel_order = lambda oid, symbol: oid == "A"      # B keeps failing

    n = g._cancel_resting_orders("buy", "position cap")

    assert n == 1, f"expected 1 confirmed cancel, reported {n}"
    assert claimed[0].order_id is None
    assert claimed[1].order_id == "B", "the unconfirmed one was freed anyway"


def test_pause_keeps_a_level_that_would_not_cancel():
    """pause() deliberately does not flatten, so a surviving order must stay claimed --
    otherwise resuming re-places on top of it."""
    ex = _Ex()
    g = _engine(ex)
    claimed = _claim_two_buys(g)
    g.active = True          # pause() returns immediately when inactive -- without this
                             # the test passes against broken code by doing nothing

    g.pause()

    assert not g.active
    for level in claimed:
        assert level.order_id is not None, "pause freed a level it could not cancel"


# --- the trend follower: an unconfirmed cancel becomes a double entry --------

def test_an_unconfirmed_entry_cancel_does_not_allow_a_second_entry():
    from trend_follower import TrendFollower

    tf = TrendFollower.__new__(TrendFollower)
    tf.symbol = "DOGEUSDT"
    tf.exchange = _Ex()
    tf._order_id = "A"
    tf._side = None
    tf._entry_price = 0.07
    tf._event_journal = None

    assert TrendFollower._cancel_entry(tf, "regime_change") is False
    assert tf._order_id == "A", (
        "the entry was forgotten while still live -- place_initial_orders' "
        "`if self._order_id is not None` guard would now let a SECOND entry through"
    )

    tf.exchange.cancel_result = True
    assert TrendFollower._cancel_entry(tf, "regime_change") is True
    assert tf._order_id is None


# --- the fee model the gates are built on -----------------------------------

def test_the_round_trip_is_priced_at_the_blended_rate():
    """The ledger says 11.8% of fill volume pays taker. Pricing the round trip at
    2 * maker understated it by 1.12x, in the profitability gate AND -- worse -- in
    every break-even price, so 'break-even' exits booked a real loss."""
    g = _engine(_Ex())

    assert g.round_trip_fee_pct == pytest.approx(0.00044720), (
        f"round trip priced at {g.round_trip_fee_pct:.8f}, expected the blended "
        f"0.00044720 (0.0447%)"
    )
    assert g.round_trip_fee_pct > 2 * g.maker_fee_pct, "still the optimistic all-maker figure"


def test_an_all_maker_book_collapses_to_the_old_number():
    """Sanity: the blend is a generalisation, not a different model."""
    ex = _Ex()
    g = GridEngine(exchange=ex, symbol="DOGEUSDT", grid_lower=0.069, grid_upper=0.071,
                   grid_count=10, capital_per_grid_pct=0.018, stop_loss_pct=0.03,
                   taker_fill_share=0.0, max_exposure_pct=0.5, leverage=5)
    assert g.round_trip_fee_pct == pytest.approx(2 * g.maker_fee_pct)


def test_break_even_clears_the_real_cost_not_the_assumed_one():
    """A long's break-even must sit above entry by the BLENDED round trip."""
    ex = _Ex()
    # _position_break_even reads the position from the exchange on purpose (AUDIT #7/#8)
    ex.get_positions = lambda s: [{"side": "long", "contracts": 5000.0, "entryPrice": 0.07}]
    g = _engine(ex)

    result = g._position_break_even()
    assert result is not None
    side, be = result
    assert side == "long"

    optimistic = 0.07000 * (1 + 2 * g.maker_fee_pct)
    assert be > optimistic, (
        f"break-even {be:.8f} does not clear the real round trip -- exits clamped here "
        f"book a loss (optimistic figure was {optimistic:.8f})"
    )
    assert be == pytest.approx(0.07000 * (1 + g.round_trip_fee_pct))
