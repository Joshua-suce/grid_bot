"""Tests for the fee-economics fixes (AUDIT.md issues #17-#20).

The 2026-08-11 session booked ~1442 fills and still ended down. Binance's own income
ledger explained why: realized +13.21 against commission -45.26. Fees were 2.2x gross
profit. Two mechanisms let that happen -- crossing orders being silently downgraded to
taker fills, and a profitability gate pinned at break-even -- and two config settings
guaranteed it stayed that way.
"""

import pytest

import ccxt

from exchange import PostOnlyWouldCross
from grid import GridEngine


class RecordingExchange:
    """Records placements. Optionally raises PostOnlyWouldCross for a given side."""

    def __init__(self, cross_side=None):
        self.placed: list[dict] = []
        self._cross_side = cross_side
        self._next_id = 0

        class _inner:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{float(amount):.0f}"

            @staticmethod
            def price_to_precision(symbol, price):
                return f"{float(price):.5f}"

        self.exchange = _inner()

    def get_positions(self, symbol):
        return []

    def get_open_orders(self, symbol):
        return []

    def get_open_order_ids(self, symbol):
        return set()

    def can_place_order(self, symbol):
        return True

    def get_balance(self):
        return 5000.0

    def cancel_order(self, order_id, symbol):
        return True

    def get_orderbook_depth(self, symbol, limit=10):
        return {"bid_vol": 0, "ask_vol": 0, "imbalance": 0, "spread_pct": 0}

    def place_limit_order(self, symbol, side, price, amount, max_attempts=3, params=None, post_only=True):
        if self._cross_side is not None and side == self._cross_side:
            raise PostOnlyWouldCross(f"{side} @ {price} would cross the spread")
        self._next_id += 1
        order = {"id": f"o{self._next_id}", "side": side, "price": price, "amount": amount}
        self.placed.append(order)
        return order


class SpyNotifier:
    def __init__(self):
        self.failures = []
        self.placements = []

    def on_order_placed(self, *a, **kw):
        self.placements.append(a)

    def on_order_failed(self, *a, **kw):
        self.failures.append(a)


class SpyJournal:
    def __init__(self):
        self.failures = []

    def order_placed(self, *a, **kw):
        pass

    def order_failed(self, *a, **kw):
        self.failures.append(a)


def make_engine(exchange, grid_count=10, min_profit_multiplier=3.0, **kw):
    defaults = dict(
        symbol="DOGEUSDT",
        grid_lower=0.0710,
        grid_upper=0.0730,
        grid_count=grid_count,
        capital_per_grid_pct=0.05,
        stop_loss_pct=0.05,
        maker_fee_pct=0.0002,
        taker_fee_pct=0.0004,
        max_exposure_pct=1.0,
        min_profit_multiplier=min_profit_multiplier,
        order_pacing_seconds=0.0,
    )
    defaults.update(kw)
    return GridEngine(exchange=exchange, **defaults)


# --- #17: crossing orders must never become taker fills --------------------

def test_crossing_level_is_skipped_not_filled_as_taker():
    """A level on the wrong side of the book is left unplaced, not crossed.

    Two sells went out below market on 2026-08-12 at 01:22:40 and 01:22:43 with
    postOnly=False and filled instantly as taker 22 seconds later, both at a loss.
    """
    ex = RecordingExchange(cross_side="sell")
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)

    grid.place_initial_orders(balance=5000)

    assert all(o["side"] == "buy" for o in ex.placed), "a crossing sell was placed anyway"
    unplaced = [lv for lv in grid.levels if lv.side == "sell" and lv.order_id is None]
    assert unplaced, "crossing sells should be left pending for a later retry"


def test_crossing_level_is_not_reported_as_a_failure():
    """It is a normal transient condition, not an error worth alerting on."""
    ex = RecordingExchange(cross_side="sell")
    notifier, journal = SpyNotifier(), SpyJournal()
    grid = make_engine(ex, event_journal=journal, notifier=notifier)
    grid.initialize(0.0720, balance=5000)

    grid.place_initial_orders(balance=5000)

    assert notifier.failures == [], "crossing level raised a Telegram alert"
    assert journal.failures == [], "crossing level was journalled as a failure"


def test_crossing_level_stays_retryable():
    """The level must keep its identity so the next pass can place it."""
    ex = RecordingExchange(cross_side="sell")
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.place_initial_orders(balance=5000)

    sells = [lv for lv in grid.levels if lv.side == "sell"]
    assert sells and all(lv.order_id is None for lv in sells)
    assert all(lv.status == "pending" for lv in sells)

    # Book moves back; the same levels now rest normally.
    ex._cross_side = None
    grid.place_initial_orders(balance=5000)
    assert any(o["side"] == "sell" for o in ex.placed)


def test_real_placement_errors_are_still_reported():
    """Only PostOnlyWouldCross is quiet -- genuine failures must still surface."""
    class BrokenExchange(RecordingExchange):
        def place_limit_order(self, *a, **kw):
            raise ccxt.ExchangeError("-1013 something genuinely wrong")

    ex = BrokenExchange()
    notifier, journal = SpyNotifier(), SpyJournal()
    grid = make_engine(ex, event_journal=journal, notifier=notifier)
    grid.initialize(0.0720, balance=5000)

    grid.place_initial_orders(balance=5000)

    assert notifier.failures, "a real placement error was swallowed"
    assert journal.failures


# --- #18/#20: the profitability gate must demand a real margin -------------

def test_level_whose_spacing_barely_covers_fees_is_skipped():
    """30 levels across this range gives 0.0069% spacing against a 0.04% round trip.

    Under the old hardcoded multiplier of 1.0 this passed the gate, because spacing
    exceeded fees by a hair. It is a guaranteed net loss once anything goes against it.
    """
    ex = RecordingExchange()
    grid = make_engine(ex, grid_count=30, min_profit_multiplier=3.0)
    grid.initialize(0.0720, balance=5000)

    grid.place_initial_orders(balance=5000)
    assert ex.placed == [], "placed levels that cannot cover their own fees"


def test_level_with_healthy_spacing_is_placed():
    ex = RecordingExchange()
    grid = make_engine(ex, grid_count=10, min_profit_multiplier=3.0)
    grid.initialize(0.0720, balance=5000)

    grid.place_initial_orders(balance=5000)
    assert ex.placed, "refused levels that clear fees 5x over"


def test_gate_tightens_as_the_multiplier_rises():
    """The same grid passes at 1.0 and fails at 5.0 -- the knob actually binds."""
    spacings = {}
    for mult in (1.0, 5.0):
        ex = RecordingExchange()
        grid = make_engine(ex, grid_count=24, min_profit_multiplier=mult)
        grid.initialize(0.0720, balance=5000)
        grid.place_initial_orders(balance=5000)
        spacings[mult] = len(ex.placed)

    assert spacings[1.0] > 0, "break-even multiplier should still place this grid"
    assert spacings[5.0] == 0, "a 5x margin requirement should reject it"


def test_profitability_gate_uses_round_trip_not_single_leg_fees():
    """A cycle pays two fees, not one. Sizing the gate on a single leg would let
    through levels that lose exactly the second fee on every completed cycle."""
    ex = RecordingExchange()
    grid = make_engine(ex, grid_count=10, min_profit_multiplier=1.0)
    price = 0.0720
    grid.initialize(price, balance=5000)

    single_leg = grid.maker_fee_pct * price
    assert grid._is_level_profitable(price) is (grid.grid_spacing > 2 * single_leg)
