"""Regression tests for the 2026-08-12 technical diagnosis.

Each test pins one defect found by tracing the 2026-08-11 production log, where the
bot recentered 89 times (median 196s apart -- exactly the recenter cooldown), logged
740 ReduceOnly rejections, and held a 9148 DOGE long for over an hour whose stop-loss
drifted DOWN with the market. See AUDIT.md issues #11-#16.
"""

import pytest

from grid import GridEngine


class FakeExchange:
    """Records placed orders instead of hitting an API. Mirrors the ccxt surface the
    engine actually touches (amount_to_precision / price_to_precision)."""

    def __init__(self, positions=None, reject_reduce_only_over=None):
        self.placed: list[dict] = []
        self.cancelled: list[str] = []
        self._positions = positions or []
        # When set, emulates Binance -2022: reject a reduceOnly order once the
        # cumulative reduceOnly quantity exceeds the open position.
        self._reject_over = reject_reduce_only_over
        self._reduce_only_total = 0.0
        self._next_id = 0

        outer = self

        class _inner:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{float(amount):.0f}"

            @staticmethod
            def price_to_precision(symbol, price):
                return f"{float(price):.5f}"

        self.exchange = _inner()
        self._outer = outer

    def get_positions(self, symbol):
        return self._positions

    def get_open_orders(self, symbol):
        return []

    def get_open_order_ids(self, symbol):
        return set()

    def can_place_order(self, symbol):
        return True

    def cancel_order(self, order_id, symbol):
        self.cancelled.append(order_id)
        return True

    def cancel_everything(self, symbol, timeout_seconds=300.0):
        return 0

    def get_orderbook_depth(self, symbol, limit=10):
        return {"bid_vol": 0, "ask_vol": 0, "imbalance": 0, "spread_pct": 0}

    def place_limit_order(self, symbol, side, price, amount, max_attempts=3, params=None, post_only=True):
        params = params or {}
        if params.get("reduceOnly") and self._reject_over is not None:
            if self._reduce_only_total + amount > self._reject_over + 1e-9:
                raise ValueError('binanceusdm {"code":-2022,"msg":"ReduceOnly Order is rejected."}')
            self._reduce_only_total += amount
        self._next_id += 1
        order = {
            "id": f"o{self._next_id}", "side": side, "price": price,
            "amount": amount, "params": params,
        }
        self.placed.append(order)
        return order


def make_engine(exchange, **kw):
    defaults = dict(
        symbol="DOGEUSDT",
        grid_lower=0.0710,
        grid_upper=0.0730,
        grid_count=10,
        capital_per_grid_pct=0.05,
        stop_loss_pct=0.05,
        max_exposure_pct=1.0,
    )
    defaults.update(kw)
    return GridEngine(exchange=exchange, **defaults)


# --- #11: reduceOnly hard-coded on every replacement sell ------------------

def test_replacement_sell_is_not_reduce_only_while_net_short():
    """A SELL while net SHORT opens exposure, so reduceOnly is illegal (-2022).

    The engine used to set reduceOnly on every replacement sell unconditionally, so
    the whole sell side was un-armable while short: 265 rejections in one session.
    """
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=0.0, short_position=5000.0, max_position_qty=8000.0)

    params, qty = grid._exit_order_params("sell", 1500.0)
    assert params is None, "sell while short opens exposure — must not be reduceOnly"
    assert qty == 1500.0


def test_replacement_sell_is_reduce_only_while_net_long():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.set_position_limit(long_position=5000.0, short_position=0.0, max_position_qty=8000.0)

    params, qty = grid._exit_order_params("sell", 1500.0)
    assert params == {"reduceOnly": True, "postOnly": False}
    assert qty == 1500.0


def test_reduce_only_quantity_is_clamped_to_remaining_position():
    """Binance also rejects a reduceOnly order larger than the position."""
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.set_position_limit(long_position=800.0, short_position=0.0, max_position_qty=8000.0)

    params, qty = grid._exit_order_params("sell", 1500.0)
    assert params == {"reduceOnly": True, "postOnly": False}
    assert qty == 800.0, "must not ask to close more than is open"


def test_buy_while_net_long_is_not_reduce_only():
    """Mirror case: a BUY adds to a long, so it can never be reduceOnly."""
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.set_position_limit(long_position=5000.0, short_position=0.0, max_position_qty=8000.0)

    params, _ = grid._exit_order_params("buy", 1000.0)
    assert params is None


def test_flat_position_never_sends_reduce_only():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.set_position_limit(long_position=0.0, short_position=0.0, max_position_qty=8000.0)

    assert grid._exit_order_params("sell", 100.0)[0] is None
    assert grid._exit_order_params("buy", 100.0)[0] is None


# --- #12: unwind over-allocated every exit level ---------------------------

def test_unwind_slices_the_position_not_the_grid_notional():
    """The unwind must distribute the ACTUAL position across exit levels.

    It used to size every level at the full grid notional, so it asked to close
    len(levels) x grid_qty against a smaller position and Binance rejected the
    overflow (105 rejections in one session).
    """
    position = 3000.0
    ex = FakeExchange(
        positions=[{"side": "long", "contracts": position, "entryPrice": 0.0723}],
        reject_reduce_only_over=position,
    )
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=position, short_position=0.0, max_position_qty=8000.0)

    grid._unwind_position_through_grid(balance=5000)

    reduce_orders = [o for o in ex.placed if o["params"].get("reduceOnly")]
    assert reduce_orders, "unwind should place exit orders"
    total = sum(o["amount"] for o in reduce_orders)
    assert total <= position + 1e-6, (
        f"unwind tried to close {total} against a {position} position"
    )


def test_unwind_places_no_order_exceeding_the_position():
    position = 500.0
    ex = FakeExchange(
        positions=[{"side": "long", "contracts": position, "entryPrice": 0.0723}],
        reject_reduce_only_over=position,
    )
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=position, short_position=0.0, max_position_qty=8000.0)

    grid._unwind_position_through_grid(balance=5000)
    for o in ex.placed:
        assert o["amount"] <= position + 1e-6


# --- #13: dead-grid false positive drove the recenter loop -----------------

def test_capped_long_with_exit_sells_is_not_a_dead_grid():
    """The exact production state that looped: long at the cap, buys blocked by the
    position limit, exit sells resting above price. That is a healthy unwind, not a
    dead grid — recentering it cancels the exits that were about to fill."""
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=9148.0, short_position=0.0, max_position_qty=8197.0)
    assert grid._block_buys, "precondition: the position cap blocked the buy side"

    for lv in grid.levels:
        lv.order_id = "x" if lv.side == "sell" else None

    grid._last_recenter_time = 0.0
    assert grid.recenter(0.0719, balance=5000, margin_pct=0.02) is False, (
        "recenter fired on a healthy one-sided unwind — this is the 196s thrash loop"
    )


def test_genuinely_dead_grid_still_recenters():
    """The real failure the check exists for: no position, no reason for the missing
    side, and price stranded below every resting sell."""
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=0.0, short_position=0.0, max_position_qty=8000.0)
    assert not grid._block_buys

    for lv in grid.levels:
        lv.order_id = "x" if lv.side == "sell" else None

    grid._last_recenter_time = 0.0
    assert grid.recenter(0.0711, balance=5000, margin_pct=0.02) is True


def test_capped_short_with_exit_buys_is_not_a_dead_grid():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=0.0, short_position=9000.0, max_position_qty=8000.0)
    assert grid._block_sells

    for lv in grid.levels:
        lv.order_id = "x" if lv.side == "buy" else None

    grid._last_recenter_time = 0.0
    assert grid.recenter(0.0729, balance=5000, margin_pct=0.02) is False


# --- #14: trailing stop-loss was not a ratchet -----------------------------

def test_long_trailing_stop_never_moves_down():
    ex = FakeExchange()
    grid = make_engine(ex)

    grid.update_trailing_sl(0.0730)
    high_water = grid.get_stop_loss_price()

    for price in (0.0725, 0.0718, 0.0705, 0.0690):
        grid.update_trailing_sl(price)
        assert grid.get_stop_loss_price() >= high_water - 1e-12, (
            f"long stop dropped from {high_water} to {grid.get_stop_loss_price()} at price {price}"
        )


def test_long_trailing_stop_still_rises_with_price():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.update_trailing_sl(0.0730)
    first = grid.get_stop_loss_price()
    grid.update_trailing_sl(0.0800)
    assert grid.get_stop_loss_price() > first


def test_short_trailing_stop_never_moves_up():
    ex = FakeExchange()
    grid = make_engine(ex)

    grid.update_trailing_sl_short(0.0710)
    low_water = grid.get_short_stop_loss_price()

    for price in (0.0715, 0.0725, 0.0740):
        grid.update_trailing_sl_short(price)
        assert grid.get_short_stop_loss_price() <= low_water + 1e-12


def test_side_flip_releases_the_ratchet():
    """reset_trailing() is the one sanctioned way to release the ratchet."""
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.update_trailing_sl(0.0730)
    assert grid.get_stop_loss_price() > 0

    grid.reset_trailing()
    assert grid._trailing_sl_price is None
    assert grid._peak_price == 0.0


# --- #15: recenter reset the trailing anchor mid-position ------------------

def test_recenter_preserves_trailing_stop_while_long_is_open():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=5000.0, short_position=0.0, max_position_qty=8000.0)

    grid.update_trailing_sl(0.0760)
    protected = grid.get_stop_loss_price()

    grid._last_recenter_time = 0.0
    grid.recenter(0.0700, balance=5000, margin_pct=0.001)

    assert grid._peak_price == pytest.approx(0.0760), "recenter wiped the high-water mark"
    assert grid.get_stop_loss_price() >= protected - 1e-12


def test_recenter_reanchors_trailing_stop_when_flat():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=0.0, short_position=0.0, max_position_qty=8000.0)
    grid.update_trailing_sl(0.0760)

    grid._last_recenter_time = 0.0
    grid.recenter(0.0700, balance=5000, margin_pct=0.001)

    assert grid._trailing_sl_price is None, "with no position open the anchor should reset"
