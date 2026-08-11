import pytest

import main as main_module
from main import (
    build_scale_out_orders,
    get_net_position,
    get_position_details,
    get_short_position,
    get_total_position,
)


class FakeExchange:
    def __init__(self, positions):
        self._positions = positions

    def get_positions(self, symbol):
        return self._positions


def test_get_total_position_counts_only_longs():
    exchange = FakeExchange([
        {"side": "long", "contracts": 10},
        {"side": "short", "contracts": 5},
    ])
    assert get_total_position(exchange, "DOGEUSDT") == 10.0


def test_get_net_position_long():
    exchange = FakeExchange([{"side": "long", "contracts": 10}])
    assert get_net_position(exchange, "DOGEUSDT") == ("long", 10.0)


def test_get_net_position_short():
    exchange = FakeExchange([{"side": "short", "contracts": 5}])
    assert get_net_position(exchange, "DOGEUSDT") == ("short", 5.0)


def test_get_net_position_flat():
    exchange = FakeExchange([])
    assert get_net_position(exchange, "DOGEUSDT") == ("", 0.0)


def test_get_net_position_negative_contracts_short():
    exchange = FakeExchange([{"side": "long", "contracts": -5}])
    assert get_net_position(exchange, "DOGEUSDT") == ("short", 5.0)


def test_get_net_position_nets_long_and_short():
    exchange = FakeExchange([
        {"side": "long", "contracts": 10},
        {"side": "short", "contracts": 4},
    ])
    assert get_net_position(exchange, "DOGEUSDT") == ("long", 6.0)


def test_get_short_position_returns_qty_and_entry():
    exchange = FakeExchange([{"side": "short", "contracts": 5, "entryPrice": 0.07}])
    assert get_short_position(exchange, "DOGEUSDT") == (5.0, 0.07)


def test_get_short_position_negative_contracts_encoding():
    exchange = FakeExchange([{"side": "long", "contracts": -5, "entryPrice": 0.07}])
    assert get_short_position(exchange, "DOGEUSDT") == (5.0, 0.07)


def test_get_short_position_ignores_longs_and_flat():
    exchange = FakeExchange([
        {"side": "long", "contracts": 10, "entryPrice": 0.06},
        {"side": "long", "contracts": -5, "entryPrice": 0.07},
    ])
    assert get_short_position(exchange, "DOGEUSDT") == (5.0, 0.07)
    assert get_short_position(FakeExchange([]), "DOGEUSDT") == (0.0, 0.0)


def test_get_short_position_weighted_entry_across_legs():
    exchange = FakeExchange([
        {"side": "short", "contracts": 100, "entryPrice": 0.08},
        {"side": "short", "contracts": 300, "entryPrice": 0.10},
    ])
    qty, entry = get_short_position(exchange, "DOGEUSDT")
    assert qty == 400.0
    assert entry == pytest.approx((100 * 0.08 + 300 * 0.10) / 400)


def test_get_position_details_includes_short_positions():
    exchange = FakeExchange([
        {"side": "short", "contracts": 5, "entryPrice": 0.07},
        {"side": "long", "contracts": 3, "entryPrice": 0.069},
    ])
    details = get_position_details(exchange, "DOGEUSDT")
    assert len(details) == 2
    assert any(p["side"] == "short" and p["qty"] == 5.0 for p in details)
    assert any(p["side"] == "long" and p["qty"] == 3.0 for p in details)


def test_build_scale_out_orders_splits_qty_at_half():
    orders = build_scale_out_orders("long", 16343.0, 0.5, trail_price=0.0700, hard_price=0.0680)
    assert orders == [("trail", 8171.5, 0.0700), ("hard", 8171.5, 0.0680)]


def test_build_scale_out_orders_uses_rounder_for_qty():
    orders = build_scale_out_orders(
        "long", 16343.0, 0.5, trail_price=0.0700, hard_price=0.0680,
        rounder=lambda q: int(q),
    )
    assert orders == [("trail", 8171, 0.0700), ("hard", 8172, 0.0680)]


def test_build_scale_out_orders_single_stop_when_levels_equal():
    orders = build_scale_out_orders("long", 1000.0, 0.5, trail_price=0.0680, hard_price=0.0680)
    assert orders == [("hard", 1000.0, 0.0680)]


def test_build_scale_out_orders_split_arms_with_startup_trail_price():
    orders = build_scale_out_orders(
        "long", 1000.0, 0.5, trail_price=0.0680, hard_price=0.0680,
        startup_trail_price=0.0687,
    )
    assert orders == [("trail", 500.0, 0.0687), ("hard", 500.0, 0.0680)]


def test_build_scale_out_orders_startup_trail_equal_to_hard_stays_single():
    orders = build_scale_out_orders(
        "long", 1000.0, 0.5, trail_price=0.0680, hard_price=0.0680,
        startup_trail_price=0.0680,
    )
    assert orders == [("hard", 1000.0, 0.0680)]


def test_build_scale_out_orders_ignores_startup_trail_when_trail_armed():
    orders = build_scale_out_orders(
        "long", 1000.0, 0.5, trail_price=0.0700, hard_price=0.0680,
        startup_trail_price=0.0687,
    )
    assert orders == [("trail", 500.0, 0.0700), ("hard", 500.0, 0.0680)]


def test_build_scale_out_orders_startup_trail_ignored_after_scale_out_done():
    orders = build_scale_out_orders(
        "long", 1000.0, 0.5, trail_price=0.0680, hard_price=0.0680,
        scale_out_done=True, startup_trail_price=0.0687,
    )
    assert orders == [("hard", 1000.0, 0.0680)]


def test_build_scale_out_orders_startup_trail_short_side():
    orders = build_scale_out_orders(
        "short", 8000.0, 0.5, trail_price=0.0740, hard_price=0.0740,
        startup_trail_price=0.0730,
    )
    assert orders == [("trail", 4000.0, 0.0730), ("hard", 4000.0, 0.0740)]


def test_build_scale_out_orders_single_stop_after_scale_out_done():
    orders = build_scale_out_orders("long", 8171.5, 0.5, trail_price=0.0700, hard_price=0.0680, scale_out_done=True)
    assert orders == [("hard", 8171.5, 0.0680)]


def test_build_scale_out_orders_short_side():
    orders = build_scale_out_orders("short", 8000.0, 0.5, trail_price=0.0720, hard_price=0.0740)
    assert orders == [("trail", 4000.0, 0.0720), ("hard", 4000.0, 0.0740)]


def test_build_scale_out_orders_zero_qty():
    assert build_scale_out_orders("long", 0.0, 0.5, trail_price=0.07, hard_price=0.068) == []


def test_build_scale_out_orders_clamps_scale_pct():
    orders = build_scale_out_orders("long", 100.0, 2.0, trail_price=0.07, hard_price=0.068)
    assert orders[0][1] == 95.0


class DirtyBookExchange:
    """Exchange stub that leaves stale orders open after cleanup, forcing the
    startup dirty-book abort path to trigger."""

    def __init__(self, config, demo=False):
        self.config = config
        self.demo = demo

    def set_leverage(self, symbol, leverage):
        pass

    def cancel_everything(self, symbol):
        return 0

    def close_all_positions(self, symbol):
        return 0

    def get_open_orders(self, symbol):
        return [{"id": "stale-1"}, {"id": "stale-2"}]


def test_run_bot_aborts_on_dirty_book(monkeypatch):
    """Startup must NOT continue past the cleanup check when the book is still dirty;
    if it did, the stub below (no get_ohlcv) would raise and fail this test."""
    monkeypatch.setattr(main_module, "Exchange", DirtyBookExchange)
    monkeypatch.setattr(main_module.settings, "telegram_enabled", False)
    monkeypatch.setattr(main_module, "setup_logging", lambda *a, **k: None)
    main_module.run_bot()
