import pytest

import main as main_module
from main import get_net_position, get_position_details, get_total_position


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


def test_get_position_details_includes_short_positions():
    exchange = FakeExchange([
        {"side": "short", "contracts": 5, "entryPrice": 0.07},
        {"side": "long", "contracts": 3, "entryPrice": 0.069},
    ])
    details = get_position_details(exchange, "DOGEUSDT")
    assert len(details) == 2
    assert any(p["side"] == "short" and p["qty"] == 5.0 for p in details)
    assert any(p["side"] == "long" and p["qty"] == 3.0 for p in details)


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
