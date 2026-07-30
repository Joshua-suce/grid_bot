import pytest

from grid import (
    GridLevel,
    GridEngine,
    calculate_grid_range,
    validate_grid_spacing,
)
import pandas as pd
import numpy as np


def make_ohlcv(n=200):
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = 80000 + np.cumsum(np.random.randn(n) * 50)
    high = close + np.abs(np.random.randn(n) * 30)
    low = close - np.abs(np.random.randn(n) * 30)
    return pd.DataFrame({
        "timestamp": dates,
        "open": close + np.random.randn(n) * 20,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100, 1000, n).astype(float),
    })


def test_grid_level_to_dict():
    level = GridLevel(price=80000.0, side="buy")
    d = level.to_dict()
    assert d["price"] == 80000.0
    assert d["side"] == "buy"
    assert d["order_id"] is None


def test_grid_level_from_dict():
    d = {"price": 80000.0, "side": "buy", "order_id": None, "status": "pending", "fill_count": 0, "total_pnl": 0.0}
    level = GridLevel.from_dict(d)
    assert level.price == 80000.0
    assert level.side == "buy"


def test_calculate_grid_range():
    df = make_ohlcv(200)
    current_price = df["close"].iloc[-1]
    lower, upper = calculate_grid_range(df, current_price, lookback_days=14, atr_multiplier=1.5)
    assert lower < current_price
    assert upper > current_price
    assert lower > 0


def test_calculate_grid_range_short_data():
    df = make_ohlcv(5)
    current_price = df["close"].iloc[-1]
    lower, upper = calculate_grid_range(df, current_price, lookback_days=14, atr_multiplier=1.5)
    assert lower < current_price
    assert upper > current_price


def test_validate_grid_spacing_ok():
    assert validate_grid_spacing(76000, 84000, 15, 0.005, 80000) is True


def test_validate_grid_spacing_too_tight():
    assert validate_grid_spacing(79900, 80100, 100, 0.005, 80000) is False


def test_validate_grid_spacing_too_few():
    assert validate_grid_spacing(78000, 82000, 1, 0.005, 80000) is False


def test_grid_engine_initialization():
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.2f}"

    grid = GridEngine(
        exchange=FakeExchange(),
        symbol="BTCUSDT",
        grid_lower=78000,
        grid_upper=82000,
        grid_count=10,
        capital_per_grid_pct=0.05,
        stop_loss_pct=0.03,
    )
    grid.initialize(80000, 1000)
    assert len(grid.levels) == 10
    assert grid.grid_spacing > 0


def test_handle_fill_fills_buy_level_replaces_at_next_sell_level():
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.2f}"

        def place_limit_order(self, symbol, side, price, amount, params=None):
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

    exchange = FakeExchange()
    grid = GridEngine(
        exchange=exchange,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )

    # build a grid with multiple buy levels and sell levels
    grid.levels = [
        GridLevel(price=100.0, side="buy", quantity=1.0, entry_price=100.0),
        GridLevel(price=105.0, side="buy", quantity=1.0, entry_price=105.0),
        GridLevel(price=110.0, side="buy", quantity=1.0, entry_price=110.0),
        GridLevel(price=115.0, side="sell", quantity=1.0, entry_price=110.0),
        GridLevel(price=120.0, side="sell", quantity=1.0, entry_price=110.0),
    ]

    filled_level = grid.levels[1]
    filled_level.order_id = "BUY-105"
    result = grid._handle_fill(filled_level, balance=1000.0)

    assert result["side"] == "buy"
    assert filled_level.side == "sell"
    assert filled_level.price == 115.0
    assert filled_level.order_id == "ORDER-SELL-11500"


def test_grid_engine_stop_loss_price():
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.2f}"

    grid = GridEngine(
        exchange=FakeExchange(),
        symbol="BTCUSDT",
        grid_lower=78000,
        grid_upper=82000,
        grid_count=10,
        capital_per_grid_pct=0.05,
        stop_loss_pct=0.03,
    )
    grid.initialize(80000, 1000)
    sl = grid.get_stop_loss_price()
    assert sl == 78000 * 0.97


def test_grid_engine_to_dict():
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.2f}"

    grid = GridEngine(
        exchange=FakeExchange(),
        symbol="BTCUSDT",
        grid_lower=78000,
        grid_upper=82000,
        grid_count=5,
        capital_per_grid_pct=0.05,
        stop_loss_pct=0.03,
    )
    grid.initialize(80000, 1000)
    d = grid.to_dict()
    assert "levels" in d
    assert len(d["levels"]) == 5
    assert d["grid_spacing"] > 0


def test_reduceonly_sell_params_and_market_close():
    """Verify reduce-only sell placements include postOnly=False in params and that
    reconcile_positions uses market close when configured.
    """
    class FakeExchange:
        def __init__(self):
            self.last_place_params = None
            self.closed = None
            # simulate underlying helper
            class _ex:
                @staticmethod
                def amount_to_precision(symbol, amount):
                    return f"{amount:.6f}"
                @staticmethod
                def price_to_precision(symbol, price):
                    return f"{price:.2f}"
            self.exchange = _ex()

        def place_limit_order(self, symbol, side, price, amount, params=None):
            self.last_place_params = params
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

        def get_positions(self, symbol):
            return [{"side": "long", "contracts": 2.0, "entryPrice": 100.0}]

        def close_position(self, symbol, side, amount):
            self.closed = {"id": f"CLOSE-{int(amount)}"}
            return self.closed

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
        use_market_close_on_replace=False,
    )
    # make a buy-level near entry so reconcile finds it
    grid.levels = [
        GridLevel(price=95.0, side="buy", quantity=2.0, entry_price=95.0),
        GridLevel(price=100.0, side="buy", quantity=2.0, entry_price=100.0),
        GridLevel(price=105.0, side="sell", quantity=2.0, entry_price=100.0),
    ]

    # First: with market-close disabled, reconcile_positions should call place_limit_order
    grid.reconcile_positions()
    assert ex.last_place_params is not None
    # Should include reduceOnly and postOnly False
    assert ex.last_place_params.get("reduceOnly") is True
    assert ex.last_place_params.get("postOnly") is False

    # Now enable market close and ensure reconcile uses close_position
    ex.last_place_params = None
    grid.use_market_close_on_replace = True
    grid.reconcile_positions()
    assert ex.closed is not None
    # ensure when market close used, no new limit params recorded
    assert ex.last_place_params is None
