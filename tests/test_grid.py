import pytest

from grid import (
    GridLevel,
    GridEngine,
    calculate_grid_range,
    calculate_dynamic_grid_count,
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


def test_dynamic_grid_count_keeps_levels_in_higher_vol():
    assert calculate_dynamic_grid_count(0.010, 20) == 20
    assert calculate_dynamic_grid_count(0.025, 20) == 19
    assert calculate_dynamic_grid_count(0.05, 20) == 18
    assert calculate_dynamic_grid_count(0.10, 4) >= 3


def test_update_volatility_calm_market_boosts_sizing():
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
    grid.update_volatility(0.0)
    assert grid._volatility_mult == pytest.approx(2.5)

    grid.update_volatility(0.005)
    assert grid._volatility_mult == pytest.approx(1.75)

    grid.update_volatility(0.05)
    assert grid._volatility_mult == pytest.approx(0.9)


def test_volatility_mult_scales_capital_per_grid():
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
        grid_count=3,
        capital_per_grid_pct=0.10,
        stop_loss_pct=0.03,
        max_exposure_pct=1.0,
    )
    grid._volatility_mult = 2.5
    usdt_per_grid = grid._calc_usdt_per_grid(1000)
    assert usdt_per_grid == pytest.approx(250.0)


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


def test_calc_usdt_per_grid_uses_larger_allocation():
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
        grid_count=3,
        capital_per_grid_pct=0.10,
        capital_per_grid_usdt=5,
        stop_loss_pct=0.03,
        leverage=2,
        max_exposure_pct=1.0,
    )
    grid._volatility_mult = 1.0
    usdt_per_grid = grid._calc_usdt_per_grid(1000)
    assert usdt_per_grid == pytest.approx(100.0)


def test_calc_usdt_per_grid_uses_fixed_override_when_larger():
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
        capital_per_grid_pct=0.01,
        capital_per_grid_usdt=20,
        stop_loss_pct=0.03,
        leverage=2,
    )
    grid._volatility_mult = 1.0
    usdt_per_grid = grid._calc_usdt_per_grid(1000)
    assert usdt_per_grid == pytest.approx(40.0)


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


def test_grid_engine_short_stop_loss_price_static():
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
    sl = grid.get_short_stop_loss_price()
    assert sl == 82000 * 1.03


def test_grid_engine_short_trailing_sl_follows_trough():
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
        trailing_sl_trigger_pct=0.05,
    )
    grid.update_trailing_sl_short(80000)
    sl1 = grid.get_short_stop_loss_price()
    assert sl1 == pytest.approx(80000 * 1.05)
    assert sl1 <= 82000 * 1.03, "trailing short SL must not exceed the static ceiling"

    grid.update_trailing_sl_short(78000)
    sl2 = grid.get_short_stop_loss_price()
    assert sl2 == pytest.approx(78000 * 1.05)
    assert sl2 < sl1, "short SL must tighten as price falls (locking in profit)"


def test_grid_engine_reset_trailing_clears_both_sides():
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
    grid.update_trailing_sl(81000)
    grid.update_trailing_sl_short(79000)
    grid.reset_trailing()
    assert grid._peak_price == 0.0
    assert grid._trough_price == 0.0
    assert grid._trailing_sl_price is None
    assert grid._trailing_sl_price_short is None


def test_grid_engine_scale_out_trail_price_long_anchors_above_hard():
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
    hard = 78000 * 0.97
    assert grid.get_scale_out_trail_price("long") == hard, "falls back to hard with no peak"

    grid.update_trailing_sl(81000)
    trail = grid.get_scale_out_trail_price("long")
    assert trail == pytest.approx(81000 * 0.97)
    assert trail > hard, "scale-out trail must sit above the hard stop once a peak is observed"

    grid.update_trailing_sl(81500)
    assert grid.get_scale_out_trail_price("long") == pytest.approx(81500 * 0.97)


def test_grid_engine_scale_out_trail_price_short_anchors_below_hard():
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
    hard = 82000 * 1.03
    assert grid.get_scale_out_trail_price("short") == hard, "falls back to hard with no trough"

    grid.update_trailing_sl_short(79000)
    trail = grid.get_scale_out_trail_price("short")
    assert trail == pytest.approx(79000 * 1.03)
    assert trail < hard, "scale-out trail must sit below the hard stop once a trough is observed"


def test_grid_engine_scale_out_trail_price_reset_returns_to_hard():
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
    grid.update_trailing_sl(81000)
    grid.update_trailing_sl_short(79000)
    grid.reset_trailing()
    assert grid.get_scale_out_trail_price("long") == 78000 * 0.97
    assert grid.get_scale_out_trail_price("short") == 82000 * 1.03


def test_get_exposure_pct_counts_short_positions():
    """Exposure must include short positions so risk gating does not see 0% exposure
    while the grid holds a short."""
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.2f}"

        def get_positions(self, symbol):
            return [{"side": "short", "contracts": 1000, "entryPrice": 0.07}]

    grid = GridEngine(
        exchange=FakeExchange(),
        symbol="DOGEUSDT",
        grid_lower=0.065,
        grid_upper=0.075,
        grid_count=10,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    pct = grid.get_exposure_pct(balance=100.0)
    assert pct == pytest.approx(1000 * 0.07 / 100.0)


def test_set_position_limit_blocks_sells_on_short():
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
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.set_position_limit(long_position=0.0, short_position=10.0, max_position_qty=10.0)
    assert grid._block_buys is False
    assert grid._block_sells is True
    assert grid._sell_scale == 0.0
    assert grid._buy_scale == 1.0


def test_set_position_limit_scales_sells_near_short_cap():
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
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.set_position_limit(long_position=0.0, short_position=7.5, max_position_qty=10.0)
    assert grid._block_sells is False
    assert grid._sell_scale == pytest.approx(0.5)
    assert grid._buy_scale == 1.0


def test_set_position_limit_cancels_resting_buys_when_long_capped():
    """Resting buy orders placed before the cap was hit keep filling and overshoot it
    (the live idle bug: 22639 DOGE vs a 17518 cap). Once the long cap is hit, the
    resting buys must be cancelled so the position stops growing.
    """
    class FakeExchange:
        def __init__(self):
            self.cancelled = []
            self.open = {"B1", "B2"}

        def get_open_order_ids(self, symbol):
            return set(self.open)

        def cancel_order(self, order_id, symbol):
            self.cancelled.append(order_id)
            self.open.discard(order_id)

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.levels = [
        GridLevel(price=100.0, side="buy", order_id="B1", quantity=10.0),
        GridLevel(price=104.0, side="buy", order_id="B2", quantity=10.0),
        GridLevel(price=108.0, side="sell", order_id="S1", quantity=10.0),
    ]

    grid.set_position_limit(long_position=50.0, short_position=0.0, max_position_qty=10.0)

    assert grid._block_buys is True
    assert ex.cancelled == ["B1", "B2"], "resting buys must be cancelled when the long cap is hit"
    assert grid.levels[0].order_id is None and grid.levels[0].status == "pending"
    assert grid.levels[1].order_id is None and grid.levels[1].status == "pending"
    assert grid.levels[2].order_id == "S1", "sell orders must be untouched"


def test_set_position_limit_cancels_resting_sells_when_short_capped():
    """Mirror: resting sells must be cancelled once the short cap is hit."""
    class FakeExchange:
        def __init__(self):
            self.cancelled = []
            self.open = {"S1", "S2"}

        def get_open_order_ids(self, symbol):
            return set(self.open)

        def cancel_order(self, order_id, symbol):
            self.cancelled.append(order_id)
            self.open.discard(order_id)

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.levels = [
        GridLevel(price=100.0, side="sell", order_id="S1", quantity=10.0),
        GridLevel(price=104.0, side="sell", order_id="S2", quantity=10.0),
        GridLevel(price=108.0, side="buy", order_id="B1", quantity=10.0),
    ]

    grid.set_position_limit(long_position=0.0, short_position=50.0, max_position_qty=10.0)

    assert grid._block_sells is True
    assert ex.cancelled == ["S1", "S2"], "resting sells must be cancelled when the short cap is hit"
    assert grid.levels[2].order_id == "B1", "buy orders must be untouched"


def test_set_position_limit_does_not_cancel_orders_below_cap():
    """A grid well below the cap must keep its resting orders."""
    class FakeExchange:
        def __init__(self):
            self.cancelled = []

        def get_open_order_ids(self, symbol):
            return {"B1"}

        def cancel_order(self, order_id, symbol):
            self.cancelled.append(order_id)

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.levels = [GridLevel(price=100.0, side="buy", order_id="B1", quantity=10.0)]

    grid.set_position_limit(long_position=1.0, short_position=0.0, max_position_qty=10.0)

    assert grid._block_buys is False
    assert ex.cancelled == []
    assert grid.levels[0].order_id == "B1"


def test_place_order_skips_sell_when_short_blocked():
    class FakeExchange:
        def __init__(self):
            class _ex:
                @staticmethod
                def amount_to_precision(symbol, amount):
                    return f"{amount:.6f}"
                @staticmethod
                def price_to_precision(symbol, price):
                    return f"{price:.2f}"
            self.exchange = _ex()

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None, max_attempts=1):
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid._block_sells = True
    level = GridLevel(price=115.0, side="sell", quantity=1.0)

    result = grid._place_order_for_level(level, balance=1000.0)

    assert result is False
    assert level.order_id is None


def test_place_order_scales_sell_quantity_by_sell_scale():
    class FakeExchange:
        def __init__(self):
            class _ex:
                @staticmethod
                def amount_to_precision(symbol, amount):
                    return f"{amount:.6f}"
                @staticmethod
                def price_to_precision(symbol, price):
                    return f"{price:.2f}"
            self.exchange = _ex()
            self.placed = []

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None, max_attempts=1):
            self.placed.append((side, price, amount))
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid._sell_scale = 0.5
    level = GridLevel(price=115.0, side="sell", quantity=1.0)

    result = grid._place_order_for_level(level, balance=1000.0)

    assert result is True
    side, price, amount = ex.placed[-1]
    assert side == "sell"
    assert float(amount) == pytest.approx(100.0 / 115.0 * 0.5, rel=1e-3)


def test_place_order_skips_when_scaled_below_min_notional():
    """Regression: position-limit scaling (_buy_scale/_sell_scale) can shrink an
    order's notional below the exchange minimum (5 USDT on Binance USDM), which
    is a guaranteed rejection (-4164). Must skip cleanly instead of spending an
    API round-trip on a placement that can never succeed (observed live: 84 of
    these rejections across the trade history).
    """
    class FakeExchange:
        def __init__(self):
            class _ex:
                @staticmethod
                def amount_to_precision(symbol, amount):
                    return f"{amount:.6f}"
                @staticmethod
                def price_to_precision(symbol, price):
                    return f"{price:.2f}"
            self.exchange = _ex()
            self.placed = []

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None, max_attempts=1):
            self.placed.append((side, price, amount))
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    # 100 USDT/grid * sell_scale 0.02 = 2 USDT notional, below the 5 USDT floor.
    grid._sell_scale = 0.02
    level = GridLevel(price=115.0, side="sell", quantity=1.0)

    result = grid._place_order_for_level(level, balance=1000.0)

    assert result is False
    assert ex.placed == [], "must not attempt a placement that's guaranteed to be rejected"


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


def test_handle_fill_occupied_replacement_keeps_levels_and_order_ids():
    """When a fill's replacement target is already occupied by an active order, the
    level must go pending at that price WITHOUT merging/dropping grid levels. Merging
    caused the grid to collapse (20 levels -> 3 in the live demo run) and lost the
    fill_count so no cycle ever completed. Occupied fills must preserve grid size and
    keep order ids tracked by exactly one level each.
    """
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.6f}"

        def place_limit_order(self, symbol, side, price, amount, params=None):
            return {"id": f"ORDER-{side.upper()}-{price:.6f}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=0.06928,
        grid_upper=0.07076,
        grid_count=18,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )

    grid.levels = [
        GridLevel(price=0.07014, side="buy", order_id="BUY-0.07014", quantity=239.0, entry_price=0.07014),
        GridLevel(price=0.07026, side="sell", order_id="SELL-0.07026", quantity=238.0, entry_price=0.07002),
        GridLevel(price=0.07034, side="sell", order_id="SELL-0.07034", quantity=237.0, entry_price=0.07002),
    ]

    grid._handle_fill(grid.levels[1], balance=5000.0)
    grid._handle_fill(grid.levels[1], balance=5000.0)

    order_ids = [level.order_id for level in grid.levels if level.order_id is not None]
    assert len(order_ids) == len(set(order_ids)), "no two levels may track the same order"
    assert "BUY-0.07014" in order_ids
    assert "SELL-0.07034" in order_ids
    assert len(grid.levels) == 3, "occupied fills must not merge/drop grid levels"


def test_dedupe_levels_merges_duplicate_price_side_slots():
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.6f}"

    grid = GridEngine(
        exchange=FakeExchange(),
        symbol="TEST",
        grid_lower=0.06928,
        grid_upper=0.07076,
        grid_count=18,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )

    grid.levels = [
        GridLevel(price=0.07014, side="buy", order_id="BUY-0.07014", quantity=239.0, entry_price=0.07014, fill_count=1, total_pnl=1.0),
        GridLevel(price=0.07014, side="buy", order_id=None, status="pending", quantity=0.0, entry_price=0.07014, fill_count=0, total_pnl=0.5),
        GridLevel(price=0.07026, side="sell", order_id="SELL-0.07026", quantity=238.0, entry_price=0.07002, fill_count=1),
    ]

    grid._dedupe_levels()

    buy_levels = [level for level in grid.levels if level.side == "buy" and level.price == 0.07014]
    assert len(buy_levels) == 1
    assert buy_levels[0].order_id == "BUY-0.07014"
    assert buy_levels[0].fill_count == 1
    assert buy_levels[0].total_pnl == pytest.approx(1.5)


def test_handle_fill_short_cycle_counts_pnl():
    """A sell fill opens a short; the replacement buy that closes it must be a
    completed cycle with profit = (short_entry - buy_exit) * qty. The level must
    remember the sell price as the short entry so the closing buy prices it right."""
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

    grid = GridEngine(
        exchange=FakeExchange(),
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )

    grid.levels = [
        GridLevel(price=105.0, side="sell", order_id="SELL-105", quantity=10.0, entry_price=0.0),
    ]

    first = grid._handle_fill(grid.levels[0], balance=1000.0)
    assert first["completed_cycle"] is False, "opening a short is not a completed cycle"
    assert first["profit"] == 0.0
    assert grid.levels[0].side == "buy"
    assert grid.levels[0].price == 100.0
    assert grid.levels[0].entry_price == 105.0, "short entry must be the sell fill price"
    assert grid.total_completed_cycles == 0

    second = grid._handle_fill(grid.levels[0], balance=1000.0)
    assert second["completed_cycle"] is True, "buy fill closing a short completes a cycle"
    assert second["profit"] == pytest.approx((105.0 - 100.0) * 10.0)
    assert grid.total_completed_cycles == 1
    assert grid.total_pnl == pytest.approx(50.0)
    assert grid.levels[0].side == "sell"
    assert grid.levels[0].entry_price == 100.0, "after the short is closed the sell level tracks the next long entry"


def test_handle_fill_long_cycle_still_counts_pnl():
    """Regression: a buy fill opening a long then the sell that closes it must
    still complete a cycle with profit = (sell_exit - buy_entry) * qty."""
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

    grid = GridEngine(
        exchange=FakeExchange(),
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )

    grid.levels = [
        GridLevel(price=100.0, side="buy", order_id="BUY-100", quantity=10.0, entry_price=100.0),
    ]

    first = grid._handle_fill(grid.levels[0], balance=1000.0)
    assert first["completed_cycle"] is False
    assert grid.levels[0].side == "sell"
    assert grid.levels[0].entry_price == 100.0, "long entry must be the buy fill price"

    second = grid._handle_fill(grid.levels[0], balance=1000.0)
    assert second["completed_cycle"] is True
    assert second["profit"] == pytest.approx((105.0 - 100.0) * 10.0)
    assert grid.total_completed_cycles == 1
    assert grid.total_pnl == pytest.approx(50.0)


def test_check_fills_processes_vanished_sell_as_fill_with_short_position():
    """When a sell order disappears and a short position exists (either encoding),
    check_fills must process it as a fill instead of marking the level dead."""
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.2f}"

        def __init__(self, positions):
            self.positions = positions

        def get_open_orders(self, symbol):
            return []

        def fetch_order(self, order_id, symbol):
            return None

        def get_positions(self, symbol):
            return self.positions

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None):
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

    grid = GridEngine(
        exchange=FakeExchange([{"side": "short", "contracts": 10.0, "entryPrice": 105.0}]),
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )

    grid.levels = [
        GridLevel(price=105.0, side="sell", order_id="SELL-105", quantity=10.0, entry_price=0.0),
    ]

    fills = grid.check_fills(balance=1000.0)
    assert len(fills) == 1
    assert fills[0]["side"] == "sell"
    assert grid.total_completed_cycles == 0, "opening fill, not a completed cycle"


def test_check_fills_vanished_sell_with_negative_contract_short():
    """Short positions encoded as side='long' with negative contracts must also be
    recognized when a sell order vanishes."""
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.2f}"

        def __init__(self, positions):
            self.positions = positions

        def get_open_orders(self, symbol):
            return []

        def fetch_order(self, order_id, symbol):
            return None

        def get_positions(self, symbol):
            return self.positions

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None):
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

    grid = GridEngine(
        exchange=FakeExchange([{"side": "long", "contracts": -10.0, "entryPrice": 105.0}]),
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )

    grid.levels = [
        GridLevel(price=105.0, side="sell", order_id="SELL-105", quantity=10.0, entry_price=0.0),
    ]

    fills = grid.check_fills(balance=1000.0)
    assert len(fills) == 1, "negative-contract short must be recognized as a short position"
    assert fills[0]["side"] == "sell"


def test_check_fills_marks_vanished_sell_dead_when_only_long_position():
    """A vanished sell order with only a long position open must stay marked dead
    (the position did not come from this sell order)."""
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.2f}"

        def __init__(self, positions):
            self.positions = positions

        def get_open_orders(self, symbol):
            return []

        def fetch_order(self, order_id, symbol):
            return None

        def get_positions(self, symbol):
            return self.positions

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None):
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

    grid = GridEngine(
        exchange=FakeExchange([{"side": "long", "contracts": 10.0, "entryPrice": 100.0}]),
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )

    grid.levels = [
        GridLevel(price=105.0, side="sell", order_id="SELL-105", quantity=10.0, entry_price=0.0),
    ]

    fills = grid.check_fills(balance=1000.0)
    assert fills == []
    assert grid.levels[0].order_id is None
    assert grid.levels[0].status == "pending"


def test_check_fills_retries_unfilled_levels():
    """A pending level that failed to place (quantity=0, no order_id) must be
    retried on the next check_fills cycle so a flaky backend does not kill the grid.
    """
    class FakeExchange:
        def __init__(self):
            class _ex:
                @staticmethod
                def amount_to_precision(symbol, amount):
                    return f"{amount:.6f}"
                @staticmethod
                def price_to_precision(symbol, price):
                    return f"{price:.2f}"
            self.exchange = _ex()
            self.placed = []

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None, max_attempts=1):
            self.placed.append((side, price, amount))
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
        replacement_cooldown=0,
    )
    grid.levels = [
        GridLevel(price=105.0, side="buy", status="pending"),
        GridLevel(price=115.0, side="sell", status="pending", quantity=1.0),
    ]

    fills = grid.check_fills(balance=1000.0)

    assert ex.placed, "failed/pending levels should be retried"
    assert fills == []


def test_place_order_adopts_existing_open_order_instead_of_duplicating():
    """If an order already rests on the exchange at a level's (price, side), the
    level must adopt it rather than stacking a duplicate (e.g. after a placement
    whose response was lost to a network timeout).
    """
    class FakeExchange:
        def __init__(self):
            class _ex:
                @staticmethod
                def amount_to_precision(symbol, amount):
                    return f"{amount:.6f}"
                @staticmethod
                def price_to_precision(symbol, price):
                    return f"{price:.2f}"
            self.exchange = _ex()
            self.placed = []

        def get_open_orders(self, symbol):
            return [{"id": "EXISTING-1", "price": 105.0, "side": "buy", "amount": 10.0}]

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None, max_attempts=1):
            self.placed.append((side, price, amount))
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    level = GridLevel(price=105.0, side="buy")

    result = grid._place_order_for_level(level, balance=1000.0)

    assert result is True
    assert level.order_id == "EXISTING-1"
    assert level.quantity == 10.0
    assert ex.placed == [], "must not place a duplicate order on top of an existing one"


def test_place_order_skips_adoption_when_order_already_tracked():
    """A pending level must NOT adopt an open order that another level already tracks,
    otherwise both levels would process the same fill (double PnL, double state).
    """
    class FakeExchange:
        def __init__(self):
            class _ex:
                @staticmethod
                def amount_to_precision(symbol, amount):
                    return f"{amount:.6f}"
                @staticmethod
                def price_to_precision(symbol, price):
                    return f"{price:.2f}"
            self.exchange = _ex()
            self.placed = []

        def get_open_orders(self, symbol):
            return [{"id": "SELL-11500", "price": 115.0, "side": "sell", "amount": 1.0}]

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None, max_attempts=1):
            self.placed.append((side, price, amount))
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    tracked = GridLevel(price=115.0, side="sell", order_id="SELL-11500", quantity=1.0)
    pending = GridLevel(price=115.0, side="sell", status="pending")
    grid.levels = [tracked, pending]

    result = grid._place_order_for_level(pending, balance=1000.0)

    assert result is False
    assert pending.order_id is None, "must not share an order id with another level"
    assert ex.placed == []


def test_recenter_aborts_on_dirty_book():
    """Recenter must refuse to rebuild the grid when orders are still open after
    pause() — re-placing on a dirty book is how duplicate levels historically piled up.
    """
    class FakeExchange:
        def __init__(self):
            class _ex:
                @staticmethod
                def amount_to_precision(symbol, amount):
                    return f"{amount:.6f}"
                @staticmethod
                def price_to_precision(symbol, price):
                    return f"{price:.2f}"
            self.exchange = _ex()

        def cancel_everything(self, symbol, timeout_seconds=300.0):
            return 1

        def get_open_order_ids(self, symbol):
            return {"stale-1"}

        def cancel_order(self, order_id, symbol):
            return True

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.active = True
    grid.levels = [GridLevel(price=100.0, side="buy", order_id="stale-1")]

    result = grid.recenter(current_price=80.0, balance=1000.0)

    assert result is False
    assert grid.active is False
    assert len(grid.levels) == 1, "levels must not be rebuilt when the book is dirty"


def test_recenter_unwinds_position_through_reduceonly_sells():
    """When a long position is open and the book is clean, recenter must place
    reduce-only limit sells across the new grid's sell levels (sized to the normal
    grid allocation) instead of market-closing the inventory — no market close call.
    """
    class FakeExchange:
        def __init__(self):
            class _ex:
                @staticmethod
                def amount_to_precision(symbol, amount):
                    return f"{amount:.6f}"
                @staticmethod
                def price_to_precision(symbol, price):
                    return f"{price:.2f}"
            self.exchange = _ex()
            self.sell_params = []
            self.buy_params = []
            self.closed = None
            self._order_id = 0

        def cancel_everything(self, symbol, timeout_seconds=300.0):
            return 0

        def get_open_order_ids(self, symbol):
            return {}

        def get_positions(self, symbol):
            return [{"side": "long", "contracts": 1000.0, "entryPrice": 0.070}]

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None, max_attempts=1):
            self._order_id += 1
            if params and params.get("reduceOnly"):
                self.sell_params.append({"side": side, "price": price, "amount": amount, "params": params})
            elif side == "buy":
                self.buy_params.append({"side": side, "price": price, "amount": amount, "params": params})
            return {"id": f"ORDER-{self._order_id}"}

        def close_position(self, symbol, side=None):
            self.closed = (symbol, side)

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.active = True

    result = grid.recenter(current_price=80.0, balance=1000.0)

    assert result is True
    assert ex.closed is None, "must not market-close the position on recenter"
    sell_levels = [l for l in grid.levels if l.side == "sell"]
    buy_levels = [l for l in grid.levels if l.side == "buy"]
    assert len(ex.sell_params) == len(sell_levels), "reduce-only sells on every sell level"
    for entry in ex.sell_params:
        assert entry["side"] == "sell"
        assert entry["params"]["reduceOnly"] is True
        assert entry["params"]["postOnly"] is False
    assert len(ex.buy_params) == len(buy_levels), "buy levels below the new center should still be placed"
    for level in sell_levels:
        assert level.order_id is not None
        assert level.fill_count == 1
        assert level.entry_price == 0.070


def test_recenter_forced_when_grid_goes_one_sided_above():
    """A grid with zero active sell orders sitting under a price above the grid is
    dead even when price is inside the margin band — recenter must force through
    instead of waiting for the band edge (the live idle bug).
    """
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.6f}"

        def __init__(self):
            self._order_id = 0

        def cancel_everything(self, symbol, timeout_seconds=300.0):
            return 0

        def get_open_order_ids(self, symbol):
            return {}

        def get_positions(self, symbol):
            return []

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None):
            self._order_id += 1
            return {"id": f"ORDER-{self._order_id}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.active = True
    # All levels consumed to buys; price 110.8 is inside the 0.008 margin band
    # [89.28, 110.88] but above grid_upper 110.0 with no sell orders left.
    grid.levels = [
        GridLevel(price=100.0, side="buy", order_id="B1"),
        GridLevel(price=102.0, side="buy", order_id="B2"),
        GridLevel(price=104.0, side="buy", order_id="B3"),
        GridLevel(price=106.0, side="buy", order_id="B4"),
        GridLevel(price=108.0, side="buy", order_id="B5"),
    ]

    result = grid.recenter(current_price=110.8, balance=1000.0, margin_pct=0.008)

    assert result is True, "one-sided grid above price must recenter inside margin band"
    assert any(l.side == "sell" for l in grid.levels), "rebuilt grid must have sell levels"


def test_recenter_forced_when_grid_goes_one_sided_below():
    """Mirror of the above: all buy orders consumed (grid is all sells) and price
    sits below grid_lower inside the margin band — recenter must fire.
    """
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.6f}"

        def __init__(self):
            self._order_id = 0

        def cancel_everything(self, symbol, timeout_seconds=300.0):
            return 0

        def get_open_order_ids(self, symbol):
            return {}

        def get_positions(self, symbol):
            return []

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None):
            self._order_id += 1
            return {"id": f"ORDER-{self._order_id}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.active = True
    grid.levels = [
        GridLevel(price=98.0, side="sell", order_id="S1"),
        GridLevel(price=100.0, side="sell", order_id="S2"),
        GridLevel(price=102.0, side="sell", order_id="S3"),
        GridLevel(price=104.0, side="sell", order_id="S4"),
        GridLevel(price=106.0, side="sell", order_id="S5"),
    ]

    result = grid.recenter(current_price=89.2, balance=1000.0, margin_pct=0.008)

    assert result is True, "one-sided grid below price must recenter inside margin band"
    assert any(l.side == "buy" for l in grid.levels), "rebuilt grid must have buy levels"


def test_recenter_still_waits_for_margin_when_grid_has_both_sides():
    """With both buy and sell orders alive and price inside the margin band, recenter
    must still refuse — the forced path is only for one-sided grids.
    """
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.6f}"

        def cancel_everything(self, symbol, timeout_seconds=300.0):
            return 0

        def get_open_order_ids(self, symbol):
            return {}

        def get_positions(self, symbol):
            return []

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None):
            return {"id": "ORDER-1"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.active = True
    grid.levels = [
        GridLevel(price=100.0, side="buy", order_id="B1"),
        GridLevel(price=104.0, side="sell", order_id="S1"),
    ]

    result = grid.recenter(current_price=110.5, balance=1000.0, margin_pct=0.008)

    assert result is False, "balanced grid inside margin band must not recenter"


def test_recenter_forced_when_grid_dead_inside_band_below_sells():
    """The live idle bug: price wedged inside the band below every resting sell with
    no active buys — no order can fill, so recenter must fire even though price is
    inside both the grid and the margin band.
    """
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.6f}"

        def __init__(self):
            self._order_id = 0

        def cancel_everything(self, symbol, timeout_seconds=300.0):
            return 0

        def get_open_order_ids(self, symbol):
            return {}

        def get_positions(self, symbol):
            return []

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None):
            self._order_id += 1
            return {"id": f"ORDER-{self._order_id}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.active = True
    # All buys consumed; sells rest at 98-106. Price 95 is inside the grid and the
    # 0.008 margin band [89.28, 110.88] but below every sell: grid is dead.
    grid.levels = [
        GridLevel(price=98.0, side="sell", order_id="S1"),
        GridLevel(price=100.0, side="sell", order_id="S2"),
        GridLevel(price=102.0, side="sell", order_id="S3"),
        GridLevel(price=104.0, side="sell", order_id="S4"),
        GridLevel(price=106.0, side="sell", order_id="S5"),
    ]

    result = grid.recenter(current_price=95.0, balance=1000.0, margin_pct=0.008)

    assert result is True, "dead one-sided grid inside band must recenter"
    assert any(l.side == "buy" for l in grid.levels), "rebuilt grid must have buy levels"


def test_recenter_forced_when_grid_dead_inside_band_above_buys():
    """Mirror: price wedged inside the band above every resting buy with no active
    sells — no order can fill, recenter must fire.
    """
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.6f}"

        def __init__(self):
            self._order_id = 0

        def cancel_everything(self, symbol, timeout_seconds=300.0):
            return 0

        def get_open_order_ids(self, symbol):
            return {}

        def get_positions(self, symbol):
            return []

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None):
            self._order_id += 1
            return {"id": f"ORDER-{self._order_id}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.active = True
    # All sells consumed; buys rest at 94-102. Price 105 is inside the grid and the
    # margin band but above every buy: grid is dead.
    grid.levels = [
        GridLevel(price=94.0, side="buy", order_id="B1"),
        GridLevel(price=96.0, side="buy", order_id="B2"),
        GridLevel(price=98.0, side="buy", order_id="B3"),
        GridLevel(price=100.0, side="buy", order_id="B4"),
        GridLevel(price=102.0, side="buy", order_id="B5"),
    ]

    result = grid.recenter(current_price=105.0, balance=1000.0, margin_pct=0.008)

    assert result is True, "dead one-sided grid inside band must recenter"
    assert any(l.side == "sell" for l in grid.levels), "rebuilt grid must have sell levels"


def test_recenter_does_not_fire_when_sells_are_below_price_inside_band():
    """A one-sided grid is NOT dead when the resting side can still trade: all sells
    at 98-106 with price 108 inside the band means sells fill on the next touch —
    recenter must refuse.
    """
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.6f}"

        def cancel_everything(self, symbol, timeout_seconds=300.0):
            return 0

        def get_open_order_ids(self, symbol):
            return {}

        def get_positions(self, symbol):
            return []

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None):
            return {"id": "ORDER-1"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=90.0,
        grid_upper=110.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )
    grid.active = True
    grid.levels = [
        GridLevel(price=98.0, side="sell", order_id="S1"),
        GridLevel(price=100.0, side="sell", order_id="S2"),
        GridLevel(price=102.0, side="sell", order_id="S3"),
        GridLevel(price=104.0, side="sell", order_id="S4"),
        GridLevel(price=106.0, side="sell", order_id="S5"),
    ]

    result = grid.recenter(current_price=108.0, balance=1000.0, margin_pct=0.008)

    assert result is False, "one-sided grid that can still trade must not recenter"


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


def test_burst_of_fills_distributes_replacements_across_slots():
    """A burst of fills on one side must NOT pile several pending levels onto the
    same occupied (price, side). The live demo run collapsed this way: 6 sells filled
    and each parked pending at the same occupied buy slot (0.07009), so when that
    order finally filled every pending level placed a taker buy at once -> position
    explosion. After a fill, the replacement must move to the next free slot instead
    of stacking on the occupied one.
    """
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.6f}"

        def place_limit_order(self, symbol, side, price, amount, params=None):
            return {"id": f"ORDER-{side.upper()}-{price:.6f}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=140.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
        replacement_cooldown=0,
    )

    grid.levels = [
        GridLevel(price=100.0, side="buy", order_id="BUY-100", quantity=10.0, entry_price=100.0),
        GridLevel(price=110.0, side="buy", order_id="BUY-110", quantity=10.0, entry_price=110.0),
        GridLevel(price=120.0, side="sell", order_id="SELL-120", quantity=10.0, entry_price=110.0),
        GridLevel(price=130.0, side="sell", order_id="SELL-130", quantity=10.0, entry_price=110.0),
        GridLevel(price=140.0, side="sell", order_id="SELL-140", quantity=10.0, entry_price=110.0),
    ]

    grid._handle_fill(grid.levels[2], balance=1000.0)  # sell @ 120 fills
    grid._handle_fill(grid.levels[3], balance=1000.0)  # sell @ 130 fills
    grid._handle_fill(grid.levels[4], balance=1000.0)  # sell @ 140 fills

    order_ids = [l.order_id for l in grid.levels if l.order_id is not None]
    assert len(order_ids) == len(set(order_ids)), "no two levels may track the same order"

    pending = [l for l in grid.levels if l.order_id is None and l.status == "pending"]
    by_slot: dict[tuple[float, str], int] = {}
    for l in pending:
        by_slot[(l.price, l.side)] = by_slot.get((l.price, l.side), 0) + 1
    assert all(count == 1 for count in by_slot.values()), (
        f"at most one pending level per (price, side), got {by_slot}"
    )

    assert len(grid.levels) == 5, "fills must not merge/drop grid levels"
    assert "BUY-100" in order_ids and "BUY-110" in order_ids


def test_load_from_dict_recovers_duplicate_slots_without_rebuild():
    """When a restart saves state while a fill replacement is parked on an occupied
    slot, two levels share the same (price, side) and one grid line is left empty.
    load_from_dict must recover by merging the duplicate and refilling the empty
    grid line -- NOT rebuilding all levels (which lost fill bookkeeping in the live
    demo run: 20 levels -> 16 -> GRID STATE CORRUPT -> rebuild on every restart).
    """
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.6f}"

    grid = GridEngine(
        exchange=FakeExchange(),
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=140.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )

    state = {
        "grid_lower": 100.0,
        "grid_upper": 140.0,
        "grid_count": 5,
        "grid_spacing": 10.0,
        "active": True,
        "total_pnl": 3.5,
        "total_fees": 0.5,
        "total_fills": 12,
        "total_completed_cycles": 4,
        "_last_recenter_time": 0.0,
        "_volatility_mult": 1.0,
        "_trailing_sl_price": None,
        "_trailing_sl_trigger": 0.05,
        "_peak_price": 0.0,
        "_trough_price": 0.0,
        "_trailing_sl_price_short": None,
        "_block_buys": False,
        "_block_sells": False,
        "_last_replacement_time": 0.0,
        "_buy_scale": 1.0,
        "_sell_scale": 1.0,
        "levels": [
            {"price": 100.0, "side": "buy", "order_id": None, "status": "pending", "fill_count": 1, "total_pnl": 0.0, "quantity": 0.0, "entry_price": 100.0},
            {"price": 110.0, "side": "buy", "order_id": "BUY-110", "status": "replaced", "fill_count": 1, "total_pnl": 0.0, "quantity": 9.0, "entry_price": 110.0},
            {"price": 120.0, "side": "sell", "order_id": "SELL-120", "status": "replaced", "fill_count": 1, "total_pnl": 1.2, "quantity": 9.0, "entry_price": 110.0},
            {"price": 120.0, "side": "sell", "order_id": None, "status": "pending", "fill_count": 1, "total_pnl": 0.0, "quantity": 0.0, "entry_price": 110.0},
            {"price": 140.0, "side": "sell", "order_id": "SELL-140", "status": "replaced", "fill_count": 1, "total_pnl": 0.8, "quantity": 9.0, "entry_price": 110.0},
        ],
    }

    grid.load_from_dict(state, current_price=110.0)

    assert not grid.state_corrupted, "recoverable duplicates must not trigger a full rebuild"
    assert len(grid.levels) == 5, "grid must be refilled back to grid_count after merge"
    prices = sorted(l.price for l in grid.levels)
    assert prices == [100.0, 110.0, 120.0, 130.0, 140.0], f"empty line 130 must be refilled, got {prices}"
    assert grid.total_fills == 12, "fill bookkeeping must survive the recovery"
    assert grid.total_pnl == 3.5, "pnl bookkeeping must survive the recovery"

    sell_120 = [l for l in grid.levels if l.price == 120.0 and l.side == "sell"]
    assert len(sell_120) == 1, "duplicate (price, side) must be merged into one level"
    assert sell_120[0].order_id == "SELL-120", "active order id must be preserved on merge"
    assert sell_120[0].fill_count == 1, "fill_count must survive the merge"


def test_load_from_dict_still_rebuilds_when_count_cannot_be_recovered():
    """If levels are fundamentally broken (e.g. count mismatch that refilling cannot
    fix), the grid must still fall back to a full rebuild instead of trading with a
    wrong level count."""
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.6f}"

    grid = GridEngine(
        exchange=FakeExchange(),
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=140.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
    )

    state = {
        "grid_lower": 100.0,
        "grid_upper": 140.0,
        "grid_count": 5,
        "grid_spacing": 10.0,
        "active": True,
        "total_pnl": 0.0,
        "total_fees": 0.0,
        "total_fills": 0,
        "total_completed_cycles": 0,
        "_last_recenter_time": 0.0,
        "_volatility_mult": 1.0,
        "_trailing_sl_price": None,
        "_trailing_sl_trigger": 0.05,
        "_peak_price": 0.0,
        "_trough_price": 0.0,
        "_trailing_sl_price_short": None,
        "_block_buys": False,
        "_block_sells": False,
        "_last_replacement_time": 0.0,
        "_buy_scale": 1.0,
        "_sell_scale": 1.0,
        "levels": [
            {"price": 100.0, "side": "buy", "order_id": None, "status": "pending", "fill_count": 0, "total_pnl": 0.0, "quantity": 0.0, "entry_price": 100.0},
            {"price": 110.0, "side": "buy", "order_id": None, "status": "pending", "fill_count": 0, "total_pnl": 0.0, "quantity": 0.0, "entry_price": 110.0},
            {"price": 120.0, "side": "sell", "order_id": None, "status": "pending", "fill_count": 0, "total_pnl": 0.0, "quantity": 0.0, "entry_price": 100.0},
            {"price": 130.0, "side": "sell", "order_id": None, "status": "pending", "fill_count": 0, "total_pnl": 0.0, "quantity": 0.0, "entry_price": 100.0},
            {"price": 140.0, "side": "sell", "order_id": None, "status": "pending", "fill_count": 0, "total_pnl": 0.0, "quantity": 0.0, "entry_price": 100.0},
            {"price": 125.0, "side": "buy", "order_id": None, "status": "pending", "fill_count": 0, "total_pnl": 0.0, "quantity": 0.0, "entry_price": 125.0},
        ],
    }

    grid.load_from_dict(state, current_price=110.0)

    assert grid.state_corrupted, "more levels than grid_count must still trigger rebuild"
    assert len(grid.levels) == 5, "rebuild must restore grid_count levels"
    assert grid.state_corrupted


def test_replacement_cooldown_is_per_level_not_global():
    """A fill on one level must not block an unrelated orphaned level from being
    replaced. REPLACEMENT_COOLDOWN used to be a single engine-wide timer reset by
    every fill, so a burst of fills anywhere in the grid froze replacement of every
    OTHER level for the whole cooldown window -- exactly when the grid should be
    trading the most. Cooldown is now tracked per level.
    """
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.2f}"

        def __init__(self):
            self.placed = []

        def get_open_orders(self, symbol):
            # level_a's replacement order (placed inside _handle_fill) is still resting.
            return [{"id": "REPL-A"}]

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None, max_attempts=1):
            self.placed.append((side, price, amount))
            return {"id": "REPL-A"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
        replacement_cooldown=100,  # long cooldown so an accidental global block would be obvious
    )

    level_a = GridLevel(price=105.0, side="buy", order_id="BUY-105", quantity=1.0, entry_price=105.0)
    level_b = GridLevel(price=110.0, side="sell", order_id=None, status="pending", quantity=1.0)
    grid.levels = [level_a, level_b]

    # Fill level_a: this used to stamp a single engine-wide cooldown timestamp.
    grid._handle_fill(level_a, balance=1000.0)
    ex.placed.clear()

    # level_b is unrelated and was never itself replaced -- it must NOT be on cooldown.
    fills = grid.check_fills(balance=1000.0)

    assert fills == []
    assert any(price == 110.0 for _, price, _ in ex.placed), (
        "an unrelated orphaned level must be replaced right after another level's fill, "
        "not throttled by a shared global cooldown timer"
    )


def test_replacement_cooldown_still_blocks_the_same_level():
    """The level that was just replaced must still respect its own cooldown."""
    class FakeExchange:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{amount:.6f}"
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{price:.2f}"

        def __init__(self):
            self.placed = []

        def get_open_orders(self, symbol):
            return []

        def can_place_order(self, symbol):
            return True

        def place_limit_order(self, symbol, side, price, amount, params=None, max_attempts=1):
            self.placed.append((side, price, amount))
            return {"id": f"ORDER-{side.upper()}-{int(price*100)}"}

    ex = FakeExchange()
    grid = GridEngine(
        exchange=ex,
        symbol="TEST",
        grid_lower=100.0,
        grid_upper=120.0,
        grid_count=5,
        capital_per_grid_pct=0.1,
        stop_loss_pct=0.03,
        replacement_cooldown=100,
    )

    level = GridLevel(price=105.0, side="buy", order_id="BUY-105", quantity=1.0, entry_price=105.0)
    grid.levels = [level]

    grid._handle_fill(level, balance=1000.0)
    # _handle_fill's own replacement placement always fires regardless of cooldown
    # (that's the immediate flip, not the orphan-retry path); simulate that placement
    # having failed so the level falls back to the orphaned/pending retry path.
    level.order_id = None
    level.status = "pending"
    ex.placed.clear()

    fills = grid.check_fills(balance=1000.0)

    assert fills == []
    assert ex.placed == [], "the level that was just replaced must still respect its own cooldown"
