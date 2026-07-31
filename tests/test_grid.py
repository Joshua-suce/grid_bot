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
