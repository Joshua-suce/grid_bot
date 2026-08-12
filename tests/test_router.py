"""Tests for the regime router (step 4).

The handoff tests are the reason this file exists. In one-way position mode there is a
single net position per symbol, so activating a strategy while another still holds one
means both write to the same position -- fees paid to cancel each other out, and the
reduce-only bookkeeping AUDIT #11 fixed goes stale with two writers. The sequence must
stop rather than press on.
"""

import pytest

from router import DEFAULT_ROUTING, StrategyRouter
from strategy import Strategy


class FakeStrategy:
    def __init__(self, name):
        self.name = name
        self.active = False
        self.state_corrupted = False
        self.total_fills = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.total_completed_cycles = 0
        self.regime = None
        self.paused = 0
        self.activated = 0
        self.orders_placed = 0
        self.position_limits = []

    def initialize(self, current_price, balance): pass
    def activate(self, balance):
        self.active = True
        self.activated += 1
    def pause(self):
        self.active = False
        self.paused += 1
    def emergency_stop(self): self.active = False
    def place_initial_orders(self, balance):
        self.orders_placed += 1
        return 1
    def check_fills(self, balance): return []
    def set_position_limit(self, long_position, short_position, max_position_qty):
        self.position_limits.append((long_position, short_position, max_position_qty))
    def get_exposure_pct(self, balance): return 0.0
    def update_volatility(self, atr_pct): pass
    def update_regime(self, regime): self.regime = regime
    def get_stop_loss_price(self): return None
    def get_short_stop_loss_price(self): return None
    def update_trailing_sl(self, current_price): pass
    def update_trailing_sl_short(self, current_price): pass
    def reset_trailing(self): pass
    def reconcile_state(self): pass
    def reconcile_positions(self): pass
    def get_tracked_order_ids(self): return set()
    def to_dict(self): return {"name": self.name}
    def load_from_dict(self, data, current_price): pass

    # grid-specific surface main.py reaches for
    def recenter(self, *a, **kw): return False
    levels: list = []


class FakeExchange:
    def __init__(self, position=0.0):
        self.position = position
        self.closes = 0
        self.close_fails = False

    def get_positions(self, symbol):
        if self.position == 0:
            return []
        return [{"side": "long" if self.position > 0 else "short",
                 "contracts": abs(self.position), "entryPrice": 0.072}]

    def close_position(self, symbol):
        if self.close_fails:
            raise RuntimeError("exchange unreachable")
        self.closes += 1
        self.position = 0.0
        return True


def make(position=0.0, min_regime_seconds=0):
    ex = FakeExchange(position)
    grid, trend = FakeStrategy("grid"), FakeStrategy("trend")
    r = StrategyRouter(
        strategies={"grid": grid, "trend": trend},
        min_regime_seconds=min_regime_seconds,
        exchange=ex, symbol="DOGEUSDT",
    )
    grid.activate(5000)
    return r, grid, trend, ex


# --- routing ---------------------------------------------------------------

@pytest.mark.parametrize("regime,expected", [
    ("ranging", "grid"), ("uncertain", "grid"),
    ("uptrend", "trend"), ("downtrend", "trend"),
])
def test_regime_maps_to_the_right_strategy(regime, expected):
    r, *_ = make()
    r.update_regime(regime)
    r.place_initial_orders(5000)
    assert r.active_name == expected


def test_unknown_regime_falls_back_to_the_default():
    r, *_ = make()
    r.update_regime("something-new")
    r.place_initial_orders(5000)
    assert r.active_name == "grid"


def test_every_routed_name_exists():
    """A typo in the routing table must not silently route into nothing."""
    r, *_ = make()
    for name in DEFAULT_ROUTING.values():
        assert name in r.strategies


# --- the handoff sequence --------------------------------------------------

def test_outgoing_strategy_is_paused_before_the_switch():
    r, grid, trend, ex = make()
    r.update_regime("uptrend")
    assert grid.paused == 1, "outgoing strategy kept trading during handoff"


def test_position_is_flattened_before_handing_over():
    r, grid, trend, ex = make(position=5000.0)
    r.update_regime("uptrend")
    r.place_initial_orders(5000)
    assert ex.closes == 1
    assert r.active_name == "trend"


def test_incoming_strategy_is_not_activated_while_a_position_remains():
    """The core safety property: never two writers on one net position."""
    r, grid, trend, ex = make(position=5000.0)
    ex.close_fails = True
    r.update_regime("uptrend")
    r.place_initial_orders(5000)

    assert r.active_name == "grid", "handed over on top of an open position"
    assert trend.activated == 0
    assert r.failed_handoffs == 1


def test_failed_handoff_retries_and_completes_once_flat():
    r, grid, trend, ex = make(position=5000.0)
    ex.close_fails = True
    r.update_regime("uptrend")
    r.place_initial_orders(5000)
    assert r.handoff_in_progress

    ex.close_fails = False
    r.place_initial_orders(5000)
    assert r.active_name == "trend"
    assert not r.handoff_in_progress


def test_no_orders_are_placed_during_a_handoff():
    r, grid, trend, ex = make(position=5000.0)
    ex.close_fails = True
    r.update_regime("uptrend")
    before = grid.orders_placed + trend.orders_placed
    r.place_initial_orders(5000)
    assert grid.orders_placed + trend.orders_placed == before


def test_unverifiable_position_is_treated_as_not_flat():
    """An API failure must never read as 'no position'."""
    class Unreadable(FakeExchange):
        def get_positions(self, symbol):
            raise RuntimeError("API down")

    r, grid, trend, _ = make()
    r.exchange = Unreadable()
    r.update_regime("uptrend")
    r.place_initial_orders(5000)
    assert r.active_name == "grid"


def test_incoming_strategy_knows_the_regime_when_activated():
    r, grid, trend, ex = make()
    r.update_regime("downtrend")
    r.place_initial_orders(5000)
    assert trend.regime == "downtrend"
    assert trend.active


# --- switch damping --------------------------------------------------------

def test_regime_must_persist_before_a_switch_is_paid_for():
    r, grid, trend, ex = make(min_regime_seconds=3600)
    r.update_regime("uptrend")
    r.place_initial_orders(5000)
    assert r.active_name == "grid", "switched on a single regime reading"
    assert grid.paused == 0


def test_regime_flapping_back_cancels_the_pending_switch():
    r, grid, trend, ex = make(min_regime_seconds=3600)
    r.update_regime("uptrend")
    r.update_regime("ranging")
    assert r._pending_name is None


# --- delegation ------------------------------------------------------------

def test_router_satisfies_the_strategy_protocol():
    r, *_ = make()
    assert isinstance(r, Strategy)


def test_grid_specific_calls_fall_through_to_the_live_strategy():
    """main.py calls grid.recenter() unconditionally; the router must not break it."""
    r, grid, trend, ex = make()
    assert r.recenter(0.072, 5000, 0.01) is False
    assert r.levels == []


def test_all_strategies_receive_position_limits_not_just_the_live_one():
    """A dormant strategy must not wake holding a stale view of the position."""
    r, grid, trend, ex = make()
    r.set_position_limit(1000.0, 0.0, 8000.0)
    assert grid.position_limits == [(1000.0, 0.0, 8000.0)]
    assert trend.position_limits == [(1000.0, 0.0, 8000.0)]


def test_metrics_sum_across_strategies():
    r, grid, trend, ex = make()
    grid.total_fills, trend.total_fills = 10, 5
    grid.total_pnl, trend.total_pnl = 1.5, -0.5
    assert r.total_fills == 15
    assert r.total_pnl == pytest.approx(1.0)


def test_emergency_stop_hits_every_strategy():
    r, grid, trend, ex = make()
    r.emergency_stop()
    assert grid.active is False and trend.active is False


# --- persistence -----------------------------------------------------------

def test_state_round_trips_including_which_strategy_was_live():
    r, grid, trend, ex = make()
    r.update_regime("uptrend")
    r.place_initial_orders(5000)
    snapshot = r.to_dict()

    r2, *_ = make()
    r2.load_from_dict(snapshot, current_price=0.072)
    assert r2.active_name == "trend"
    assert r2.switches == r.switches


def test_restoring_an_unknown_strategy_name_falls_back_to_default():
    r, *_ = make()
    r.load_from_dict({"router": {"active_name": "does-not-exist"}}, 0.072)
    assert r.active_name == "grid"


def test_default_must_be_a_real_strategy():
    with pytest.raises(ValueError, match="not in"):
        StrategyRouter(strategies={"grid": FakeStrategy("grid")}, default="trend")


# --- main.py wiring --------------------------------------------------------

def test_install_strategy_returns_the_bare_engine_in_grid_mode(monkeypatch):
    """Default mode must be byte-identical to before the router existed."""
    import main
    from config import settings

    monkeypatch.setattr(settings, "strategy_mode", "grid")
    engine = object()
    assert main._install_strategy(engine, None, None, None) is engine


def test_install_strategy_wraps_in_a_router_when_asked(monkeypatch):
    import main
    from config import settings
    from grid import GridEngine

    monkeypatch.setattr(settings, "strategy_mode", "router")

    class _Ex:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount): return f"{float(amount):.0f}"
            @staticmethod
            def price_to_precision(symbol, price): return f"{float(price):.5f}"

    engine = GridEngine(
        exchange=_Ex(), symbol="DOGEUSDT", grid_lower=0.071, grid_upper=0.073,
        grid_count=10, capital_per_grid_pct=0.018, stop_loss_pct=0.03,
    )
    wrapped = main._install_strategy(engine, _Ex(), None, None)
    assert isinstance(wrapped, StrategyRouter)
    assert wrapped.strategies["grid"] is engine
    assert "trend" in wrapped.strategies
    assert wrapped.active_name == "grid", "must start on the grid, not mid-handoff"


def test_router_state_carries_grid_bounds_for_mains_restore_path(monkeypatch):
    """main.py restores via saved_state['grid']['grid_lower']. A router-mode state file
    must stay readable, or switching modes would strand it."""
    import main
    from config import settings
    from grid import GridEngine

    monkeypatch.setattr(settings, "strategy_mode", "router")

    class _Ex:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount): return f"{float(amount):.0f}"
            @staticmethod
            def price_to_precision(symbol, price): return f"{float(price):.5f}"

    engine = GridEngine(
        exchange=_Ex(), symbol="DOGEUSDT", grid_lower=0.071, grid_upper=0.073,
        grid_count=10, capital_per_grid_pct=0.018, stop_loss_pct=0.03,
    )
    state = main._install_strategy(engine, _Ex(), None, None).to_dict()
    for key in ("grid_lower", "grid_upper", "grid_count"):
        assert key in state, f"router state lost '{key}' -- main.py could not restore it"
    assert "router" in state and "strategies" in state
