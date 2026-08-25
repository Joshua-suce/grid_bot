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
from trend_follower import TrendFollower


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
    def emergency_stop(self, reason="emergency"): self.active = False
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
    def get_hard_stop_loss_price(self): return None
    def get_short_hard_stop_loss_price(self): return None
    def update_trailing_sl(self, current_price): pass
    def update_trailing_sl_short(self, current_price): pass
    def reset_trailing(self): pass
    def reconcile_state(self): pass
    def reconcile_positions(self): pass
    def get_tracked_order_ids(self): return set()
    def get_spread_pct(self): return 0.0
    peak_price: float = 0.0
    def to_dict(self): return {"name": self.name}
    def load_from_dict(self, data, current_price): pass

    # grid-specific surface main.py reaches for
    def recenter(self, *a, **kw): return False
    levels: list = []


class FakeExchange:
    def __init__(self, position=0.0):
        self.position = position
        self.closes = 0
        self.close_args = []
        self.close_fails = False

    def get_positions(self, symbol):
        if self.position == 0:
            return []
        return [{"side": "long" if self.position > 0 else "short",
                 "contracts": abs(self.position), "entryPrice": 0.072}]

    def get_price(self, symbol):
        return 0.072

    def close_position(self, symbol, side, amount, max_attempts=None):
        """Signature mirrors the real Exchange exactly -- see AUDIT #38."""
        if self.close_fails:
            raise RuntimeError("exchange unreachable")
        self.closes += 1
        self.close_args.append((symbol, side, amount))
        self.position = 0.0
        return True


def make(position=0.0, min_regime_seconds=0, handoff_grace_seconds=0):
    """handoff_grace_seconds defaults to 0 so the force-close path is exercised
    without waiting; tests for the graceful path pass a real grace explicitly."""
    ex = FakeExchange(position)
    grid, trend = FakeStrategy("grid"), FakeStrategy("trend")
    r = StrategyRouter(
        strategies={"grid": grid, "trend": trend},
        min_regime_seconds=min_regime_seconds,
        handoff_grace_seconds=handoff_grace_seconds,
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
    """Once the grace period is spent, an unresolved position is closed outright."""
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=0)
    r.update_regime("uptrend")
    r.place_initial_orders(5000)
    assert ex.closes == 1
    assert r.active_name == "trend"
    assert r.forced_flattens >= 1


def test_a_healthy_position_is_not_dumped_to_make_the_switch():
    """AUDIT #29. The router used to market-close whatever the grid was holding the
    moment a regime was confirmed. Across 90 days of DOGE that dumped 64,767 DOGE over
    18 handoffs for -46.16 realised -- roughly half the router's entire shortfall
    against simply pausing the grid. The grid accumulates inventory *expecting* to
    unwind it through its own levels; flattening it realises exactly the loss those
    levels exist to avoid. So inside the grace period the switch waits."""
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=3600)
    r.update_regime("uptrend")
    r.place_initial_orders(5000)

    assert ex.closes == 0, "dumped an open position instead of waiting for it to unwind"
    assert r.active_name == "grid"
    assert r.handoff_in_progress, "the switch should still be pending, not abandoned"
    assert grid.active, "outgoing strategy must keep working to reach flat"


def test_the_switch_completes_for_free_once_the_position_unwinds_naturally():
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=3600)
    r.update_regime("uptrend")
    assert r.active_name == "grid"

    ex.position = 0.0            # the grid's own levels closed it out
    r.update_regime("uptrend")

    assert r.active_name == "trend"
    assert ex.closes == 0, "paid to close a position that had already gone flat"
    assert r.forced_flattens == 0


def test_the_grace_period_expires_rather_than_waiting_for_ever():
    """Waiting is the cheap path, not an excuse to never switch."""
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=60)
    r.update_regime("uptrend")
    assert r.active_name == "grid"

    r._handoff_started -= 61      # grace spent
    r.update_regime("uptrend")

    assert r.active_name == "trend"
    assert ex.closes == 1
    assert r.forced_flattens == 1


def test_the_grace_clock_is_not_restarted_by_repeated_ticks():
    """_begin_handoff is reached on every iteration while a switch is pending. If it
    reset the clock each time, the deadline would move away faster than time passed and
    the router would wait for ever holding a position it meant to hand over."""
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=60)
    r.update_regime("uptrend")
    started = r._handoff_started

    for _ in range(5):
        r.update_regime("uptrend")
    assert r._handoff_started == started


def test_waiting_strategy_may_close_but_not_open():
    """The wait only ends if the outgoing strategy actually reaches flat. A grid left
    at its normal cap keeps refilling the side it is meant to be working down, so the
    grace period would expire into the forced dump this was written to avoid. During a
    pending handoff the cap is clamped to what is already open: exits still fill,
    nothing new does."""
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=3600)
    r.update_regime("uptrend")
    r.set_position_limit(5000.0, 0.0, 20000.0)

    _, _, cap = grid.position_limits[-1]
    assert cap == 5000.0, "outgoing strategy was still allowed to add exposure"


def test_position_limit_is_untouched_when_no_handoff_is_pending():
    r, grid, trend, ex = make()
    r.set_position_limit(5000.0, 0.0, 20000.0)
    assert grid.position_limits[-1] == (5000.0, 0.0, 20000.0)


def test_a_pending_switch_is_cancelled_if_the_regime_comes_back():
    """Nothing to hand over if the regime returns to the strategy already trading --
    and the grace clock must stop, or the next trend inherits a spent one and dumps."""
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=3600)
    r.update_regime("uptrend")
    assert r.handoff_in_progress

    r.update_regime("ranging")

    assert not r.handoff_in_progress
    assert r.active_name == "grid"
    assert ex.closes == 0
    assert r._handoff_started == 0.0


def test_incoming_strategy_is_not_activated_while_a_position_remains():
    """The core safety property: never two writers on one net position."""
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=0)
    ex.close_fails = True
    r.update_regime("uptrend")
    r.place_initial_orders(5000)

    assert r.active_name == "grid", "handed over on top of an open position"
    assert trend.activated == 0
    # >=1 not ==1: the handoff is retried from update_regime as well as from
    # place_initial_orders (AUDIT #28), so the counter tracks attempts, not events.
    # What matters is that none of them handed over.
    assert r.failed_handoffs >= 1


def test_failed_handoff_retries_and_completes_once_flat():
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=0)
    ex.close_fails = True
    r.update_regime("uptrend")
    r.place_initial_orders(5000)
    assert r.handoff_in_progress

    ex.close_fails = False
    r.place_initial_orders(5000)
    assert r.active_name == "trend"
    assert not r.handoff_in_progress


def test_the_incoming_strategy_places_nothing_while_a_handoff_is_pending():
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=0)
    ex.close_fails = True
    r.update_regime("uptrend")
    before = trend.orders_placed
    r.place_initial_orders(5000)
    assert trend.orders_placed == before, "incoming strategy traded on top of a position"
    assert not grid.active, "a forced close must stand the outgoing strategy down"


def test_the_outgoing_strategy_keeps_working_while_waiting_for_flat():
    """It is still live for a reason: it is unwinding. Refusing to let it re-arm its
    own exits would strand the position and guarantee the forced close."""
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=3600)
    r.update_regime("uptrend")
    before = grid.orders_placed

    r.place_initial_orders(5000)

    assert grid.orders_placed > before
    assert trend.orders_placed == 0


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


# --- #28: the handoff must not depend on place_initial_orders --------------

def test_handoff_completes_without_place_initial_orders_ever_being_called():
    """main.py calls place_initial_orders once, at startup, before the loop.

    Every other strategy call in its loop sits behind `if grid.active:`, so the moment
    anything pauses the outgoing strategy -- the force-close path here, an emergency
    stop, a failed flatten -- none of them run again. A handoff driven only from
    place_initial_orders could never advance past that point: the bot would stop
    trading permanently on the first confirmed trend. update_regime is the one
    unconditional per-iteration call, so it must be able to carry a handoff to
    completion on its own.
    """
    r, grid, trend, ex = make(position=5000.0)

    r.update_regime("uptrend")   # the ONLY call -- no place_initial_orders anywhere

    assert r.active_name == "trend", "handoff stalled without place_initial_orders"
    assert not r.handoff_in_progress
    assert trend.active
    assert ex.closes == 1, "position was not flattened before handing over"


def test_update_regime_driven_handoff_still_refuses_to_hand_over_dirty():
    """Driving from update_regime must not weaken the flat-before-handover rule."""
    r, grid, trend, ex = make(position=5000.0)
    ex.close_fails = True

    r.update_regime("uptrend")

    assert r.active_name == "grid"
    assert trend.activated == 0
    assert r.handoff_in_progress, "should still be pending, not abandoned"


def test_update_regime_handoff_survives_an_unreadable_balance():
    """_current_balance falls back to 0.0 rather than aborting the handoff."""
    class NoBalance(FakeExchange):
        def get_balance(self):
            raise RuntimeError("balance endpoint down")

    r, grid, trend, _ = make()
    r.exchange = NoBalance()
    r.update_regime("uptrend")

    assert r.active_name == "trend"
    assert trend.active


# --- handoff acceleration (AUDIT #145) --------------------------------------

class BalancedExchange(FakeExchange):
    def get_balance(self):
        return 4866.53


class TrackingStrategy(FakeStrategy):
    """A strategy with a ladder to accelerate, unlike the plain FakeStrategy above."""
    def __init__(self, name):
        super().__init__(name)
        self.accel_calls = []

    def accelerate_handoff_exit(self, price, balance):
        self.accel_calls.append((price, balance))


def test_the_outgoing_strategy_is_offered_a_chance_to_accelerate():
    """Every deferred tick during a handoff must give the outgoing ladder a chance to
    reprice toward the market, not just wait silently on the grace clock."""
    ex = BalancedExchange(position=5000.0)
    grid, trend = TrackingStrategy("grid"), FakeStrategy("trend")
    r = StrategyRouter(
        strategies={"grid": grid, "trend": trend},
        min_regime_seconds=0, handoff_grace_seconds=3600, exchange=ex, symbol="DOGEUSDT",
    )
    grid.activate(5000)

    r.update_regime("uptrend")

    assert grid.accel_calls == [(0.072, 4866.53)], (
        "the outgoing strategy was never asked to accelerate its exit"
    )


def test_a_strategy_without_a_ladder_is_left_alone():
    """FakeStrategy has no accelerate_handoff_exit -- getattr must no-op, not raise.
    The real trend follower has had one since AUDIT #148
    (test_the_real_trend_follower_is_actually_reached below covers that); this pins
    the getattr fallback itself, for whatever future strategy still has nothing to
    accelerate."""
    r, grid, trend, ex = make(position=5000.0, handoff_grace_seconds=3600)

    r.update_regime("uptrend")   # must not raise AttributeError

    assert r.handoff_in_progress


def test_acceleration_is_not_offered_once_the_handoff_completes():
    """Once flat, the switch happens and there is nothing left to accelerate."""
    ex = BalancedExchange(position=5000.0)
    grid, trend = TrackingStrategy("grid"), FakeStrategy("trend")
    r = StrategyRouter(
        strategies={"grid": grid, "trend": trend},
        min_regime_seconds=0, handoff_grace_seconds=3600, exchange=ex, symbol="DOGEUSDT",
    )
    grid.activate(5000)
    r.update_regime("uptrend")
    assert grid.accel_calls, "sanity: it was offered a chance while still holding"

    ex.position = 0.0
    calls_before = len(grid.accel_calls)
    r.update_regime("uptrend")

    assert r.active_name == "trend"
    assert len(grid.accel_calls) == calls_before, "accelerated an exit after going flat"


class TrendExchange:
    """Enough surface for a real TrendFollower to hold a position and trade against,
    wired through a real StrategyRouter. Shared by every test that needs the real
    class instead of a fake with a deliberately-present accelerate_handoff_exit."""
    def __init__(self):
        self.price = 0.2182
        self.positions: list[dict] = []
        self._orders: dict[str, dict] = {}
        self._next = 0

        class _inner:
            @staticmethod
            def price_to_precision(symbol, price):
                return f"{float(price):.5f}"

            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{float(amount):.0f}"

        self.exchange = _inner()

    def get_price(self, symbol):
        return self.price

    def get_positions(self, symbol):
        return self.positions

    def get_balance(self):
        return 4866.53

    def get_open_order_ids(self, symbol):
        return {i for i, o in self._orders.items() if o["status"] == "open"}

    def fetch_order(self, order_id, symbol):
        return self._orders.get(order_id)

    def cancel_order(self, order_id, symbol):
        if order_id in self._orders:
            self._orders[order_id]["status"] = "canceled"
        return True

    def place_limit_order(self, symbol, side, price, amount, max_attempts=3,
                           params=None, post_only=True, allow_taker_fallback=False):
        self._next += 1
        oid = f"o{self._next}"
        order = {"id": oid, "side": side, "price": price, "amount": amount,
                 "filled": 0.0, "average": price, "status": "open"}
        self._orders[oid] = order
        return order

    def close_position(self, symbol, side, amount, max_attempts=None):
        self.positions = []
        return True


def test_the_real_trend_follower_is_actually_reached():
    """AUDIT #148. Every test above proves the WIRING with a fake that deliberately
    has accelerate_handoff_exit -- the actual regression was that the real trend
    follower never had one, so this reached for it via getattr and got nothing,
    every single time. A trend position sat through a full 31-minute handoff wait
    with zero "HANDOFF EXIT ACCELERATED" log lines, and the router forced a market
    close at grace expiry -- exactly what deferring the close exists to avoid.

    Wires a genuine TrendFollower in as the outgoing strategy, holding a real
    position the exchange says is not flat, and checks ITS take-profit actually gets
    repriced during the deferred tick -- not a stand-in that only proves the router
    calls something.
    """
    ex = TrendExchange()
    trend = TrendFollower(exchange=ex, symbol="ADAUSDT", min_hold_seconds=0)
    trend.active = True
    trend._side = "long"
    trend._entry_price = 0.2179
    trend._qty = 114.0

    grid = FakeStrategy("grid")
    r = StrategyRouter(
        strategies={"grid": grid, "trend": trend},
        min_regime_seconds=0, handoff_grace_seconds=3600, exchange=ex, symbol="ADAUSDT",
    )
    r.active_name = "trend"

    # The position exists on the exchange too, not just in the strategy's own
    # bookkeeping -- _flat_on_exchange reads the exchange, not the strategy.
    ex.positions = [{"side": "long", "contracts": 114.0, "entryPrice": 0.2179}]

    r.update_regime("ranging")   # targets "grid" -- trend must unwind first

    assert r.handoff_in_progress, "sanity: the handoff must still be pending, not forced closed"
    assert trend._take_profit_price is not None, (
        "the real trend follower's accelerate_handoff_exit was never reached"
    )
    assert any(o["side"] == "sell" for o in ex._orders.values()), (
        "no reduce-only exit was actually rested on the book"
    )


def test_a_cancelled_handoff_undoes_the_real_trend_followers_acceleration():
    """AUDIT #151. Continues the scenario above one step further: the regime that
    triggered the handoff flaps back before the position ever reaches flat. The
    router cancels the pending switch -- but the take-profit
    accelerate_handoff_exit manufactured a moment ago (trend_take_profit_r=0.0 here,
    the documented common case: no target at all) must not silently survive that
    cancellation and cap a position that is supposed to keep riding the trend.
    """
    ex = TrendExchange()
    trend = TrendFollower(exchange=ex, symbol="ADAUSDT", min_hold_seconds=0)
    trend.active = True
    trend._side = "long"
    trend._entry_price = 0.2179
    trend._qty = 114.0
    assert trend.take_profit_r == 0.0, "sanity: no target configured at all"

    grid = FakeStrategy("grid")
    r = StrategyRouter(
        strategies={"grid": grid, "trend": trend},
        min_regime_seconds=0, handoff_grace_seconds=3600, exchange=ex, symbol="ADAUSDT",
    )
    r.active_name = "trend"
    ex.positions = [{"side": "long", "contracts": 114.0, "entryPrice": 0.2179}]

    r.update_regime("ranging")   # begins a trend->grid handoff, accelerates
    assert r.handoff_in_progress
    assert trend._take_profit_price is not None, "sanity: acceleration fired"

    r.update_regime("uptrend")   # regime flaps back -- cancels the pending handoff

    assert r.handoff_in_progress is False
    assert r.active_name == "trend"
    assert trend._take_profit_price is None, (
        "a take-profit manufactured for a handoff that never completed survived it "
        "-- the position is now capped even though take_profit_r=0.0 promises no cap"
    )
    assert trend._tp_order_id is None, "the manufactured resting order was never disarmed"


# --- AUDIT #155/#156: the pending-switch confirmation clock survives a restart ----

import time as _time


def test_pending_switch_clock_round_trips_through_to_dict():
    r, grid, trend, ex = make(min_regime_seconds=900)
    r._pending_name = "trend"
    r._pending_since = _time.time() - 300

    saved = r.to_dict()

    r2, _, _, _ = make(min_regime_seconds=900)
    r2.load_from_dict(saved, current_price=0.072)

    assert r2._pending_name == "trend"
    assert r2._pending_since == pytest.approx(r._pending_since)


def test_a_stale_pending_clock_is_not_restored():
    """Down long enough that the gap could span a real reversal -- the elapsed time
    must not be trusted as continued confirmation the router never actually observed.
    """
    r, grid, trend, ex = make(min_regime_seconds=900)
    r._pending_name = "trend"
    r._pending_since = _time.time() - (StrategyRouter.PENDING_STALE_SECONDS + 60)
    saved = r.to_dict()

    r2, _, _, _ = make(min_regime_seconds=900)
    r2.load_from_dict(saved, current_price=0.072)

    assert r2._pending_name is None
    assert r2._pending_since == 0.0


def test_a_pending_clock_just_inside_the_staleness_window_is_restored():
    r, grid, trend, ex = make(min_regime_seconds=900)
    r._pending_name = "trend"
    r._pending_since = _time.time() - (StrategyRouter.PENDING_STALE_SECONDS - 60)
    saved = r.to_dict()

    r2, _, _, _ = make(min_regime_seconds=900)
    r2.load_from_dict(saved, current_price=0.072)

    assert r2._pending_name == "trend"


def test_a_future_pending_timestamp_is_rejected():
    r, grid, trend, ex = make(min_regime_seconds=900)
    r._pending_name = "trend"
    r._pending_since = _time.time() + 3600
    saved = r.to_dict()

    r2, _, _, _ = make(min_regime_seconds=900)
    r2.load_from_dict(saved, current_price=0.072)

    assert r2._pending_name is None
    assert r2._pending_since == 0.0


def test_a_pending_target_no_longer_in_strategies_is_dropped():
    r, grid, trend, ex = make(min_regime_seconds=900)
    r._pending_name = "trend"
    r._pending_since = _time.time() - 100
    saved = r.to_dict()
    saved["router"]["pending_name"] = "some_removed_strategy"

    r2, _, _, _ = make(min_regime_seconds=900)
    r2.load_from_dict(saved, current_price=0.072)

    assert r2._pending_name is None
    assert r2._pending_since == 0.0


def test_a_restored_pending_clock_lets_a_still_agreeing_regime_switch_immediately():
    """The whole point: a regime that had already held past min_regime_seconds before
    a restart should not need another full min_regime_seconds after it -- the very
    next observation that still agrees should complete the switch right away,
    instead of restarting a 15-minute wait from a blank clock."""
    r, grid, trend, ex = make(min_regime_seconds=900)
    r._pending_name = "trend"
    r._pending_since = _time.time() - 950   # already past min_regime_seconds=900
    saved = r.to_dict()

    r2, grid2, trend2, ex2 = make(min_regime_seconds=900)
    r2.load_from_dict(saved, current_price=0.072)
    assert r2.active_name == "grid"

    r2.update_regime("uptrend")   # target is still "trend" -- clock is already satisfied

    assert r2.handoff_in_progress or r2.active_name == "trend", (
        "the restored pending clock was not honoured -- update_regime treated this "
        "as a brand-new pending switch instead of continuing the one from before "
        "the restart"
    )


def test_a_restored_pending_clock_does_not_switch_early_on_its_own():
    """Restoring the clock must not itself trigger a switch -- only a subsequent
    update_regime() call (a fresh, real observation) can complete it."""
    r, grid, trend, ex = make(min_regime_seconds=900)
    r._pending_name = "trend"
    r._pending_since = _time.time() - 890
    saved = r.to_dict()

    r2, _, _, _ = make(min_regime_seconds=900)
    r2.load_from_dict(saved, current_price=0.072)

    assert r2.active_name == "grid"
    assert not r2.handoff_in_progress
