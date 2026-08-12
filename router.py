"""Regime router -- step 4 of the multi-strategy plan.

Routes control between strategies by market regime: the grid trades chop, the trend
follower trades direction. Before this existed the bot detected trends and responded by
switching *off*, leaving capital idle through exactly the conditions another strategy
could work in.

The router itself satisfies the `Strategy` protocol and delegates to whichever strategy
is live, so main.py drives it exactly as it drove a bare GridEngine. Anything not on
the protocol (`recenter`, `grid_lower`, `levels`, ...) falls through `__getattr__` to
the active strategy, which is why TrendFollower implements harmless equivalents of
those -- see strategy.GRID_SPECIFIC_MEMBERS.

THE HANDOFF IS THE DANGEROUS PART
Switching is not "start the other one". In one-way position mode there is a single net
position per symbol, so if the outgoing strategy's orders are still resting when the
incoming one opens a position, they fill against it: both strategies pay fees to
cancel each other out, and the reduce-only bookkeeping that AUDIT #11 fixed goes stale
because two writers now own one position.

So the sequence is strict, and it stops on failure rather than pressing on:

    pause outgoing        cancel its resting orders
    flatten              close the position it was holding
    verify flat          re-read the exchange; the exchange is the truth
    hand over            only now activate the incoming strategy

If the verify step still sees a position, the router stays in the paused state and
retries on the next tick. It never activates a strategy on top of another's position.

SWITCHING COSTS REAL MONEY
Every handoff pays a taker fee to flatten plus the spread to re-enter. A router that
flips on regime noise bleeds on transitions alone, which is why `min_regime_seconds`
requires a regime to persist before it is acted on -- on top of TrendFilter's own
`confirmation_seconds`. This is deliberately conservative: the backtest evidence in
AUDIT.md shows this project's noise floor exceeds the effect sizes involved, so a
router that switches rarely is the defensible default.
"""

from __future__ import annotations

import time

from loguru import logger

# Which strategy handles which regime. Regimes absent here route to the default.
DEFAULT_ROUTING = {
    "ranging": "grid",
    "uncertain": "grid",
    "uptrend": "trend",
    "downtrend": "trend",
}


class StrategyRouter:
    """Presents one Strategy to main.py; switches which one is really trading."""

    def __init__(
        self,
        strategies: dict,
        routing: dict | None = None,
        default: str = "grid",
        min_regime_seconds: int = 900,
        exchange=None,
        symbol: str = "",
        notifier: object | None = None,
        event_journal: object | None = None,
    ):
        if default not in strategies:
            raise ValueError(f"default strategy '{default}' is not in {sorted(strategies)}")
        self.strategies = strategies
        self.routing = dict(routing or DEFAULT_ROUTING)
        self.default = default
        self.min_regime_seconds = min_regime_seconds
        self.exchange = exchange
        self.symbol = symbol
        self._notifier = notifier
        self._event_journal = event_journal

        self.active_name = default
        self._regime = "uncertain"
        self._pending_name: str | None = None
        self._pending_since = 0.0
        self._handoff_target: str | None = None
        self.switches = 0
        self.failed_handoffs = 0

    # --- the live strategy -------------------------------------------------

    @property
    def strategy(self):
        return self.strategies[self.active_name]

    def _target_for(self, regime: str) -> str:
        name = self.routing.get(regime, self.default)
        return name if name in self.strategies else self.default

    def __getattr__(self, name: str):
        """Delegate anything not defined here to the live strategy.

        Covers the grid-specific surface main.py still calls. Guarded against
        recursion during __init__ before `strategies` exists.
        """
        if name.startswith("_") or name in ("strategies", "active_name"):
            raise AttributeError(name)
        try:
            strategies = object.__getattribute__(self, "strategies")
            active = object.__getattribute__(self, "active_name")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(strategies[active], name)

    # --- regime handling ---------------------------------------------------

    def update_regime(self, regime: str) -> None:
        """Record the regime and switch if it has held long enough to be worth paying
        the handoff for. Every strategy is told the regime, not just the live one, so
        an incoming strategy already knows its direction when it is activated."""
        self._regime = regime
        for s in self.strategies.values():
            s.update_regime(regime)

        target = self._target_for(regime)
        if target == self.active_name:
            self._pending_name = None
            self._pending_since = 0.0
            return

        now = time.time()
        if self._pending_name != target:
            self._pending_name = target
            self._pending_since = now
            if self.min_regime_seconds > 0:
                logger.info(
                    "ROUTER | regime={} suggests '{}' (holding {}s before switching)",
                    regime, target, self.min_regime_seconds,
                )

        # Falls through on the first sighting too, so min_regime_seconds=0 means
        # "switch immediately" rather than "switch on the second consecutive reading".
        if (now - self._pending_since) >= self.min_regime_seconds:
            self._begin_handoff(target)

        # Drive the handoff to completion here, not only from place_initial_orders.
        #
        # main.py calls place_initial_orders exactly once, at startup, before the loop
        # begins; every other strategy call in the loop sits behind `if grid.active:`.
        # Since _begin_handoff pauses the outgoing strategy, active goes False and none
        # of those run -- so a handoff driven only from place_initial_orders would never
        # advance and the bot would stop trading permanently on the first confirmed
        # trend. update_regime is the one call made unconditionally every iteration, so
        # progress is anchored to it. Caught by the router backtest (AUDIT #28).
        if self._handoff_target is not None:
            self._continue_handoff(self._current_balance())

    def _current_balance(self) -> float:
        """Balance for activating the incoming strategy.

        update_regime has no balance argument, so read it from the exchange. A failure
        here must not abort the handoff -- 0.0 still lets activate() run, and the next
        loop iteration supplies a real figure.
        """
        if self.exchange is None:
            return 0.0
        try:
            return float(self.exchange.get_balance())
        except Exception as e:
            logger.debug("ROUTER | balance read failed during handoff ({})", e)
            return 0.0

    def _begin_handoff(self, target: str) -> None:
        logger.info("ROUTER | handoff {} -> {} starting", self.active_name, target)
        self._handoff_target = target
        self.strategy.pause()

    def _flat_on_exchange(self) -> bool:
        """Re-read the exchange rather than trusting internal bookkeeping.

        Returns True only when genuinely flat. On an API failure it returns False --
        an unknown position must never be treated as no position.
        """
        if self.exchange is None:
            return True
        try:
            for p in self.exchange.get_positions(self.symbol):
                if abs(float(p.get("contracts", 0) or 0)) > 0:
                    return False
            return True
        except Exception as e:
            logger.warning("ROUTER | could not verify flat ({}) -- assuming not flat", e)
            return False

    def _continue_handoff(self, balance: float) -> None:
        """Drive the pause -> flatten -> verify -> activate sequence.

        Called every iteration while a handoff is in progress. Each stage is retried on
        the next tick rather than forced, so a failed close cannot leave two strategies
        sharing one position.
        """
        target = self._handoff_target
        if target is None:
            return

        if not self._flat_on_exchange():
            try:
                self.exchange.close_position(self.symbol)
                logger.info("ROUTER | flattening before handoff to {}", target)
            except Exception as e:
                self.failed_handoffs += 1
                logger.error("ROUTER | flatten failed ({}) -- handoff deferred", e)
                return
            if not self._flat_on_exchange():
                self.failed_handoffs += 1
                logger.warning("ROUTER | still not flat after close -- retrying next tick")
                return

        previous = self.active_name
        self.active_name = target
        self._handoff_target = None
        self._pending_name = None
        self._pending_since = 0.0
        self.switches += 1

        incoming = self.strategy
        incoming.reset_trailing()
        incoming.update_regime(self._regime)
        incoming.activate(balance)
        logger.info(
            "ROUTER | handoff complete {} -> {} | regime={} | switches={}",
            previous, target, self._regime, self.switches,
        )
        if self._notifier:
            try:
                self._notifier.send(
                    f"<b>STRATEGY SWITCH</b>\n{previous} -> {target}\nRegime: {self._regime}"
                )
            except Exception:
                pass

    @property
    def handoff_in_progress(self) -> bool:
        return self._handoff_target is not None

    # --- Strategy protocol -------------------------------------------------

    @property
    def active(self) -> bool:
        return self.strategy.active

    @active.setter
    def active(self, value: bool) -> None:
        self.strategy.active = value

    @property
    def state_corrupted(self) -> bool:
        return any(s.state_corrupted for s in self.strategies.values())

    def initialize(self, current_price: float, balance: float) -> None:
        for s in self.strategies.values():
            s.initialize(current_price, balance)

    def activate(self, balance: float) -> None:
        self.strategy.activate(balance)

    def pause(self) -> None:
        for s in self.strategies.values():
            s.pause()

    def emergency_stop(self) -> None:
        for s in self.strategies.values():
            s.emergency_stop()
        self._handoff_target = None

    def place_initial_orders(self, balance: float) -> int:
        """Advance any in-flight handoff first, so the outgoing strategy's position is
        gone before the incoming one is asked to place anything."""
        if self.handoff_in_progress:
            self._continue_handoff(balance)
            return 0
        return self.strategy.place_initial_orders(balance)

    def check_fills(self, balance: float) -> list[dict]:
        return self.strategy.check_fills(balance)

    def set_position_limit(
        self, long_position: float, short_position: float, max_position_qty: float,
    ) -> None:
        # Every strategy is kept current: a dormant one must not wake with a stale view
        # of the position, which is how reduce-only rejections start (AUDIT #11).
        for s in self.strategies.values():
            s.set_position_limit(long_position, short_position, max_position_qty)

    def get_exposure_pct(self, balance: float) -> float:
        return self.strategy.get_exposure_pct(balance)

    def update_volatility(self, atr_pct: float) -> None:
        for s in self.strategies.values():
            s.update_volatility(atr_pct)

    def get_stop_loss_price(self):
        return self.strategy.get_stop_loss_price()

    def get_short_stop_loss_price(self):
        return self.strategy.get_short_stop_loss_price()

    def get_hard_stop_loss_price(self):
        return self.strategy.get_hard_stop_loss_price()

    def get_short_hard_stop_loss_price(self):
        return self.strategy.get_short_hard_stop_loss_price()

    def update_trailing_sl(self, current_price: float) -> None:
        self.strategy.update_trailing_sl(current_price)

    def update_trailing_sl_short(self, current_price: float) -> None:
        self.strategy.update_trailing_sl_short(current_price)

    def reset_trailing(self) -> None:
        self.strategy.reset_trailing()

    def reconcile_state(self) -> None:
        self.strategy.reconcile_state()

    def reconcile_positions(self) -> None:
        self.strategy.reconcile_positions()

    def get_tracked_order_ids(self) -> set:
        ids: set = set()
        for s in self.strategies.values():
            ids |= s.get_tracked_order_ids()
        return ids

    # --- metrics (summed across strategies) --------------------------------

    @property
    def total_fills(self) -> int:
        return sum(s.total_fills for s in self.strategies.values())

    @property
    def total_pnl(self) -> float:
        return sum(s.total_pnl for s in self.strategies.values())

    @property
    def total_fees(self) -> float:
        return sum(s.total_fees for s in self.strategies.values())

    @property
    def total_completed_cycles(self) -> int:
        return sum(s.total_completed_cycles for s in self.strategies.values())

    # --- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise every strategy, plus which one is live.

        The default strategy's own keys are also merged at the top level. main.py
        restores a grid by reading saved_state["grid"]["grid_lower"] directly, so
        without them a router-mode state file could not be reloaded by a bot that had
        been switched back to grid mode -- and a state file that only one mode can read
        is a trap. `router` and `strategies` do not collide with any grid key.
        """
        base = dict(self.strategies[self.default].to_dict())
        base["router"] = {
            "active_name": self.active_name,
            "regime": self._regime,
            "handoff_target": self._handoff_target,
            "switches": self.switches,
            "failed_handoffs": self.failed_handoffs,
        }
        base["strategies"] = {name: s.to_dict() for name, s in self.strategies.items()}
        return base

    def load_from_dict(self, data: dict, current_price: float) -> None:
        router_state = (data or {}).get("router") or {}
        name = router_state.get("active_name", self.default)
        self.active_name = name if name in self.strategies else self.default
        self._regime = str(router_state.get("regime", "uncertain"))
        self._handoff_target = router_state.get("handoff_target")
        if self._handoff_target not in self.strategies:
            self._handoff_target = None
        self.switches = int(router_state.get("switches", 0))
        self.failed_handoffs = int(router_state.get("failed_handoffs", 0))

        sub_states = (data or {}).get("strategies") or {}
        if sub_states:
            for sub_name, sub_state in sub_states.items():
                if sub_name in self.strategies:
                    self.strategies[sub_name].load_from_dict(sub_state, current_price)
        elif data:
            # A state file written in grid mode: the top level *is* the grid's state.
            # Restoring it lets a bot switched from grid to router keep its levels
            # instead of silently rebuilding the book.
            self.strategies[self.default].load_from_dict(data, current_price)
