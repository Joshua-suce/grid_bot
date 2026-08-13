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

# Public attributes that belong to the router itself. Everything else assigned on a
# router instance is forwarded to the live strategy -- see __setattr__.
_ROUTER_OWNED = frozenset({
    "strategies", "routing", "default", "min_regime_seconds", "handoff_grace_seconds",
    "exchange", "symbol", "active_name",
    "switches", "failed_handoffs", "deferred_ticks", "forced_flattens",
})


class StrategyRouter:
    """Presents one Strategy to main.py; switches which one is really trading."""

    def __init__(
        self,
        strategies: dict,
        routing: dict | None = None,
        default: str = "grid",
        min_regime_seconds: int = 900,
        handoff_grace_seconds: int = 21600,
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
        # How long to let the outgoing strategy unwind naturally before force-closing.
        self.handoff_grace_seconds = handoff_grace_seconds
        self.exchange = exchange
        self.symbol = symbol
        self._notifier = notifier
        self._event_journal = event_journal

        self.active_name = default
        self._regime = "uncertain"
        self._pending_name: str | None = None
        self._pending_since = 0.0
        self._handoff_target: str | None = None
        self._handoff_started = 0.0
        self.switches = 0
        self.failed_handoffs = 0
        self.deferred_ticks = 0
        self.forced_flattens = 0

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

        Private names are NOT forwarded: `_regime`, `_pending_name` and friends belong
        to the router, and forwarding them would mean an internal typo silently read a
        strategy's unrelated attribute instead of failing. Callers must use public
        members -- test_strategy.py enforces that main.py does (AUDIT #31).
        """
        if name.startswith("_") or name in ("strategies", "active_name"):
            raise AttributeError(name)
        try:
            strategies = object.__getattribute__(self, "strategies")
            active = object.__getattribute__(self, "active_name")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(strategies[active], name)

    def __setattr__(self, name: str, value) -> None:
        """Route writes to the live strategy, mirroring __getattr__.

        Without this, `grid.total_fills = old_fills` -- which main.py does when it
        rebuilds the engine after recovery -- lands in the router's own __dict__. From
        then on every read finds that stale shadow copy first and __getattr__ is never
        consulted, so the counters freeze at whatever was restored and the strategy's
        real figures never surface again. Same for `peak_price`, where the shadow would
        strand the stop ratchet's anchor on the wrong object.
        """
        if (name.startswith("_")
                or name in _ROUTER_OWNED
                or "strategies" not in self.__dict__):
            object.__setattr__(self, name, value)
            return

        declared = getattr(type(self), name, None)
        if isinstance(declared, property):
            if declared.fset is not None:
                object.__setattr__(self, name, value)   # the router's own setter
                return
            # A read-only aggregate like total_fills, which reads through to the live
            # strategy. Writes belong there too -- shadowing them on the router would
            # freeze the figure the property was written to expose.
        setattr(self.strategies[self.active_name], name, value)

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
            if self._handoff_target is not None:
                # The regime came back to the strategy already trading. Nothing to hand
                # over -- drop the pending switch so the grace clock stops and the
                # outgoing strategy's position stops being treated as inventory to
                # unwind. Cheaper than completing a switch we would only reverse.
                logger.info(
                    "ROUTER | handoff to {} cancelled -- regime returned to {}",
                    self._handoff_target, self.active_name,
                )
                self._handoff_target = None
                self._handoff_started = 0.0
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
        # The moment anything stands the outgoing strategy down -- the force-close path,
        # a failed flatten, an emergency stop -- active goes False and none of those run
        # again, so a handoff driven only from place_initial_orders would never advance
        # and the bot would stop trading permanently. update_regime is the one call made
        # unconditionally every iteration, so progress is anchored to it. Caught by the
        # router backtest (AUDIT #28).
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
        """Start a switch. Deliberately does NOT pause or flatten yet.

        The outgoing strategy keeps working until it is naturally flat. A grid
        accumulates inventory expecting to unwind it through its own levels; pausing
        and market-closing that inventory realises the loss the levels existed to
        avoid. Measured on DOGE 1h/90d: 18 of 44 handoffs dumped an open position,
        64,767 DOGE force-liquidated, -46.16 realised at those moments alone -- about
        half the router's total shortfall versus simply pausing the grid (AUDIT #29).

        So: wait for flat, and only force the close once handoff_grace_seconds has
        passed. The safety rule is unchanged -- the incoming strategy is activated only
        when the position is confirmed flat, so there is never a moment with two
        writers on one net position.
        """
        logger.info(
            "ROUTER | handoff {} -> {} starting (waiting for flat, grace {}s)",
            self.active_name, target, self.handoff_grace_seconds,
        )
        if self._handoff_target == target:
            # Already waiting for this one. Restarting the clock here would push the
            # deadline out every iteration and the grace period would never expire.
            return
        self._handoff_target = target
        self._handoff_started = time.time()

    def _close_live_position(self) -> bool:
        """Market-close whatever the exchange reports open. True if a close was sent.

        The router tracks no position of its own, so side and size come from the
        exchange -- which is the authority the flat check already trusts.
        """
        if self.exchange is None:
            return False
        for p in self.exchange.get_positions(self.symbol):
            qty = float(p.get("contracts", 0) or 0)
            if qty == 0:
                continue
            side = p.get("side", "")
            if qty < 0:
                side = "short" if side == "long" else "long"
                qty = abs(qty)
            self.exchange.close_position(self.symbol, side, abs(qty))
            return True
        return False

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
        """Wait for flat, then hand over. Force the close only once grace expires.

        Called every iteration while a handoff is pending. The outgoing strategy stays
        active and keeps unwinding through its own levels; we take the switch the
        moment it happens to be flat, which costs nothing. Only if it never gets there
        within handoff_grace_seconds do we pause and market-close it.

        Every stage is retried on the next tick rather than forced, so a failed close
        can never leave two strategies sharing one position.
        """
        target = self._handoff_target
        if target is None:
            return

        if not self._flat_on_exchange():
            waited = time.time() - self._handoff_started
            if waited < self.handoff_grace_seconds:
                # Still holding inventory and still inside grace: let the outgoing
                # strategy keep working it down through its own exits. Dumping here is
                # what cost -46.16 across 18 handoffs (AUDIT #29).
                self.deferred_ticks += 1
                return

            logger.warning(
                "ROUTER | handoff to {} still not flat after {:.0f}s grace — forcing close",
                target, waited,
            )
            self.forced_flattens += 1
            self.strategy.pause()
            try:
                # Same signature defect as trend_follower (AUDIT #38): this passed the
                # symbol alone, so the forced flatten raised TypeError every time, was
                # caught below, and the handoff deferred forever -- the grace period
                # would expire and then never actually resolve. The router holds no
                # position bookkeeping of its own, so read the side and size off the
                # exchange, which is the source of truth here anyway.
                closed = self._close_live_position()
                if not closed:
                    self.failed_handoffs += 1
                    logger.error("ROUTER | nothing to flatten or close failed -- handoff deferred")
                    return
                logger.info("ROUTER | flattening before handoff to {}", target)
            except Exception as e:
                self.failed_handoffs += 1
                logger.error("ROUTER | flatten failed ({}) -- handoff deferred", e)
                return
            if not self._flat_on_exchange():
                self.failed_handoffs += 1
                logger.warning("ROUTER | still not flat after close -- retrying next tick")
                return

        # Flat (naturally or forced): stand the outgoing strategy down and switch.
        self.strategy.pause()
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

    def emergency_stop(self, reason: str = "emergency") -> None:
        for s in self.strategies.values():
            s.emergency_stop(reason)
        self._handoff_target = None

    def place_initial_orders(self, balance: float) -> int:
        """Advance any in-flight handoff first, so the outgoing strategy's position is
        gone before the incoming one is asked to place anything."""
        if self.handoff_in_progress:
            self._continue_handoff(balance)
            if self.handoff_in_progress:
                # Still waiting for flat. The outgoing strategy stays live on purpose,
                # so let it keep re-arming its own exits -- refusing here would strand
                # the inventory it is trying to work down and guarantee the forced
                # close this design exists to avoid. What stops it *adding* exposure is
                # the cap clamp in set_position_limit, not silence here.
                if self.strategy.active:
                    return self.strategy.place_initial_orders(balance)
            # Just completed: activate() already placed for the incoming strategy.
            return 0
        return self.strategy.place_initial_orders(balance)

    def check_fills(self, balance: float) -> list[dict]:
        return self.strategy.check_fills(balance)

    def set_position_limit(
        self, long_position: float, short_position: float, max_position_qty: float,
    ) -> None:
        if self.handoff_in_progress:
            # Waiting for the outgoing strategy to reach flat. Clamp the cap to what is
            # already open so it can still close through its own levels but cannot open
            # anything new -- otherwise a grid keeps refilling the side it is meant to
            # be working down and never gets flat, and the grace period expires into
            # exactly the forced dump this was written to avoid.
            max_position_qty = min(max_position_qty, max(long_position, short_position))

        # Every strategy is kept current: a dormant one must not wake with a stale view
        # of the position, which is how reduce-only rejections start (AUDIT #11).
        for s in self.strategies.values():
            s.set_position_limit(long_position, short_position, max_position_qty)

    def block_side(self, side: str, reason: str) -> None:
        # Broadcast, for the same reason set_position_limit does: the position is NET
        # and shared, so an unprotected long is unprotected no matter which strategy
        # would be the one to add to it.
        for s in self.strategies.values():
            s.block_side(side, reason)

    def get_exposure_pct(self, balance: float) -> float:
        return self.strategy.get_exposure_pct(balance)

    def update_volatility(self, atr_pct: float) -> None:
        for s in self.strategies.values():
            s.update_volatility(atr_pct)

    def get_spread_pct(self) -> float:
        return self.strategy.get_spread_pct()

    @property
    def peak_price(self) -> float:
        return self.strategy.peak_price

    @peak_price.setter
    def peak_price(self, value: float) -> None:
        self.strategy.peak_price = value

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
            "forced_flattens": self.forced_flattens,
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
        self.forced_flattens = int(router_state.get("forced_flattens", 0))
        if self._handoff_target is not None:
            self._handoff_started = time.time()

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
