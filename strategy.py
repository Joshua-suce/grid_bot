"""The interface a trading strategy must satisfy to be driven by main.py.

Step 2 of the multi-strategy plan. This module adds an abstraction and changes no
behaviour: `GridEngine` already satisfies `Strategy` as written, which is the point --
the protocol was derived from what main.py actually calls on it, not invented and then
imposed. `test_strategy.py` asserts that conformance so it cannot silently rot.

WHY A PROTOCOL AND NOT A BASE CLASS
`GridEngine` is 1600 lines of live-tested logic with real state files behind it.
Reparenting it onto an ABC would mean touching its constructor and MRO for no
behavioural gain. A structural (duck-typed) Protocol asserts the same contract without
modifying the class at all, so this step carries no runtime risk.

WHAT IS DELIBERATELY *NOT* HERE
main.py also reaches for grid-specific members: `recenter`, `grid_lower`/`grid_upper`/
`grid_count`/`grid_spacing`, `levels`, `log_sl_status`, `log_analytics`,
`get_scale_out_trail_price`, `update_orderbook`. Those describe a *ladder of resting
orders* and mean nothing to, say, a trend follower that holds one position.

Keeping them out of the protocol is what makes it useful: it draws the real line
between "every strategy needs this" and "this is how a grid happens to work". Step 3
(a second strategy) and step 4 (the regime router) have to deal with that list --
either by generalising each item or by having main.py stop calling it -- and the list
above is exactly the remaining work. See AUDIT.md.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Strategy(Protocol):
    """What main.py needs from anything that trades.

    Quantities are in base units (contracts/coins), prices in quote currency, and
    balances in USDT, matching the live exchange wrapper.
    """

    # --- lifecycle ---------------------------------------------------------

    active: bool
    """True when the strategy is armed and may place orders."""

    state_corrupted: bool
    """Set by load_from_dict when restored state is unusable; main.py deletes the
    state file and rebuilds rather than trading on it."""

    def initialize(self, current_price: float, balance: float) -> None:
        """Compute whatever structure the strategy needs at `current_price`.

        Must not place orders -- main.py arms separately so it can gate on the trend
        filter and risk checks between construction and the first live order.
        """
        ...

    def activate(self, balance: float) -> None:
        """Arm the strategy. Sets active=True."""
        ...

    def pause(self) -> None:
        """Stop trading and cancel resting orders, keeping any open position.

        Deliberately does NOT flatten: the position stays under stop-loss protection.
        A regime router that needs flat before handing over must close explicitly --
        see AUDIT.md on the handoff state machine.
        """
        ...

    def emergency_stop(self) -> None:
        """Cancel everything immediately. Called on shutdown and kill-switch trips."""
        ...

    # --- trading -----------------------------------------------------------

    def place_initial_orders(self, balance: float) -> int:
        """Place whatever orders the strategy wants resting. Returns the count placed.

        Must be idempotent: main.py calls it every iteration to re-arm levels that
        were skipped (crossing, cooldown, min-notional), so it must not duplicate
        orders that already exist.
        """
        ...

    def check_fills(self, balance: float) -> list[dict]:
        """Detect and process fills since the last call.

        Returns one dict per fill with at least `price`, `side`, `quantity`, `profit`,
        `fee` and `completed_cycle`.
        """
        ...

    # --- risk interface ----------------------------------------------------

    def set_position_limit(
        self, long_position: float, short_position: float, max_position_qty: float,
    ) -> None:
        """Report the live position and the cap, so the strategy can block the side
        that would breach it and size down as it approaches.

        Also how the strategy learns whether `reduceOnly` is legal on a given side --
        getting this stale is what produced 740 rejections (AUDIT #11).
        """
        ...

    def get_exposure_pct(self, balance: float) -> float:
        """Committed notional as a fraction of equity."""
        ...

    def update_volatility(self, atr_pct: float) -> None:
        """Supply current ATR as a fraction of price, for volatility-scaled sizing."""
        ...

    def update_regime(self, regime: str) -> None:
        """Report the current market regime: ranging / uptrend / downtrend / uncertain.

        Added so a strategy can learn the regime through the interface rather than the
        router reaching into concrete types. A grid ignores it (main.py already gates
        it externally); a trend follower needs it to pick a side.
        """
        ...

    # --- stops -------------------------------------------------------------

    def get_stop_loss_price(self) -> float | None:
        """Stop price for a long position, or None if not applicable."""
        ...

    def get_short_stop_loss_price(self) -> float | None:
        ...

    def get_hard_stop_loss_price(self) -> float | None:
        """Static (non-trailing) stop for a long -- the last line of defence.

        Must also be a ratchet: it may tighten while a position is open but never
        loosen. A grid derives it from its lower bound, which recentering moves, so
        without the ratchet a recenter pushes the stop away from the position it
        protects (AUDIT #25).
        """
        ...

    def get_short_hard_stop_loss_price(self) -> float | None:
        ...

    def update_trailing_sl(self, current_price: float) -> None:
        """Advance the long trailing stop. Must be a ratchet: never returns a lower
        stop while the position stays open (AUDIT #14)."""
        ...

    def update_trailing_sl_short(self, current_price: float) -> None:
        ...

    def reset_trailing(self) -> None:
        """Release the ratchet. Only legitimate once the position is closed."""
        ...

    # --- reconciliation ----------------------------------------------------

    def reconcile_state(self) -> None:
        """Verify tracked orders still exist on the exchange; repair what drifted."""
        ...

    def reconcile_positions(self) -> None:
        """Match open exchange positions to internal bookkeeping."""
        ...

    def get_tracked_order_ids(self) -> set:
        """Every order id this strategy believes it owns, so main.py can spot orphans
        on the exchange that belong to nobody."""
        ...

    # --- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise for the state file. Must round-trip through load_from_dict."""
        ...

    def load_from_dict(self, data: dict, current_price: float) -> None:
        """Restore from a state file, setting state_corrupted if unusable."""
        ...

    # --- metrics -----------------------------------------------------------

    total_fills: int
    total_pnl: float
    total_fees: float
    total_completed_cycles: int


# Members main.py currently uses that are specific to a ladder-of-orders strategy.
# Step 3/4 must either generalise each or stop calling it. Kept here as data so the
# conformance test can report the remaining coupling instead of it living in prose.
GRID_SPECIFIC_MEMBERS = (
    "recenter",
    "grid_lower",
    "grid_upper",
    "grid_count",
    "grid_spacing",
    "levels",
    "log_sl_status",
    "log_analytics",
    "get_scale_out_trail_price",
    "update_orderbook",
)
