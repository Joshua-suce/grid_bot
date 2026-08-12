"""Trend-following strategy -- step 3 of the multi-strategy plan.

The complement to the grid, not a replacement for it. A grid earns in chop and bleeds
in sustained direction; this does the reverse. It exists so the regime router
(router.py) has something to switch *to* when the trend filter fires, instead of the
account sitting idle through every trend -- which is what happens today.

MECHANICS
  entry   one position in the direction of the confirmed regime, sized as a fraction
          of equity and clamped by the position cap
  exit    whichever comes first: the trailing stop, or the regime ceasing to support
          the side being held
  stop    ATR-scaled trailing stop, ratcheted -- it never moves against the position
          while that position is open (the defect AUDIT #14 fixed in the grid; the
          same mistake is easy to repeat here, so it is enforced and tested)

FEES
Entries and exits cross the spread deliberately (`allow_taker_fallback=True`). That is
the opposite of the rule for grid levels, and correct for opposite reasons: a grid
level earns one grid spacing, so a taker fee consumes a large share of it. A trend
position targets multiples of ATR, so 0.04% is noise, and refusing to cross would mean
missing entries -- a far larger cost than the fee.

WHAT THIS DOES NOT DO
No pyramiding, no partial exits, no re-entry after a stop within the same regime.
One position, one stop, one exit. Those additions all have parameters, and the
backtest evidence in AUDIT.md shows this project's noise floor already exceeds the
effect sizes being chased -- so extra knobs would be unfalsifiable rather than useful.
"""

from __future__ import annotations

import time

from loguru import logger

MIN_NOTIONAL_USDT = 5.0

LONG_REGIMES = {"uptrend"}
SHORT_REGIMES = {"downtrend"}


class TrendFollower:
    """Holds at most one position, in the direction of the prevailing regime.

    Implements the `Strategy` protocol, plus no-op equivalents of the grid-specific
    members main.py reaches for, so it can be dropped in behind the router without
    main.py needing to know which strategy is live.
    """

    def __init__(
        self,
        exchange,
        symbol: str,
        capital_pct: float = 0.10,
        stop_loss_pct: float = 0.03,
        trailing_sl_trigger_pct: float = 0.05,
        atr_stop_multiplier: float = 2.0,
        leverage: int = 1,
        max_exposure_pct: float = 0.50,
        min_hold_seconds: int = 300,
        event_journal: object | None = None,
        notifier: object | None = None,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.capital_pct = capital_pct
        self.stop_loss_pct = stop_loss_pct
        self.atr_stop_multiplier = atr_stop_multiplier
        self.leverage = leverage
        self.max_exposure_pct = max_exposure_pct
        self.min_hold_seconds = min_hold_seconds
        self._trailing_sl_trigger = trailing_sl_trigger_pct

        # --- Strategy protocol state ---
        self.active = False
        self.state_corrupted = False
        self.total_fills = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.total_completed_cycles = 0

        # --- position / order tracking ---
        self._regime = "uncertain"
        self._side: str | None = None          # "long" | "short" | None
        self._entry_price = 0.0
        self._qty = 0.0
        self._order_id: str | None = None
        self._entry_time = 0.0

        # --- stops (ratcheted, see update_trailing_sl) ---
        self._peak_price = 0.0
        self._trough_price = 0.0
        self._trailing_sl_price: float | None = None
        self._trailing_sl_price_short: float | None = None

        self._atr_pct = 0.02
        self._net_long_qty = 0.0
        self._net_short_qty = 0.0
        self._max_position_qty = 0.0
        self._event_journal = event_journal
        self._notifier = notifier

    # --- helpers -----------------------------------------------------------

    def _round_price(self, price: float) -> float:
        return float(self.exchange.exchange.price_to_precision(self.symbol, price))

    def _round_amount(self, amount: float) -> float:
        return float(self.exchange.exchange.amount_to_precision(self.symbol, amount))

    def _desired_side(self) -> str | None:
        if self._regime in LONG_REGIMES:
            return "long"
        if self._regime in SHORT_REGIMES:
            return "short"
        return None

    def _stop_distance(self, price: float) -> float:
        """ATR-scaled, floored at stop_loss_pct so a quiet market cannot produce a stop
        so tight that noise closes the position immediately."""
        atr_stop = price * self._atr_pct * self.atr_stop_multiplier
        return max(atr_stop, price * self.stop_loss_pct)

    # --- lifecycle ---------------------------------------------------------

    def initialize(self, current_price: float, balance: float) -> None:
        self._peak_price = current_price
        self._trough_price = current_price
        logger.info(
            "TREND FOLLOWER INITIALIZED | {} @ {} | capital_pct={:.1%} atr_stop={}x",
            self.symbol, current_price, self.capital_pct, self.atr_stop_multiplier,
        )

    def activate(self, balance: float) -> None:
        self.active = True
        logger.info("TREND FOLLOWER ACTIVATED | regime={}", self._regime)

    def pause(self) -> None:
        """Cancel the resting entry order but keep any open position.

        Same contract as GridEngine.pause(): the position stays under stop protection.
        A router that needs flat must close explicitly.
        """
        if not self.active:
            return
        self._cancel_entry("pause")
        self.active = False
        logger.info("TREND FOLLOWER PAUSED | position={} qty={}", self._side, self._qty)

    def emergency_stop(self) -> None:
        self._cancel_entry("emergency_stop")
        try:
            self.exchange.cancel_everything(self.symbol, timeout_seconds=30)
        except Exception as e:
            logger.error("TREND FOLLOWER | cancel_everything failed: {}", e)
        self.active = False
        logger.error("TREND FOLLOWER EMERGENCY STOP")

    def _cancel_entry(self, reason: str) -> None:
        if self._order_id is None:
            return
        try:
            self.exchange.cancel_order(self._order_id, self.symbol)
        except Exception as e:
            logger.warning("TREND FOLLOWER | cancel {} failed: {}", self._order_id, e)
        if self._event_journal:
            self._event_journal.order_cancelled(
                self.symbol, self._side or "", self._entry_price, self._order_id, reason,
            )
        self._order_id = None

    # --- trading -----------------------------------------------------------

    def place_initial_orders(self, balance: float) -> int:
        """Open a position in the regime's direction, or close one the regime no longer
        supports. Idempotent: a no-op when already correctly positioned.
        """
        if not self.active:
            return 0

        want = self._desired_side()

        # Holding a side the regime no longer supports -> exit.
        if self._side is not None and want != self._side:
            if self._close_position("regime_change"):
                return 1
            return 0

        if self._side is not None or want is None:
            return 0
        if self._order_id is not None:
            return 0

        price = self.exchange.get_price(self.symbol)
        qty = self._entry_qty(balance, price)
        if qty <= 0:
            return 0

        side = "buy" if want == "long" else "sell"
        try:
            order = self.exchange.place_limit_order(
                self.symbol, side, self._round_price(price), qty,
                max_attempts=1, post_only=False, allow_taker_fallback=True,
            )
        except Exception as e:
            logger.error("TREND FOLLOWER | entry {} {} @ {} failed: {}", side, qty, price, e)
            if self._event_journal:
                self._event_journal.order_failed(self.symbol, side, price, qty, str(e))
            return 0

        self._order_id = order.get("id")
        logger.info(
            "TREND ENTRY PLACED | {} {} {} @ {} | regime={} (id={})",
            side.upper(), qty, self.symbol, price, self._regime, self._order_id,
        )
        if self._event_journal:
            self._event_journal.order_placed(self.symbol, side, price, qty, self._order_id)
        if self._notifier:
            self._notifier.on_order_placed(self.symbol, side, price, qty, self._order_id)
        return 1

    def _entry_qty(self, balance: float, price: float) -> float:
        if price <= 0 or balance <= 0:
            return 0.0
        notional = balance * self.capital_pct * self.leverage
        max_notional = balance * self.max_exposure_pct
        notional = min(notional, max_notional)
        qty = self._round_amount(notional / price)
        if self._max_position_qty > 0:
            qty = min(qty, self._round_amount(self._max_position_qty))
        if qty * price < MIN_NOTIONAL_USDT:
            logger.debug(
                "TREND FOLLOWER | entry notional {:.2f} below {:.2f} minimum -- skipping",
                qty * price, MIN_NOTIONAL_USDT,
            )
            return 0.0
        return qty

    def check_fills(self, balance: float) -> list[dict]:
        """Detect the entry filling, and enforce the trailing stop."""
        fills: list[dict] = []
        if self._order_id is not None:
            open_ids = self.exchange.get_open_order_ids(self.symbol)
            if self._order_id not in open_ids:
                order = self.exchange.fetch_order(self._order_id, self.symbol)
                if order and order.get("status") == "closed":
                    fills.append(self._record_entry(order))
                elif order is None or order.get("status") == "canceled":
                    self._order_id = None

        if self._side is not None:
            price = self.exchange.get_price(self.symbol)
            if self._side == "long":
                self.update_trailing_sl(price)
                stop = self.get_stop_loss_price()
                if stop is not None and price <= stop and self._can_exit():
                    exit_fill = self._close_position("trailing_stop")
                    if exit_fill:
                        fills.append(exit_fill)
            else:
                self.update_trailing_sl_short(price)
                stop = self.get_short_stop_loss_price()
                if stop is not None and price >= stop and self._can_exit():
                    exit_fill = self._close_position("trailing_stop")
                    if exit_fill:
                        fills.append(exit_fill)
        return fills

    def _can_exit(self) -> bool:
        """Guard against stopping out on the same tick the position opened.

        Without it a stop placed inside the current bid/ask closes instantly and the
        strategy pays two taker fees for zero exposure.
        """
        return (time.time() - self._entry_time) >= self.min_hold_seconds

    def _record_entry(self, order: dict) -> dict:
        price = float(order.get("average") or order.get("price") or 0.0)
        qty = float(order.get("filled") or order.get("amount") or 0.0)
        self._side = "long" if order.get("side") == "buy" else "short"
        self._entry_price = price
        self._qty = qty
        self._entry_time = time.time()
        self._order_id = None
        self.total_fills += 1

        # Anchor the ratchet at entry so the first stop is a real level, not zero.
        self._peak_price = price
        self._trough_price = price
        self._trailing_sl_price = None
        self._trailing_sl_price_short = None
        if self._side == "long":
            self.update_trailing_sl(price)
        else:
            self.update_trailing_sl_short(price)

        logger.info(
            "TREND ENTRY FILLED | {} {} @ {} | stop={}",
            self._side.upper(), qty, price,
            self.get_stop_loss_price() if self._side == "long" else self.get_short_stop_loss_price(),
        )
        return {
            "price": price, "side": order.get("side"), "quantity": qty,
            "profit": 0.0, "fee": 0.0, "completed_cycle": False,
        }

    def _close_position(self, reason: str) -> dict | None:
        if self._side is None:
            return None
        entry, qty, side = self._entry_price, self._qty, self._side
        try:
            self.exchange.close_position(self.symbol)
        except Exception as e:
            logger.error("TREND FOLLOWER | close failed: {}", e)
            return None

        exit_price = self.exchange.get_price(self.symbol)
        direction = 1.0 if side == "long" else -1.0
        profit = (exit_price - entry) * qty * direction

        self.total_fills += 1
        self.total_completed_cycles += 1
        self.total_pnl += profit
        logger.info(
            "TREND EXIT | {} {} @ {} (entry {}) | reason={} | pnl={:.6f}",
            side.upper(), qty, exit_price, entry, reason, profit,
        )
        if self._notifier:
            self._notifier.on_position_closed(self.symbol, side, qty, exit_price, profit)

        self._side = None
        self._entry_price = 0.0
        self._qty = 0.0
        self._entry_time = 0.0
        self.reset_trailing()
        return {
            "price": exit_price, "side": "sell" if side == "long" else "buy",
            "quantity": qty, "profit": profit, "fee": 0.0, "completed_cycle": True,
        }

    # --- risk interface ----------------------------------------------------

    def set_position_limit(
        self, long_position: float, short_position: float, max_position_qty: float,
    ) -> None:
        self._net_long_qty = max(0.0, long_position)
        self._net_short_qty = max(0.0, short_position)
        self._max_position_qty = max(0.0, max_position_qty)

    def get_exposure_pct(self, balance: float) -> float:
        if balance <= 0 or self._side is None:
            return 0.0
        return (self._qty * self._entry_price) / balance

    def update_volatility(self, atr_pct: float) -> None:
        if atr_pct > 0:
            self._atr_pct = atr_pct

    def update_regime(self, regime: str) -> None:
        self._regime = regime

    # --- stops (ratcheted) -------------------------------------------------

    def update_trailing_sl(self, current_price: float) -> None:
        """Long stop. Monotonic while the position is open -- it may rise, never fall.

        This is the AUDIT #14 defect restated: a stop recomputed from the current price
        each tick follows the market down and stops protecting anything.
        """
        if current_price > self._peak_price:
            self._peak_price = current_price
        if self._peak_price <= 0:
            return
        candidate = self._peak_price - self._stop_distance(self._peak_price)
        trigger = self._peak_price * (1 - self._trailing_sl_trigger)
        candidate = max(candidate, trigger)
        if self._trailing_sl_price is None:
            self._trailing_sl_price = candidate
        else:
            self._trailing_sl_price = max(self._trailing_sl_price, candidate)

    def update_trailing_sl_short(self, current_price: float) -> None:
        if self._trough_price <= 0 or current_price < self._trough_price:
            self._trough_price = current_price
        if self._trough_price <= 0:
            return
        candidate = self._trough_price + self._stop_distance(self._trough_price)
        trigger = self._trough_price * (1 + self._trailing_sl_trigger)
        candidate = min(candidate, trigger)
        if self._trailing_sl_price_short is None:
            self._trailing_sl_price_short = candidate
        else:
            self._trailing_sl_price_short = min(self._trailing_sl_price_short, candidate)

    def get_stop_loss_price(self) -> float | None:
        return self._trailing_sl_price

    def get_short_stop_loss_price(self) -> float | None:
        return self._trailing_sl_price_short

    def reset_trailing(self) -> None:
        self._peak_price = 0.0
        self._trough_price = 0.0
        self._trailing_sl_price = None
        self._trailing_sl_price_short = None

    # --- reconciliation ----------------------------------------------------

    def reconcile_state(self) -> None:
        if self._order_id is None:
            return
        if self._order_id in self.exchange.get_open_order_ids(self.symbol):
            return
        order = self.exchange.fetch_order(self._order_id, self.symbol)
        if order and order.get("status") == "closed":
            logger.info("TREND FOLLOWER | entry {} filled externally", self._order_id)
            self._record_entry(order)
        else:
            logger.warning("TREND FOLLOWER | entry {} vanished, clearing", self._order_id)
            self._order_id = None

    def reconcile_positions(self) -> None:
        """Adopt whatever the exchange actually holds. The exchange is the truth."""
        positions = self.exchange.get_positions(self.symbol)
        live = None
        for p in positions:
            contracts = float(p.get("contracts", 0) or 0)
            side = p.get("side", "")
            if contracts < 0:
                side, contracts = ("short" if side == "long" else "long"), abs(contracts)
            if contracts > 0:
                live = (side, contracts, float(p.get("entryPrice", 0) or 0))
                break

        if live is None:
            if self._side is not None:
                logger.warning("TREND FOLLOWER | tracked {} but exchange is flat -- clearing", self._side)
                self._side = None
                self._qty = 0.0
                self._entry_price = 0.0
                self.reset_trailing()
            return

        side, qty, entry = live
        if self._side != side or abs(self._qty - qty) > 1e-9:
            logger.warning(
                "TREND FOLLOWER | adopting exchange position {} {} @ {} (tracked {} {})",
                side, qty, entry, self._side, self._qty,
            )
            self._side = side
            self._qty = qty
            self._entry_price = entry or self.exchange.get_price(self.symbol)
            if self._entry_time == 0.0:
                self._entry_time = time.time()
            self._peak_price = self._peak_price or self._entry_price
            self._trough_price = self._trough_price or self._entry_price

    def get_tracked_order_ids(self) -> set:
        return {self._order_id} if self._order_id else set()

    # --- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "side": self._side,
            "entry_price": self._entry_price,
            "qty": self._qty,
            "order_id": self._order_id,
            "entry_time": self._entry_time,
            "regime": self._regime,
            "peak_price": self._peak_price,
            "trough_price": self._trough_price,
            "trailing_sl_price": self._trailing_sl_price,
            "trailing_sl_price_short": self._trailing_sl_price_short,
            "atr_pct": self._atr_pct,
            "active": self.active,
            "total_fills": self.total_fills,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "total_completed_cycles": self.total_completed_cycles,
        }

    def load_from_dict(self, data: dict, current_price: float) -> None:
        try:
            self._side = data.get("side")
            self._entry_price = float(data.get("entry_price", 0.0))
            self._qty = float(data.get("qty", 0.0))
            self._order_id = data.get("order_id")
            self._entry_time = float(data.get("entry_time", 0.0))
            self._regime = str(data.get("regime", "uncertain"))
            self._peak_price = float(data.get("peak_price", 0.0))
            self._trough_price = float(data.get("trough_price", 0.0))
            tsl = data.get("trailing_sl_price")
            tss = data.get("trailing_sl_price_short")
            self._trailing_sl_price = None if tsl is None else float(tsl)
            self._trailing_sl_price_short = None if tss is None else float(tss)
            self._atr_pct = float(data.get("atr_pct", 0.02))
            self.active = bool(data.get("active", False))
            self.total_fills = int(data.get("total_fills", 0))
            self.total_pnl = float(data.get("total_pnl", 0.0))
            self.total_fees = float(data.get("total_fees", 0.0))
            self.total_completed_cycles = int(data.get("total_completed_cycles", 0))
        except (TypeError, ValueError) as e:
            logger.error("TREND FOLLOWER | corrupt state ({}), starting fresh", e)
            self.state_corrupted = True
            return

        if self._side is not None and self._qty <= 0:
            logger.warning("TREND FOLLOWER | state claims a {} with no quantity -- clearing", self._side)
            self._side = None

    # --- grid-specific surface main.py still calls -------------------------
    # Implemented as harmless equivalents so the router can delegate blindly and
    # main.py needs no changes. See strategy.GRID_SPECIFIC_MEMBERS.

    levels: list = []

    @property
    def grid_lower(self) -> float:
        return 0.0

    @property
    def grid_upper(self) -> float:
        return 0.0

    @property
    def grid_count(self) -> int:
        return 0

    @property
    def grid_spacing(self) -> float:
        return 0.0

    def recenter(self, current_price: float, balance: float, margin_pct: float = 0.01) -> bool:
        """A single position has nothing to recentre."""
        return False

    def update_orderbook(self, *args, **kwargs) -> None:
        return None

    def get_scale_out_trail_price(self, *args, **kwargs) -> float | None:
        """No scale-out: the position exits in one piece at the trailing stop."""
        return None

    def log_sl_status(self) -> None:
        if self._side is None:
            return
        stop = self.get_stop_loss_price() if self._side == "long" else self.get_short_stop_loss_price()
        logger.info(
            "TREND SL STATUS | side={} entry={} peak={} trough={} sl={}",
            self._side, self._entry_price, self._peak_price, self._trough_price, stop,
        )

    def log_analytics(self, *args, **kwargs) -> None:
        logger.info(
            "TREND ANALYTICS | fills={} cycles={} pnl={:.6f} regime={} side={}",
            self.total_fills, self.total_completed_cycles, self.total_pnl,
            self._regime, self._side,
        )
