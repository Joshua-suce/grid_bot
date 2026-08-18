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

# One definition for the production path. This was an independent copy of grid.py's,
# so a symbol change had two places to remember and no way to notice missing one.
from grid import MIN_NOTIONAL_USDT  # noqa: E402  (AUDIT #107)

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
        capital_usdt: float = 0.0,
        stop_loss_pct: float = 0.03,
        trailing_sl_trigger_pct: float = 0.05,
        atr_stop_multiplier: float = 2.0,
        trail_atr_multiplier: float = 0.0,
        take_profit_r: float = 0.0,
        leverage: int = 1,
        max_exposure_pct: float = 0.50,
        min_hold_seconds: int = 300,
        event_journal: object | None = None,
        notifier: object | None = None,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.capital_pct = capital_pct
        # Fixed own-capital sizing, mirroring CAPITAL_PER_GRID_USDT. When set it
        # WINS over the percent path rather than competing with it (AUDIT #110).
        self.capital_usdt = max(0.0, capital_usdt)
        self.stop_loss_pct = stop_loss_pct
        self.atr_stop_multiplier = atr_stop_multiplier
        # The trail may ride WIDER than the stop the trade opened with. They were one
        # number, and that coupling is what put the take-profit out of reach: the target
        # sits at take_profit_r x the opening stop while the trail follows one stop-width
        # behind the extreme, so price has to run R widths without ever giving back one.
        # Measured on 62 days of DOGEUSDT 5m, target 3R, 70 handoffs: at a trail equal to
        # the stop the target was hit 3 times out of 70 and the realised win:loss came to
        # 1.46:1. At 3x it was 5 of 45 and 2.44:1 (AUDIT #105).
        #
        # 0 means "same as the opening stop", which is the behaviour this replaces.
        self.trail_atr_multiplier = trail_atr_multiplier or atr_stop_multiplier
        # Reward expressed in units of the risk actually taken on this trade ("R").
        # take_profit_r=3 means the target sits three times as far from entry as the
        # opening stop does, so a winner pays for three losers. 0 keeps the original
        # behaviour: no target, ride the trailing stop for as far as the trend runs.
        self.take_profit_r = max(0.0, take_profit_r)
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
        # Fixed at entry from the OPENING stop, never re-derived. The stop ratchets as
        # price runs, so a target recomputed from the live stop would creep toward entry
        # and shrink the reward it was set to guarantee.
        self._initial_risk = 0.0
        self._take_profit_price: float | None = None

        # --- stops (ratcheted, see update_trailing_sl) ---
        self._peak_price = 0.0
        self._trough_price = 0.0
        self._trailing_sl_price: float | None = None
        self._trailing_sl_price_short: float | None = None

        self._atr_pct = 0.02
        self._net_long_qty = 0.0
        self._net_short_qty = 0.0
        self._max_position_qty = 0.0
        self._blocked_sides: dict[str, str] = {}
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

    def _stop_distance(self, price: float, multiplier: float | None = None) -> float:
        """ATR-scaled, floored at stop_loss_pct so a quiet market cannot produce a stop
        so tight that noise closes the position immediately.

        `multiplier` defaults to the opening stop's. The ratchet passes the trail's,
        which may be wider -- see trail_atr_multiplier.
        """
        mult = self.atr_stop_multiplier if multiplier is None else multiplier
        atr_stop = price * self._atr_pct * mult
        return max(atr_stop, price * self.stop_loss_pct)

    def _ratchet_distance(self, price: float, opening: bool) -> float:
        """The opening stop defines 1R, so it must keep using the entry multiplier even
        though the same method sets it. Only later moves use the wider trail."""
        return self._stop_distance(
            price, None if opening else self.trail_atr_multiplier)

    # --- lifecycle ---------------------------------------------------------

    def initialize(self, current_price: float, balance: float) -> None:
        self._peak_price = current_price
        self._trough_price = current_price
        logger.info(
            "TREND FOLLOWER INITIALIZED | {} @ {} | capital_pct={:.1%} atr_stop={}x",
            self.symbol, current_price, self.capital_pct, self.atr_stop_multiplier,
        )

    def activate(self, balance: float) -> None:
        """Go live and open straight away if the regime already supports a side.

        GridEngine.activate places its ladder here rather than waiting to be driven, so
        this does the same. Without it the router hands over on a confirmed trend and
        the bot stands flat until something else happens to call place_initial_orders.
        """
        self.active = True
        logger.info("TREND FOLLOWER ACTIVATED | regime={}", self._regime)
        self.place_initial_orders(balance)

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

    def emergency_stop(self, reason: str = "emergency") -> None:
        """`reason` only picks the log level -- see GridEngine.emergency_stop."""
        self._cancel_entry("emergency_stop")
        try:
            self.exchange.cancel_everything(self.symbol, timeout_seconds=30)
        except Exception as e:
            logger.error("TREND FOLLOWER | cancel_everything failed: {}", e)
        self.active = False
        if reason == "shutdown":
            logger.info("TREND FOLLOWER STOPPED | shutdown")
        else:
            logger.error("TREND FOLLOWER EMERGENCY STOP ({})", reason)

    def get_spread_pct(self) -> float:
        """No orderbook depth is read by this strategy, so there is nothing to report.

        Present because main.py logs the spread every iteration and the router forwards
        the call to whichever strategy is live (AUDIT #31).
        """
        return 0.0

    @property
    def peak_price(self) -> float:
        return self._peak_price

    @peak_price.setter
    def peak_price(self, value: float) -> None:
        self._peak_price = float(value)

    def _cancel_entry(self, reason: str) -> bool:
        """Returns True only when the entry order is confirmed gone.

        Clearing `_order_id` on an unconfirmed cancel lets place_initial_orders through
        its `if self._order_id is not None: return 0` guard, so it opens a SECOND entry
        at full size while the first is still live -- and the survivor fills untracked,
        outside check_fills and outside the stop (AUDIT #51).
        """
        if self._order_id is None:
            return True
        try:
            confirmed = self.exchange.cancel_order(self._order_id, self.symbol)
        except Exception as e:
            logger.warning("TREND FOLLOWER | cancel {} failed: {}", self._order_id, e)
            confirmed = False
        if not confirmed:
            logger.error(
                "TREND FOLLOWER | entry {} could NOT be confirmed cancelled — keeping it "
                "claimed rather than risking a duplicate entry", self._order_id,
            )
            return False
        if self._event_journal:
            self._event_journal.order_cancelled(
                self.symbol, self._side or "", self._entry_price, self._order_id, reason,
            )
        self._order_id = None
        return True

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
        if side in self._blocked_sides:
            logger.warning(
                "TREND ENTRY BLOCKED | {} entry withheld — {} (AUDIT #50)",
                side.upper(), self._blocked_sides[side],
            )
            return 0
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
        if self.capital_usdt > 0:
            notional = self.capital_usdt * self.leverage
        else:
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

    def check_fills(self, balance: float, open_orders: list[dict] | None = None) -> list[dict]:
        """Detect the entry filling, and enforce the trailing stop."""
        fills: list[dict] = []
        if self._order_id is not None:
            open_ids = ({o["id"] for o in open_orders} if open_orders is not None
                        else self.exchange.get_open_order_ids(self.symbol))
            if self._order_id not in open_ids:
                order = self.exchange.fetch_order(self._order_id, self.symbol)
                if order and order.get("status") == "closed":
                    fills.append(self._record_entry(order))
                elif order is None or order.get("status") == "canceled":
                    self._order_id = None

        if self._side is not None:
            price = self.exchange.get_price(self.symbol)
            # The target is checked FIRST, and deliberately. Stop and target sit on
            # opposite sides of entry so they cannot both be live on one tick -- except
            # on a gap, where price has jumped clean past one of them. Taking the target
            # there would book a win the market never offered; testing the stop's side
            # of the move first keeps the pessimistic reading.
            tp = self._take_profit_price
            hit_target = tp is not None and (
                price >= tp if self._side == "long" else price <= tp)

            if self._side == "long":
                self.update_trailing_sl(price)
                stop = self.get_stop_loss_price()
                hit_stop = stop is not None and price <= stop
            else:
                self.update_trailing_sl_short(price)
                stop = self.get_short_stop_loss_price()
                hit_stop = stop is not None and price >= stop

            if (hit_stop or hit_target) and self._can_exit():
                exit_fill = self._close_position(
                    "trailing_stop" if hit_stop else "take_profit")
                if exit_fill:
                    fills.append(exit_fill)

        # Re-arm. Entry lives in place_initial_orders, and main.py's live loop calls
        # that exactly once, at startup, before the loop begins -- so in router mode
        # the trend follower would take over on a confirmed trend and then never open
        # anything, leaving the bot flat through the whole move (AUDIT #30). check_fills
        # is called every iteration whenever a strategy is live, so the entry is armed
        # from here too. Both paths are idempotent: place_initial_orders no-ops while a
        # position or a resting entry order exists, so being driven twice in one
        # iteration (as the backtester does) still opens only one position.
        if self.active and self._side is None and self._order_id is None:
            self.place_initial_orders(balance)

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
            opening_stop = self.get_stop_loss_price()
        else:
            self.update_trailing_sl_short(price)
            opening_stop = self.get_short_stop_loss_price()

        # One R = the distance to the stop this trade actually opened with, so the
        # target is stated in the same units the risk is. Measured from the stop rather
        # than from stop_loss_pct because the stop is the greater of an ATR band and
        # that floor -- in any market with real movement it is the ATR band that binds,
        # and a target built off the floor would sit far inside the noise.
        self._initial_risk = abs(price - opening_stop) if opening_stop else 0.0
        self._take_profit_price = None
        if self.take_profit_r > 0 and self._initial_risk > 0:
            offset = self._initial_risk * self.take_profit_r
            self._take_profit_price = self._round_price(
                price + offset if self._side == "long" else price - offset)

        logger.info(
            "TREND ENTRY FILLED | {} {} @ {} | stop={} | target={} ({}R)",
            self._side.upper(), qty, price, opening_stop,
            self._take_profit_price if self._take_profit_price else "none",
            self.take_profit_r if self.take_profit_r > 0 else 0,
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
            # Exchange.close_position(symbol, side, amount) -- `side` is the POSITION
            # side ("long"/"short"), which it converts to the closing order side.
            #
            # This used to pass the symbol alone, which is a TypeError against the real
            # Exchange: side and amount have no defaults. The `except Exception` below
            # swallowed it, so every exit this strategy has -- the trailing stop and the
            # regime-change exit -- logged "close failed" and returned None, and the
            # early return here happens BEFORE _side is cleared, so the follower stayed
            # wedged believing it still held the position: place_initial_orders returns
            # 0 while _side is set, so it never exited and never re-entered again.
            #
            # No test caught it because every fake exchange in the suite and in
            # backtest.py declared close_position(self, symbol) -- only the real class
            # has the three-argument form. test_exchange_contract.py now pins the fakes
            # to the real signature (AUDIT #38).
            self.exchange.close_position(self.symbol, side, abs(qty))
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
        self._initial_risk = 0.0
        self._take_profit_price = None
        self.reset_trailing()
        return {
            "price": exit_price, "side": "sell" if side == "long" else "buy",
            "quantity": qty, "profit": profit, "fee": 0.0, "completed_cycle": True,
            # Why the trade ended, not just that it did. Without this the reason exists
            # only in a log line, so nothing downstream -- and no test -- can tell a
            # stop-out from a target hit. A gap through the stop mislabelled
            # "take_profit" would then corrupt every later attempt to measure which exit
            # actually pays, which is the same class of error as pricing a level's round
            # trip instead of the position's (AUDIT #80).
            "reason": reason,
        }

    # --- risk interface ----------------------------------------------------

    def set_position_limit(
        self, long_position: float, short_position: float, max_position_qty: float,
    ) -> None:
        self._net_long_qty = max(0.0, long_position)
        self._net_short_qty = max(0.0, short_position)
        self._max_position_qty = max(0.0, max_position_qty)
        # Cleared every iteration, exactly as the grid recomputes _block_buys/_block_sells
        # here: main.py re-blocks below if the position is STILL unprotected, so the block
        # lasts precisely as long as the condition and lifts on its own when stops return.
        self._blocked_sides.clear()

    def block_side(self, side: str, reason: str) -> None:
        """Withhold ENTRIES on `side`. Exits run through _close_position, which never
        consults this -- an unprotected position must still be closable (AUDIT #50)."""
        if side in ("buy", "sell") and side not in self._blocked_sides:
            self._blocked_sides[side] = reason
            logger.warning("SIDE BLOCKED | {} — {} (AUDIT #50)", side, reason)

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
        candidate = self._peak_price - self._ratchet_distance(
            self._peak_price, self._trailing_sl_price is None)
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
        candidate = self._trough_price + self._ratchet_distance(
            self._trough_price, self._trailing_sl_price_short is None)
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

    def get_hard_stop_loss_price(self) -> float | None:
        """No separate hard leg: the position exits in one piece at the trailing stop,
        which is already ratcheted, so it is its own last line of defence."""
        return self._trailing_sl_price

    def get_short_hard_stop_loss_price(self) -> float | None:
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

    # An unbounded range, not a zero-width one at the origin. main.py tests
    # `price < grid.grid_lower` and `price > grid.grid_upper` to detect price escaping
    # the ladder; with 0.0 for both, the upper test is true at every price, and while
    # the follower was flat that logged GRID EXIT and journalled an event once per
    # iteration for as long as it stayed live. A strategy with no ladder is never
    # outside it (AUDIT #31).

    @property
    def grid_lower(self) -> float:
        return 0.0

    @property
    def grid_upper(self) -> float:
        return float("inf")

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

    def reset_levels_to_pending(self, *args, **kwargs) -> int:
        """No ladder to reset -- the position, if any, is real and stays."""
        return 0

    def get_scale_out_trail_price(self, *args, **kwargs) -> float | None:
        """No scale-out: the position exits in one piece at the trailing stop."""
        return None

    def one_side_notional(self, balance: float) -> float:
        """No ladder, so nothing accumulates a rung at a time.

        main.py's startup check compares this against the position cap to catch a grid
        whose outer rungs could never fill. A trend follower opens one position sized by
        _entry_qty and bounded by the same cap, so there is no rung-stranding failure to
        warn about -- 0.0 reads as "nothing to check" (AUDIT #66).
        """
        return 0.0

    def log_sl_status(self, side: str = "long") -> None:
        """`side` is accepted and ignored: this strategy holds at most one position and
        already knows which way it is facing. main.py passes it positionally
        (`grid.log_sl_status(position_side)`), so dropping the parameter made every
        stop-status log a TypeError while the follower was live (AUDIT #31)."""
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
