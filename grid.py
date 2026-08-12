from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from loguru import logger

from exchange import Exchange


@dataclass
class GridLevel:
    price: float
    side: str  # "buy" or "sell"
    order_id: str | None = None
    status: str = "pending"  # pending, replaced
    fill_count: int = 0
    total_pnl: float = 0.0
    quantity: float = 0.0
    entry_price: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GridLevel":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


TIMEFRAME_CANDLES_PER_DAY = {
    "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
    "1h": 24, "4h": 6, "1d": 1,
}

# Binance USDM futures minimum order notional for most pairs. When position-limit
# scaling (_buy_scale/_sell_scale, see set_position_limit) shrinks an order below
# this, the exchange guarantees a "-4164 notional must be no smaller than 5"
# rejection -- skip cleanly instead of spending an API round-trip + error log +
# Telegram alert on a placement that can never succeed.
MIN_NOTIONAL_USDT = 5.0


def calculate_grid_range(
    ohlcv: pd.DataFrame,
    current_price: float,
    lookback_days: int = 14,
    atr_multiplier: float = 1.5,
    timeframe: str = "1h",
) -> tuple[float, float]:
    candles_per_day = TIMEFRAME_CANDLES_PER_DAY.get(timeframe, 24)
    recent = ohlcv.tail(lookback_days * candles_per_day)
    if len(recent) < 20:
        price_range = current_price * 0.05
        return current_price - price_range, current_price + price_range

    from trend_filter import atr as calc_atr
    atr_series = calc_atr(recent["high"], recent["low"], recent["close"], period=14)
    current_atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else current_price * 0.02

    lower = current_price - (current_atr * atr_multiplier)
    upper = current_price + (current_atr * atr_multiplier)

    logger.info(
        "GRID RANGE CALCULATED | lower={} upper={} ATR={} current={}",
        round(lower, 8), round(upper, 8), round(current_atr, 8), round(current_price, 8),
    )
    return lower, upper


def calculate_dynamic_grid_count(atr_pct: float, base_count: int) -> int:
    if atr_pct < 0.02:
        return base_count
    elif atr_pct < 0.03:
        return max(base_count - 1, 3)
    else:
        return max(base_count - 2, 3)


def validate_grid_spacing(lower: float, upper: float, count: int, min_spacing_pct: float, price: float) -> bool:
    if count < 2:
        return False
    spacing = (upper - lower) / max(1, count - 1)
    spacing_pct = spacing / price
    if spacing_pct < min_spacing_pct * 0.99:
        logger.warning(
            "GRID SPACING TOO TIGHT | spacing={} ({:.4f}%) < min ({:.4f}%)",
            round(spacing, 8), spacing_pct * 100, min_spacing_pct * 100,
        )
        return False
    return True


class GridEngine:
    def __init__(
        self,
        exchange: Exchange,
        symbol: str,
        grid_lower: float,
        grid_upper: float,
        grid_count: int,
        capital_per_grid_pct: float,
        stop_loss_pct: float,
        maker_fee_pct: float = 0.0002,
        taker_fee_pct: float = 0.0004,
        recenter_cooldown: int = 300,
        replacement_cooldown: int = 60,
        order_pacing_seconds: float = 0.6,
        capital_per_grid_usdt: float = 0.0,
        leverage: int = 1,
        trailing_sl_trigger_pct: float = 0.05,
        max_exposure_pct: float = 0.50,
        use_market_close_on_replace: bool = False,
        event_journal: object | None = None,
        notifier: object | None = None,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.grid_lower = grid_lower
        self.grid_upper = grid_upper
        self.grid_count = grid_count
        self.capital_per_grid_pct = capital_per_grid_pct
        self.capital_per_grid_usdt = capital_per_grid_usdt
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.recenter_cooldown = recenter_cooldown
        self.replacement_cooldown = replacement_cooldown
        self.order_pacing_seconds = order_pacing_seconds
        self.max_exposure_pct = max_exposure_pct
        self.use_market_close_on_replace = use_market_close_on_replace
        self.grid_spacing = (grid_upper - grid_lower) / max(1, grid_count - 1)
        self.levels: list[GridLevel] = []
        self.active = False
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.total_fills = 0
        self.total_completed_cycles = 0
        self._last_recenter_time = 0.0
        self._volatility_mult = 1.0
        self._trailing_sl_price: float | None = None
        self._trailing_sl_trigger: float = trailing_sl_trigger_pct
        self._peak_price = 0.0
        self._trough_price = 0.0
        self._trailing_sl_price_short: float | None = None
        self._last_orderbook: dict = {}
        self._event_journal = event_journal
        self._notifier = notifier
        self._block_buys = False
        self._block_sells = False
        # Last net position seen by set_position_limit(), used to decide whether an
        # order actually *reduces* a position (reduceOnly is only legal then -- see
        # _reduce_only_qty). Refreshed from the exchange every main-loop iteration.
        self._net_long_qty: float = 0.0
        self._net_short_qty: float = 0.0
        self._last_replacement_time: float = 0.0
        # Per-level cooldown tracking (keyed by object id, not persisted): a fill on
        # one level used to reset a single engine-wide timer that gated ALL orphan/
        # cancelled-order replacement across the whole grid for `replacement_cooldown`
        # seconds. During a burst of fills across several levels that meant the timer
        # kept getting pushed back and unrelated levels sat unreplaced for the whole
        # burst -- exactly when the grid should be trading the most. Tracking cooldown
        # per level instead means a fill/replacement on level A no longer blocks level
        # B from being replaced.
        self._level_cooldowns: dict[int, float] = {}
        self.state_corrupted: bool = False
        self._buy_scale: float = 1.0
        self._sell_scale: float = 1.0
        self._min_profit_multiplier: float = 1.0
        self._open_orders_fetch_time: float = 0.0
        self._open_orders_map: dict[tuple[float, str], dict] = {}
        self._warned_small_fixed_allocation = False

    def _round_price(self, price: float) -> float:
        """Round price to exchange tick size."""
        return float(self.exchange.exchange.price_to_precision(self.symbol, price))

    def update_volatility(self, atr_pct: float) -> None:
        if atr_pct < 0.01:
            # In calmer markets, increase grid sizing aggressively to keep the bot active.
            self._volatility_mult = min(2.5, 1.0 + (0.01 - atr_pct) * 150)
        elif atr_pct > 0.03:
            # Reduce sizing gently even in very high volatility so the bot keeps trading.
            self._volatility_mult = max(0.8, 1.0 - (atr_pct - 0.03) * 5)
        else:
            self._volatility_mult = 1.0

    def set_position_limit(self, long_position: float, short_position: float, max_position_qty: float) -> None:
        """Block new buys when the long position would exceed the limit and new sells
        when the short position would exceed it; scale order sizes down near the cap."""
        old_block_buys = self._block_buys
        old_buy_scale = self._buy_scale
        old_block_sells = self._block_sells
        old_sell_scale = self._sell_scale

        self._net_long_qty = max(0.0, long_position)
        self._net_short_qty = max(0.0, short_position)

        self._block_buys, self._buy_scale = self._position_limit_state(long_position, max_position_qty)
        self._block_sells, self._sell_scale = self._position_limit_state(short_position, max_position_qty)

        # The cap is enforced on new placements only; resting orders placed before the
        # cap was hit keep filling and overshoot it. Once a side is blocked, cancel the
        # resting orders on that side so the position cannot keep growing past the cap.
        if self._block_buys:
            self._cancel_resting_orders("buy", "position_limit")
        if self._block_sells:
            self._cancel_resting_orders("sell", "position_limit")

        if self._block_buys and not old_block_buys:
            logger.warning(
                "POSITION LIMIT | long {} >= {} — buy orders blocked",
                round(long_position, 2), round(max_position_qty, 2),
            )
        elif self._buy_scale != old_buy_scale and self._buy_scale < 1.0:
            logger.info("BUY SCALE | long={:.1f}/{:.1f} | scale={:.2f}", long_position, max_position_qty, self._buy_scale)

        if self._block_sells and not old_block_sells:
            logger.warning(
                "POSITION LIMIT | short {} >= {} — sell orders blocked",
                round(short_position, 2), round(max_position_qty, 2),
            )
        elif self._sell_scale != old_sell_scale and self._sell_scale < 1.0:
            logger.info("SELL SCALE | short={:.1f}/{:.1f} | scale={:.2f}", short_position, max_position_qty, self._sell_scale)

    @staticmethod
    def _position_limit_state(current_position: float, max_position_qty: float) -> tuple[bool, float]:
        """Return (blocked, scale) for one side given the current open position."""
        if max_position_qty <= 0:
            return current_position >= max_position_qty, 0.0
        if current_position >= max_position_qty:
            return True, 0.0
        ratio = current_position / max_position_qty
        if ratio < 0.5:
            return False, 1.0
        return False, 1.0 - (ratio - 0.5) / 0.5

    def _reduce_only_qty(self, side: str) -> float:
        """Return how much quantity an order on `side` could legally close right now.

        Binance rejects a reduceOnly order (-2022) whenever it cannot reduce the net
        position: a reduceOnly SELL with no long open, or one whose size exceeds the
        remaining long, is refused outright. In one-way position mode the sell side of
        a grid is an *exit* while long but an *entry* while short, so reduceOnly can
        never be a constant -- it has to be derived from the live net position.

        Returns 0.0 when no position on the closing side exists, i.e. the order opens
        exposure and must be sent WITHOUT reduceOnly.
        """
        if side == "sell":
            return self._net_long_qty
        if side == "buy":
            return self._net_short_qty
        return 0.0

    def _exit_order_params(self, side: str, quantity: float) -> tuple[dict | None, float]:
        """Build order params + quantity for a level that may be closing a position.

        Returns (params, quantity). params is None for a normal opening order.
        Quantity is clamped to the closable size when the order is reduceOnly, because
        Binance also rejects a reduceOnly order larger than the remaining position.
        """
        closable = self._reduce_only_qty(side)
        if closable <= 0:
            return None, quantity
        return {"reduceOnly": True, "postOnly": False}, min(quantity, closable)

    def _cancel_resting_orders(self, side: str, reason: str) -> int:
        """Cancel resting orders on one side so a position-capped grid stops growing."""
        targets = [l for l in self.levels if l.side == side and l.order_id is not None]
        if not targets:
            return 0
        try:
            still_open = self.exchange.get_open_order_ids(self.symbol)
        except Exception as e:
            logger.error("POSITION LIMIT | could not fetch open orders to cancel {} orders: {}", side, e)
            return 0
        cancelled = 0
        for level in targets:
            if level.order_id in still_open:
                try:
                    self.exchange.cancel_order(level.order_id, self.symbol)
                except Exception as e:
                    logger.error(
                        "POSITION LIMIT | failed to cancel {} @ {}: {}", level.side, level.price, e,
                    )
            if self._event_journal:
                self._event_journal.order_cancelled(self.symbol, level.side, level.price, level.order_id, reason)
            if self._notifier:
                self._notifier.on_order_cancelled(self.symbol, level.side, level.price, level.order_id, reason)
            level.order_id = None
            level.status = "pending"
            cancelled += 1
        logger.warning(
            "POSITION LIMIT | cancelled {} resting {} orders ({})", cancelled, side, reason,
        )
        return cancelled

    def update_orderbook(self) -> None:
        self._last_orderbook = self.exchange.get_orderbook_depth(self.symbol)

    def get_tracked_order_ids(self) -> set[str]:
        """Return set of order IDs currently tracked by grid levels."""
        return {l.order_id for l in self.levels if l.order_id is not None}

    def get_exposure_pct(self, balance: float) -> float:
        if balance <= 0:
            return 0.0
        exposure_usdt = 0.0
        try:
            positions = self.exchange.get_positions(self.symbol)
            for pos in positions:
                qty = float(pos.get("contracts", 0) or 0)
                entry = float(pos.get("entryPrice", 0) or 0)
                if qty == 0 or entry <= 0:
                    continue
                if qty < 0:
                    qty = abs(qty)
                exposure_usdt += qty * entry
        except Exception:
            for level in self.levels:
                if level.quantity <= 0:
                    continue
                notional = level.quantity * (level.entry_price if level.entry_price else level.price)
                exposure_usdt += notional
        return exposure_usdt / balance

    def _cycle_pnl(self, quantity: float, entry_price: float, exit_price: float, is_short: bool = False) -> float:
        if quantity <= 0 or entry_price <= 0 or exit_price <= 0:
            return 0.0
        delta = exit_price - entry_price
        if is_short:
            return -delta * quantity
        return delta * quantity

    def _cycle_fee(self, quantity: float, buy_price: float, sell_price: float, is_taker: bool = False) -> float:
        fee_rate = self.taker_fee_pct if is_taker else self.maker_fee_pct
        return (buy_price + sell_price) * quantity * fee_rate

    def initialize(self, current_price: float, balance: float, dynamic_spacing: bool = True) -> None:
        if dynamic_spacing:
            self._initialize_dynamic(current_price)
        else:
            self._initialize_uniform(current_price)

        logger.info(
            "GRID INITIALIZED | {} levels | avg_spacing={} | range=[{}-{}]",
            len(self.levels), round(self.grid_spacing, 8),
            round(self.grid_lower, 8), round(self.grid_upper, 8),
        )

    def _initialize_uniform(self, current_price: float) -> None:
        levels = []
        for i in range(self.grid_count):
            price = self._round_price(self.grid_lower + i * self.grid_spacing)
            side = "buy" if price < current_price else "sell"
            lvl = GridLevel(price=price, side=side)
            if side == "buy":
                lvl.entry_price = price
            levels.append(lvl)
        self.levels = levels

    def _initialize_dynamic(self, current_price: float) -> None:
        half_count = self.grid_count // 2

        if current_price <= self.grid_lower + self.grid_spacing * 0.5:
            self._initialize_uniform(current_price)
            return
        if current_price >= self.grid_upper - self.grid_spacing * 0.5:
            self._initialize_uniform(current_price)
            return

        buy_prices = np.linspace(self.grid_lower, current_price, half_count, endpoint=False)
        sell_prices = np.linspace(current_price, self.grid_upper, self.grid_count - half_count + 1)[1:]

        raw_prices = [self._round_price(p) for p in np.concatenate([buy_prices, sell_prices])]

        seen = {}
        prices = []
        for p in raw_prices:
            if p not in seen:
                seen[p] = True
                prices.append(p)

        if len(prices) != len(raw_prices):
            logger.warning(
                "GRID DEDUP | {} duplicate prices removed after tick rounding ({} -> {})",
                len(raw_prices) - len(prices), len(raw_prices), len(prices),
            )

        levels = []
        for price in prices:
            side = "buy" if price < current_price else "sell"
            lvl = GridLevel(price=price, side=side)
            if side == "buy":
                lvl.entry_price = price
            levels.append(lvl)

        if len(levels) < 2:
            self._initialize_uniform(current_price)
            return

        if len(levels) != self.grid_count:
            logger.warning(
                "GRID COUNT MISMATCH | expected={} actual={} after tick dedup — adjusting",
                self.grid_count, len(levels),
            )
            self.grid_count = len(levels)

        spacings = [levels[i+1].price - levels[i].price for i in range(len(levels)-1)]
        self.grid_spacing = sum(spacings) / len(spacings)

        self.levels = levels

        logger.info(
            "DYNAMIC GRID | {} levels | price concentration near {}",
            len(levels), round(current_price, 8),
        )

    def _calc_usdt_per_grid(self, balance: float) -> float:
        """Return the USDT notional per grid level.

        If both a fixed USDT allocation and a percentage-based allocation are configured,
        use the larger of the two so a small fixed override does not undercut the
        chosen percentage allocation.
        """
        pct_allocation = balance * self.capital_per_grid_pct * self._volatility_mult
        if self.capital_per_grid_usdt > 0:
            fixed_allocation = self.capital_per_grid_usdt * self.leverage * self._volatility_mult
            if fixed_allocation < pct_allocation:
                if not self._warned_small_fixed_allocation:
                    self._warned_small_fixed_allocation = True
                    logger.warning(
                        "CAPITAL_PER_GRID_USDT ({:.2f}) is smaller than percent-based allocation ({:.2f}); "
                        "using the larger value for per-grid sizing.",
                        fixed_allocation, pct_allocation,
                    )
            raw = max(fixed_allocation, pct_allocation)
            current_total = raw * self.grid_count
            target_total = balance * self.max_exposure_pct
            if current_total > target_total:
                raw = target_total / self.grid_count
            return raw
        return pct_allocation

    def _is_level_profitable(self, level_price: float) -> bool:
        expected_profit = self.grid_spacing
        worst_fees = 2 * self.maker_fee_pct * level_price
        return expected_profit > worst_fees * self._min_profit_multiplier

    def _existing_open_order(self, price: float, side: str) -> dict | None:
        """Return an open exchange order already resting at the same price+side, if any.

        Prevents duplicate grid levels when a placement actually succeeded on the
        exchange but the response was lost (e.g. network timeout): the level is then
        retried without an order_id, and without this check a second order would be
        stacked on top of the first.
        """
        now = time.time()
        if (now - self._open_orders_fetch_time) >= 2.0 or not self._open_orders_map:
            try:
                orders = self.exchange.get_open_orders(self.symbol)
            except Exception:
                self._open_orders_fetch_time = 0.0
                return None
            mapping: dict[tuple[float, str], dict] = {}
            for order in orders:
                o_price = order.get("price")
                o_side = order.get("side")
                if o_price is not None and o_side is not None:
                    mapping[(float(o_price), o_side)] = order
            self._open_orders_map = mapping
            self._open_orders_fetch_time = now
        return self._open_orders_map.get((float(price), side))

    def _place_order_for_level(self, level: GridLevel, balance: float) -> bool:
        if level.side == "buy" and self._block_buys:
            logger.debug("SKIP BUY ORDER | position limit reached")
            return False
        if level.side == "sell" and self._block_sells:
            logger.debug("SKIP SELL ORDER | short position limit reached")
            return False
        if not self._is_level_profitable(level.price):
            logger.info(
                "SKIP ORDER @ {} | spacing={:.8f} < min_profit={:.8f} ({}x fees)",
                level.price, self.grid_spacing,
                2 * self.maker_fee_pct * level.price * self._min_profit_multiplier,
                self._min_profit_multiplier,
            )
            return False
        existing = self._existing_open_order(level.price, level.side)
        if existing and existing.get("id"):
            if any(l is not level and l.order_id == existing["id"] for l in self.levels):
                logger.debug(
                    "SKIP ADOPT | order {} @ {} {} already tracked by another level",
                    existing["id"], level.price, level.side,
                )
                return False
            logger.warning(
                "ADOPTED existing open order {} @ {} {} — avoiding duplicate placement",
                existing["id"], level.price, level.side,
            )
            level.order_id = existing["id"]
            level.status = "pending"
            amount = existing.get("amount")
            if amount is not None:
                level.quantity = float(amount)
            return True
        if not self.exchange.can_place_order(self.symbol):
            return False
        usdt_per_grid = self._calc_usdt_per_grid(balance)
        quantity = usdt_per_grid / level.price
        if level.side == "buy":
            quantity *= self._buy_scale
        else:
            quantity *= self._sell_scale
        quantity = self.exchange.exchange.amount_to_precision(self.symbol, quantity)
        if float(quantity) <= 0:
            if self._event_journal:
                self._event_journal.order_failed(self.symbol, level.side, level.price, 0.0, "quantity_zero")
            if self._notifier:
                self._notifier.on_order_failed(self.symbol, level.side, level.price, 0.0, "quantity_zero")
            return False
        notional = float(quantity) * level.price
        if notional < MIN_NOTIONAL_USDT:
            scale = self._buy_scale if level.side == "buy" else self._sell_scale
            logger.debug(
                "SKIP ORDER @ {} | notional {:.2f} USDT < exchange minimum {:.2f} USDT "
                "(scale={:.2f}) — would be guaranteed-rejected, not attempting",
                level.price, notional, MIN_NOTIONAL_USDT, scale,
            )
            return False
        try:
            order = self.exchange.place_limit_order(self.symbol, level.side, level.price, float(quantity), max_attempts=1)
            if "id" not in order:
                raise ValueError("Order response missing 'id'")
            level.order_id = order["id"]
            level.status = "pending"
            level.quantity = float(quantity)
            if self._event_journal:
                self._event_journal.order_placed(self.symbol, level.side, level.price, float(quantity), order["id"])
            if self._notifier:
                self._notifier.on_order_placed(self.symbol, level.side, level.price, float(quantity), order["id"])
            return True
        except Exception as e:
            logger.error("Failed to place order at {}: {}", level.price, e)
            if self._event_journal:
                self._event_journal.order_failed(self.symbol, level.side, level.price, float(quantity), str(e))
            if self._notifier:
                self._notifier.on_order_failed(self.symbol, level.side, level.price, float(quantity), str(e))
            return False
        finally:
            self._open_orders_fetch_time = 0.0

    def place_initial_orders(self, balance: float) -> int:
        placed = 0
        failed = 0
        first = True
        for level in self.levels:
            if level.order_id is not None:
                continue
            if self.order_pacing_seconds > 0 and not first:
                time.sleep(self.order_pacing_seconds)
            first = False
            if self._place_order_for_level(level, balance):
                placed += 1
            else:
                failed += 1

        logger.info(
            "PLACED {} initial grid orders ({} failed) | vol_mult={:.2f} | exposure={:.1%}",
            placed, failed, self._volatility_mult, self.get_exposure_pct(balance),
        )
        return placed

    def reconcile_state(self) -> None:
        """Verify all pending/replaced orders exist on the exchange. Mark missing ones dead."""
        balance = self.exchange.get_balance()
        open_ids = self.exchange.get_open_order_ids(self.symbol)
        reconciled = 0
        for level in self.levels:
            if level.order_id is None:
                continue
            if level.order_id in open_ids:
                continue
            order = self.exchange.fetch_order(level.order_id, self.symbol)
            if order and order.get("status") == "closed":
                logger.info("Reconcile: order {} was filled externally — processing as fill", level.order_id)
                self._handle_fill(level, balance)
                reconciled += 1
                continue
            if order and order.get("status") == "canceled":
                logger.info("Reconcile: order {} was cancelled, marking for replacement", level.order_id)
                if self._event_journal:
                    self._event_journal.order_cancelled(self.symbol, level.side, level.price, level.order_id, "reconcile_cancelled")
                level.order_id = None
                level.status = "pending"
                reconciled += 1
                continue
            logger.warning(
                "Reconcile: order {} ({}) not found on exchange (status={}), marking dead",
                level.order_id, level.side, order.get("status") if order else None,
            )
            if self._event_journal:
                self._event_journal.order_cancelled(self.symbol, level.side, level.price, level.order_id, "reconcile_dead")
            level.order_id = None
            level.status = "pending"
            reconciled += 1
        if reconciled:
            logger.warning("Reconciled {} dead orders from state", reconciled)
        orphaned = [l for l in self.levels if l.order_id is None and l.status == "pending"]
        if orphaned:
            for level in orphaned:
                if level.price < self.grid_lower or level.price > self.grid_upper:
                    logger.debug(
                        "RECONCILE SKIP ORPHAN | {} @ {} outside grid bounds [{}-{}]",
                        level.side, level.price, self.grid_lower, self.grid_upper,
                    )
                    continue
                if self._place_order_for_level(level, balance):
                    logger.info("Reconcile: placed replacement order @ {} {}", level.price, level.side)

    def reconcile_positions(self) -> None:
        """Match open exchange positions to grid levels and place the opposing hedge order.

        This path now supports both long and short inventory so the grid can continue to
        recover and defend positions in either direction without treating a short as flat.
        """
        positions = self.exchange.get_positions(self.symbol)
        for pos in positions:
            side = pos.get("side", "")
            amt = float(pos.get("contracts", 0) or 0)
            entry = float(pos.get("entryPrice", 0) or 0)
            if amt == 0 or entry <= 0:
                continue

            if side == "long" and amt < 0:
                side = "short"
                amt = abs(amt)
            elif side == "short" and amt < 0:
                side = "long"
                amt = abs(amt)

            if side not in {"long", "short"}:
                continue

            if side == "long":
                target_side = "buy"
                hedge_side = "sell"
                base_price = entry
            else:
                target_side = "sell"
                hedge_side = "buy"
                base_price = entry

            best_level = None
            best_diff = float("inf")
            for level in self.levels:
                if level.side != target_side:
                    continue
                diff = abs(level.price - base_price)
                if diff < best_diff:
                    best_diff = diff
                    best_level = level

            if best_level is None:
                logger.warning("RECONCILE | no {} level found for {} position @ {}", target_side, side, round(entry, 8))
                continue

            if side == "long":
                hedge_price = self._round_price(best_level.price + self.grid_spacing)
            else:
                hedge_price = self._round_price(best_level.price - self.grid_spacing)

            if hedge_price < self.grid_lower or hedge_price > self.grid_upper:
                logger.warning(
                    "RECONCILE | {} price {} outside grid — position unprotected",
                    hedge_side, hedge_price,
                )
                continue

            if best_level.order_id is not None:
                if not self.exchange.cancel_order(best_level.order_id, self.symbol):
                    logger.warning(
                        "RECONCILE | could not cancel order {} for level @ {} — skipping repurpose "
                        "(order may still be open; will retry next reconcile)",
                        best_level.order_id, best_level.price,
                    )
                    continue

            best_level.fill_count += 1
            best_level.status = "replaced"
            best_level.quantity = amt
            best_level.entry_price = entry
            best_level.side = hedge_side
            best_level.price = hedge_price

            qty = self.exchange.exchange.amount_to_precision(self.symbol, amt)
            if float(qty) <= 0:
                continue
            try:
                # Market-closing the whole position on every reconcile realized large
                # losses whenever the grid restarted or reconnected with an open bag
                # (e.g. 08-01 -12.98, 08-03 -5.92, 08-05 -8.43). Default to a
                # reduce-only limit hedge instead; opt into the market close only
                # if explicitly configured.
                if self.use_market_close_on_replace:
                    try:
                        order = self.exchange.close_position(self.symbol, side, abs(amt))
                        if order and "id" in order:
                            best_level.order_id = order["id"]
                            best_level.status = "replaced"
                            logger.info(
                                "RECONCILE | position {} @ {} -> CLOSED MARKET (id={})",
                                side, round(entry, 8), best_level.order_id,
                            )
                            continue
                    except Exception as e:
                        logger.warning("RECONCILE | market close failed, falling back to limit: {}", e)
                params = {"reduceOnly": True, "postOnly": False} if hedge_side == "sell" else None
                order = self.exchange.place_limit_order(self.symbol, hedge_side, hedge_price, float(qty), params=params)
                if "id" not in order:
                    raise ValueError("Order response missing 'id'")
                best_level.order_id = order["id"]
                self._open_orders_fetch_time = 0.0
                logger.info(
                    "RECONCILE | position {} @ {} -> {} order @ {} (qty={})",
                    side, round(entry, 8), hedge_side, hedge_price, qty,
                )
            except Exception as e:
                logger.error("RECONCILE | failed to place {} order: {}", hedge_side, e)
                best_level.order_id = None
                best_level.status = "pending"

        self.levels.sort(key=lambda l: l.price)

        for level in self.levels:
            if level.side != "sell" or level.order_id is not None:
                continue
            if level.quantity <= 0 or level.price <= 0:
                continue
            qty = self.exchange.exchange.amount_to_precision(self.symbol, level.quantity)
            if float(qty) <= 0:
                continue
            try:
                params = {"reduceOnly": True, "postOnly": False}
                order = self.exchange.place_limit_order(self.symbol, "sell", level.price, float(qty), params=params)
                if "id" not in order:
                    raise ValueError("Order response missing 'id'")
                level.order_id = order["id"]
                self._open_orders_fetch_time = 0.0
                level.status = "replaced"
                logger.info(
                    "RECONCILE | orphaned sell level @ {} — placed sell order (qty={})",
                    level.price, qty,
                )
            except Exception as e:
                logger.error("RECONCILE | failed to place orphaned sell order: {}", e)

    def _unwind_position_through_grid(self, balance: float) -> None:
        """Spread any open inventory across the new grid's exit-side levels as
        reduce-only limit orders so recenter lets the position unwind through the
        grid as price recovers instead of market-closing it (which realized large
        losses on every downward breakout).
        """
        try:
            positions = self.exchange.get_positions(self.symbol)
        except Exception as e:
            logger.error("UNWIND | could not fetch positions: {}", e)
            return

        for pos in positions:
            side = pos.get("side", "")
            amt = float(pos.get("contracts", 0) or 0)
            entry = float(pos.get("entryPrice", 0) or 0)
            if amt == 0 or entry <= 0:
                continue
            if side == "long" and amt < 0:
                side = "short"
                amt = abs(amt)
            elif side == "short" and amt < 0:
                side = "long"
                amt = abs(amt)
            if side not in {"long", "short"}:
                continue

            exit_side = "sell" if side == "long" else "buy"
            levels = [l for l in self.levels if l.side == exit_side and l.order_id is None]
            if not levels:
                logger.warning(
                    "UNWIND | no free {} levels to absorb {} {} position @ {} — position rides unhedged",
                    exit_side, side, amt, round(entry, 8),
                )
                continue

            # Slice the ACTUAL position across the exit levels. Sizing each level at
            # the normal grid notional instead meant the unwind tried to sell
            # len(levels) x grid_qty against a position often far smaller than that:
            # Binance accepted orders until the cumulative reduceOnly quantity reached
            # the position and rejected the rest (-2022), so every recenter logged a
            # burst of guaranteed-fail placements (105 in one session).
            remaining = amt
            per_level = amt / len(levels)
            placed = 0
            for level in levels:
                try:
                    if remaining <= 0:
                        break
                    slice_qty = min(per_level, remaining)
                    qty = self.exchange.exchange.amount_to_precision(self.symbol, slice_qty)
                    if float(qty) <= 0:
                        continue
                    if float(qty) * level.price < MIN_NOTIONAL_USDT:
                        logger.debug(
                            "UNWIND | slice {} @ {} below {} USDT minimum — skipping level",
                            qty, level.price, MIN_NOTIONAL_USDT,
                        )
                        continue
                    params = {"reduceOnly": True, "postOnly": False}
                    order = self.exchange.place_limit_order(self.symbol, exit_side, level.price, float(qty), params=params)
                    if "id" not in order:
                        raise ValueError("Order response missing 'id'")
                    level.order_id = order["id"]
                    level.status = "replaced"
                    level.quantity = float(qty)
                    level.entry_price = entry
                    level.fill_count = 1
                    remaining -= float(qty)
                    placed += 1
                    logger.info(
                        "UNWIND | {} {} reduce-only {} @ {} (qty={}, entry={})",
                        side, amt, exit_side.upper(), level.price, qty, round(entry, 8),
                    )
                except Exception as e:
                    logger.error("UNWIND | failed to place {} @ {}: {}", exit_side, level.price, e)
            logger.info(
                "UNWIND | {} {} position rides through {} {} levels (placed={})",
                side, amt, len(levels), exit_side, placed,
            )

    def _is_on_cooldown(self, level: GridLevel) -> bool:
        """Check if this specific level is still within its post-fill/cancel cooldown.

        Cooldown is tracked per level (by object identity) so a fill on one level
        cannot throttle replacement of other levels during a burst.
        """
        if self.replacement_cooldown <= 0:
            return False
        last = self._level_cooldowns.get(id(level))
        if last is None:
            return False
        return (time.time() - last) < self.replacement_cooldown

    def _mark_cooldown(self, level: GridLevel) -> None:
        now = time.time()
        self._last_replacement_time = now  # kept for state-file/telemetry compatibility
        self._level_cooldowns[id(level)] = now

    def check_fills(self, balance: float) -> list[dict]:
        open_orders = self.exchange.get_open_orders(self.symbol)
        open_ids = {o["id"] for o in open_orders}
        fills = []

        for level in self.levels:
            if level.order_id is None:
                continue
            if level.order_id not in open_ids:
                order = self.exchange.fetch_order(level.order_id, self.symbol)
                if order is None:
                    positions = self.exchange.get_positions(self.symbol)
                    has_long = False
                    has_short = False
                    for p in positions:
                        side = p.get("side", "")
                        contracts = float(p.get("contracts", 0) or 0)
                        if side == "long" and contracts < 0:
                            side, contracts = "short", abs(contracts)
                        elif side == "short" and contracts < 0:
                            side, contracts = "long", abs(contracts)
                        if contracts > 0:
                            if side == "long":
                                has_long = True
                            elif side == "short":
                                has_short = True
                    if level.side == "buy" and has_long:
                        logger.warning(
                            "Order {} gone and buy level + long position exists — processing as fill",
                            level.order_id,
                        )
                        fills.append(self._handle_fill(level, balance))
                    elif level.side == "sell" and has_short:
                        logger.warning(
                            "Order {} gone and sell level + short position exists — processing as fill",
                            level.order_id,
                        )
                        fills.append(self._handle_fill(level, balance))
                    elif has_long or has_short:
                        logger.warning(
                            "Order {} gone but {} level — position may be from another order, marking dead",
                            level.order_id, level.side,
                        )
                        if self._event_journal:
                            self._event_journal.order_cancelled(self.symbol, level.side, level.price, level.order_id, "order_dead_orphan")
                        level.order_id = None
                        level.status = "pending"
                    else:
                        logger.warning(
                            "Order {} gone and no position — marking level dead",
                            level.order_id,
                        )
                        if self._event_journal:
                            self._event_journal.order_cancelled(self.symbol, level.side, level.price, level.order_id, "order_dead_no_position")
                        level.order_id = None
                        level.status = "pending"
                    continue
                if order.get("status") == "canceled":
                    logger.debug("Order {} was cancelled, placing replacement", level.order_id)
                    if self._event_journal:
                        self._event_journal.order_cancelled(self.symbol, level.side, level.price, level.order_id, "fill_check_cancelled")
                    level.order_id = None
                    level.status = "pending"
                    if not self._is_on_cooldown(level):
                        self._place_order_for_level(level, balance)
                    else:
                        logger.debug("SKIP REPLACEMENT (cooldown) | {} @ {}", level.side, level.price)
                    continue
                fills.append(self._handle_fill(level, balance))

        orphaned = [l for l in self.levels if l.order_id is None and l.status == "pending" and (l.quantity > 0 or l.fill_count == 0)]
        if orphaned:
            placed_slots: set[tuple[float, str]] = set()
            first = True
            for level in orphaned:
                if level.price < self.grid_lower or level.price > self.grid_upper:
                    logger.debug(
                        "SKIP ORPHAN | {} @ {} outside grid bounds [{}-{}]",
                        level.side, level.price, self.grid_lower, self.grid_upper,
                    )
                    continue
                if self._is_on_cooldown(level):
                    logger.debug("SKIP ORPHAN (cooldown) | {} @ {}", level.side, level.price)
                    continue
                slot = (level.price, level.side)
                if slot in placed_slots:
                    logger.debug(
                        "SKIP ORPHAN | {} @ {} — slot already placed this cycle",
                        level.side, level.price,
                    )
                    continue
                if self.order_pacing_seconds > 0 and not first:
                    time.sleep(self.order_pacing_seconds)
                first = False
                if self._place_order_for_level(level, balance):
                    placed_slots.add(slot)
                    logger.info("Replaced orphaned level @ {} {}", level.price, level.side)

        return fills

    def _handle_fill(self, level: GridLevel, balance: float, is_taker: bool = False) -> dict:
        # Any fill on a level that has already filled once completes the position
        # opened by the previous fill on the same level: a sell closes the long,
        # a buy closes the short. fill_count == 0 means this fill merely opens a
        # new position (long on a buy level, short on a sell level).
        completed_cycle = level.fill_count > 0
        self._mark_cooldown(level)

        if completed_cycle:
            if level.side == "buy":
                entry_price = level.entry_price if level.entry_price else (level.price + self.grid_spacing)
            else:
                entry_price = level.entry_price if level.entry_price else (level.price - self.grid_spacing)
            exit_price = level.price
            qty = level.quantity if level.quantity > 0 else (self._calc_usdt_per_grid(balance) / max(level.price, 1e-12))
            profit = self._cycle_pnl(qty, entry_price, exit_price, is_short=level.side == "buy")
            fee = self._cycle_fee(qty, entry_price, exit_price, is_taker=is_taker)
        else:
            profit = 0.0
            fee = 0.0

        fill_record = {
            "price": level.price,
            "side": level.side,
            "timestamp": pd.Timestamp.now().isoformat(),
            "profit": profit,
            "fee": fee,
            "quantity": level.quantity if level.quantity > 0 else 0.0,
            "completed_cycle": completed_cycle,
        }

        level.fill_count += 1
        self.total_fills += 1

        if completed_cycle:
            level.total_pnl += profit
            self.total_pnl += profit
            self.total_fees += fee
            self.total_completed_cycles += 1

        logger.info(
            "FILL #{} | {} @ {} | qty={} profit={:.6f} fees={:.6f} net={:.6f} | cycle={}",
            self.total_fills, level.side.upper(), level.price,
            level.quantity, profit, fee, profit - fee, "complete" if completed_cycle else "open",
        )

        new_side = "sell" if level.side == "buy" else "buy"
        fill_price = level.price

        def _slot_pending(price: float) -> bool:
            return any(
                l is not level and l.price == price and l.side == new_side
                and l.order_id is None and l.status == "pending"
                and l.fill_count > 0 and l.quantity > 0
                for l in self.levels
            )

        sorted_levels = sorted(self.levels, key=lambda l: l.price)
        current_idx = None
        for i, lv in enumerate(sorted_levels):
            if lv is level:
                current_idx = i
                break

        # Snap to the nearest existing level of the replacement side whose slot is not
        # already claimed by another pending level. This keeps orders on grid lines
        # while guaranteeing at most one pending level per price+side, so a burst of
        # fills cannot pile several levels onto the same slot.
        if current_idx is not None:
            if new_side == "sell":
                new_price = level.price + self.grid_spacing
                for j in range(current_idx + 1, len(sorted_levels)):
                    if sorted_levels[j].side == "sell" and not _slot_pending(sorted_levels[j].price):
                        new_price = sorted_levels[j].price
                        break
            elif new_side == "buy":
                new_price = level.price - self.grid_spacing
                for j in range(current_idx - 1, -1, -1):
                    if sorted_levels[j].side == "buy" and not _slot_pending(sorted_levels[j].price):
                        new_price = sorted_levels[j].price
                        break
            else:
                new_price = level.price + self.grid_spacing if new_side == "sell" else level.price - self.grid_spacing
        else:
            new_price = level.price + self.grid_spacing if new_side == "sell" else level.price - self.grid_spacing
        new_price = self._round_price(new_price)

        # If the snapped slot is still claimed by another pending level (rare, e.g. a
        # burst of fills), step outward by grid spacing until a free slot is found.
        # Bounded by grid_count+2 iterations: if grid_spacing rounds to zero at the
        # exchange's price precision (a very tight spacing on a low-tick-size symbol),
        # new_price would never change and this would spin forever without a cap.
        step = self.grid_spacing if new_side == "sell" else -self.grid_spacing
        _max_steps = self.grid_count + 2
        _steps = 0
        while _slot_pending(new_price) and self.grid_lower <= new_price <= self.grid_upper:
            new_price = self._round_price(new_price + step)
            _steps += 1
            if _steps > _max_steps:
                logger.warning(
                    "REPLACEMENT SLOT SEARCH | gave up after {} steps @ {} (spacing {} may be below "
                    "tick precision) — level will not place order",
                    _max_steps, new_price, self.grid_spacing,
                )
                new_price = self.grid_upper + step if step > 0 else self.grid_lower + step
                break

        if new_price < self.grid_lower or new_price > self.grid_upper:
            logger.warning("Replacement price {} outside grid bounds — level will not place order", new_price)
            level.side = new_side
            level.price = new_price
            level.order_id = None
            level.status = "pending"
            return fill_record

        occupied = any(
            l is not level and l.price == new_price and l.order_id is not None and l.status in ("pending", "replaced")
            for l in self.levels
        )
        if occupied:
            logger.info(
                "SKIP REPLACEMENT | level at {} already occupied by active order — will retry", new_price,
            )
            level.side = new_side
            level.price = new_price
            level.order_id = None
            level.status = "pending"
            return fill_record

        qty = level.quantity if level.quantity > 0 else (self._calc_usdt_per_grid(balance) / max(new_price, 1e-12))
        quantity = self.exchange.exchange.amount_to_precision(self.symbol, qty)

        level.side = new_side
        level.price = new_price
        level.quantity = float(quantity)
        # Track the price of the position this level now implies: after a buy fill
        # the resting sell closes a long entered at fill_price, and after a sell
        # fill the resting buy closes a short entered at fill_price.
        level.entry_price = fill_price

        try:
            # reduceOnly is only legal when this order actually closes an open
            # position. Hard-coding it on every sell meant that while the grid was
            # net SHORT -- where a sell ADDS exposure -- every single replacement was
            # rejected with -2022, so the sell side could never re-arm and the grid
            # decayed into a one-sided book (265 such rejections in one session).
            params, adj_qty = self._exit_order_params(new_side, float(quantity))
            if adj_qty <= 0:
                raise ValueError("replacement quantity resolved to zero")
            quantity = self.exchange.exchange.amount_to_precision(self.symbol, adj_qty)
            level.quantity = float(quantity)
            order = self.exchange.place_limit_order(self.symbol, new_side, new_price, float(quantity), params=params)
            if "id" not in order:
                raise ValueError("Order response missing 'id'")
            level.order_id = order["id"]
            level.status = "replaced"
            self._open_orders_fetch_time = 0.0
        except Exception as e:
            logger.error("Failed to place replacement order at {}: {} — will retry next cycle", new_price, e)
            level.order_id = None
            level.status = "pending"

        self.levels.sort(key=lambda l: l.price)
        return fill_record

    def pause(self) -> None:
        if not self.active:
            return
        cancelled = self.exchange.cancel_everything(self.symbol, timeout_seconds=30)
        still_open = self.exchange.get_open_order_ids(self.symbol)
        for level in self.levels:
            if level.order_id is not None:
                if level.order_id in still_open:
                    logger.warning("PAUSE | order {} still open after cancel_everything — force cancelling", level.order_id)
                    self.exchange.cancel_order(level.order_id, self.symbol)
                if self._event_journal:
                    self._event_journal.order_cancelled(self.symbol, level.side, level.price, level.order_id, "pause")
                if self._notifier:
                    self._notifier.on_order_cancelled(self.symbol, level.side, level.price, level.order_id, "pause")
                level.order_id = None
                level.status = "pending"
        self.active = False
        if self._event_journal:
            self._event_journal.grid_paused(self.symbol, "manual/pause", cancelled)
        logger.info("GRID PAUSED | {} orders cancelled via cancel_everything", cancelled)

    def activate(self, balance: float) -> None:
        if self.active:
            return
        placed = self.place_initial_orders(balance)
        total_open = len(self.get_tracked_order_ids())
        self.active = total_open > 0
        if self._event_journal:
            self._event_journal.grid_activated(self.symbol, self.grid_lower, self.grid_upper, self.grid_count, total_open)
        if self.active:
            logger.info("GRID ACTIVATED | {} orders active", total_open)
        else:
            logger.warning("GRID NOT ACTIVATED | no orders could be placed or restored")

    def emergency_stop(self) -> None:
        logger.error("EMERGENCY STOP | cancelling all orders")
        cancelled = self.exchange.cancel_everything(self.symbol)
        for level in self.levels:
            if level.order_id is not None:
                if self._event_journal:
                    self._event_journal.order_cancelled(self.symbol, level.side, level.price, level.order_id, "emergency_stop")
                if self._notifier:
                    self._notifier.on_order_cancelled(self.symbol, level.side, level.price, level.order_id, "emergency_stop")
                level.order_id = None
                level.status = "pending"
        self.active = False
        if self._event_journal:
            self._event_journal.grid_paused(self.symbol, "emergency_stop", cancelled)

    def recenter(self, current_price: float, balance: float, margin_pct: float = 0.01) -> bool:
        now = time.time()
        if (now - self._last_recenter_time) < self.recenter_cooldown:
            return False

        lower_margin = self.grid_lower * (1 - margin_pct)
        upper_margin = self.grid_upper * (1 + margin_pct)

        in_margin_band = current_price >= lower_margin and current_price <= upper_margin

        active_buys = [l for l in self.levels if l.side == "buy" and l.order_id is not None]
        active_sells = [l for l in self.levels if l.side == "sell" and l.order_id is not None]

        # A grid that has gone one-sided is dead even inside the margin band: if
        # every sell level has been consumed and price sits above the grid, or every
        # buy level consumed and price sits below, no order can ever fill. Recenter
        # immediately instead of waiting for price to hit the margin edge.
        stranded_above = current_price > self.grid_upper and not active_sells
        stranded_below = current_price < self.grid_lower and not active_buys

        # A one-sided grid can also be dead INSIDE the band: with no active buys and
        # price below every resting sell (or vice versa) nothing on the book fills
        # until price crosses the whole grid.
        #
        # But "one-sided" is the NORMAL, HEALTHY state while unwinding inventory: a
        # capped long legitimately has buys blocked and exit sells resting above
        # price, and those sells fill as soon as price ticks up. Treating that as
        # dead made recenter fire on every cooldown expiry forever (median 196s
        # apart, 89 times in one session), and because recenter pauses the grid it
        # CANCELLED the very exit orders that were about to fill -- the position
        # could never unwind. Only call it dead when the missing side is missing for
        # no reason: not deliberately blocked by the position cap, and with no
        # inventory whose exit orders explain the imbalance.
        buys_intentionally_absent = self._block_buys or self._net_long_qty > 0
        sells_intentionally_absent = self._block_sells or self._net_short_qty > 0
        dead_inside = (
            (
                not active_buys and active_sells and not buys_intentionally_absent
                and current_price < min(l.price for l in active_sells)
            )
            or (
                not active_sells and active_buys and not sells_intentionally_absent
                and current_price > max(l.price for l in active_buys)
            )
        )

        if in_margin_band and not stranded_above and not stranded_below and not dead_inside:
            return False

        if stranded_above or stranded_below:
            logger.warning(
                "ONE-SIDED GRID | price {} outside [{}-{}] with no active {} orders — forcing recenter inside margin band",
                round(current_price, 8), round(self.grid_lower, 8), round(self.grid_upper, 8),
                "sell" if stranded_above else "buy",
            )
        elif dead_inside:
            logger.warning(
                "DEAD GRID INSIDE BAND | price {} with {} active buys and {} active sells all {} price — no fill possible, recentering",
                round(current_price, 8), len(active_buys), len(active_sells),
                "below" if not active_buys else "above",
            )

        logger.info(
            "RECENTERING GRID | price {} outside [{}-{}] (margin {:.1%})",
            round(current_price, 8), round(self.grid_lower, 8), round(self.grid_upper, 8), margin_pct,
        )

        self.pause()

        try:
            still_open = self.exchange.get_open_order_ids(self.symbol)
        except Exception as e:
            logger.error("RECENTER ABORTED | could not verify clean book: {}", e)
            return False
        if still_open:
            logger.error(
                "RECENTER ABORTED | {} orders still open after pause (write path down) — "
                "not rebuilding on a dirty book",
                len(still_open),
            )
            return False

        half_range = (self.grid_upper - self.grid_lower) / 2
        self.grid_lower = current_price - half_range
        self.grid_upper = current_price + half_range
        self.grid_spacing = (self.grid_upper - self.grid_lower) / max(1, self.grid_count - 1)

        self.levels = []
        self.initialize(current_price, balance, dynamic_spacing=True)
        self._unwind_position_through_grid(balance)
        self.place_initial_orders(balance)
        self.active = True
        self._last_recenter_time = now
        # Only re-anchor the trailing stop when there is no position to protect.
        # Resetting unconditionally handed the ratchet back to the market on every
        # recenter: with recenter firing repeatedly the peak was continuously reset
        # to the current price, so a long's stop tracked price downward instead of
        # holding its high-water mark.
        if self._net_long_qty <= 0 and self._net_short_qty <= 0:
            self._peak_price = current_price
            self._trough_price = current_price
            self._trailing_sl_price = None
            self._trailing_sl_price_short = None

        logger.info(
            "GRID RECENTERED | new range [{}-{}] | spacing={}",
            round(self.grid_lower, 8), round(self.grid_upper, 8), round(self.grid_spacing, 8),
        )
        return True

    def update_trailing_sl(self, current_price: float) -> None:
        """Raise the long trailing stop toward price. Never lowers it.

        A trailing stop is a ratchet: while a position stays open it may only move in
        the position's favour. This used to recompute the level from grid_lower and a
        _peak_price that recenter() reset to the current price, so in a downtrend the
        "stop" walked DOWN with the market (observed: 0.07003 -> 0.06905 across one
        continuously-open long) and could never be hit -- precisely when it was the
        only thing protecting the position. reset_trailing() on a side flip is the
        one place the ratchet is deliberately released.
        """
        if current_price > self._peak_price:
            self._peak_price = current_price
        static_sl = self.grid_lower * (1 - self.stop_loss_pct)
        if self._peak_price > 0:
            candidate = max(static_sl, self._peak_price * (1 - self._trailing_sl_trigger))
        else:
            candidate = static_sl
        if self._trailing_sl_price is None:
            self._trailing_sl_price = candidate
        else:
            self._trailing_sl_price = max(self._trailing_sl_price, candidate)

    def update_trailing_sl_short(self, current_price: float) -> None:
        """Lower the short trailing stop toward price. Never raises it (see above)."""
        if self._trough_price == 0.0 or current_price < self._trough_price:
            self._trough_price = current_price
        static_sl = self.grid_upper * (1 + self.stop_loss_pct)
        if self._trough_price > 0:
            candidate = min(static_sl, self._trough_price * (1 + self._trailing_sl_trigger))
        else:
            candidate = static_sl
        if self._trailing_sl_price_short is None:
            self._trailing_sl_price_short = candidate
        else:
            self._trailing_sl_price_short = min(self._trailing_sl_price_short, candidate)

    def get_stop_loss_price(self) -> float:
        if self._trailing_sl_price is not None:
            return self._trailing_sl_price
        return self.grid_lower * (1 - self.stop_loss_pct)

    def get_short_stop_loss_price(self) -> float:
        if self._trailing_sl_price_short is not None:
            return self._trailing_sl_price_short
        return self.grid_upper * (1 + self.stop_loss_pct)

    def get_scale_out_trail_price(self, side: str = "long") -> float:
        """Trailing price for the scale-out leg.

        Anchored at stop-loss distance from the running peak/trough so the scale-out
        split arms immediately at fresh start. The trigger-distance trail
        (peak*(1-trigger)) stays pinned to the static hard level while
        TRAILING_SL_TRIGGER_PCT > STOP_LOSS_PCT, which collapses the split to a
        single hard stop until price rises ~2%+ above the grid. Falls back to the
        static hard level when no anchor has been observed yet.
        """
        if side == "short":
            hard = self.grid_upper * (1 + self.stop_loss_pct)
            if self._trough_price <= 0:
                return hard
            return min(hard, self._trough_price * (1 + self.stop_loss_pct))
        hard = self.grid_lower * (1 - self.stop_loss_pct)
        if self._peak_price <= 0:
            return hard
        return max(hard, self._peak_price * (1 - self.stop_loss_pct))

    def reset_trailing(self) -> None:
        """Reset trailing peak/trough tracking, e.g. when the position side flips."""
        self._peak_price = 0.0
        self._trough_price = 0.0
        self._trailing_sl_price = None
        self._trailing_sl_price_short = None

    def log_sl_status(self, side: str = "long") -> None:
        if side == "short":
            logger.info(
                "SL STATUS | side=short | trigger={}% | trough={} | sl={}",
                round(self._trailing_sl_trigger * 100, 2),
                round(self._trough_price, 8),
                round(self.get_short_stop_loss_price(), 8),
            )
            return
        logger.info(
            "SL STATUS | side=long | trigger={}% | peak={} | sl={}",
            round(self._trailing_sl_trigger * 100, 2),
            round(self._peak_price, 8),
            round(self.get_stop_loss_price(), 8),
        )

    def log_analytics(self, balance: float = 0.0) -> None:
        net_pnl = self.total_pnl - self.total_fees
        avg_pnl_per_cycle = net_pnl / max(1, self.total_completed_cycles)
        total_range = self.grid_upper - self.grid_lower
        mid_price = (self.grid_upper + self.grid_lower) / 2
        range_pct = total_range / mid_price * 100 if mid_price > 0 else 0.0
        filled_levels = sum(1 for l in self.levels if l.fill_count > 0)
        pending_orders = sum(1 for l in self.levels if l.status == "pending")
        replaced_orders = sum(1 for l in self.levels if l.status == "replaced")
        buy_levels = sum(1 for l in self.levels if l.side == "buy" and l.status in ("pending", "replaced"))
        sell_levels = sum(1 for l in self.levels if l.side == "sell" and l.status in ("pending", "replaced"))
        exposure = self.get_exposure_pct(balance) if balance > 0 else 0.0

        logger.info(
            "GRID ANALYTICS | fills={} cycles={} | gross={:.6f} fees={:.6f} net={:.6f} | "
            "avg_cycle={:.6f} | filled={}/{} | pending={} replaced={} | "
            "buys={} sells={} | range={:.2f}% | vol_mult={:.2f} | "
            "sl={} | exposure={:.1%} | spread={:.4f}%",
            self.total_fills, self.total_completed_cycles, self.total_pnl, self.total_fees, net_pnl,
            avg_pnl_per_cycle, filled_levels, len(self.levels),
            pending_orders, replaced_orders,
            buy_levels, sell_levels, range_pct,
            self._volatility_mult, self.get_stop_loss_price(),
            exposure, self._last_orderbook.get("spread_pct", 0) * 100,
        )

    def _rebuild_levels(self, current_price: float | None = None) -> None:
        self.grid_spacing = (self.grid_upper - self.grid_lower) / max(1, self.grid_count - 1)
        levels = []
        for i in range(self.grid_count):
            price = self._round_price(self.grid_lower + i * self.grid_spacing)
            side = "buy" if current_price is not None and price < current_price else "sell"
            lvl = GridLevel(price=price, side=side)
            if side == "buy":
                lvl.entry_price = price
            levels.append(lvl)
        self.levels = levels

    def get_unrealized_pnl(self, current_price: float) -> float:
        pnl = 0.0
        for level in self.levels:
            qty = level.quantity if level.quantity > 0 else 0.0
            if qty <= 0:
                continue
            if level.side == "sell":
                if level.status in ("replaced", "pending"):
                    entry_price = level.entry_price if level.entry_price else level.price
                    gross = (current_price - entry_price) * qty
                    pnl += gross
        return pnl

    def to_dict(self) -> dict:
        return {
            "grid_lower": self.grid_lower,
            "grid_upper": self.grid_upper,
            "grid_count": self.grid_count,
            "grid_spacing": self.grid_spacing,
            "active": self.active,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "total_fills": self.total_fills,
            "total_completed_cycles": self.total_completed_cycles,
            "_last_recenter_time": self._last_recenter_time,
            "_trailing_sl_price": self._trailing_sl_price,
            "_trailing_sl_trigger": self._trailing_sl_trigger,
            "_peak_price": self._peak_price,
            "_trough_price": self._trough_price,
            "_trailing_sl_price_short": self._trailing_sl_price_short,
            "_volatility_mult": self._volatility_mult,
            "_block_buys": self._block_buys,
            "_block_sells": self._block_sells,
            "_last_replacement_time": self._last_replacement_time,
            "_buy_scale": self._buy_scale,
            "_sell_scale": self._sell_scale,
            "levels": [l.to_dict() for l in self.levels],
        }

    def _dedupe_levels(self) -> None:
        """Merge duplicate price+side slots while preserving active bookkeeping.

        This stabilizes both restore-time and runtime state where a fill/replacement
        path can leave the same grid price-side slot represented twice.
        """
        merged: dict[tuple[float, str], GridLevel] = {}
        for level in self.levels:
            key = (self._round_price(level.price), level.side)
            existing = merged.get(key)
            if existing is None:
                merged[key] = level
                continue

            existing.fill_count = max(existing.fill_count, level.fill_count)
            existing.total_pnl += level.total_pnl
            if level.quantity > existing.quantity:
                existing.quantity = level.quantity
            if level.order_id is not None and existing.order_id is None:
                existing.order_id = level.order_id
                existing.status = level.status
                existing.entry_price = level.entry_price
                existing.quantity = level.quantity
            elif existing.order_id is None and level.order_id is None and level.status == "replaced":
                existing.status = level.status
                existing.entry_price = level.entry_price
                existing.quantity = level.quantity
            elif level.status == "replaced" and existing.status != "replaced":
                existing.status = level.status
                existing.order_id = level.order_id or existing.order_id
                existing.entry_price = level.entry_price or existing.entry_price
                existing.quantity = level.quantity or existing.quantity

        self.levels = sorted(merged.values(), key=lambda x: x.price)

    def load_from_dict(self, data: dict, current_price: float | None = None) -> None:
        self.grid_lower = data["grid_lower"]
        self.grid_upper = data["grid_upper"]
        self.grid_count = data["grid_count"]
        self.grid_spacing = data["grid_spacing"]
        self.active = data["active"]
        self.total_pnl = data.get("total_pnl", 0.0)
        self.total_fees = data.get("total_fees", 0.0)
        self.total_fills = data.get("total_fills", 0)
        self.total_completed_cycles = data.get("total_completed_cycles", 0)
        self._last_recenter_time = data.get("_last_recenter_time", 0.0)
        self._trailing_sl_price = data.get("_trailing_sl_price", None)
        self._trailing_sl_trigger = data.get("_trailing_sl_trigger", self._trailing_sl_trigger)
        self._peak_price = data.get("_peak_price", 0.0)
        self._trough_price = data.get("_trough_price", 0.0)
        self._trailing_sl_price_short = data.get("_trailing_sl_price_short", None)
        self._volatility_mult = data.get("_volatility_mult", 1.0)
        self._block_buys = data.get("_block_buys", False)
        self._block_sells = data.get("_block_sells", False)
        self._last_replacement_time = data.get("_last_replacement_time", 0.0)
        self._buy_scale = data.get("_buy_scale", 1.0)
        self._sell_scale = data.get("_sell_scale", 1.0)
        raw_levels = [GridLevel.from_dict(l) for l in data.get("levels", [])]
        seen = {}
        for l in raw_levels:
            key = (l.price, l.side)
            if key not in seen:
                seen[key] = l
            else:
                existing = seen[key]
                existing.fill_count = max(existing.fill_count, l.fill_count)
                existing.total_pnl += l.total_pnl
                if l.order_id is not None and existing.order_id is None:
                    existing.order_id = l.order_id
                    existing.status = l.status
                    existing.entry_price = l.entry_price
                    existing.quantity = l.quantity
                elif l.status == "replaced" and existing.status != "replaced":
                    existing.status = l.status
                    existing.entry_price = l.entry_price
                    existing.quantity = l.quantity
        self.levels = sorted(seen.values(), key=lambda x: x.price)
        self._dedupe_levels()
        if len(raw_levels) != len(self.levels):
            logger.warning(
                "DEDUPLICATED {} levels with duplicate price+side ({} -> {})",
                len(raw_levels) - len(self.levels), len(raw_levels), len(self.levels),
            )

        self._refill_missing_grid_lines(current_price)

        if len(self.levels) != self.grid_count:
            self.state_corrupted = True
            logger.warning(
                "GRID STATE CORRUPT | expected {} levels but restored {} — "
                "rebuilding levels from grid bounds",
                self.grid_count, len(self.levels),
            )
            self._rebuild_levels(current_price)

    def _refill_missing_grid_lines(self, current_price: float | None = None) -> None:
        """Re-add pending levels on empty grid lines after duplicate price+side
        levels were merged. A fill whose replacement parked on an occupied slot
        leaves a hole in the grid; refilling it preserves fill/cycle bookkeeping
        instead of rebuilding every level from the grid bounds."""
        occupied = {self._round_price(l.price) for l in self.levels}
        missing = 0
        for i in range(self.grid_count):
            if len(self.levels) >= self.grid_count:
                break
            price = self._round_price(self.grid_lower + i * self.grid_spacing)
            if price in occupied:
                continue
            side = "buy" if current_price is not None and price < current_price else "sell"
            lvl = GridLevel(price=price, side=side)
            if side == "buy":
                lvl.entry_price = price
            self.levels.append(lvl)
            occupied.add(price)
            missing += 1
        self.levels.sort(key=lambda x: x.price)
        if missing:
            logger.warning(
                "RECOVERED grid state | refilled {} empty grid line(s) after dedupe, "
                "preserving {} existing levels' bookkeeping",
                missing, len(self.levels) - missing,
            )
