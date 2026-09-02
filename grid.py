from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from loguru import logger

from exchange import Exchange, PostOnlyWouldCross


@dataclass
class GridLevel:
    price: float
    side: str  # "buy" or "sell"
    order_id: str | None = None
    status: str = "pending"  # pending, replaced, awaiting_counter
    fill_count: int = 0
    total_pnl: float = 0.0
    quantity: float = 0.0
    entry_price: float = 0.0
    # Set when this rung filled but its counter-slot was already live. That order is
    # this fill's exit, so the rung places nothing until the slot frees -- re-arming on
    # the same side instead accumulated 8215 DOGE at one price (AUDIT #61).
    awaiting_side: str | None = None
    awaiting_price: float | None = None

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
    mode: str = "atr",
    realised_window: int = 24,
    realised_multiplier: float = 5.0,
) -> tuple[float, float]:
    """Half-width of the ladder, from recent volatility.

    Two estimators. "atr" is ATR(14) x atr_multiplier, the long-standing behaviour.
    "realised" is the mean CANDLE RANGE over realised_window bars x realised_multiplier.

    The second one predicts the next 24 hours' actual span better, which is what the
    range is FOR: too wide and the rungs sit where price never goes -- 2026-08-18 ran a
    2.83% ladder while price travelled 0.43% in four hours and filled once. Measured
    walk-forward on DOGEUSDT 1h, three folds of 363 test bars each:

        fold   ATRx3.5 MAE   realised MAE   better by
           1       1.4004%        1.3535%        3.3%
           2       1.3363%        1.2473%        6.7%
           3       1.0183%        0.8310%       18.4%

    Better in every fold, and the multiplier that falls out is stable across them
    (4.94-5.07), which is why it is a constant and not something to tune (AUDIT #111).
    """
    candles_per_day = TIMEFRAME_CANDLES_PER_DAY.get(timeframe, 24)
    recent = ohlcv.tail(lookback_days * candles_per_day)
    if len(recent) < 20:
        price_range = current_price * 0.05
        return current_price - price_range, current_price + price_range

    from trend_filter import atr as calc_atr
    atr_series = calc_atr(recent["high"], recent["low"], recent["close"], period=14)
    current_atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else current_price * 0.02

    half = current_atr * atr_multiplier
    if mode == "realised":
        window = recent.tail(realised_window)
        if len(window) >= max(2, realised_window // 2):
            mean_range = float(
                ((window["high"] - window["low"]) / window["close"]).mean())
            if mean_range > 0:
                # realised_multiplier was fitted against the FULL next-24h span, and
                # this function returns a HALF-width. Halve it or the ladder comes out
                # twice as wide as the estimator says -- which is the exact failure the
                # mode exists to fix, so it would have been quiet and wrong.
                half = current_price * mean_range * realised_multiplier / 2.0
                logger.info(
                    "GRID RANGE | realised: mean 1h range {:.3%} over {} bars x {}/2 "
                    "= {:.3%} half-width (ATR path would have given {:.3%})",
                    mean_range, len(window), realised_multiplier,
                    half / current_price, current_atr * atr_multiplier / current_price,
                )

    # A zero half-width puts every rung on one line and leaves grid_spacing at 0, which
    # disables the fee-floor and deformation checks that divide by it. Both estimators
    # can reach it: ATR is 0.0 (not NaN) on a perfectly flat window, and so is the mean
    # candle range. Fall back to the same 5% the too-few-candles path uses -- "not
    # enough information to size this" is the same answer in both cases.
    if half <= 0:
        logger.warning(
            "GRID RANGE | measured volatility is zero over the lookback — falling back "
            "to a 5% half-width rather than collapsing every rung onto one price")
        half = current_price * 0.05

    lower = current_price - half
    upper = current_price + half

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


def order_was_filled(order: dict | None) -> bool:
    """Did this order actually execute? Allowlist, never a denylist.

    An absent order is ambiguous and a present one is only unambiguous when the exchange
    says it completed. check_fills used to ask `status == "canceled"` and treat every
    other answer as a fill -- so EXPIRED, REJECTED, and even NEW all booked profit.

    Measured live on 2026-08-15 at 23:58:45. A recenter placed three reduce-only unwind
    buys; Binance EXPIRED all three with executedQty=0 (reduce-only orders that can no
    longer reduce are expired, not cancelled). The grid booked FILL #2/#3/#4 for
    +0.375 +0.238 +0.095, three completed cycles, three rows in the trade journal, and
    three trades against the daily counter. userTrades for that window is empty: nothing
    executed. The risk layer printed the contradiction in the same second --
    "verified pnl=+0.00 | grid estimated +0.71" -- and the +0.71 then sat in every
    status line for the next three and a half hours (AUDIT #75).

    This is trail_stop_fired's rule (AUDIT #26) applied to the path that books PnL.
    """
    if (order or {}).get("status") not in ("closed", "filled"):
        return False
    # A terminal order that moved no quantity is not a fill either, whatever it is
    # called. `filled` is ccxt's normalised executedQty; absent means unknown, and an
    # unknown quantity on a status that already says "closed" is treated as genuine.
    filled = (order or {}).get("filled")
    if filled is None:
        filled = ((order or {}).get("info") or {}).get("executedQty")
    if filled is None:
        return True
    try:
        return float(filled) > 0
    except (TypeError, ValueError):
        return True


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
    # Class-level so it survives GridEngine.__new__(GridEngine). Several tests build the
    # engine that way on purpose -- tests/test_startup_ladder_teardown.py reproduces the
    # exact 05:00:56 startup shape by setting only the fields recenter reads -- and an
    # __init__-only attribute raises AttributeError there instead of defaulting. It is
    # transient by design: a warning clock, not state, so it is not persisted and a
    # restart legitimately starts it at zero.
    _last_deform_warn_time = 0.0

    # Same reason, and the same mistake made twice: _flip_captures_a_spread reaches
    # _position_break_even, which reads this cache, and tests/test_oneway_starvation.py
    # also builds its engine with __new__. Four of its tests raised AttributeError the
    # first time the gate landed (AUDIT #119).
    _break_even_cache: "tuple[str, float] | None" = None
    _break_even_time = 0.0

    # Same defensive placement as the two above, on principle rather than a specific
    # incident this time: no test currently builds an engine with __new__ and reaches
    # accelerate_handoff_exit, but _last_deform_warn_time and _break_even_cache/_time
    # were both bitten by exactly that combination once each, and this field has the
    # identical shape (a transient __init__-only clock, read before any __init__ has
    # necessarily run). Cheaper to default it here now than to rediscover the pattern
    # a third time (AUDIT #145).
    _last_handoff_accel_time = 0.0
    HANDOFF_ACCEL_COOLDOWN_SECONDS = 60.0

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
        taker_fill_share: float = 0.118,
        recenter_cooldown: int = 300,
        replacement_cooldown: int = 60,
        order_pacing_seconds: float = 0.6,
        capital_per_grid_usdt: float = 0.0,
        leverage: int = 1,
        trailing_sl_trigger_pct: float = 0.05,
        max_exposure_pct: float = 0.50,
        use_market_close_on_replace: bool = False,
        min_profit_multiplier: float = 3.0,
        rung_loss_cap_pct: float = 0.01,
        max_open_loss_usdt: float = 0.0,
        daily_profit_lock_usdt: float = 0.0,
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
        self.taker_fill_share = min(1.0, max(0.0, taker_fill_share))
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
        # The netted position the account actually holds. total_pnl is realised from
        # THIS, not from per-level round trips, so the bot's figures agree with the
        # exchange (AUDIT #80).
        self._pos_qty = 0.0
        self._pos_entry = 0.0
        self._mirror_mismatch_polls = 0
        self.total_fills = 0
        self.total_completed_cycles = 0
        self._last_recenter_time = 0.0
        self._last_handoff_accel_time = 0.0
        self._volatility_mult = 1.0
        self._trailing_sl_price: float | None = None
        self._trailing_sl_trigger: float = trailing_sl_trigger_pct
        self._peak_price = 0.0
        self._break_even_cache: tuple[str, float] | None = None
        self._break_even_time = 0.0
        self._be_block_logged: set[tuple[str, float]] = set()
        self._trough_price = 0.0
        self._trailing_sl_price_short: float | None = None
        # Ratcheted static stop levels. See get_hard_stop_loss_price -- recenter moves
        # grid_lower/grid_upper, and without these the hard stop follows it away from an
        # open position.
        self._hard_sl_price: float | None = None
        self._hard_sl_price_short: float | None = None
        self._last_orderbook: dict = {}
        self._event_journal = event_journal
        self._notifier = notifier
        self._block_buys = False
        self._block_sells = False
        # Loss budget: once the OPEN position's unrealised loss reaches
        # max_open_loss_usdt, the side that would ADD to it is blocked (0 disables).
        # See apply_open_loss_guard.
        self.max_open_loss_usdt = max_open_loss_usdt
        self._loss_block_buys = False
        self._loss_block_sells = False
        # Profit budget for the DAY: once daily_realized_pnl reaches
        # daily_profit_lock_usdt, BOTH sides stop opening/adding new exposure (0
        # disables). One shared flag, not a buy/sell pair like the loss guard above --
        # this isn't defending a specific position, it blocks either direction alike.
        # See apply_profit_lock_guard.
        self.daily_profit_lock_usdt = daily_profit_lock_usdt
        self._profit_lock_active = False
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
        # How many times the round-trip fee a level's spacing must cover before the
        # level is worth placing (see _is_level_profitable). This was hardcoded to 1.0,
        # i.e. break-even plus epsilon: a level clearing its own fees by a hair passed
        # the gate, so a grid whose spacing was only ~2.6x the round-trip fee placed
        # every level and handed ~39% of gross back to the exchange. Fees are the
        # dominant cost at this trade frequency, so the floor belongs in config.
        self._min_profit_multiplier: float = min_profit_multiplier
        self._rung_loss_cap_pct: float = max(0.0, rung_loss_cap_pct)
        self._regime: str = "uncertain"
        self._open_orders_fetch_time: float = 0.0
        self._open_orders_map: dict[tuple[float, str], dict] = {}
        self._warned_small_fixed_allocation = False
        self._exposure_ceiling_trimming = False
        self._last_exposure_trim = 0.0

    def _round_price_toward(self, value: float, direction: int) -> float:
        """Round to exchange precision WITHOUT crossing `value`.

        `direction` -1 means the result must not end up above `value`; +1 means it must
        not end up below. Ordinary round-to-nearest crosses profitability boundaries: a
        short's break-even of 0.07021909 rounds to 0.07022 at five decimals, which is
        above it, so covering there is a loss by 0.0000009 -- small per unit, and on
        9,916 DOGE it is the difference between a winning exit and a losing one
        (AUDIT #41).
        """
        price = self._round_price(value)
        if (direction < 0 and price <= value) or (direction > 0 and price >= value):
            return price

        # Rounding crossed the boundary. Back off by a step that DOUBLES until the
        # rounded result lands on the safe side -- the exchange's tick size is not
        # exposed, and a fixed epsilon is either too small to move a coarse tick or
        # needlessly wide on a fine one.
        step = max(abs(value) * 1e-9, 1e-12)
        for _ in range(60):
            candidate = self._round_price(value - step if direction < 0 else value + step)
            if (direction < 0 and candidate <= value) or (direction > 0 and candidate >= value):
                return candidate
            step *= 2
        return price

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

    def update_regime(self, regime: str) -> None:
        """Part of the Strategy protocol; recorded but not acted on.

        A grid does not change behaviour by regime -- main.py (and the router) gate it
        externally by pausing, which cancels resting orders. Storing the value keeps it
        available for logging and avoids a caller having to special-case grids.
        """
        self._regime = regime

    def set_position_limit(self, long_position: float, short_position: float, max_position_qty: float) -> None:
        """Block new buys when the long position would exceed the limit and new sells
        when the short position would exceed it; scale order sizes down near the cap."""
        old_block_buys = self._block_buys
        old_buy_scale = self._buy_scale
        old_block_sells = self._block_sells
        old_sell_scale = self._sell_scale

        self._net_long_qty = max(0.0, long_position)
        self._net_short_qty = max(0.0, short_position)

        # Gate on what the ladder has COMMITTED, not just what has filled (AUDIT #138).
        committed_long, committed_short = self._committed_exposure(
            long_position, short_position)
        self._block_buys, self._buy_scale = self._position_limit_state(
            committed_long, max_position_qty)
        self._block_sells, self._sell_scale = self._position_limit_state(
            committed_short, max_position_qty)

        # Cancel on the FILLED position, NOT the committed one.
        #
        # These are two different jobs and merging them oscillates. The block above is
        # a placement gate: it stops NEW orders that would breach the cap. Cancelling
        # because of a commitment would remove the very orders that created it -- next
        # poll the commitment is gone, the side unblocks, the ladder re-places, and it
        # blocks again. A loop that burns API calls and never converges.
        #
        # So the cancel keeps its original, blunter trigger: the position that has
        # ALREADY filled is at or past the cap, so pull the pending adds behind it
        # (AUDIT #138).
        filled_long_over, _ = self._position_limit_state(long_position, max_position_qty)
        filled_short_over, _ = self._position_limit_state(short_position, max_position_qty)
        if filled_long_over:
            self._cancel_resting_orders("buy", "position_limit")
        if filled_short_over:
            self._cancel_resting_orders("sell", "position_limit")

        if self._block_buys and not old_block_buys:
            logger.warning(
                "POSITION LIMIT | long {} (committed {} incl. resting buys) >= {} "
                "— buy orders blocked",
                round(long_position, 2), round(committed_long, 2),
                round(max_position_qty, 2),
            )
        # Rounded to the precision the message itself prints. The cap moves with equity
        # every iteration, so an exact comparison logged BUY SCALE on every single loop
        # -- 0.53, 0.53, 0.52, 0.53 -- burying the lines that matter (AUDIT #33).
        elif round(self._buy_scale, 2) != round(old_buy_scale, 2) and self._buy_scale < 1.0:
            logger.info("BUY SCALE | long={:.1f}/{:.1f} | scale={:.2f}", long_position, max_position_qty, self._buy_scale)

        if self._block_sells and not old_block_sells:
            logger.warning(
                "POSITION LIMIT | short {} (committed {} incl. resting sells) >= {} "
                "— sell orders blocked",
                round(short_position, 2), round(committed_short, 2),
                round(max_position_qty, 2),
            )
        elif round(self._sell_scale, 2) != round(old_sell_scale, 2) and self._sell_scale < 1.0:
            logger.info("SELL SCALE | short={:.1f}/{:.1f} | scale={:.2f}", short_position, max_position_qty, self._sell_scale)

    def _resting_qty(self, side: str) -> float:
        """Quantity of orders this ladder currently has live on one side."""
        return sum(l.quantity for l in self.levels
                   if l.side == side and l.order_id is not None and l.quantity > 0)

    def _committed_exposure(self, long_position: float, short_position: float
                            ) -> tuple[float, float]:
        """What each side would hold if every resting order on it filled.

        The cap was enforced against the FILLED position only, so resting orders were
        invisible to it: six sell rungs of ~125 USDT could sit under a 243 USDT cap
        and the gate reported "not blocked" until the moment they all filled. On
        2026-08-20 the ADA short book reached ~1,103 USDT against that 243 cap -- 4.5x
        over -- and the hard stop closed it for -74.84, which is 105% of the account's
        entire -71.17 loss for the period. 179 maker grid cycles earned +2.08 in the
        same window; one uncapped position gave back thirty years of that.

        Cancelling the resting side once blocked (below) is a backstop, not a bound:
        the overshoot happens BETWEEN the cap being reached and the next poll, and a
        run through six rungs takes less than one 10s interval.

        One-way netting matters here. A resting sell against an open long REDUCES the
        position; only the part beyond the long can create short. Counting it as new
        short would block the exits and weld the position in place, which is the
        failure the AUDIT #32 break-even rule already produces on its own.
        """
        net = long_position - short_position          # signed, + is long

        # ONLY the side that would ADD to the current position is gated. The side that
        # reduces it is how the position gets closed, and blocking that welds it.
        #
        # The first version of this gated both sides on net resting volume, and it
        # broke the router handoff live on 2026-08-24. router.set_position_limit
        # deliberately clamps the cap to the open size during a handoff -- "it can
        # still close through its own levels but cannot open anything new" -- so with
        # long 114 against a clamped cap of 114 and three sell rungs totalling 338
        # resting, committed_short came to 224 and the SELL side blocked. The grid
        # could no longer place the orders that would get it flat, which is precisely
        # the state the handoff clamp exists to avoid, and the grace expires into the
        # forced market dump AUDIT #29 measured at -46.16 across 18 handoffs.
        #
        # Gating the adding side alone still bounds the 2026-08-20 breach: that was a
        # SHORT growing through resting sells while already short, so sells were the
        # adding side and are gated. Overshoot past flat is bounded by the next poll,
        # when the position has flipped and those sells become the adding side.
        if net > 0:                                   # long: buys add, sells exit
            return max(0.0, net + self._resting_qty("buy")), 0.0
        if net < 0:                                   # short: sells add, buys exit
            return 0.0, max(0.0, -net + self._resting_qty("sell"))
        # Flat: either side opens, so both are gated on what they would open.
        return self._resting_qty("buy"), self._resting_qty("sell")

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

    def apply_open_loss_guard(self, price: float) -> None:
        """Refuse to ADD to a position whose open loss has eaten the loss budget.

        The cap bounds how BIG a position can get; nothing bounded how much adverse
        room it was handed. A trend that runs one way through the ladder fills every
        rung on one side -- each an intentional average -- and then hands the whole
        capped position to the hard stop as a single taker print: -74.39 of the
        account's -68.37 net over 2026-07-22..08-20, against +2.61 earned by 178
        maker cycles. The wins are structurally small (one spacing); this makes the
        losses structurally small too: once the open loss reaches the budget, the
        adverse side stops digging. Reducing stays legal -- exits are how the
        position resolves, and blocking them would weld the loss in place.

        Stateless by design: recomputed from (pos, entry, price) every call, so it
        needs no persistence and self-clears when price recovers or the position
        closes.
        """
        if self.max_open_loss_usdt <= 0:
            self._loss_block_buys = False
            self._loss_block_sells = False
            return

        if self._pos_qty > 0:
            loss = (self._pos_entry - price) * self._pos_qty
            adverse = "buy"
        elif self._pos_qty < 0:
            loss = (price - self._pos_entry) * -self._pos_qty
            adverse = "sell"
        else:
            self._loss_block_buys = False
            self._loss_block_sells = False
            return

        trip = loss >= self.max_open_loss_usdt
        if adverse == "buy":
            blocked, was = trip, self._loss_block_buys
            self._loss_block_buys = trip
        else:
            blocked, was = trip, self._loss_block_sells
            self._loss_block_sells = trip

        if blocked and not was:
            logger.warning(
                "OPEN LOSS BUDGET | {} position down {:.2f} USDT >= {:.2f} budget — "
                "{} orders blocked (no longer averaging into the move); exits stay "
                "open",
                "long" if adverse == "buy" else "short", loss,
                self.max_open_loss_usdt, adverse.upper(),
            )
            self._cancel_resting_orders(adverse, "open_loss_budget")
        elif not blocked and was:
            logger.info(
                "OPEN LOSS BUDGET | {} back inside budget ({:.2f} USDT) — {} orders "
                "unblocked",
                "long" if adverse == "buy" else "short", loss, adverse.upper(),
            )

    def apply_profit_lock_guard(self, daily_realized_pnl: float) -> None:
        """Stop OPENING new exposure once the day's realised profit hits the budget.

        Every guard above bounds LOSSES; nothing bounded the profit side, so a good
        day's gains just rode as continued exposure with no mechanism to lock them in
        -- the same shape apply_open_loss_guard exists for, mirrored: hours of small
        grid profit erased in under a minute by one bad move. Once the day's REALISED
        P&L reaches daily_profit_lock_usdt, both sides stop adding new exposure. Unlike
        apply_open_loss_guard, which only blocks the one side adverse to whichever
        position currently exists, this is symmetric -- it isn't defending a specific
        position, it's refusing to risk a day already won, regardless of which
        direction the next trade would open. Reducing stays legal: exits are how any
        open position resolves, and blocking them would weld it in place.

        Realised only, deliberately -- unrealised marks are not yet locked in, so
        (unlike the loss guard, which counts unrealised loss) they do not count toward
        this budget.

        Stateless by design: recomputed from daily_realized_pnl every call, so it needs
        no persistence and self-clears the moment the figure drops back under budget --
        including at the next UTC daily reset, once daily_reset_check has rolled the
        day's realised P&L back to zero.
        """
        if self.daily_profit_lock_usdt <= 0:
            self._profit_lock_active = False
            return

        trip = daily_realized_pnl >= self.daily_profit_lock_usdt
        was = self._profit_lock_active
        self._profit_lock_active = trip

        if trip and not was:
            logger.warning(
                "PROFIT LOCK | day's profit {:.2f} USDT >= {:.2f} budget — BUY and SELL "
                "entries blocked for the rest of the day (today's gain is protected); "
                "exits stay open",
                daily_realized_pnl, self.daily_profit_lock_usdt,
            )
            if self._notifier:
                self._notifier.on_profit_lock(daily_realized_pnl, self.daily_profit_lock_usdt)
            # Mirror apply_open_loss_guard: blocking new entries in _place_order_for_level
            # only stops rungs that don't already have a resting order (place_initial_orders
            # skips levels with order_id is not None). Without cancelling here, any entry
            # already resting on either side at the moment the lock trips stays live and can
            # still fill, adding new exposure -- exactly what this guard exists to prevent.
            #
            # AUDIT #168-adjacent fix, 2026-08-31: cancelling BOTH sides unconditionally was
            # wrong. In one-way mode a side is only ever an entry OR the position's own exit
            # -- never both at once -- and which one flips with which way the position is
            # held (see _reduce_only_qty's docstring). Tearing down both sides while SHORT
            # cancelled the resting BUY legs that were the position's only exit path, leaving
            # it un-closeable by anything but a full restart: 13:45 profit lock trips, ladder
            # goes to zero orders, DORMANT WITH EXPOSURE fires three times over 45 minutes
            # before the watchdog force-restarts the bot to re-lay it. Only cancel the side
            # that is actually adding to the CURRENT position; the opposite side, if a
            # position is held, is that position's exit and must stay live. Flat has no
            # exit to protect, so both sides are fair game there.
            if self._pos_qty > 0:
                self._cancel_resting_orders("buy", "profit_lock")
            elif self._pos_qty < 0:
                self._cancel_resting_orders("sell", "profit_lock")
            else:
                self._cancel_resting_orders("buy", "profit_lock")
                self._cancel_resting_orders("sell", "profit_lock")
        elif not trip and was:
            logger.info(
                "PROFIT LOCK | released — day's profit {:.2f} USDT back under {:.2f} "
                "budget — entries unblocked",
                daily_realized_pnl, self.daily_profit_lock_usdt,
            )

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

    def _refresh_net_counters(self) -> bool:
        """Re-read the exchange's net position into the reduce-only mirror. AUDIT #98.

        _net_long_qty/_net_short_qty are written ONLY by set_position_limit, which
        main.py calls at :1773 -- more than a hundred lines AFTER check_fills, in the
        same iteration. So every order placed during the fill sweep is sized against the
        position as it stood BEFORE the fill that triggered it, and the worst case is the
        common one: the exit for a level is placed in the same breath as its own entry.

        2026-08-17 14:06:46, fill #47 bought 1782 into a 302 long. One second later the
        exit went out as SELL 302 -- the pre-fill size -- against a real 1789. It filled
        at 14:50:30 for 302 and booked +0.045; at 1778 the same crossing was worth about
        +0.27. Again at 15:19:57: fill #51 bought 1778 into 1494, and the exit went out
        at 1494 against a real 3272.

        This asks the exchange rather than incrementing a local guess, because an
        increment cannot see a PARTIAL fill. That same 1782 order had already put 295
        into the position before it completed, so adding its full size would have claimed
        2084 against a real 1789 -- and a reduce-only order larger than the position is
        rejected -2022, which turns a missed opportunity into an unplaceable exit.

        Only called on a poll that saw a fill (11 of ~640 iterations that session), so the
        extra read costs nothing measurable. Failure is not an error: the mirror
        set_position_limit left is exactly what this code used before, so a bad read
        degrades to the old behaviour rather than blocking the fill.
        """
        try:
            positions = self.exchange.get_positions(self.symbol) or []
        except Exception as e:
            logger.debug(
                "NET COUNTER REFRESH | positions unreadable ({}) — keeping the mirror "
                "set_position_limit left behind", e,
            )
            return False

        long_qty = short_qty = 0.0
        for pos in positions:
            qty = float(pos.get("contracts", 0) or 0)
            side = str(pos.get("side", "")).lower()
            # One-way mode spells a short either as side='short' or as a negative
            # contracts count; get_position_breakdown normalises both and so must this,
            # or a short reads as a long of the same size and reduceOnly goes out
            # backwards.
            if qty < 0:
                side, qty = ("short" if side == "long" else "long"), abs(qty)
            if qty <= 0:
                continue
            if side == "short":
                short_qty += qty
            else:
                long_qty += qty

        self._net_long_qty, self._net_short_qty = long_qty, short_qty
        return True

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
        failed = 0
        for level in targets:
            if level.order_id in still_open:
                # cancel_order returns False for "status unknown" and only True when the
                # order is confirmed gone (OrderNotFound counts as gone). A level cleared
                # on an unconfirmed cancel is the worst outcome available here: the order
                # is still live AND the level goes back to "pending", so the next
                # place_initial_orders lays a SECOND order at the same price -- growing
                # exposure at the exact moment the cap says to shrink it (AUDIT #51).
                try:
                    confirmed = self.exchange.cancel_order(level.order_id, self.symbol)
                except Exception as e:
                    logger.error(
                        "POSITION LIMIT | failed to cancel {} @ {}: {}", level.side, level.price, e,
                    )
                    confirmed = False
                if not confirmed:
                    failed += 1
                    continue                      # keep order_id: retried next iteration
            if self._event_journal:
                self._event_journal.order_cancelled(self.symbol, level.side, level.price, level.order_id, reason)
            if self._notifier:
                self._notifier.on_order_cancelled(self.symbol, level.side, level.price, level.order_id, reason)
            level.order_id = None
            level.status = "pending"
            cancelled += 1
        if failed:
            logger.error(
                "POSITION LIMIT | cancelled {} resting {} orders ({}) but {} could NOT be "
                "confirmed cancelled — those levels stay claimed and will retry",
                cancelled, side, reason, failed,
            )
        else:
            logger.warning(
                "POSITION LIMIT | cancelled {} resting {} orders ({})", cancelled, side, reason,
            )
        return cancelled

    def update_orderbook(self) -> None:
        self._last_orderbook = self.exchange.get_orderbook_depth(self.symbol)

    def get_spread_pct(self) -> float:
        """Last observed bid/ask spread as a fraction of price, 0.0 if never read.

        Public because main.py logs it. It used to read `_last_orderbook` directly,
        which the router refuses to forward -- see AUDIT #31.
        """
        return float(self._last_orderbook.get("spread_pct", 0.0) or 0.0)

    @property
    def peak_price(self) -> float:
        """Highest price seen while the current long has been open -- the anchor the
        trailing and hard stops ratchet against. Public for the same reason: main.py
        carries it across the engine rebuild in recovery, and reaching into
        `_peak_price` through the router raised AttributeError every iteration."""
        return self._peak_price

    @peak_price.setter
    def peak_price(self, value: float) -> None:
        self._peak_price = float(value)

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

    def _apply_to_position(self, side: str, qty: float, price: float) -> float:
        """Net this fill into the running position and return the P&L it REALISED.

        Binance USDM one-way mode holds a single netted position, so money is realised
        only by a fill that REDUCES it. A level completing its own round trip is a
        ladder event, not necessarily a cash event, and the two came apart badly on
        2026-08-15: fill #2 covered the short and realised +0.25 while the grid called
        it 'open' and booked nothing, then fill #3 ADDED to a long while the grid
        called it 'complete' and booked a +0.25 that never happened. Reported +0.41
        against an actual +0.24 -- 69% high (AUDIT #80).

        Level-local (exit-entry)*qty arithmetic cannot be repaired by picking better
        prices or quantities, because in a netted account the level is simply not the
        thing that holds the position.
        """
        if qty <= 0 or price <= 0:
            return 0.0
        signed = qty if side == "buy" else -qty

        # Opening, or adding to what is already there: no money changes hands, the
        # entry just re-averages.
        if self._pos_qty == 0 or (self._pos_qty > 0) == (signed > 0):
            total = self._pos_qty + signed
            self._pos_entry = (
                (self._pos_entry * self._pos_qty + price * signed) / total
                if total else 0.0
            )
            self._pos_qty = total
            return 0.0

        closing = min(abs(signed), abs(self._pos_qty))
        direction = 1.0 if self._pos_qty > 0 else -1.0
        realized = (price - self._pos_entry) * closing * direction
        self._pos_qty += signed
        if abs(self._pos_qty) < 1e-9:
            self._pos_qty, self._pos_entry = 0.0, 0.0
        elif (self._pos_qty > 0) != (direction > 0):
            # Sold/bought clean through zero -- the remainder is a NEW position opened
            # at this price, not a continuation of the one just closed.
            self._pos_entry = price
        return realized

    # A positions reply is one of three things, and conflating any two of them has
    # cost this bot money in both directions. AUDIT #134.
    POSITIONS_FLAT = "flat"
    POSITIONS_HOLDS = "holds"
    POSITIONS_UNREADABLE = "unreadable"

    def _read_positions(self) -> "tuple[str, list]":
        """Classify what the account says: flat, holding, or unreadable.

        `[]` and a zero-size row both mean "this account holds nothing" -- Binance
        returns zero-size rows for symbols you are not in, so treating those as
        ambiguous would make a genuinely flat account permanently unreadable.

        A raise, or a size that will not parse, is NOT evidence of flatness. Acting on
        a reply we could not understand is how a live position's cost basis gets
        thrown away and its eventual close booked as pure profit (AUDIT #80).

        Note this classifies SIZE only. A row that holds size but carries no usable
        entry price is still HOLDS -- the account is not flat just because we cannot
        price what it holds.
        """
        try:
            positions = self.exchange.get_positions(self.symbol) or []
        except Exception as exc:
            logger.warning("POSITIONS UNREADABLE | {}", exc)
            return self.POSITIONS_UNREADABLE, []
        for pos in positions:
            raw = pos.get("contracts")
            if raw in (None, ""):
                raw = (pos.get("info") or {}).get("positionAmt")
            try:
                qty = float(raw or 0)
            except (TypeError, ValueError):
                logger.warning("POSITIONS UNREADABLE | unparseable size {!r}", raw)
                return self.POSITIONS_UNREADABLE, positions
            if abs(qty) > 0:
                return self.POSITIONS_HOLDS, positions
        return self.POSITIONS_FLAT, positions

    def detect_external_close(self, price: float) -> dict | None:
        """The exchange closed our position and nothing told the ledger.

        The hard stop-market leg fires ON THE EXCHANGE. The bot never initiates it, so
        it never passes through _handle_fill: the ledger keeps a position that no
        longer exists, total_pnl never books the loss, and no journal row is written.

        2026-08-20, from the Binance execution ledger: stop_hard was 6 taker
        executions for -76.98, one of them -74.84 -- 105% of the account's entire
        -71.17 for the period. cycle_pnl for the same window showed 47 wins, 1 loss
        and +10.96, and the -74.84 produced no row at all. Meanwhile the phantom
        position stayed on the books, where the next fill gets priced against it
        (AUDIT #143).

        reconcile_position_entry cannot cover this: it returns early when the exchange
        reads flat, because adopting "flat" was the AUDIT #80 hazard. So flat is
        handled here instead, and only when CORROBORATED by a second read -- one bad
        HTTP reply must not book a close that never happened (AUDIT #134's rule).

        Returns a fill-shaped dict so the caller journals it exactly like any other
        fill, or None when there is nothing to report. `profit` is an ESTIMATE priced
        at `price`: detection happens within a poll of the close, but the exact exit
        is the exchange's and only the income reconciler knows it. The estimate makes
        total_pnl approximately right instead of definitely wrong, and AUDIT #139's
        divergence alarm still watches the gap.
        """
        if abs(self._pos_qty) <= 1e-9 or price <= 0:
            return None

        state, _ = self._read_positions()
        if state != self.POSITIONS_FLAT:
            return None
        second, _ = self._read_positions()
        if second != self.POSITIONS_FLAT:
            logger.warning(
                "EXTERNAL CLOSE UNCONFIRMED | first read flat, second said {} -- "
                "keeping the ledger of {} (AUDIT #143)", second, round(self._pos_qty, 4),
            )
            return None

        qty, entry = self._pos_qty, self._pos_entry
        # Long closes by selling, short closes by buying.
        side = "sell" if qty > 0 else "buy"
        profit = (price - entry) * qty if entry > 0 else 0.0

        logger.error(
            "EXTERNAL CLOSE | the exchange closed {} {} @ entry {} and the ladder was "
            "never told -- booking an estimated {:+.4f} at {} and clearing the ledger. "
            "A stop leg firing is the usual cause (AUDIT #143)",
            "LONG" if qty > 0 else "SHORT", round(abs(qty), 4), round(entry, 8),
            profit, price,
        )
        self.total_fills += 1
        self.total_completed_cycles += 1
        self.total_pnl += profit
        self.seed_position(0.0, 0.0)
        if self._event_journal:
            self._event_journal.risk_check("external_close", abs(profit), 0.0, "BOOKED")
        return {
            "price": price,
            "side": side,
            "quantity": abs(qty),
            "profit": profit,
            "fee": 0.0,          # the taker fee is the exchange's; the reconciler has it
            "completed_cycle": True,
            "external": True,
            "estimated": True,
        }

    def _seed_position_from_exchange(self) -> None:
        """Adopt whatever the account already holds before trading starts.

        The exchange is the authority on the position; the ledger is a local mirror.
        Starting a session believing we are flat when 5348 DOGE of short is open --
        exactly the state this bot woke up in on 2026-08-15 -- books the eventual
        close of that inheritance as profit the session never made (AUDIT #80).

        The mirror image of that cost money too. 2026-08-20 06:02:31: the restored
        ledger said short 531 @ 0.1919 (an old session, a price level 12% away) while
        the account was genuinely flat, because this function only ever adopted the
        exchange from INSIDE its loop over open positions. A BUY 116 @ 0.2148 was then
        priced against that phantom short for (0.1919 - 0.2148) * 116 = -2.6564, and
        that number went into trades_demo.csv as a realised cycle that never happened.

        So a flat account is authority too -- but only once CORROBORATED by a second,
        independent read. One bad HTTP reply must not be enough to discard a real
        position's cost basis, which is the failure the AUDIT #80 tests above guard
        (AUDIT #134).
        """
        state, positions = self._read_positions()

        if state == self.POSITIONS_UNREADABLE:
            logger.warning(
                "POSITION SEED SKIPPED | positions unreadable -- ledger starts from "
                "saved state",
            )
            return

        if state == self.POSITIONS_HOLDS:
            for pos in positions:
                qty = float(pos.get("contracts") or 0)
                if qty <= 0:
                    continue
                signed = -qty if str(pos.get("side", "")).lower() == "short" else qty
                entry = float(pos.get("entryPrice") or 0)
                if entry <= 0:
                    continue
                if abs(signed - self._pos_qty) > 1e-9:
                    logger.info(
                        "POSITION SEEDED | ledger had {}, exchange holds {} @ {} -- "
                        "adopting the exchange's",
                        round(self._pos_qty, 4), round(signed, 4), entry,
                    )
                self.seed_position(signed, entry)
                return
            # Holds size, but nothing we can price. Adopting a size with no cost basis
            # books its close as pure profit; clearing denies a position that exists.
            # Leave the saved ledger alone and say so.
            logger.warning(
                "POSITION SEED INCOMPLETE | the account holds a position with no "
                "usable entry price -- keeping the saved ledger rather than guessing",
            )
            return

        # Flat. If the ledger already agrees there is nothing to corroborate and no
        # reason to spend a second API call.
        if abs(self._pos_qty) <= 1e-9:
            return

        second, _ = self._read_positions()
        if second != self.POSITIONS_FLAT:
            logger.warning(
                "POSITION SEED FLAT UNCONFIRMED | first read said flat, second said "
                "{} -- keeping the saved ledger of {} (AUDIT #134)",
                second, round(self._pos_qty, 4),
            )
            return

        logger.warning(
            "POSITION SEEDED FLAT | ledger had {} but two independent reads agree the "
            "account holds nothing -- clearing it. A fill priced against a phantom "
            "position reports profit the session never made (AUDIT #134)",
            round(self._pos_qty, 4),
        )
        self.seed_position(0.0, 0.0)

    # Polls of ledger-vs-exchange size disagreement tolerated before the mirror is
    # rebuilt outright. check_fills attributes a fill on the very next poll, so
    # anything past a couple of polls is a wrong ledger, not a race (AUDIT #88).
    MIRROR_MISMATCH_TOLERANCE_POLLS = 3

    def reconcile_position_entry(self) -> bool:
        """Re-adopt the exchange's average entry when our mirror has drifted. AUDIT #88.

        _seed_position_from_exchange runs once, in activate(). After that _pos_entry is
        a local mirror maintained only from fills THIS ladder placed and observed -- and
        the exchange moves the position without asking: the scale-out trailing leg, the
        hard stop, and any reduce-only close all change the blended average entry and
        none of them arrive through _handle_fill.

        Measured on 2026-08-16 22:21. The exchange held LONG 8516 @ 0.06952357 and the
        ladder sold 1796 at 0.06959, which is 0.0000664 above that entry -- gross
        +0.119. _apply_to_position reported +0.001293, and the income reconciler
        independently verified +0.09. So the mirror's entry had drifted roughly
        0.0000657 high and the engine booked 1% of a round trip it actually won.

        _position_break_even already treats this data the right way -- it re-reads the
        exchange every two seconds precisely because "internal per-level bookkeeping is
        exactly the thing that drifts" (AUDIT #7/#8). It decides whether an order may
        lose money. _apply_to_position decides what the money WAS, off the same
        quantity, and never re-read at all.

        ONLY the entry is adopted, and only while the quantities already agree. A
        quantity disagreement means a fill exists that check_fills has not attributed
        yet; the exchange's position already contains it, so adopting there would apply
        it twice -- once here and once when the fill is processed. That case is logged
        and left for the next poll, by which time the quantities agree.

        Returns True when an adjustment was made (for tests and for the caller's log).
        """
        try:
            positions = self.exchange.get_positions(self.symbol) or []
        except Exception:
            return False                        # unreadable: keep the mirror we have

        signed, entry = 0.0, 0.0
        for pos in positions:
            qty = float(pos.get("contracts") or 0)
            if qty <= 0:
                continue
            signed = -qty if str(pos.get("side", "")).lower() == "short" else qty
            entry = float(pos.get("entryPrice") or 0)
            break

        if signed == 0.0 or entry <= 0:
            return False                        # flat, or no usable entry to adopt

        if abs(signed - self._pos_qty) > max(1e-9, abs(signed) * 1e-6):
            # A mismatch has two causes and they need opposite handling. TRANSIENTLY it
            # means check_fills has not attributed a fill yet: the exchange's position
            # already contains it, so adopting here and then processing the fill applies
            # the same trade twice. PERSISTENTLY it means the mirror is simply wrong.
            #
            # The first draft of this treated both as the first case and deferred
            # unconditionally, which never converges -- and that is not hypothetical.
            # 2026-08-17 04:33, after a full session with the fix live:
            #
            #     ledger   LONG  1389 @ 0.06981
            #     exchange SHORT 5342 @ 0.07015
            #
            # Different SIDE, 6731 apart. The quantities were never going to agree, so
            # the resync never ran, and _apply_to_position kept pricing closes against a
            # long that did not exist: fills #31/#34/#35 booked +0.74/+0.48/+0.73 while
            # the income reconciler verified -0.02/-0.02/-0.03 on the same cycles.
            #
            # check_fills runs every poll, so an in-flight fill is attributed by the
            # next one. Three consecutive polls of disagreement is not a race, it is a
            # wrong ledger, and then the exchange's position is adopted whole.
            self._mirror_mismatch_polls += 1
            if self._mirror_mismatch_polls < self.MIRROR_MISMATCH_TOLERANCE_POLLS:
                logger.debug(
                    "POSITION MIRROR BEHIND | ledger {} vs exchange {} — poll {}/{}, "
                    "a fill may still be unattributed (AUDIT #88)",
                    round(self._pos_qty, 4), round(signed, 4),
                    self._mirror_mismatch_polls, self.MIRROR_MISMATCH_TOLERANCE_POLLS,
                )
                return False
            logger.warning(
                "POSITION MIRROR REBUILT | ledger held {} @ {} but the exchange holds "
                "{} @ {} after {} polls — adopting the exchange's. Realised P&L was "
                "being computed against a position that does not exist (AUDIT #88)",
                round(self._pos_qty, 4), round(self._pos_entry, 8),
                round(signed, 4), round(entry, 8), self._mirror_mismatch_polls,
            )
            self._pos_qty, self._pos_entry = signed, entry
            self._mirror_mismatch_polls = 0
            return True

        self._mirror_mismatch_polls = 0
        # Same quantity, different average: pure drift, and the exchange is right.
        if abs(entry - self._pos_entry) <= entry * 1e-9:
            return False
        drift = entry - self._pos_entry
        logger.info(
            "POSITION ENTRY RESYNCED | {} -> {} ({:+.8f}) on {} — realised P&L was "
            "being computed against a stale average (AUDIT #88)",
            round(self._pos_entry, 8), round(entry, 8), drift, round(signed, 4),
        )
        self._pos_entry = entry
        return True

    def seed_position(self, qty: float, entry: float) -> None:
        """Adopt a position the exchange already holds, so the ledger starts truthful.

        Without this a restart books the eventual close of a pre-existing position as
        pure profit, because the ledger believes it opened flat.
        """
        self._pos_qty = float(qty)
        self._pos_entry = float(entry) if qty else 0.0

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

    def one_side_notional(self, balance: float) -> float:
        """USDT one side of the ladder commits if every rung on it fills.

        The position cap lives in main.py (set_position_limit), not here, so the engine
        exposes the figure and the caller compares it (AUDIT #66).
        """
        return self._calc_usdt_per_grid(balance) * (self.grid_count / 2)

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

        # Offset each side by HALF a step so the price sits in the middle of one
        # spacing, not two.
        #
        # AUDIT #44. The old construction was
        #     buys  = linspace(lower, price, n, endpoint=False)   -> step (price-lower)/n
        #     sells = linspace(price, upper, m+1)[1:]             -> step (upper-price)/m
        # which puts the last buy one FULL step below the price and the first sell one
        # full step above it. The gap straddling the price was therefore always exactly
        # twice the spacing everywhere else -- the widest hole in the ladder, parked
        # permanently where the price actually is.
        #
        # Measured on the 2026-08-13 17:00 run: levels ... 0.06954 | 0.07006 ... a 0.748%
        # centre gap against 0.37-0.39% everywhere else. The price spent 77 minutes
        # inside it, ranging 0.06973-0.06994, and the bot recorded zero fills. It needed
        # a full spacing of movement to trade when it should have needed half.
        sell_count = self.grid_count - half_count
        if half_count < 1 or sell_count < 1:
            # A one-level "grid" has no two sides to balance. grid_count can shrink to
            # this after tick-rounding dedup, and dividing by half_count would raise.
            self._initialize_uniform(current_price)
            return
        buy_step = (current_price - self.grid_lower) / half_count
        sell_step = (self.grid_upper - current_price) / sell_count
        buy_prices = [current_price - buy_step * (i + 0.5) for i in range(half_count)][::-1]
        sell_prices = [current_price + sell_step * (i + 0.5) for i in range(sell_count)]

        raw_prices = [self._round_price(p) for p in buy_prices + sell_prices]

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

        CAPITAL_PER_GRID_USDT, when set, is AUTHORITATIVE: it means "commit this much of
        my own capital per trade", and the notional is that times leverage.

        It used to be `max(fixed, percent)`, which made the setting silently inert
        whenever the percent path happened to be larger -- and on a 4930 balance it
        always was: the fixed path asked for 25 and the percent path returned 88.74, so
        every order was 3.5x the configured size. The bot logged the override once and
        carried on, which reads as a note rather than "your setting is being ignored"
        (AUDIT #63).

        Percent-based sizing remains the default when CAPITAL_PER_GRID_USDT is 0.
        """
        pct_allocation = balance * self.capital_per_grid_pct * self._volatility_mult
        if self.capital_per_grid_usdt <= 0:
            return pct_allocation

        # The configured size is a CEILING, not a target. Volatility may shrink an order
        # but never grow it past what was asked for.
        #
        # Unbounded, the calm-market multiplier reaches 2.5 and turned a configured 50
        # USDT into 102.50 -- ten rungs of which is 1025 against a 587 position cap, so
        # the cap blocked the ladder in exactly the quiet markets the multiplier exists
        # to keep it working in. Shrinking in violent markets is kept: that reduces risk
        # (AUDIT #64).
        sizing_mult = min(1.0, self._volatility_mult)
        raw = self.capital_per_grid_usdt * self.leverage * sizing_mult
        if not self._warned_small_fixed_allocation:
            self._warned_small_fixed_allocation = True
            logger.info(
                "PER-GRID SIZING | CAPITAL_PER_GRID_USDT={:.2f} x {}x leverage = {:.2f} "
                "USDT notional per order (percent-based would have been {:.2f})",
                self.capital_per_grid_usdt, self.leverage, raw, pct_allocation,
            )

        # Total committed notional still cannot exceed the exposure ceiling.
        current_total = raw * self.grid_count
        target_total = balance * self.max_exposure_pct
        if current_total > target_total:
            raw = target_total / self.grid_count
            # Log the transition, not every call: this path is reachable from every
            # order placement AND from check_fills' per-poll orphan sweep, so an
            # unthrottled warning here repeated on every single 10s poll for as long
            # as the configured size stayed over the ceiling -- one line per level per
            # cycle, for the life of the mismatch, drowning out everything else in the
            # log. _warned_small_fixed_allocation (above) already gets this right for
            # its own sibling log; this one never did.
            if not self._exposure_ceiling_trimming:
                self._exposure_ceiling_trimming = True
                logger.warning(
                    "PER-GRID SIZING | {} rungs at the configured size would commit {:.2f} "
                    "USDT, over the {:.0%} exposure ceiling — trimming to {:.2f} per order "
                    "(further trims logged only if the trimmed size itself changes)",
                    self.grid_count, current_total, self.max_exposure_pct, raw,
                )
            elif abs(raw - self._last_exposure_trim) > 1e-9:
                logger.warning(
                    "PER-GRID SIZING | trim changed: {} rungs would commit {:.2f} USDT, "
                    "over the {:.0%} exposure ceiling — now trimming to {:.2f} per order "
                    "(was {:.2f})",
                    self.grid_count, current_total, self.max_exposure_pct, raw,
                    self._last_exposure_trim,
                )
            self._last_exposure_trim = raw
        elif self._exposure_ceiling_trimming:
            self._exposure_ceiling_trimming = False
            logger.info(
                "PER-GRID SIZING | back within the {:.0%} exposure ceiling — no longer trimming",
                self.max_exposure_pct,
            )
        return raw

    @property
    def round_trip_fee_pct(self) -> float:
        """Cost of one completed cycle as a fraction of price, at the BLENDED rate.

        Every gate here used to price the round trip at `2 * maker_fee`, on the reasoning
        that both legs rest as post-only maker orders. The Binance income ledger says
        otherwise: 11.8% of fill volume over 15 days paid the taker rate, making the real
        round trip 0.0447% against the 0.0400% assumed -- 1.12x. The gap is structural,
        not drift. Reduce-only exits are placed `postOnly=False` on purpose (a queued
        exit that never fills is worse than a crossed one), stop-losses always cross, and
        reconcile/unwind close at market.

        Understating it hurts twice: `MIN_PROFIT_MULTIPLIER=3.0` bought a real 2.68x
        margin rather than 3.0x, and -- worse -- every break-even price was 12% short, so
        exits clamped to "break-even" booked a small genuine loss (AUDIT #51).
        """
        per_side = (self.maker_fee_pct * (1.0 - self.taker_fill_share)
                    + self.taker_fee_pct * self.taker_fill_share)
        return 2.0 * per_side

    def _is_level_profitable(self, level_price: float) -> bool:
        """Is one round trip at this level worth more than the fees it will pay?

        A completed cycle earns one grid_spacing of price movement and pays two fees
        (entry + exit), priced at the blended rate -- see round_trip_fee_pct.
        """
        expected_profit = self.grid_spacing
        round_trip_fees = self.round_trip_fee_pct * level_price
        return expected_profit > round_trip_fees * self._min_profit_multiplier

    def _position_break_even(self) -> tuple[str, float] | None:
        """Return (side, break_even_price) for the open position, or None when flat.

        The exchange nets everything into one position at one blended average entry,
        so *any* sell below that average realises a loss on a long, no matter which
        grid level the engine has paired it with internally. Break-even here means the
        average entry plus the round trip's fees.

        Read from the exchange, cached briefly: internal per-level bookkeeping is
        exactly the thing that drifts (AUDIT #7/#8), and this decides whether an order
        is allowed to lose money.
        """
        now = time.time()
        if (now - self._break_even_time) < 2.0:
            return self._break_even_cache
        try:
            positions = self.exchange.get_positions(self.symbol)
        except Exception:
            # Unknown position: do not claim break-even. The caller treats None as
            # "no constraint", which is the pre-existing behaviour, not a new risk.
            return self._break_even_cache

        result = None
        for pos in positions:
            side = pos.get("side", "")
            qty = float(pos.get("contracts", 0) or 0)
            entry = float(pos.get("entryPrice", 0) or 0)
            if qty == 0 or entry <= 0:
                continue
            if qty < 0:
                side = "short" if side == "long" else "long"
                qty = abs(qty)
            fees = self.round_trip_fee_pct
            if side == "long":
                result = ("long", entry * (1 + fees))
            elif side == "short":
                result = ("short", entry * (1 - fees))
            break

        self._break_even_cache = result
        self._break_even_time = now
        return result

    def _would_realise_a_loss(self, side: str, price: float) -> bool:
        """Would an order on `side` at `price` close part of the open position at a loss?

        This is the guard AUDIT #32 added. On 2026-08-12 a downward recenter rebuilt the
        ladder below a long held at 0.06962 and then sold into it at 0.06928 and 0.06954
        -- reduce-only orders the bot placed itself, for -0.88 realised, over half that
        session's entire loss. The position was not in trouble: the stop was far below
        and price recovered within minutes.

        A grid is allowed to sit on inventory and wait; that is what its levels are for.
        What it must not do is voluntarily book a loss to keep the ladder tidy. If price
        never comes back, the hard stop -- not an exit ladder priced below cost -- is
        what closes the position.
        """
        be = self._position_break_even()
        if be is None:
            return False
        pos_side, break_even = be
        if pos_side == "long" and side == "sell":
            return price < break_even
        if pos_side == "short" and side == "buy":
            return price > break_even
        return False

    def _flip_captures_a_spread(self, price: float, side: str) -> bool:
        """Would flipping a held rung onto its OWN line close inventory for real profit?

        A grid earns one spacing per cycle. It earns that because the counter sits a
        spacing away from the fill. When a held rung instead comes back FLIPPED on its
        own line, the two legs are the same price and the cycle captures whatever the
        average entry has drifted -- typically a fraction of a rung -- while still paying
        a full round trip of fees.

        Measured on ADAUSDT 2026-08-19 04:52-08:07. Four flips, all onto the line the
        rung had just filled at:

            05:03:58  FILL #1 | SELL @ 0.1743
            05:10:17  RUNG FLIPPED | SELL 0.1743 -> BUY
            05:10:18  ORDER PLACED | BUY 717.0 ADAUSDT @ 0.1743

        Sold at 0.1743, bought back at 0.1743. Across the session: 8 fills, 7 with
        profit=0.000000, gross 0.05 against fees 0.20, net -0.15. The one cycle that
        earned anything made 0.053 -- from the drift between the average short entry
        (0.174874) and the close (0.1748), not from a rung -- against 0.05 of round-trip
        fees. Net 0.003 where the 0.2507% spacing should have paid 0.263.

        _books_a_loss is not enough of a gate: it only asks whether the leg loses money,
        and a leg that beats break-even by a hundredth of a rung passes it while still
        handing the exchange more in fees than it takes in spread. This asks for the fee
        floor -- the same bar every other cycle in the ladder has to clear (AUDIT #119).

        Flat means there is no inventory to close and nothing to undercut, so the flip is
        just a re-siting and is allowed. A flip that ADDS exposure rather than closing is
        the position cap's business, not this gate's.
        """
        be = self._position_break_even()
        if be is None:
            return True
        pos_side, break_even = be
        if break_even <= 0:
            return True
        floor = self.round_trip_fee_pct * self._min_profit_multiplier
        if pos_side == "short" and side == "buy":
            return (break_even - price) / break_even >= floor
        if pos_side == "long" and side == "sell":
            return (price - break_even) / break_even >= floor
        return True

    def _stuck_exit_ceiling(self, side: str) -> float | None:
        """Worst price an exit may take while the ladder is stuck, or None. AUDIT #122.

        AUDIT #32's rule -- never book a loss, the levels will unwind the inventory --
        rests on those levels being able to TRADE while they wait. Waiting is free only
        while the other side of the ladder still earns. Once the position has eaten the
        position cap that side is blocked, no rung on it can be placed, and waiting earns
        nothing: it is a directional bet with the income switched off.

        ADAUSDT 2026-08-19, the sequence in full:

            14:56:27  POSITION LIMIT | short 6307.0 >= 5608.77 -- sell orders blocked
            15:06:28  KILL SWITCH: price 0.1776 above stop loss 0.17753
            16:30:22  SKIP BUY @ 0.1752 | below break-even 0.17475982
            16:31:42  POSITION LIMIT | short 6307.0 >= 5433.6 -- sell orders blocked
            19:44:58  Cancelled 0 open orders   (empty book for 3h13m)

        At 14:56 price was 0.176 -- 0.59% past break-even. Refusing that loss is what
        produced a 3.3% one by 19:44, plus three hours in which no rung could trade at
        all. The hard stop is not a substitute: it fired at 15:06 and the position was
        still open at shutdown.

        So: while the OPPOSITE side is cap-blocked, an exit may price up to
        rung_loss_cap_pct past break-even. Outside that state nothing changes, and
        rung_loss_cap_pct = 0 restores the old behaviour exactly.

        A buy exits a short and the SELL side is what grows that short, so the buy side
        is stuck exactly when sells are capped. The long case is the mirror.
        """
        if self._rung_loss_cap_pct <= 0:
            return None
        be = self._position_break_even()
        if be is None:
            return None
        pos_side, break_even = be
        if break_even <= 0:
            return None
        if pos_side == "short" and side == "buy" and self._block_sells:
            return break_even * (1.0 + self._rung_loss_cap_pct)
        if pos_side == "long" and side == "sell" and self._block_buys:
            return break_even * (1.0 - self._rung_loss_cap_pct)
        return None

    def _nearest_legal_exit(self, level: "GridLevel") -> float | None:
        """The closest price this level can sit at without booking a loss, or None.

        AUDIT #42. The #32 guard was right about the economics and catastrophic about
        the consequence: it returned False and left the level DEAD. Nothing re-sited it,
        nothing replaced it, so the ladder kept a permanent hole exactly where trading
        happens -- next to the price.

        On 2026-08-13 that hole was the whole strategy. A short at 0.07024719 left the
        buy level at 0.07034 permanently blocked; it was the only level within 1.2% of
        the price, so from 10:46 to 15:29 -- four hours and forty-three minutes -- the
        bot logged the same skip every fifteen seconds and did not trade once. Seven
        fills in seven hours, six of them inside one 90-second burst.

        Waiting out inventory does not require refusing to quote. It requires quoting at
        a price that does not lose: break-even. So the level moves there instead of
        dying, provided the move stays inside the grid and does not crowd a neighbour
        past the fee floor.
        """
        be = self._position_break_even()
        if be is None:
            return None
        pos_side, break_even = be

        # Break-even alone is not enough: the level also has to REST. A buy above the
        # market crosses, post-only rejects it, and _place_order_for_level retries it
        # forever at debug level -- the same silent dormancy in a different disguise.
        # So take the stricter of "does not lose" and "is a valid maker price".
        try:
            price_now = float(self.exchange.get_price(self.symbol))
        except Exception:
            price_now = None

        # Round AWAY from the loss: a short's cover must land at or below break-even,
        # a long's exit at or above it. Rounding to nearest crosses the line (#41).
        if pos_side == "short" and level.side == "buy":
            limit = break_even if price_now is None else min(break_even, price_now)
            target = self._round_price_toward(limit, -1)
        elif pos_side == "long" and level.side == "sell":
            limit = break_even if price_now is None else max(break_even, price_now)
            target = self._round_price_toward(limit, +1)
        else:
            return None

        if target <= 0 or not (self.grid_lower <= target <= self.grid_upper):
            return None

        # Pending levels count. place_initial_orders walks the whole ladder every
        # iteration, so a pending neighbour is not an empty price -- it is an order about
        # to exist, and moving on top of it is #34's deformation either way.
        #
        # This also bounds the dormancy risk that refusing creates: a neighbour inside
        # the fee floor is a neighbour within ~0.13% of the target, and it quotes. The
        # #42 incident was the opposite case -- the blocked level was the ONLY one within
        # 1.2% of the price, so refusing meant the grid quoted nothing at all.
        floor = target * self.round_trip_fee_pct * self._min_profit_multiplier
        for other in self.levels:
            if other is level:
                continue
            if abs(other.price - target) < floor:
                return None                     # would deform the ladder (#34)
        return target

    def accelerate_handoff_exit(self, current_price: float, balance: float) -> bool:
        """During a router handoff, bring the nearest exit-side rung to a price it can
        actually trade at instead of leaving it wherever ordinary grid spacing put it.

        router._continue_handoff lets the outgoing strategy keep unwinding through its
        own levels for up to handoff_grace_seconds before forcing a market close --
        AUDIT #29 measured that force-close at -46.16 across 18 handoffs, so waiting is
        the right default. But nothing ever repriced the level it is waiting on: on
        2026-08-24 a long's nearest exit sat at 0.2195 while price held 0.2181-0.2183
        for the whole 620s of the run, 0.6% away and going nowhere, with the grace
        clock racing toward the same forced dump this design exists to avoid.

        Only the EXIT side is ever touched here -- the side that reduces the position
        -- and _nearest_legal_exit never returns worse than the position's own
        break-even, so this can only bring a still-profitable exit closer to market,
        never manufacture the loss the force-close would take anyway (AUDIT #145).

        The replacement is placed through _place_exit_reprice, NOT
        _place_order_for_level: that path is the generic ladder-opening placement --
        it recomputes quantity from capital sizing and never sets reduceOnly, because
        an ordinary rung might be opening exposure, not closing it. This call is
        ALWAYS closing a specific, already-known slice of the position (the rung's
        own quantity), and the first version of this used the generic path anyway --
        on a 10-unit position with a correctly-sized 10-unit exit rung it placed a
        naked, non-reduceOnly SELL for 114.6 units, sized purely from capital and
        totally unrelated to the position, with nothing stopping it from flipping the
        position short if it filled (AUDIT #149).
        """
        if abs(self._pos_qty) <= 1e-9:
            return False                          # already flat -- the handoff completes on its own
        now = time.time()
        if (now - self._last_handoff_accel_time) < self.HANDOFF_ACCEL_COOLDOWN_SECONDS:
            return False
        exit_side = "sell" if self._pos_qty > 0 else "buy"
        resting = [l for l in self.levels if l.side == exit_side and l.order_id is not None]
        if not resting:
            return False
        nearest = min(resting, key=lambda l: abs(l.price - current_price))
        target = self._nearest_legal_exit(nearest)
        if target is None:
            return False
        # A real improvement, not just "legal" -- _nearest_legal_exit also fires when
        # the level is already fine and simply far away, which is exactly this case.
        improves = ((exit_side == "sell" and target < nearest.price)
                    or (exit_side == "buy" and target > nearest.price))
        if not improves:
            return False

        # Claim the cooldown before any I/O: a failed cancel below must not retry
        # every single poll (the same discipline _cancel_resting_orders follows).
        self._last_handoff_accel_time = now
        try:
            still_open = self.exchange.get_open_order_ids(self.symbol)
        except Exception as e:
            logger.error("HANDOFF ACCEL | could not fetch open orders: {}", e)
            return False
        if nearest.order_id in still_open:
            try:
                confirmed = self.exchange.cancel_order(nearest.order_id, self.symbol)
            except Exception as e:
                logger.error(
                    "HANDOFF ACCEL | failed to cancel {} @ {}: {}",
                    nearest.side, nearest.price, e,
                )
                return False
            if not confirmed:
                return False                      # order_id kept; retried after the cooldown
        else:
            # Missing from the open-orders snapshot is ambiguous: filled, or
            # cancelled by something else. check_fills disambiguates via fetch_order
            # before deciding which; treating both the same here silently dropped a
            # genuine fill's P&L/journal/position bookkeeping and reclaimed the
            # level as if nothing had traded (AUDIT #150).
            order = self.exchange.fetch_order(nearest.order_id, self.symbol)
            if order is not None and order_was_filled(order):
                logger.warning(
                    "HANDOFF ACCEL | {} @ {} filled instead of waiting to be "
                    "repriced -- processing as a fill, not a cancel (AUDIT #150)",
                    nearest.side, nearest.price,
                )
                self._handle_fill(nearest, balance)
                return True

        old_price, old_id = nearest.price, nearest.order_id
        if self._event_journal:
            self._event_journal.order_cancelled(self.symbol, exit_side, old_price, old_id, "handoff_accel")
        if self._notifier:
            self._notifier.on_order_cancelled(self.symbol, exit_side, old_price, old_id, "handoff_accel")
        nearest.order_id = None
        nearest.status = "pending"
        nearest.price = target
        logger.warning(
            "HANDOFF EXIT ACCELERATED | {} {} -> {} | {:.2f}% from market and idle -- "
            "repricing toward break-even so the handoff can close through it instead "
            "of waiting on the grace deadline (AUDIT #145)",
            exit_side.upper(), old_price, target,
            abs(old_price - current_price) / current_price * 100,
        )
        return self._place_exit_reprice(nearest, exit_side, balance)

    def _place_exit_reprice(self, level: "GridLevel", side: str, balance: float) -> bool:
        """Place the repriced exit accelerate_handoff_exit just cleared -- reduceOnly
        and clamped to the real closable quantity, never the generic ladder-opening
        path (AUDIT #149).

        Refreshes the reduce-only mirror right before using it, the same discipline
        _handle_fill's own replacement branch follows (AUDIT #98): the position may
        have moved since this level was last sized, and a stale mirror here is
        exactly how #98 sized a replacement against a position that no longer
        existed.
        """
        self._refresh_net_counters()
        params, adj_qty = self._exit_order_params(side, level.quantity)
        if params is None or adj_qty <= 0:
            logger.error(
                "HANDOFF ACCEL | {} has no closable position on the exchange right "
                "now despite pos_qty={} -- refusing to place an unprotected order",
                side.upper(), self._pos_qty,
            )
            return False
        quantity = self.exchange.exchange.amount_to_precision(self.symbol, adj_qty)
        if float(quantity) <= 0 or float(quantity) * level.price < MIN_NOTIONAL_USDT:
            return False
        try:
            order_params = dict(params)
            order_params["purpose"] = "handoff_accel"
            order = self.exchange.place_limit_order(
                self.symbol, side, level.price, float(quantity), max_attempts=1,
                params=order_params,
            )
            if "id" not in order:
                raise ValueError("Order response missing 'id'")
        except Exception as e:
            logger.error("HANDOFF ACCEL | failed to place repriced exit @ {}: {}", level.price, e)
            return False
        level.order_id = order["id"]
        level.status = "pending"
        level.quantity = float(quantity)
        if self._event_journal:
            self._event_journal.order_placed(self.symbol, side, level.price, float(quantity), order["id"])
        if self._notifier:
            self._notifier.on_order_placed(self.symbol, side, level.price, float(quantity), order["id"])
        return True

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

    def _current_price_or_none(self) -> float | None:
        """Live price for the release gate, or None if it cannot be read.

        None degrades to counter-slot-only release rather than guessing: re-arming a
        rung on a fabricated price is how a rung refills at the price it just filled at.
        """
        try:
            price = float(self.exchange.get_price(self.symbol))
        except Exception as e:
            logger.debug("RELEASE GATE | price unavailable ({}) — counter-slot only", e)
            return None
        return price if price > 0 else None

    def _price_has_cleared(self, level: GridLevel, price: float) -> bool:
        """Has price moved a full spacing past this rung, so it can rest again?

        This is the gate that separates the two failure modes seen live. #58 let a
        filled rung re-arm on its own side immediately: with price sitting ON the rung
        it refilled six times and took 8215 DOGE, 96% of the cap, at one price. #61 then
        refused to re-arm at all until the counter-slot freed -- and since every counter
        is another rung of the same ladder, held rungs waited on held rungs and the book
        drained from 10 orders to 5 in one session.

        A rung may come back, but only once price has genuinely left it (AUDIT #62).
        """
        gap = abs(self.grid_spacing)
        if gap <= 0:
            return False
        # A full spacing EITHER WAY frees the rung.
        #
        # This used to ask a filled BUY to see price rise a spacing and a filled SELL to
        # see price FALL one -- so in a one-way market the side the trend is eating can
        # never come back. Sells fill as price climbs, and the counter they wait on sits
        # below price and never frees either. Measured 2026-08-15 00:00-03:23: DOGE
        # ground +1.1%, three sells filled, none re-armed, the book went 8 orders -> 6
        # and the ladder had nothing left near price to trade (AUDIT #77).
        #
        # Widening this is only safe because the release path now re-sides the rung to
        # match where price actually is -- see _release_awaiting_levels. Re-arming a sell
        # whose price has fallen below the market would post a crossing order.
        return abs(price - level.price) >= gap

    def _ladder_holes(self) -> list[float]:
        """Grid lines the ladder should have but no level occupies any more.

        Every fill MIGRATES its level to the counter line (_handle_fill snaps to the
        nearest free line of the opposite side), so the set of lines the ladder covers
        is not conserved. Two levels can converge on one line while the line they came
        from is left with no owner, and nothing ever puts an order back on it.

        Measured 2026-08-15 09:40-14:53. Rung 0.07001 was vacated when a filled sell
        migrated down to 0.06987 -- a line a held rung was already parked on, invisible
        to _handle_fill's occupancy test because that only looks for queued
        replacements, not levels in 'awaiting_counter'. The held rung was then stranded
        (own line taken, counter line taken) and 0.07001 stayed empty for 5h13m while
        price crossed it 62 times (AUDIT #79).
        """
        gap = abs(self.grid_spacing)
        if gap <= 0:
            return []
        occupied = sorted({l.price for l in self.levels})

        # The ladder is NOT a uniform lattice. _initialize_dynamic concentrates rungs
        # near price, so grid_spacing is an AVERAGE and neighbouring lines legitimately
        # sit anywhere from 1.0 to ~1.7 of it apart. This used to demand a clean
        # multiple of the average, which rejected a real 1.733-spacing hole as "ragged"
        # and left the ladder a line short for an entire run (AUDIT #82).
        #
        # A hole is therefore just a gap wide enough to seat another rung, and the rung
        # goes in the middle of it. No lattice is assumed. The upper bound keeps rungs
        # from being invented inside a stretch the ladder legitimately spans.
        holes: list[float] = []
        for a, b in zip(occupied, occupied[1:]):
            width = (b - a) / gap
            if 1.5 <= width <= 3.0:
                holes.append(self._round_price((a + b) / 2))
        return holes

    def _repair_ladder(self, current_price: float | None) -> None:
        """Move a doubled-up rung onto a line the ladder has lost.

        _release_awaiting_levels can only rehome a rung that is HELD, so a hole is
        repaired only when a stranded 'awaiting_counter' rung happens to be sitting
        beside it. On a saw-tooth that fired exactly ONCE in a whole run: the ladder
        dropped a line and stayed down, and it is always the line nearest price that
        goes, because that is where the fills are (AUDIT #82).

        Fills migrate levels between lines, so a lost line always has a doubled-up line
        somewhere to pay for it. Only a rung with no live order is moved -- one resting
        on the exchange is left alone, since relocating it would need a cancel and this
        runs every poll.
        """
        if current_price is None:
            return
        holes = self._ladder_holes()
        if not holes:
            return

        occupants: dict[float, list[GridLevel]] = {}
        for lvl in self.levels:
            occupants.setdefault(lvl.price, []).append(lvl)
        claimed = {(l.price, l.side) for l in self.levels if l.order_id is not None}

        for hole in sorted(holes, key=lambda h: abs(h - current_price)):
            side = "buy" if hole < current_price else "sell"
            if (hole, side) in claimed:
                continue
            spare = next(
                (lvl for group in occupants.values() if len(group) > 1
                 for lvl in group
                 if lvl.order_id is None
                 and lvl.status in ("pending", "awaiting_counter")),
                None,
            )
            if spare is None:
                return                      # nothing free to pay for the hole
            occupants[spare.price].remove(spare)
            logger.info(
                "LADDER REPAIRED | {} {} -> {} {} — two rungs shared that line while "
                "this one had none",
                spare.side.upper(), spare.price, side.upper(), hole,
            )
            spare.price = hole
            spare.side = side
            spare.status = "pending"
            spare.order_id = None
            spare.awaiting_side = None
            spare.awaiting_price = None
            occupants.setdefault(hole, []).append(spare)
            claimed.add((hole, side))

    def _release_awaiting_levels(self, current_price: float | None = None) -> None:
        """Bring held rungs back, either as the counter leg or at their own rung.

        Three ways out of the hold:
          1. The counter-slot frees -- the rung becomes the replacement it was meant
             to be, which is what #61 intended.
          2. Price moves a full spacing away from the rung itself -- the classic grid
             re-arm. Without this the ladder starves, because in a ladder of buys below
             and sells above, EVERY fill's counter-target is another live rung.
          3. Both are occupied but the ladder has a vacant line -- take it. Fills
             migrate levels between lines, so two can pile onto one line and leave
             another with no owner; without this the hole is permanent (#79).

        Slots claimed earlier in this pass are tracked, or two rungs release onto the
        same price: a just-released level has order_id None, so an occupancy check that
        only looks at live orders sees the slot as free twice. Observed live at
        16:20:51 -- BUY 0.06922 and BUY 0.06939 both released to SELL 0.06955.
        """
        claimed = {
            (l.price, l.side) for l in self.levels
            if l.order_id is not None and l.status in ("pending", "replaced")
        }

        for level in self.levels:
            if level.status != "awaiting_counter" or level.awaiting_price is None:
                continue

            counter = (level.awaiting_price, level.awaiting_side or level.side)
            if counter not in claimed:
                logger.info(
                    "COUNTER SLOT FREED | {} {} -> {} {} — re-arming the held rung",
                    level.side.upper(), level.price, counter[1].upper(), counter[0],
                )
                claimed.add(counter)
                level.side = counter[1]
                level.price = counter[0]
                level.status = "pending"
                level.order_id = None
                level.awaiting_side = None
                level.awaiting_price = None
                continue

            # A rung's side is decided by where price is, not by what it was last time.
            # Below the market a grid line is a bid; above it, an offer. Re-arming a
            # stale SELL that price has since climbed past would post a crossing order,
            # which is why the clearance test used to be one-directional (AUDIT #77).
            #
            # A FLIP onto this line additionally has to capture a real spread. The flip
            # exists so a rung price has climbed past can still rest (#77), but on its
            # own line the two legs of the cycle are the same price: it closes inventory
            # for whatever the average entry has drifted and pays a full round trip of
            # fees to do it. _flip_captures_a_spread holds it to the fee floor, the same
            # bar every other cycle clears. Blocked flips fall through to the hole path
            # below, and failing that stay held -- an idle rung costs nothing, a
            # zero-spread rung costs a fee every time (AUDIT #119).
            own_side = "buy" if current_price is not None and level.price < current_price else "sell"
            own = (level.price, own_side)
            flipping = own_side != level.side
            if (current_price is not None
                    and own not in claimed
                    and self._price_has_cleared(level, current_price)
                    and (not flipping
                         or self._flip_captures_a_spread(level.price, own_side))):
                if flipping:
                    logger.info(
                        "RUNG FLIPPED | {} {} -> {} — price {} is now on the other side "
                        "of this line, so it comes back as the side that can rest there",
                        level.side.upper(), level.price, own_side.upper(), current_price,
                    )
                else:
                    logger.info(
                        "RUNG RE-ARMED | {} {} — price {} has moved a full spacing clear "
                        "while its counter {} stays busy",
                        level.side.upper(), level.price, current_price, counter[0],
                    )
                claimed.add(own)
                level.side = own_side
                level.status = "pending"
                level.order_id = None
                level.awaiting_side = None
                level.awaiting_price = None
                continue

            # 3. Neither line is available, but the ladder is missing one somewhere --
            #    take the missing line nearest price. Without this the rung is stranded
            #    for good: its counter is occupied, its own line was taken by a level
            #    that migrated onto it after a fill, and the line that level vacated has
            #    no owner left to re-arm it. That is how the book ends up with a hole in
            #    exactly the place price is trading (AUDIT #79).
            if current_price is None:
                continue
            holes = [
                h for h in self._ladder_holes()
                if (h, "buy" if h < current_price else "sell") not in claimed
            ]
            if not holes:
                continue
            target = min(holes, key=lambda h: abs(h - current_price))
            target_side = "buy" if target < current_price else "sell"
            logger.info(
                "LADDER HOLE FILLED | {} {} -> {} {} — no level was left holding that "
                "line, and price {} is trading across it",
                level.side.upper(), level.price, target_side.upper(), target, current_price,
            )
            claimed.add((target, target_side))
            level.price = target
            level.side = target_side
            level.status = "pending"
            level.order_id = None
            level.awaiting_side = None
            level.awaiting_price = None

    def _place_order_for_level(self, level: GridLevel, balance: float) -> bool:
        if level.status == "awaiting_counter":
            # Its exit is already resting. Placing here would add exposure the rung has
            # no matching exit for -- the 8215 DOGE accumulation (AUDIT #61).
            return False
        if level.side == "buy" and self._block_buys:
            logger.debug("SKIP BUY ORDER | position limit reached")
            return False
        if level.side == "sell" and self._block_sells:
            logger.debug("SKIP SELL ORDER | short position limit reached")
            return False
        if level.side == "buy" and self._loss_block_buys:
            logger.debug("SKIP BUY ORDER | open loss budget exhausted")
            return False
        if level.side == "sell" and self._loss_block_sells:
            logger.debug("SKIP SELL ORDER | open loss budget exhausted")
            return False
        if self._profit_lock_active:
            # Mirror apply_profit_lock_guard's own trip-time cancellation (fixed
            # earlier this session for the 2026-08-31 incident): in one-way netted
            # mode a side is only ever an entry OR the current position's own exit,
            # never both, and which one flips with which way the position is held.
            # Blocking BOTH sides unconditionally here -- even the side that only
            # ever REDUCES the current position -- starves that exit right when a
            # recenter also wipes the whole ladder to rebuild it: every level in the
            # fresh grid, including the ones that would have been the position's
            # only remaining exit, reaches this gate as a "new" order and gets
            # refused right alongside genuine new entries.
            #
            # 2026-09-01 incident: profit lock tripped at 16:39, then a recenter at
            # 18:29 rebuilt the ladder around a 49 ADA long with this gate still
            # blocking both sides -- all 12 fresh levels failed to place (0 placed),
            # leaving the position with no working exit at all. DORMANT WITH
            # EXPOSURE fired 15 minutes later; the raw exchange stop-loss eventually
            # closed the position with nothing on the ladder watching, so the bot
            # could only book an ESTIMATED P&L (-0.13) that turned out to differ
            # from the exchange-verified figure (-0.16) once reconciliation caught
            # up -- exactly the "P&L not adding up" symptom this fix closes.
            if self._pos_qty > 0:
                adding_side = "buy"
            elif self._pos_qty < 0:
                adding_side = "sell"
            else:
                adding_side = None  # flat: both sides would open new exposure
            if adding_side is None or level.side == adding_side:
                logger.debug("SKIP {} ORDER | daily profit lock active", level.side.upper())
                return False
        if self._would_realise_a_loss(level.side, level.price):
            be = self._position_break_even()
            ceiling = self._stuck_exit_ceiling(level.side)
            if ceiling is not None and (
                (level.side == "buy" and level.price <= ceiling)
                or (level.side == "sell" and level.price >= ceiling)
            ):
                logger.warning(
                    "STUCK LADDER EXIT | {} @ {} loses against break-even {}, but the "
                    "other side is cap-blocked so no rung can trade. Taking the small "
                    "loss inside the {:.2%} cap rather than holding for the hard stop "
                    "(AUDIT #122)",
                    level.side.upper(), level.price,
                    round(be[1], 8) if be else None, self._rung_loss_cap_pct,
                )
                self._be_block_logged.discard((level.side, level.price))
            else:
                moved = self._nearest_legal_exit(level)
                if moved is not None and moved != level.price:
                    logger.info(
                        "MOVED {} {} -> {} | the open {} makes the original price a loss; "
                        "quoting at break-even instead of leaving the level dead (AUDIT #42)",
                        level.side.upper(), level.price, moved, be[0] if be else None,
                    )
                    self._be_block_logged.discard((level.side, level.price))
                    level.price = moved
                else:
                    # Genuinely nowhere legal to sit. Say so ONCE -- this used to repeat
                    # every poll: ~1,300 identical lines in one session (AUDIT #42).
                    key = (level.side, level.price)
                    if key not in self._be_block_logged:
                        self._be_block_logged.add(key)
                        logger.info(
                            "SKIP {} @ {} | below break-even {} on the open {} and nowhere "
                            "legal to move it — level idle until the position resolves "
                            "(AUDIT #32/#42)",
                            level.side.upper(), level.price,
                            round(be[1], 8) if be else None, be[0] if be else None,
                        )
                    return False
        self._be_block_logged.discard((level.side, level.price))
        if not self._is_level_profitable(level.price):
            logger.info(
                "SKIP ORDER @ {} | spacing={:.8f} < min_profit={:.8f} ({}x fees)",
                level.price, self.grid_spacing,
                self.round_trip_fee_pct * level.price * self._min_profit_multiplier,
                self._min_profit_multiplier,
            )
            return False
        existing = self._existing_open_order(level.price, level.side)
        if existing and existing.get("id"):
            if any(l is not level and l.order_id == existing["id"] for l in self.levels):
                # This level can never place while it sits here, so it is dead weight in
                # the ladder -- one fewer rung earning. It was logged at DEBUG, invisible
                # at the bot's own level, which is how a grid ran 57 minutes on 9 of 10
                # orders without a word (AUDIT #58). Throttled per slot: the same #42
                # mistake of 1,300 identical lines is not worth repeating.
                key = ("dup", level.side, level.price)
                if key not in self._be_block_logged:
                    self._be_block_logged.add(key)
                    logger.warning(
                        "LEVEL STRANDED | {} @ {} duplicates order {} already tracked by "
                        "another level — this rung cannot place and is idle",
                        level.side.upper(), level.price, existing["id"],
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
        try:
            # can_place_order chains through get_open_order_count -> get_open_orders ->
            # a real network call that can raise after retries exhaust; amount_to_precision
            # is a ccxt call too. Both used to sit outside any try here, so a transient
            # failure on one level aborted place_initial_orders' whole loop over
            # self.levels partway through instead of just failing this one rung the way
            # every other per-level failure in this same loop already does (AUDIT #164).
            if not self.exchange.can_place_order(self.symbol):
                return False
            usdt_per_grid = self._calc_usdt_per_grid(balance)
            quantity = usdt_per_grid / level.price
            if level.side == "buy":
                quantity *= self._buy_scale
            else:
                quantity *= self._sell_scale
            quantity = self.exchange.exchange.amount_to_precision(self.symbol, quantity)
            # A level reaches this generic, freshly-capital-sized path whether it is a
            # brand-new rung or one _release_awaiting_levels just handed back after its
            # counter-slot freed, its rung re-armed, or a ladder hole was patched. In
            # any of those cases level.side can be this position's own exit -- but
            # nothing here asked, so it went out sized off fresh capital instead of the
            # position it was actually closing, and without reduceOnly. _handle_fill's
            # replacement (the ordinary, every-fill path) never places without asking
            # _exit_order_params this same question first; this path never asked it,
            # so a released rung could silently flip the position's direction and open
            # brand-new exposure that no position-limit/loss/profit-lock gate above
            # ever evaluated as an entry, because it never looked like one.
            exit_params, quantity = self._exit_order_params(level.side, float(quantity))
            quantity = self.exchange.exchange.amount_to_precision(self.symbol, quantity)
        except Exception as e:
            logger.error("PLACE ORDER PRECHECK FAILED @ {} {} | {}", level.side.upper(), level.price, e)
            return False
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
            # Same tag convention as _handle_fill's replacement: reduceOnly when this
            # order actually closes known inventory, grid_entry when it doesn't.
            order_params = dict(exit_params or {})
            order_params["purpose"] = "grid_exit" if order_params.get("reduceOnly") else "grid_entry"
            order = self.exchange.place_limit_order(
                self.symbol, level.side, level.price, float(quantity), max_attempts=1,
                params=order_params,
            )
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
        except PostOnlyWouldCross:
            # Not a failure: the level is momentarily on the wrong side of the book.
            # Leave it pending and unplaced so the next pass retries it. No error log,
            # no journal entry, no Telegram alert -- and crucially no taker fill.
            logger.debug(
                "SKIP CROSSING LEVEL | {} @ {} would cross — leaving unplaced for retry",
                level.side, level.price,
            )
            return False
        except Exception as e:
            logger.error("Failed to place order at {}: {}", level.price, e)
            if self._event_journal:
                self._event_journal.order_failed(self.symbol, level.side, level.price, float(quantity), str(e))
            if self._notifier:
                self._notifier.on_order_failed(self.symbol, level.side, level.price, float(quantity), str(e))
            return False
        finally:
            self._open_orders_fetch_time = 0.0

    def _wrong_side_of(self, level: GridLevel, current_price: float | None) -> bool:
        """Can this rung's order not be posted at all, as tagged?

        post-only rejects anything that would cross: a sell under the bid, a buy over the
        ask, both with -2019. Such a rung is not a failure to be retried, it is an order
        the exchange cannot accept while price is where it is.
        """
        if current_price is None or current_price <= 0:
            return False
        return ((level.side == "sell" and level.price < current_price)
                or (level.side == "buy" and level.price > current_price))

    def _reside_safe_levels(self, current_price: float | None) -> int:
        """Re-side idle rungs stranded on the wrong side of spot. Returns how many.

        AUDIT #115 re-sided levels in reset_levels_to_pending, which main.py calls ONLY
        when the exchange reports no position. With a position open reconcile_state runs
        instead, and that reconciles order ids and re-places orphans without ever
        re-deriving level.side -- so the same defect survived on the holding path.

        Observed 2026-08-19 00:19: the restored ladder carried 0.06994 and 0.0703 tagged
        "sell" with spot at 0.07004, and startup reported "PLACED 2 initial grid orders
        (1 failed)". The same line appears on every start whose spot sits above a rung the
        file calls a sell.

        Flipping while holding is only safe in the direction that does NOT add exposure.
        Short: a flip TO buy reduces, so allow it; a flip TO sell would add, so refuse.
        Long: the mirror. That asymmetry is the whole reason this could not simply reuse
        the flat-path rule -- there, with no inventory, no flip can add anything.

        Rungs in awaiting_counter are never touched: they hold inventory whose exit is
        already queued, and re-siding one would strand that exit (AUDIT #118).
        """
        if current_price is None or current_price <= 0:
            return 0
        resided = 0
        for level in self.levels:
            if level.order_id is not None or level.status != "pending":
                continue
            want = "buy" if level.price < current_price else "sell"
            if level.side == want:
                continue
            if self._pos_qty < 0 and want == "sell":
                continue                      # would add to the short
            if self._pos_qty > 0 and want == "buy":
                continue                      # would add to the long
            level.side = want
            level.entry_price = level.price if want == "buy" else 0.0
            level.awaiting_side = None
            level.awaiting_price = None
            resided += 1
        return resided

    def place_initial_orders(self, balance: float) -> int:
        placed = 0
        failed = 0
        held = 0
        stranded = 0
        first = True
        _price_now = self._current_price_or_none()
        self._release_awaiting_levels(_price_now)
        self._repair_ladder(_price_now)
        resided = self._reside_safe_levels(_price_now)
        if resided:
            logger.info(
                "LEVEL SIDES RESYNCED | {} idle rung(s) re-sided against {} — they were "
                "tagged for the wrong side of the market and could not have been posted",
                resided, round(_price_now, 8),
            )
        for level in self.levels:
            if level.order_id is not None:
                continue
            if level.status == "awaiting_counter":
                # Not a failure -- its exit is already on the book. Counted separately so
                # a held rung is never mistaken for one that could not place.
                held += 1
                continue
            if self._wrong_side_of(level, _price_now):
                # Also not a failure. _reside_safe_levels could not flip this one without
                # adding to an open position, so post-only would reject it every attempt.
                # Counting it as "failed" is what made a structural, self-clearing state
                # look like a broken order path on every single start.
                stranded += 1
                continue
            if self.order_pacing_seconds > 0 and not first:
                time.sleep(self.order_pacing_seconds)
            first = False
            if self._place_order_for_level(level, balance):
                placed += 1
            else:
                failed += 1

        logger.info(
            "PLACED {} initial grid orders ({} failed, {} awaiting counter, {} stranded "
            "the wrong side of {}) | vol_mult={:.2f} | exposure={:.1%}",
            placed, failed, held, stranded,
            round(_price_now, 8) if _price_now else "?",
            self._volatility_mult, self.get_exposure_pct(balance),
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

            # One spacing in the favourable direction FROM THE NEAREST LEVEL -- which is
            # not the same as "profitable". The nearest level can sit the wrong side of
            # the entry, and then one step still lands at a loss. Replaying the real
            # 2026-08-13 state: a short entered at 0.07024719, nearest sell level
            # 0.07059, hedge 0.07030 -- above break-even, so covering there books -0.52.
            # Clamp to break-even, the same rule the ordinary ladder follows since
            # AUDIT #32, and bounds-check the clamped price rather than the raw one.
            fees = self.round_trip_fee_pct
            if side == "long":
                hedge_price = self._round_price(best_level.price + self.grid_spacing)
                breakeven_bound = self._round_price_toward(entry * (1 + fees), +1)
                hedge_price = max(hedge_price, breakeven_bound)
            else:
                hedge_price = self._round_price(best_level.price - self.grid_spacing)
                breakeven_bound = self._round_price_toward(entry * (1 - fees), -1)
                hedge_price = min(hedge_price, breakeven_bound)

            if hedge_price < self.grid_lower or hedge_price > self.grid_upper:
                logger.warning(
                    "RECONCILE | {} price {} outside grid — position unprotected",
                    hedge_side, hedge_price,
                )
                continue

            # A NEAR neighbour -- not an exact duplicate -- used to sail through
            # unchecked. On 2026-08-29 a short's hedge computed at 0.2001 landed 0.0001
            # from a live BUY @ 0.2, 0.05% apart and under the round-trip fee floor: a
            # pair neither leg could ever clear a profit on. The position behind that
            # hedge was real (-1984 ADA, not dust), so the one thing that could have
            # fixed it -- a full ladder rebuild -- refuses to run while any real
            # position is open (recenter's "flat" gate), and it sat deformed for the
            # rest of that session.
            #
            # AUDIT #41 still holds: the cover must go out even beside a near
            # neighbour, so this only tries to move WHERE it lands, never whether it
            # does. Nudge away from the neighbour, capped at the same break-even bound
            # already computed above -- moving further from loss is always safe;
            # moving toward it never passes where the clamp already stopped. If no
            # clear price remains inside that bound, hedge_price is left exactly as
            # computed above and this is a no-op.
            floor = abs(hedge_price) * fees * self._min_profit_multiplier
            crowd = next(
                (l for l in self.levels
                 if l is not best_level and l.side == hedge_side
                 and 1e-12 < abs(l.price - hedge_price) < floor),
                None,
            )
            if crowd is not None:
                nudged = hedge_price - floor if crowd.price > hedge_price else hedge_price + floor
                nudged = self._round_price(nudged)
                nudged = min(nudged, breakeven_bound) if hedge_side == "buy" else max(nudged, breakeven_bound)
                clear = (
                    self.grid_lower <= nudged <= self.grid_upper
                    and abs(nudged - crowd.price) >= floor
                    and not any(
                        l is not best_level and l.side == hedge_side and abs(l.price - nudged) < floor
                        for l in self.levels
                    )
                )
                if clear:
                    logger.info(
                        "RECONCILE | {} hedge for the {} nudged {} -> {} — a {} level {} "
                        "away sat under the fee floor and this cover still needs to "
                        "clear its own",
                        hedge_side, side, hedge_price, nudged, crowd.side, round(crowd.price, 8),
                    )
                    hedge_price = nudged

            # Do not re-site a level onto a line another level already holds. Every other
            # re-siting path in this file checks for that -- _nearest_legal_exit refuses a
            # target within the fee floor of a neighbour, _refill_missing_grid_lines
            # refuses to manufacture a pair that cannot clear its own fees -- and this one
            # did not.
            #
            # The clamp above pins the hedge to break-even, which is exactly where the
            # ordinary ladder also wants to quote, so the collision is the common case
            # rather than a rare one. On the 2026-08-17 05:00 restart it put the hedge at
            # 0.07008 where a restored grid buy already rested: two levels on one line,
            # tightest spacing 0.00%, and the next poll declared the ladder DEFORMED and
            # tore down all 16 orders four seconds after GRID ACTIVATED.
            #
            # Deliberately narrow: the SAME line and the SAME side, not merely a close
            # neighbour. A near neighbour is a spacing question the ladder already has
            # machinery for, and refusing there would suppress a legitimate full-size
            # cover -- with the AUDIT #41 state (short 9916 @ 0.07024719) the break-even
            # cover lands at 0.07021 with a grid buy 0.00008 away, and that cover must
            # still go out. An exact same-side duplicate is different in kind: it is not
            # tight spacing, it is two orders where the ladder believes there is one.
            #
            # Skipping is not "leaving the position unhedged": the level already on that
            # line is a resting order on the closing side at the same price, which is the
            # job this hedge was going to do.
            occupant = next(
                (l for l in self.levels
                 if l is not best_level and l.side == hedge_side
                 and abs(l.price - hedge_price) <= 1e-12),
                None,
            )
            if occupant is not None:
                logger.info(
                    "RECONCILE | {} hedge for the {} would land on {}, the line the {} "
                    "level already holds — leaving that level to unwind the position "
                    "rather than stacking a second order on its line",
                    hedge_side, side, hedge_price, occupant.side,
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
            # Every other placement path in this file checks notional against the
            # exchange floor before spending an API call on an order that is
            # guaranteed-rejected (-4164) -- _place_order_for_level since AUDIT #164's
            # neighbour fix. This one didn't, and reconcile_positions re-runs on every
            # poll that still sees the position, so a hedge that rounds under the
            # floor (dust left behind by partial fills, or a tiny position on a low-
            # price symbol) logged the same ERROR and burned the same rate-limit
            # budget every single cycle instead of once.
            if float(qty) * hedge_price < MIN_NOTIONAL_USDT:
                logger.debug(
                    "RECONCILE SKIP HEDGE | {} @ {} notional {:.2f} < {:.2f} minimum — "
                    "would be guaranteed-rejected, not attempting",
                    hedge_side, hedge_price, float(qty) * hedge_price, MIN_NOTIONAL_USDT,
                )
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
                # reduceOnly on BOTH sides. This used to set it only when hedging a
                # long, so covering a SHORT went out as a plain buy: if the position had
                # already closed between the read and the order, that opens a fresh
                # long of the same size instead of closing anything (AUDIT #41).
                params = {"reduceOnly": True, "postOnly": False, "purpose": "reconcile"}
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

        # A reduceOnly SELL can only reduce a LONG. Placing one while the account is
        # flat or short is rejected outright (-2022) -- the exact failure AUDIT #11
        # fixed elsewhere. With the 2026-08-13 short open this loop fired three
        # guaranteed rejections on every reconcile (AUDIT #41).
        long_open = 0.0
        try:
            for pos in self.exchange.get_positions(self.symbol):
                qty_p = float(pos.get("contracts", 0) or 0)
                side_p = pos.get("side", "")
                if qty_p < 0:
                    side_p = "short" if side_p == "long" else "long"
                    qty_p = abs(qty_p)
                if side_p == "long":
                    long_open += qty_p
        except Exception as e:
            logger.debug("RECONCILE | could not read positions for orphan sells ({})", e)
            long_open = 0.0

        for level in self.levels:
            if level.side != "sell" or level.order_id is not None:
                continue
            if level.quantity <= 0 or level.price <= 0:
                continue
            if long_open <= 0:
                logger.debug(
                    "RECONCILE | skipping reduce-only sell @ {} — no long open to reduce",
                    level.price,
                )
                continue
            qty = self.exchange.exchange.amount_to_precision(self.symbol, level.quantity)
            if float(qty) <= 0:
                continue
            if float(qty) * level.price < MIN_NOTIONAL_USDT:
                logger.debug(
                    "RECONCILE SKIP ORPHAN | {} @ {} notional {:.2f} < {:.2f} minimum — "
                    "would be guaranteed-rejected, not attempting",
                    level.side, level.price, float(qty) * level.price, MIN_NOTIONAL_USDT,
                )
                continue
            try:
                params = {"reduceOnly": True, "postOnly": False, "purpose": "reconcile"}
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
            free = [l for l in self.levels if l.side == exit_side and l.order_id is None]

            # Only levels that actually clear cost. Unwinding through the grid exists to
            # avoid realising the loss a market close would have taken; a level priced
            # below break-even does the same thing, just slower. On 2026-08-12 a
            # downward recenter placed reduce-only sells at 0.06928 and 0.06954 against
            # a long held at 0.06962 -- both filled, -0.88 realised, over half that
            # session's loss, on a position whose stop was far below and which price
            # recovered past within minutes (AUDIT #32).
            fees = self.round_trip_fee_pct
            break_even = entry * (1 + fees) if side == "long" else entry * (1 - fees)
            levels = [
                l for l in free
                if ((l.price >= break_even) if side == "long" else (l.price <= break_even))
            ]
            skipped = len(free) - len(levels)
            if skipped:
                logger.info(
                    "UNWIND | {} of {} {} level(s) sit below break-even {} (entry {}) — "
                    "leaving them empty rather than booking the loss",
                    skipped, len(free), exit_side, round(break_even, 8), round(entry, 8),
                )
            if not levels:
                logger.warning(
                    "UNWIND | no {} level above break-even {} to absorb {} {} @ {} — "
                    "position waits for price to recover; the hard stop is the backstop",
                    exit_side, round(break_even, 8), side, amt, round(entry, 8),
                )
                continue

            # Slice the ACTUAL position across the exit levels. Sizing each level at
            # the normal grid notional instead meant the unwind tried to sell
            # len(levels) x grid_qty against a position often far smaller than that:
            # Binance accepted orders until the cumulative reduceOnly quantity reached
            # the position and rejected the rest (-2022), so every recenter logged a
            # burst of guaranteed-fail placements (105 in one session).
            #
            # Splitting across EVERY qualifying level up front (the original approach)
            # could size each slice under the exchange's MIN_NOTIONAL_USDT floor even
            # when the undivided amount -- or a split across fewer levels -- would clear
            # it comfortably: every slice then failed the per-level notional check below
            # and got skipped, leaving the position with NO exit resting on the ladder at
            # all. 2026-09-01: a 49 ADA long split across 3 levels came out to ~$3.20 a
            # slice, under the $5 floor, so all 3 were silently skipped (DEBUG-only log)
            # and the position sat for 15+ minutes until the raw stop-loss -- the backstop
            # of last resort -- fired and closed it externally, with only an ESTIMATED P&L
            # to show for it (AUDIT #144). Sort by distance from break-even (nearest first
            # -- those are the ones most likely to fill as price recovers) and shrink the
            # level count until each slice clears the floor, rather than dividing thinner
            # and thinner across levels the position can't actually support.
            levels.sort(key=lambda l: l.price, reverse=(side == "short"))
            level_count = len(levels)
            while level_count > 1:
                worst_price = min(l.price for l in levels[:level_count])
                if (amt / level_count) * worst_price >= MIN_NOTIONAL_USDT:
                    break
                level_count -= 1
            levels = levels[:level_count]

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
                    params = {"reduceOnly": True, "postOnly": False, "purpose": "unwind"}
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

    def check_fills(self, balance: float, open_orders: list[dict] | None = None) -> list[dict]:
        # `open_orders` lets the caller pass a book it has already read this iteration.
        # A fill is inferred from ABSENCE here, so the snapshot must be no older than the
        # caller's own -- main.py re-reads it if enforce_order_limit cancelled anything.
        if open_orders is None:
            open_orders = self.exchange.get_open_orders(self.symbol)
        open_ids = {o["id"] for o in open_orders}
        fills = []

        # A snapshot, not the live list: _handle_fill ends with self.levels.sort(),
        # an in-place reorder of this exact list while this loop's iterator is
        # walking it by index. Iterating the live list let that sort skip an
        # unvisited level (moved to an index already passed) while revisiting the
        # just-processed one under its brand-new order_id -- observed live as the
        # fresh replacement immediately mis-read as "gone" and marked dead inside the
        # same check_fills() call that had just placed it.
        for level in list(self.levels):
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
                status = order.get("status")
                if status == "open":
                    # Still live -- it just missed the open-orders snapshot (a race, or a
                    # page boundary). Touching it would place a duplicate.
                    continue
                if not order_was_filled(order):
                    # ANY terminal status that is not a completed fill: cancelled,
                    # expired, rejected. This used to test `== "canceled"` and treat
                    # everything else as a fill, which is how three reduce-only orders
                    # Binance EXPIRED with executedQty=0 were booked as +0.71 of profit
                    # that never traded (AUDIT #75).
                    logger.info(
                        "ORDER {} | {} {} @ {} did not fill — replacing the level",
                        str(status).upper(), level.side, level.quantity, level.price,
                    )
                    if self._event_journal:
                        self._event_journal.order_cancelled(
                            self.symbol, level.side, level.price, level.order_id,
                            f"fill_check_{status}",
                        )
                    level.order_id = None
                    level.status = "pending"
                    if not self._is_on_cooldown(level):
                        self._place_order_for_level(level, balance)
                    else:
                        logger.debug("SKIP REPLACEMENT (cooldown) | {} @ {}", level.side, level.price)
                    continue
                fills.append(self._handle_fill(level, balance))

        # Held rungs come back when the counter frees, or when price clears them.
        _price_now = self._current_price_or_none()
        self._release_awaiting_levels(_price_now)
        self._repair_ladder(_price_now)
        # After the fills above are attributed the ledger should agree with the
        # exchange. Where the average entry has drifted anyway -- a stop leg or a
        # reduce-only close moved the position without passing through _handle_fill --
        # adopt the exchange's, so the NEXT fill's realised P&L is measured against a
        # true average rather than a stale one (AUDIT #88).
        self.reconcile_position_entry()

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

        qty = level.quantity if level.quantity > 0 else (
            self._calc_usdt_per_grid(balance) / max(level.price, 1e-12))

        # EVERY fill pays a fee. Charging only the ones the ladder called 'complete'
        # made half of 2026-08-15's trading free in the bot's books -- 0.10 booked
        # against 0.15 actually paid (AUDIT #80).
        fee = qty * level.price * (self.taker_fee_pct if is_taker else self.maker_fee_pct)

        # Money follows the netted position, not the ladder's idea of a round trip.
        profit = self._apply_to_position(level.side, qty, level.price)

        # This fill just moved the position, and the replacement order placed further
        # down this same method is sized against the reduce-only mirror. Refresh it here
        # or that exit is sized against the position as it was before this fill (#98).
        self._refresh_net_counters()

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
        self.total_pnl += profit
        self.total_fees += fee

        if completed_cycle:
            level.total_pnl += profit
            self.total_completed_cycles += 1

        logger.info(
            # This ladder's own count, NOT the account's. main.py journals
            # router.total_fills, which sums every strategy, so the two differ by
            # whatever the trend follower has done -- 31 here against fill#37 in
            # the journal on 2026-08-20 06:02:31. Both were right and both were
            # called "fill" (AUDIT #136).
            "GRID FILL #{} | {} @ {} | qty={} profit={:.6f} fees={:.6f} net={:.6f} | cycle={}",
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
            # The counter-slot is already live, so THAT order is the exit this fill
            # needs. This rung has no work to do until it fills.
            #
            # Two wrong answers, both observed live:
            #
            #   1. Move the level onto the occupied price (the original behaviour). A
            #      one-way trip -- _place_order_for_level then finds an order already
            #      tracked by another level and returns False forever. On 2026-08-14 a
            #      buy filled at 0.07017, its sell targeted the occupied 0.07035, and the
            #      ladder ran 57 minutes on 9 of 10 rungs with a hole at the price.
            #
            #   2. Re-arm on the SAME side (AUDIT #58's first attempt). That breaks the
            #      alternation a grid depends on. With price pinned at the rung it just
            #      buys again: six consecutive BUY fills at 0.06945 on 2026-08-14,
            #      0 -> 8215 DOGE, 96% of the position cap, every one booking
            #      profit=-0.000000 and paying a fee. Doubling down is worse than idling.
            #
            # So: hold the rung, place nothing, and remember what it is waiting for. The
            # level re-arms as the counter side the moment that slot frees (AUDIT #61).
            logger.info(
                "REPLACEMENT SLOT TAKEN | {} @ {} is already live and is this fill's exit "
                "— holding {} {} until it frees rather than adding exposure",
                new_side.upper(), new_price, level.side.upper(), level.price,
            )
            level.order_id = None
            level.status = "awaiting_counter"
            level.awaiting_side = new_side
            level.awaiting_price = new_price
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

            # AUDIT #49. This path called place_limit_order DIRECTLY, so it never saw
            # _block_buys/_block_sells or the size taper -- every one of those lives in
            # _place_order_for_level, which the replacement path skips entirely.
            #
            # The position cap is therefore advisory here: each fill immediately arms
            # another order regardless of it, that order fills, and the next replacement
            # does the same. In a trend the position ratchets past the cap without limit.
            # On 2026-08-08 it reached 31,761 DOGE against a 17,467 cap -- 1.8x through
            # a limit the logs were simultaneously reporting as enforced -- and the day
            # closed -50.49, 60% of that fortnight's entire loss.
            #
            # `params is None` means _exit_order_params found nothing to reduce, i.e.
            # this order OPENS exposure. Exits stay unconditional: refusing those would
            # trap inventory, which is the #42 mistake.
            #
            # This branch has checked the position cap since #49 above, but
            # _loss_block_buys/_loss_block_sells and _profit_lock_active -- the other
            # two guards _place_order_for_level checks before it will ever open
            # exposure -- were never asked here. All three exist to stop ADDING
            # exposure, exactly what params is None means this order does, so skipping
            # the other two let a fill's own replacement re-open exposure the instant
            # after the open-loss budget or the daily profit lock had just told every
            # OTHER placement path in the file to stop.
            if params is None:
                blocked, reason = False, ""
                if new_side == "buy" and self._block_buys:
                    blocked, reason = True, "position limit reached"
                elif new_side == "sell" and self._block_sells:
                    blocked, reason = True, "position limit reached"
                elif new_side == "buy" and self._loss_block_buys:
                    blocked, reason = True, "open loss budget exhausted"
                elif new_side == "sell" and self._loss_block_sells:
                    blocked, reason = True, "open loss budget exhausted"
                elif self._profit_lock_active:
                    # Mirror _place_order_for_level's own adding_side check: a side is
                    # only ever this position's entry OR its exit, never both, so
                    # profit lock must not block the side that would only ever reduce
                    # it.
                    if self._pos_qty > 0:
                        adding_side = "buy"
                    elif self._pos_qty < 0:
                        adding_side = "sell"
                    else:
                        adding_side = None
                    if adding_side is None or new_side == adding_side:
                        blocked, reason = True, "daily profit lock active"
                if blocked:
                    logger.warning(
                        "REPLACEMENT BLOCKED | {} @ {} — {} (level left pending, AUDIT #49)",
                        new_side.upper(), new_price, reason,
                    )
                    level.order_id = None
                    level.status = "pending"
                    return fill_record
                # Same taper _place_order_for_level applies as the cap is approached.
                adj_qty *= self._buy_scale if new_side == "buy" else self._sell_scale

            if adj_qty <= 0:
                raise ValueError("replacement quantity resolved to zero")
            quantity = self.exchange.exchange.amount_to_precision(self.symbol, adj_qty)
            if float(quantity) * new_price < MIN_NOTIONAL_USDT:
                logger.debug(
                    "SKIP REPLACEMENT | {} @ {} notional {:.2f} < {:.2f} minimum",
                    new_side, new_price, float(quantity) * new_price, MIN_NOTIONAL_USDT,
                )
                level.order_id = None
                level.status = "pending"
                return fill_record
            level.quantity = float(quantity)
            # The counter-leg of a cycle: reduce-only when it closes a position, plain
            # when it re-opens the other side of the ladder. Tagged either way so the
            # ledger can separate cycle economics from forced exits (AUDIT #56).
            params = dict(params or {})
            params["purpose"] = "grid_exit" if params.get("reduceOnly") else "grid_entry"
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
        # Pausing does NOT flatten -- see Strategy.pause -- so any position survives it.
        # Cancelling the stop book here leaves that position naked for as long as it takes
        # _refresh_sl_stops to notice, and that runs on a 120s verification timer.
        #
        # ADAUSDT 2026-08-19, holding SHORT 6307:
        #
        #   14:56:28  STOP-MARKET PLACED | BUY 6307.0 ADAUSDT @ stop=0.17753
        #   15:05:39  ONE-SIDED GRID | ... forcing recenter inside margin band
        #   15:05:42  CANCEL EVERYTHING | 8 total orders confirmed cancelled
        #   15:06:28  KILL SWITCH: price 0.1776 above stop loss 0.1775302773632203
        #
        # recenter() calls pause(). Seven limit orders were open; the eighth was the
        # stop. It was gone 46 seconds before price crossed it, and the short rode from
        # 0.1767 to 0.1806 with nothing on the exchange to close it.
        #
        # emergency_stop has guarded exactly this since the shutdown fix. pause never
        # did. The gap was raised earlier the same day and dismissed on the grounds that
        # _refresh_sl_stops rebuilds the stops -- true, and not within 46 seconds
        # (AUDIT #123).
        holding = self._has_open_position()
        cancelled = self.exchange.cancel_everything(
            self.symbol, timeout_seconds=30, keep_stops=holding,
        )
        if holding:
            logger.warning(
                "STOPS LEFT ARMED | pausing with a position open, so its stop-loss legs "
                "stay on the exchange rather than going down with the ladder",
            )
        still_open = self.exchange.get_open_order_ids(self.symbol)
        for level in self.levels:
            if level.order_id is not None:
                if level.order_id in still_open:
                    logger.warning("PAUSE | order {} still open after cancel_everything — force cancelling", level.order_id)
                    if not self.exchange.cancel_order(level.order_id, self.symbol):
                        # Pausing does not flatten (see Strategy.pause), so a level that
                        # is still live must stay claimed -- otherwise resuming re-places
                        # it on top of the survivor (AUDIT #51).
                        logger.error(
                            "PAUSE | order {} @ {} could NOT be confirmed cancelled — "
                            "level stays claimed", level.order_id, level.price,
                        )
                        continue
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
        self._seed_position_from_exchange()
        placed = self.place_initial_orders(balance)
        total_open = len(self.get_tracked_order_ids())
        self.active = total_open > 0
        if self._event_journal:
            self._event_journal.grid_activated(self.symbol, self.grid_lower, self.grid_upper, self.grid_count, total_open)
        if self.active:
            logger.info("GRID ACTIVATED | {} orders active", total_open)
        else:
            logger.warning("GRID NOT ACTIVATED | no orders could be placed or restored")

    def _has_open_position(self) -> bool:
        """Is there a live position that would be left unprotected by cancelling stops?

        Asks the exchange, falling back to the last figures set_position_limit saw. On
        an unreadable book it answers True: assuming a position exists costs a few
        orphan stop orders, assuming none exists costs an unhedged position (AUDIT #65).
        """
        try:
            for pos in self.exchange.get_positions(self.symbol):
                if abs(float(pos.get("contracts", 0) or 0)) > 0:
                    return True
            return False
        except Exception as e:
            cached = self._net_long_qty > 0 or self._net_short_qty > 0
            logger.warning(
                "POSITION CHECK FAILED | {} — assuming {} so stops are not cancelled "
                "out from under a live position", e, "a position" if cached else "none",
            )
            return True

    def emergency_stop(self, reason: str = "emergency") -> None:
        """Cancel everything. `reason` only picks the log level.

        This runs on the kill switch AND from main.py's `finally:` block on a normal
        Ctrl+C, and it logged ERROR either way -- so every clean shutdown ended in two
        red EMERGENCY STOP lines and looked like a crash. Real faults have to stand out
        from routine ones or the log stops being readable (AUDIT #35).
        """
        # A position that outlives the bot must keep its stops. cancel_everything takes
        # the algo orders too, so every shutdown used to strand an unprotected position:
        # pause/shutdown deliberately does not flatten (Strategy.pause, AUDIT #37), the
        # stops were cancelled anyway, and the position then rode naked for as long as
        # the bot stayed down. Found live -- a 2522 DOGE short sat unhedged after a clean
        # Ctrl+C, and an 8215 DOGE long before it.
        #
        # Grid orders still go: they are this process's working state and would be
        # duplicated on restart. Stops are not working state, they are protection, and
        # #54's reconciler adopts live stop legs on restart rather than re-placing them
        # (AUDIT #65).
        holding = self._has_open_position()
        if reason == "shutdown":
            logger.info("SHUTDOWN | cancelling grid orders")
        else:
            logger.error("EMERGENCY STOP | cancelling grid orders ({})", reason)

        if holding:
            cancelled = self.exchange.cancel_all_open_orders(self.symbol)
            logger.warning(
                "STOPS LEFT ARMED | a position is still open, so its stop-loss legs stay "
                "on the exchange — it remains protected while the bot is down",
            )
        else:
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

        # A two-sided grid can be just as dead as a one-sided one. The checks above all
        # ask "is a whole side missing"; none of them notices a ladder that still has
        # buys and sells but has deformed so badly that the levels near the price are
        # gone. That is what a restored grid looked like on 2026-08-12 23:18 -- seven
        # buys crammed into the bottom, three sells at the top, a 1.51% hole where the
        # price actually was, and no recenter trigger for 45 minutes because price was
        # comfortably inside the range the whole time (AUDIT #34).
        #
        # Only while flat. Recentring cancels resting orders, and with inventory open
        # those orders are the exits -- the same mistake that made recenter fire 89
        # times in one session and prevented a position from ever unwinding.
        #
        # _net_long_qty/_net_short_qty are written ONLY by set_position_limit, which
        # main.py calls at main.py:1593 -- 139 lines AFTER it calls recenter at
        # main.py:1454, in the same iteration. So on the first poll after a restart they
        # are both still 0.0 while a restored position is wide open, and this guard read
        # "flat" on precisely the iteration where a restored position is most likely to
        # exist. On 2026-08-17 05:00:56, four seconds after GRID ACTIVATED, that let a
        # deformation rebuild cancel 16 orders with 5342 DOGE of short open.
        #
        # _pos_qty is the authoritative mirror and is already correct here: activate()
        # seeds it from the exchange before the loop runs at all.
        #
        # "Flat" also has to cover a position too small to protect, or dust becomes a
        # permanent veto on the one repair this gate exists to allow. Below
        # MIN_NOTIONAL_USDT there is no stop-loss to strand -- STOP-LOSS UNPROTECTABLE
        # already refuses to place one at any size -- so a rebuild sacrifices nothing
        # a real position would have lost. On 2026-08-29 a 6 ADA (~1.16 USDT) remainder
        # left two exit rungs 0.05% apart, under the fee floor; "flat" stayed False for
        # that alone, so DEFORMED LADDER (HOLDING) logged every 5 minutes for the bot's
        # entire 5-hour run and not one order filled. min_notional_price falls back to
        # +inf for a non-positive current_price, purely defensively -- ladder_defects
        # already returns [] in that case, so `flat` cannot affect the outcome then,
        # but a bad price still must not be read as "any quantity is free."
        min_notional_price = current_price if current_price > 0 else float("inf")
        flat = (
            (self._net_long_qty <= 0 or self._net_long_qty * min_notional_price < MIN_NOTIONAL_USDT)
            and (self._net_short_qty <= 0 or self._net_short_qty * min_notional_price < MIN_NOTIONAL_USDT)
            and (abs(self._pos_qty) <= 1e-9 or abs(self._pos_qty) * min_notional_price < MIN_NOTIONAL_USDT)
        )
        # Detect in every state; rebuild only when flat.
        #
        # This was `ladder_defects(current_price) if flat else []`, which made the check
        # invisible for as long as any inventory was open -- and a grid holds inventory
        # most of the time. On 2026-08-18 the ladder carried a 1.41% hole around the
        # price from 17:57, when the 0.07027 sell filled and left SHORT 1778 open, until
        # the 20:25 shutdown: 2h28m with no warning, no fills, and the nearest sell nine
        # ticks above the high price reached.
        #
        # The stale grid_spacing hid it from the other direction too. ladder_defects
        # flags a hole wider than 2x spacing; against the saved 0.00065864 that bar was
        # 0.001317 and the 0.00099 hole cleared it, so the defect was invisible even on
        # the flat path. Against the measured 0.00028286 the bar is 0.000566 and it
        # registers.
        #
        # Rebuilding while holding stays forbidden. recenter pauses the grid, and with a
        # position open the resting orders are its exits; cancelling them is what made
        # this fire 89 times in one session and left a position unable to unwind. The
        # non-destructive repair already runs every poll and is untouched by this. What
        # was missing was any signal at all (AUDIT #116).
        deformed = self.ladder_defects(current_price)
        if deformed and not flat:
            if (now - self._last_deform_warn_time) >= 300.0:
                self._last_deform_warn_time = now
                logger.warning(
                    "DEFORMED LADDER (HOLDING) | {} — not rebuilding with {:g} open, "
                    "since recentering would cancel the position's exit orders; the "
                    "per-poll repair keeps working on it",
                    "; ".join(deformed), self._pos_qty,
                )
            deformed = []

        if (in_margin_band and not stranded_above and not stranded_below
                and not dead_inside and not deformed):
            return False

        if deformed:
            logger.warning(
                "DEFORMED LADDER | {} — the grid is no longer evenly spaced around {}, "
                "rebuilding it", "; ".join(deformed), round(current_price, 8),
            )

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
        # Same check activate() makes (line ~3237): a fresh ladder with nothing
        # actually resting is not active, it is dormant wearing the active flag.
        # Setting this unconditionally meant a rebuild that failed to place a single
        # order -- the exchange down, every level rejected -- still marked the grid
        # active, so nothing downstream (the watchdog, main.py's own health checks)
        # could tell the difference between "working ladder" and "empty book,
        # believed working" until something else noticed no fills were ever coming.
        total_open = len(self.get_tracked_order_ids())
        self.active = total_open > 0
        if not self.active:
            logger.warning("RECENTER PRODUCED NO ORDERS | grid left inactive rather than reporting active with an empty book")
        self._last_recenter_time = now
        # Only re-anchor the trailing stop when there is no position to protect.
        # Resetting unconditionally handed the ratchet back to the market on every
        # recenter: with recenter firing repeatedly the peak was continuously reset
        # to the current price, so a long's stop tracked price downward instead of
        # holding its high-water mark.
        #
        # _pos_qty is included for the same reason the 'flat' gate a few dozen lines
        # above (this same function) ORs it in: _net_long_qty/_net_short_qty are the
        # exchange-position mirror, refreshed by set_position_limit every main-loop
        # iteration and by _refresh_net_counters on a fill -- but a bad read degrades
        # that mirror to its last value rather than raising (see its own docstring),
        # while _pos_qty is this ladder's own locally-updated tracker, current the
        # instant _apply_to_position runs. Checking only the mirror here reproduced
        # the exact staleness class the 'flat' gate was already hardened against, in
        # the same function, a few lines away -- trusting only the read that can go
        # stale to decide whether to discard the peak/trough high-water mark.
        if (self._net_long_qty <= 0 and self._net_short_qty <= 0
                and abs(self._pos_qty) <= 1e-9):
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

        AUDIT #168: the floor comes from get_hard_stop_loss_price() rather than a
        second grid_lower*(1-stop_loss_pct) computed inline here. get_stop_loss_price()
        returns _trailing_sl_price (set below) once armed, which is almost immediately
        after a position opens -- so a duplicated, un-anchored formula here would have
        left the scale-out trail leg quoting the old grid_lower-only level forever,
        even after the hard leg tightened to the position's average entry.
        """
        if current_price > self._peak_price:
            self._peak_price = current_price
        static_sl = self.get_hard_stop_loss_price()
        if self._peak_price > 0:
            candidate = max(static_sl, self._peak_price * (1 - self._trailing_sl_trigger))
        else:
            candidate = static_sl
        if self._trailing_sl_price is None:
            self._trailing_sl_price = candidate
        else:
            self._trailing_sl_price = max(self._trailing_sl_price, candidate)

    def update_trailing_sl_short(self, current_price: float) -> None:
        """Lower the short trailing stop toward price. Never raises it (see above).

        AUDIT #168: mirrors update_trailing_sl -- static_sl comes from
        get_short_hard_stop_loss_price(), not a duplicated inline formula.
        """
        if self._trough_price == 0.0 or current_price < self._trough_price:
            self._trough_price = current_price
        static_sl = self.get_short_hard_stop_loss_price()
        if self._trough_price > 0:
            candidate = min(static_sl, self._trough_price * (1 + self._trailing_sl_trigger))
        else:
            candidate = static_sl
        if self._trailing_sl_price_short is None:
            self._trailing_sl_price_short = candidate
        else:
            self._trailing_sl_price_short = min(self._trailing_sl_price_short, candidate)

    def get_hard_stop_loss_price(self) -> float:
        """Static stop for a long, ratcheted while the position is open.

        The raw level is grid_lower * (1 - stop_loss_pct), so it follows grid_lower --
        and recenter() moves grid_lower. Observed live on 2026-08-12 at 14:06: a
        recenter with a 7108 DOGE long open dropped the hard stop from 0.06825994 to
        0.06696984, pushing it 1.9% further from a position it was meant to protect.

        This is AUDIT #15 one layer down. #15 stopped recenter resetting the *trailing*
        anchor; the hard leg still tracked grid_lower freely. A stop may tighten while a
        position is open, never loosen. Released by reset_trailing() once flat.

        AUDIT #168: candidate now also considers the position's own average entry
        price (self._pos_entry, maintained by _apply_to_position/seed_position),
        not just grid_lower. grid_lower is frozen while a position is open (AUDIT
        #116 blocks recenter()), so a stop anchored to it alone could not track a
        position built up through a long drawdown of grid dip-buys -- observed
        2026-08-30, hours of grid profit erased in under a minute when the stop
        finally hit, because it was still sitting at the pre-drawdown grid_lower
        level instead of near the position's real (falling) average cost. Using
        max(...) here can only ever tighten the floor relative to the old
        grid_lower-only figure -- a long only fills at prices >= grid_lower, so
        _pos_entry is never below it in the normal case -- and the _pos_qty > 0 /
        _pos_entry > 0 guard falls back to the exact old behaviour whenever the
        ledger has no usable entry price yet, so this cannot make the stop worse.
        """
        candidate = self.grid_lower * (1 - self.stop_loss_pct)
        if self._pos_qty > 0 and self._pos_entry > 0:
            candidate = max(candidate, self._pos_entry * (1 - self.stop_loss_pct))
        if self._net_long_qty <= 0:
            self._hard_sl_price = candidate
            return candidate
        if self._hard_sl_price is None:
            self._hard_sl_price = candidate
        else:
            self._hard_sl_price = max(self._hard_sl_price, candidate)
        return self._hard_sl_price

    def block_side(self, side: str, reason: str) -> None:
        """Refuse to OPEN exposure on `side` until the next set_position_limit().

        AUDIT #50. Used when the position has no live stop-loss: an unprotected
        position must not also be a growing one. Exits are unaffected -- the block
        flags only gate orders that add exposure (see #49) -- so inventory can still
        unwind while uncovered.
        """
        if side == "buy" and not self._block_buys:
            self._block_buys = True
            logger.warning("SIDE BLOCKED | buy — {} (AUDIT #50)", reason)
        elif side == "sell" and not self._block_sells:
            self._block_sells = True
            logger.warning("SIDE BLOCKED | sell — {} (AUDIT #50)", reason)

    def get_short_hard_stop_loss_price(self) -> float:
        """Mirror of get_hard_stop_loss_price for a short: may fall, never rise.

        AUDIT #168: mirrors the long-side change -- see get_hard_stop_loss_price.
        """
        candidate = self.grid_upper * (1 + self.stop_loss_pct)
        if self._pos_qty < 0 and self._pos_entry > 0:
            candidate = min(candidate, self._pos_entry * (1 + self.stop_loss_pct))
        if self._net_short_qty <= 0:
            self._hard_sl_price_short = candidate
            return candidate
        if self._hard_sl_price_short is None:
            self._hard_sl_price_short = candidate
        else:
            self._hard_sl_price_short = min(self._hard_sl_price_short, candidate)
        return self._hard_sl_price_short

    def get_stop_loss_price(self) -> float:
        if self._trailing_sl_price is not None:
            return self._trailing_sl_price
        return self.get_hard_stop_loss_price()

    def get_short_stop_loss_price(self) -> float:
        if self._trailing_sl_price_short is not None:
            return self._trailing_sl_price_short
        return self.get_short_hard_stop_loss_price()

    def get_scale_out_trail_price(self, side: str = "long") -> float:
        """Trailing price for the scale-out leg.

        Anchored at stop-loss distance from the running peak/trough so the scale-out
        split arms immediately at fresh start. The trigger-distance trail
        (peak*(1-trigger)) stays pinned to the static hard level while
        TRAILING_SL_TRIGGER_PCT > STOP_LOSS_PCT, which collapses the split to a
        single hard stop until price rises ~2%+ above the grid. Falls back to the
        static hard level when no anchor has been observed yet.

        AUDIT #168: `hard` comes from get_hard_stop_loss_price()/
        get_short_hard_stop_loss_price() rather than a third inline copy of
        grid_lower*(1-stop_loss_pct) -- see update_trailing_sl.
        """
        if side == "short":
            hard = self.get_short_hard_stop_loss_price()
            if self._trough_price <= 0:
                return hard
            return min(hard, self._trough_price * (1 + self.stop_loss_pct))
        hard = self.get_hard_stop_loss_price()
        if self._peak_price <= 0:
            return hard
        return max(hard, self._peak_price * (1 - self.stop_loss_pct))

    def reset_trailing(self) -> None:
        """Release both ratchets, e.g. when the position closes or the side flips.

        The hard-stop ratchet is released here too: with no position open there is
        nothing to protect, so the static level should track grid_lower again.
        """
        self._peak_price = 0.0
        self._trough_price = 0.0
        self._trailing_sl_price = None
        self._trailing_sl_price_short = None
        self._hard_sl_price = None
        self._hard_sl_price_short = None

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
            exposure, self.get_spread_pct() * 100,
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
            "pos_qty": self._pos_qty,
            "pos_entry": self._pos_entry,
            "total_fills": self.total_fills,
            "total_completed_cycles": self.total_completed_cycles,
            "_last_recenter_time": self._last_recenter_time,
            "_last_handoff_accel_time": self._last_handoff_accel_time,
            "_trailing_sl_price": self._trailing_sl_price,
            "_trailing_sl_trigger": self._trailing_sl_trigger,
            "_peak_price": self._peak_price,
            "_trough_price": self._trough_price,
            "_trailing_sl_price_short": self._trailing_sl_price_short,
            # AUDIT #50. These are RATCHETS -- a long's hard stop may only rise, a
            # short's may only fall -- and they exist because recenter() moves
            # grid_lower/grid_upper, which would otherwise drag the stop away from an
            # open position. Every other stop anchor was persisted; these two were not,
            # so each restart reset them to None and the next call re-derived the stop
            # from the CURRENT grid bounds. Restarting with a position open therefore
            # loosened its stop, silently, which is the one direction a ratchet exists
            # to forbid.
            "_hard_sl_price": self._hard_sl_price,
            "_hard_sl_price_short": self._hard_sl_price_short,
            "_volatility_mult": self._volatility_mult,
            "_block_buys": self._block_buys,
            "_block_sells": self._block_sells,
            "_last_replacement_time": self._last_replacement_time,
            "_buy_scale": self._buy_scale,
            "_sell_scale": self._sell_scale,
            "levels": [l.to_dict() for l in self.levels],
        }

    def ladder_defects(self, current_price: float) -> list[str]:
        """Ways the ladder has stopped being a ladder. Empty list means healthy.

        A grid only works if its levels are evenly spaced around the price: price sits
        between two adjacent levels, one spacing away from each. Fills, replacements,
        recentres and the duplicate merge each move levels independently, and over a
        session they can deform the ladder into something that is still ten levels but
        no longer a grid.

        Measured on the 2026-08-12 23:18 restart -- 45 minutes, zero fills, from a
        restored ladder that looked like this at a price of 0.06945:

            0.06771 0.06797 0.06800 0.06823 0.06829 0.06850 0.06876  ...  0.06981 ...
                      ^ 0.04% apart      ^ 0.09% apart        ^--- 1.51% hole ---^

        Seven buys crammed into the bottom with two pairs closer together than the fee
        floor (so neither pair can ever profit), and a hole five times the nominal
        spacing exactly where the price was. DOGE moved 0.446% during that run and the
        nearest sell was 0.52% away; on an even ladder it would have been 0.42% away and
        that move would have filled it. The deformation cost a real fill (AUDIT #34).
        """
        if len(self.levels) < 2 or self.grid_spacing <= 0 or current_price <= 0:
            return []

        prices = sorted(l.price for l in self.levels)
        defects: list[str] = []

        floor = self.round_trip_fee_pct * self._min_profit_multiplier
        too_close = [
            (a, b) for a, b in zip(prices, prices[1:])
            if a > 0 and (b - a) / a < floor
        ]
        if too_close:
            defects.append(
                f"{len(too_close)} level pair(s) closer than the "
                f"{floor * 100:.2f}% fee floor (tightest "
                f"{min((b - a) / a for a, b in too_close) * 100:.2f}%)"
            )

        below = [p for p in prices if p <= current_price]
        above = [p for p in prices if p > current_price]
        if below and above:
            gap = min(above) - max(below)
            if gap > 2 * self.grid_spacing:
                defects.append(
                    f"a {gap / current_price * 100:.2f}% hole around the price "
                    f"({gap / self.grid_spacing:.1f}x the {self.grid_spacing:.8f} spacing)"
                )

        return defects

    def reset_levels_to_pending(self, current_price: float | None = None) -> int:
        """Clear every level's order id and return it to a pending buy at its own price.

        Called when the exchange reports no position at startup: whatever the state file
        believed about resting orders and half-finished cycles is stale, and the ladder
        should be rebuilt from its own prices.

        Reverting a replaced sell to a buy at its entry price is what makes this more
        than a loop over `levels`: two levels can land on the same price+side, and until
        this lived in the engine that duplicate was saved to the state file and only
        noticed on the *next* start ("DEDUPLICATED 1 levels ... 10 -> 9", then a refill
        that rebuilt the lost line from scratch). Merging and refilling here keeps the
        ladder whole in the session that damaged it, and keeps main.py out of GridLevel
        internals -- which the router cannot forward safely anyway (AUDIT #31).

        Returns the number of levels reset.
        """
        reset = 0
        resided = 0
        for level in self.levels:
            level.order_id = None
            if level.side == "sell" and level.status == "replaced":
                level.side = "buy"
                level.price = level.entry_price if level.entry_price else level.price
                level.status = "pending"
                reset += 1
            elif level.status != "pending":
                level.status = "pending"
                reset += 1

            # Re-derive which side of the market this line sits on.
            #
            # The saved side was decided against spot as it was when the ladder was
            # built. Once price drifts past a rung, that rung is tagged for the wrong
            # side of the market and can never be placed: post-only rejects a sell below
            # the bid and a buy above the ask with -2019, on every attempt, forever.
            #
            # Observed 2026-08-18. The restored ladder's centre was 0.069940 and spot
            # opened at 0.07027, so 0.06994 was still a "sell" sitting BELOW the market:
            #     17:43:10  PLACED 7 initial grid orders (1 failed, 0 awaiting counter)
            #     17:43:15  PLACED 0 initial grid orders (1 failed, 0 awaiting counter)
            # the same level both times. Three sell rungs covered spot instead of four,
            # and when the innermost filled at 17:57 the book above price was 0.07060 and
            # 0.07093 only. Price reached 0.07051 -- nine ticks short -- and nothing
            # filled for 2h28m.
            #
            # Safe HERE and nowhere else: main.py calls this only when the exchange
            # reports no position, so no level holds inventory. With a position open a
            # side flip would turn a reduce-only exit into an order that ADDS exposure,
            # which is why the reconcile path beside it leaves sides alone (AUDIT #115).
            #
            # `> 0` and not merely `is not None`: a failed price read surfacing as 0.0
            # would otherwise re-tag the whole ladder as sells.
            if current_price is not None and current_price > 0:
                want = "buy" if level.price < current_price else "sell"
                if level.side != want:
                    level.side = want
                    level.entry_price = level.price if want == "buy" else 0.0
                    # The flat account proves whatever cycle queued this is over; an exit
                    # left queued against a re-sided level waits on inventory that no
                    # longer exists.
                    level.awaiting_side = None
                    level.awaiting_price = None
                    resided += 1

        if resided:
            logger.warning(
                "LEVEL SIDES RESYNCED | {} level(s) were tagged for the wrong side of "
                "{} and could never have been placed post-only",
                resided, round(current_price, 8),
            )

        before = len(self.levels)
        self._dedupe_levels()
        if len(self.levels) != before:
            logger.warning(
                "RESET LEVELS | merged {} duplicate price+side level(s) created by the reset",
                before - len(self.levels),
            )
            self._refill_missing_grid_lines(current_price)
        self.levels.sort(key=lambda l: l.price)

        # Restoring level *prices* is only worth doing while they still form a ladder.
        # Nothing is at risk here -- this path runs when the exchange reports no
        # position -- so a deformed ladder is rebuilt rather than resurrected. Per-level
        # fill counts are lost; they are statistics, and a grid with a hole where the
        # price sits does not trade (AUDIT #34).
        if current_price and current_price > 0:
            defects = self.ladder_defects(current_price)
            if defects:
                logger.warning(
                    "RESET LEVELS | restored ladder is deformed ({}) — rebuilding it "
                    "around {} instead of trading a broken grid",
                    "; ".join(defects), current_price,
                )
                self.initialize(current_price, balance=0.0)
                return len(self.levels)

        return reset

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
        # grid_count is NOT restored. It comes from the config this engine was built
        # with, and taking it from the file silently undid AUDIT #76 one layer down:
        # main.py deliberately constructs with settings.grid_count and logs
        # "GRID COUNT CHANGED | saved state has 14 rungs, config says 4 -- rebuilding
        # the ladder at 4", and then this line put 14 straight back. Observed live on
        # 2026-08-18 06:22, which ran a 14-rung ladder committing 875 USDT a side --
        # 89% of the position cap -- against the 250 the config asked for (AUDIT #108).
        #
        # The BOUNDS still come from state on purpose: that is where the live orders
        # and the open position actually sit. The level rebuild below spreads the
        # configured count across them.
        saved_count = int(data.get("grid_count") or self.grid_count)
        self.grid_spacing = data["grid_spacing"]
        self.active = data["active"]
        self.total_pnl = data.get("total_pnl", 0.0)
        self.total_fees = data.get("total_fees", 0.0)
        self._pos_qty = float(data.get("pos_qty", 0.0) or 0.0)
        self._pos_entry = float(data.get("pos_entry", 0.0) or 0.0)
        self.total_fills = data.get("total_fills", 0)
        self.total_completed_cycles = data.get("total_completed_cycles", 0)
        self._last_recenter_time = data.get("_last_recenter_time", 0.0)
        self._last_handoff_accel_time = data.get("_last_handoff_accel_time", 0.0)
        self._trailing_sl_price = data.get("_trailing_sl_price", None)
        self._trailing_sl_trigger = data.get("_trailing_sl_trigger", self._trailing_sl_trigger)
        self._peak_price = data.get("_peak_price", 0.0)
        self._trough_price = data.get("_trough_price", 0.0)
        self._trailing_sl_price_short = data.get("_trailing_sl_price_short", None)
        self._hard_sl_price = data.get("_hard_sl_price", None)
        self._hard_sl_price_short = data.get("_hard_sl_price_short", None)
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
            if saved_count != self.grid_count:
                # A deliberate config change, not damage. Marking it corrupt would have
                # main.py delete the state file and throw away the PnL and risk history
                # alongside a ladder that is doing exactly what it was told.
                logger.info(
                    "GRID COUNT APPLIED | state holds {} levels, config asks for {} — "
                    "rebuilding across the saved bounds",
                    len(self.levels), self.grid_count,
                )
            else:
                self.state_corrupted = True
                logger.warning(
                    "GRID STATE CORRUPT | expected {} levels but restored {} — "
                    "rebuilding levels from grid bounds",
                    self.grid_count, len(self.levels),
                )
            self._rebuild_levels(current_price)
        else:
            # _rebuild_levels recomputes grid_spacing itself; this is the path where it
            # does not run, and nothing else repairs the restored scalar.
            self._resync_spacing_to_levels()

    def _resync_spacing_to_levels(self) -> None:
        """Derive grid_spacing from the ladder that actually exists.

        grid_spacing is restored verbatim from the file while the levels beside it are
        deduped, refilled and re-sided. Nothing reconciled the two: _rebuild_levels
        recomputes it but only fires when the level count disagrees with config, and
        _refill_missing_grid_lines deliberately inserts into the ladder's ACTUAL gaps
        without touching it. So a state file that survives a GRID_COUNT change carries
        the OLD count's spacing forever.

        Observed live 2026-08-18: the file held 0.0006586412, which is exactly
        (upper-lower)/3 -- the value _rebuild_levels writes for grid_count=4 -- against a
        restored 8-rung ladder whose real mean gap was 0.0002828571. 2.33x too wide, and
        the four rungs that ladder was built from (0.06895/0.06961/0.07027/0.07093) were
        all still in it.

        It is not a cosmetic field. Replacements are posted at level.price +/-
        grid_spacing (see _handle_fill), so the exit for the 0.07027 fill went to 0.06961
        -- two rungs down, 0.94% away -- instead of the adjacent 0.06994. Price then
        ranged 0.07022-0.07051 for 2h28m and touched nothing. The profit floor compares
        against grid_spacing too, so every cycle was scored on a gap the ladder does not
        have (AUDIT #114).

        Mean, not a step: _initialize_dynamic builds a deliberately non-uniform ladder,
        so no single value indexes it. This mirrors what that function computes.
        """
        if len(self.levels) < 2:
            return
        gaps = [self.levels[i + 1].price - self.levels[i].price
                for i in range(len(self.levels) - 1)]
        measured = sum(gaps) / len(gaps)
        if measured <= 0:
            logger.warning(
                "GRID SPACING | {} levels average a gap of {} — keeping the restored "
                "{}", len(self.levels), measured, self.grid_spacing,
            )
            return
        stale = self.grid_spacing
        self.grid_spacing = measured
        if stale > 0 and abs(stale - measured) / measured > 0.05:
            logger.warning(
                "GRID SPACING RESYNCED | state held {:.8f} but the {} restored levels "
                "average {:.8f} ({:.2f}x) — replacement orders and the profit floor "
                "would have used the stale value",
                stale, len(self.levels), measured, stale / measured,
            )

    def _refill_missing_grid_lines(self, current_price: float | None = None) -> None:
        """Re-add pending levels on empty grid lines after duplicate price+side
        levels were merged. A fill whose replacement parked on an occupied slot
        leaves a hole in the grid; refilling it preserves fill/cycle bookkeeping
        instead of rebuilding every level from the grid bounds."""
        # Refill into the ladder's ACTUAL gaps, widest first.
        #
        # This used to walk a uniform template -- grid_lower + i * grid_spacing -- and
        # add any template price not already occupied. But _initialize_dynamic builds a
        # deliberately NON-uniform ladder, concentrated near the price, so the template
        # never lines up with it: every template price lands a few ticks off a real one
        # and gets inserted as a near-duplicate. That is where the 0.04% and 0.09% level
        # pairs in the 2026-08-12 state file came from -- 0.06800 beside 0.06797,
        # 0.06829 beside 0.06823 -- neither pair able to clear the 0.12% fee floor, and
        # the levels that should have covered the middle of the range never placed at
        # all. AUDIT #34 rebuilt ladders deformed this way; this is the deformity
        # itself (AUDIT #36).
        floor_pct = self.round_trip_fee_pct * self._min_profit_multiplier
        missing = 0
        while len(self.levels) < self.grid_count:
            prices = sorted(self._round_price(l.price) for l in self.levels)
            if len(prices) < 2:
                break
            gap, pair = 0.0, None
            for a, b in zip(prices, prices[1:]):
                if b - a > gap:
                    gap, pair = b - a, (a, b)
            if pair is None:
                break
            a, b = pair
            price = self._round_price((a + b) / 2)
            # A refill must never manufacture a pair that cannot clear its own fees --
            # that is the defect being fixed, not a side effect of it. When the widest
            # remaining gap is too narrow to split, the ladder is simply full: stop and
            # run with fewer levels rather than wedge one in.
            if price <= a or price >= b:
                break
            if (price - a) / a < floor_pct or (b - price) / price < floor_pct:
                break
            side = "buy" if current_price is not None and price < current_price else "sell"
            lvl = GridLevel(price=price, side=side)
            if side == "buy":
                lvl.entry_price = price
            self.levels.append(lvl)
            missing += 1
        self.levels.sort(key=lambda x: x.price)
        if missing:
            logger.warning(
                "RECOVERED grid state | refilled {} empty grid line(s) after dedupe, "
                "preserving {} existing levels' bookkeeping",
                missing, len(self.levels) - missing,
            )
