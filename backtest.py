"""Historical replay harness for the grid engine.

The point of this module is to answer parameter questions -- "is GRID_COUNT=10 better
than 20?", "does MIN_PROFIT_MULTIPLIER=3 actually help?" -- without spending live days
and live fees finding out. Every number in AUDIT.md issues #17-#20 was derived from
arithmetic plus a single live session; this is the instrument that checks them.

Design rule: the backtest drives the **real** GridEngine. SimulatedExchange implements
the same 13-method surface the engine calls on the live Exchange, so what runs here is
the actual order placement, fill handling, replacement, recentering, position-cap and
post-only logic -- not a second implementation that could agree with the first while
both are wrong.

WHAT IS MODELLED
  - Limit orders resting on a book, filled when price trades through them
  - Post-only rejection when an order would cross (AUDIT #17)
  - reduceOnly rejection when it would not reduce (Binance -2022)
  - Minimum notional rejection (Binance -4164)
  - Maker fees on resting fills, taker fees on market closes
  - One-way position mode with a single blended average entry, as on the live account
  - Trailing and hard stop-loss
  - Optional trend-filter regime gating

WHAT IS NOT MODELLED -- read this before trusting a result
  - Slippage and order-book depth. A resting limit order fills at exactly its price.
    Real fills on a thin book are worse, so results here are optimistic.
  - Partial fills. An order fills all-or-nothing.
  - Funding fees. On the live account these were +3.05 over the audited window, i.e.
    small and favourable, but they are a real term this omits.
  - Latency. One candle is one loop iteration; the live bot polls every 10s and can be
    up to a full interval late seeing a fill.
  - Intra-candle path. Only OHLC is known, so the fill order within a candle is an
    approximation (see SimulatedExchange.step).
  - main.py's kill switches, recovery mode and daily-loss gating.

Results are therefore an upper bound on what the same configuration would have earned,
not a forecast. Use them to compare configurations against each other -- that
comparison is fair, because every configuration gets the same optimistic treatment.
"""

from __future__ import annotations

import time as _real_time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

import grid as grid_module
from exchange import PostOnlyWouldCross
from grid import GridEngine

MIN_NOTIONAL_USDT = 5.0


# --------------------------------------------------------------------------
# Virtual clock
# --------------------------------------------------------------------------

class VirtualClock:
    """Simulated wall clock, advanced one candle at a time.

    GridEngine gates replacements and recentering on time.time(): replacement_cooldown
    (20s) and recenter_cooldown (180s). A backtest replays months in seconds, so under
    the real clock those cooldowns would never expire, replacements would be blocked
    for the whole run, and the result would be meaningless. Installing this as the
    engine's `time` module keeps cooldowns behaving as they do live, measured in
    simulated seconds.
    """

    def __init__(self, start: float = 0.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        """Order pacing must not really sleep -- just advance simulated time."""
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _clock_installed:
    """Context manager swapping GridEngine's `time` module for a VirtualClock."""

    def __init__(self, clock: VirtualClock):
        self.clock = clock
        self._saved = None

    def __enter__(self) -> VirtualClock:
        self._saved = grid_module.time
        grid_module.time = self.clock
        return self.clock

    def __exit__(self, *exc) -> None:
        grid_module.time = self._saved


# --------------------------------------------------------------------------
# Simulated exchange
# --------------------------------------------------------------------------

@dataclass
class SimOrder:
    id: str
    side: str
    price: float
    amount: float
    reduce_only: bool = False
    status: str = "open"


class SimulatedExchange:
    """Implements the surface GridEngine calls on the live Exchange.

    Position accounting is one-way with a single blended average entry, matching the
    live account (the bot sets no positionSide, so Binance keeps one net position per
    symbol). Realized PnL is computed from that position, independently of the engine's
    own per-level bookkeeping -- which is the whole point, since those two disagreeing
    is what AUDIT.md issue #7 was about. run_backtest() reports both so the gap stays
    visible.
    """

    def __init__(
        self,
        symbol: str = "DOGEUSDT",
        starting_balance: float = 5000.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        price_decimals: int = 5,
        amount_decimals: int = 0,
    ):
        self.symbol = symbol
        self.starting_balance = starting_balance
        self.cash = starting_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        self.position_qty = 0.0      # signed: >0 long, <0 short
        self.position_entry = 0.0

        self.price = 0.0
        self._orders: dict[str, SimOrder] = {}
        self._next_id = 0

        # metrics
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.fill_log: list[dict] = []
        self.rejected_crossing = 0
        self.rejected_reduce_only = 0
        self.rejected_min_notional = 0
        self.maker_fills = 0
        self.taker_fills = 0

        sim = self

        class _Precision:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{float(amount):.{amount_decimals}f}"

            @staticmethod
            def price_to_precision(symbol, price):
                return f"{float(price):.{price_decimals}f}"

        self.exchange = _Precision()

    # --- state -----------------------------------------------------------

    @property
    def unrealized_pnl(self) -> float:
        if self.position_qty == 0:
            return 0.0
        return (self.price - self.position_entry) * self.position_qty

    @property
    def equity(self) -> float:
        return self.cash + self.unrealized_pnl

    def get_balance(self) -> float:
        return self.equity

    def get_positions(self, symbol: str) -> list[dict]:
        if self.position_qty == 0:
            return []
        return [{
            "side": "long" if self.position_qty > 0 else "short",
            "contracts": abs(self.position_qty),
            "entryPrice": self.position_entry,
            "unrealizedPnl": self.unrealized_pnl,
        }]

    def get_orderbook_depth(self, symbol: str, limit: int = 10) -> dict:
        # No depth model: report a balanced book so imbalance-based logic is neutral
        # rather than silently biased by a fabricated number.
        return {"bid_vol": 0.0, "ask_vol": 0.0, "imbalance": 0.0, "spread_pct": 0.0}

    def can_place_order(self, symbol: str) -> bool:
        return True

    # --- orders ----------------------------------------------------------

    def get_open_orders(self, symbol: str) -> list[dict]:
        return [
            {"id": o.id, "side": o.side, "price": o.price, "amount": o.amount}
            for o in self._orders.values() if o.status == "open"
        ]

    def get_open_order_ids(self, symbol: str) -> set:
        return {o.id for o in self._orders.values() if o.status == "open"}

    def fetch_order(self, order_id: str, symbol: str) -> dict | None:
        o = self._orders.get(order_id)
        if o is None:
            return None
        return {"id": o.id, "side": o.side, "price": o.price,
                "amount": o.amount, "status": o.status}

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        o = self._orders.get(order_id)
        if o and o.status == "open":
            o.status = "canceled"
        return True

    def cancel_everything(self, symbol: str, timeout_seconds: float = 300.0) -> int:
        n = 0
        for o in self._orders.values():
            if o.status == "open":
                o.status = "canceled"
                n += 1
        return n

    def place_limit_order(
        self, symbol: str, side: str, price: float, amount: float,
        max_attempts: int = 3, params: dict | None = None, post_only: bool = True,
        allow_taker_fallback: bool = False,
    ) -> dict:
        params = dict(params or {})
        if "postOnly" in params:
            post_only = bool(params.pop("postOnly"))
        reduce_only = bool(params.get("reduceOnly", False))
        price = float(price)
        amount = float(amount)

        if amount <= 0:
            raise ValueError("amount must be positive")

        if amount * price < MIN_NOTIONAL_USDT:
            self.rejected_min_notional += 1
            raise ValueError('binance {"code":-4164,"msg":"Order\'s notional must be no smaller than 5"}')

        # Binance -2022: reduceOnly is only legal when the order actually reduces.
        if reduce_only:
            closable = max(0.0, self.position_qty) if side == "sell" else max(0.0, -self.position_qty)
            if closable <= 0 or amount > closable + 1e-9:
                self.rejected_reduce_only += 1
                raise ValueError('binance {"code":-2022,"msg":"ReduceOnly Order is rejected."}')

        # Post-only: reject anything that would take liquidity (AUDIT #17).
        if post_only and self._would_cross(side, price):
            if not allow_taker_fallback:
                self.rejected_crossing += 1
                raise PostOnlyWouldCross(
                    f"{side} @ {price} would cross the spread; not placing as taker"
                )
            self._execute(side, price, amount, self.taker_fee, reduce_only)
            self._next_id += 1
            return {"id": f"x{self._next_id}", "side": side, "price": price, "amount": amount}

        self._next_id += 1
        oid = f"o{self._next_id}"
        self._orders[oid] = SimOrder(
            id=oid, side=side, price=price, amount=amount, reduce_only=reduce_only,
        )
        return {"id": oid, "side": side, "price": price, "amount": amount}

    def close_position(self, symbol: str) -> bool:
        """Market close at the current price, charged at the taker rate."""
        if self.position_qty == 0:
            return True
        side = "sell" if self.position_qty > 0 else "buy"
        self._execute(side, self.price, abs(self.position_qty), self.taker_fee, True)
        return True

    def _would_cross(self, side: str, price: float) -> bool:
        if self.price <= 0:
            return False
        return price >= self.price if side == "buy" else price <= self.price

    # --- matching --------------------------------------------------------

    def step(self, o: float, h: float, l: float, c: float) -> None:
        """Advance one candle, filling any resting order the price traded through.

        Only OHLC is known, so the intra-candle path is approximated by the standard
        convention: an up candle is assumed to reach its low before its high, a down
        candle its high before its low. That is the conservative reading for a grid --
        it fills the far side of the book first and so books the *less* favourable
        sequence when both sides could have filled in one candle.
        """
        self.price = o
        if c >= o:
            self._fill_buys_down_to(l)
            self._fill_sells_up_to(h)
        else:
            self._fill_sells_up_to(h)
            self._fill_buys_down_to(l)
        self.price = c

    def _fill_buys_down_to(self, low: float) -> None:
        # Highest buy is hit first on the way down.
        resting = sorted(
            [o for o in self._orders.values() if o.status == "open" and o.side == "buy"],
            key=lambda o: -o.price,
        )
        for order in resting:
            if order.price >= low:
                self.price = order.price
                self._fill(order)

    def _fill_sells_up_to(self, high: float) -> None:
        resting = sorted(
            [o for o in self._orders.values() if o.status == "open" and o.side == "sell"],
            key=lambda o: o.price,
        )
        for order in resting:
            if order.price <= high:
                self.price = order.price
                self._fill(order)

    def _fill(self, order: SimOrder) -> None:
        order.status = "closed"
        self.maker_fills += 1
        self._execute(order.side, order.price, order.amount, self.maker_fee, order.reduce_only)

    def _execute(self, side: str, price: float, qty: float, fee_rate: float, reduce_only: bool) -> None:
        if fee_rate == self.taker_fee:
            self.taker_fills += 1
        fee = qty * price * fee_rate
        self.fees_paid += fee
        self.cash -= fee

        signed = qty if side == "buy" else -qty
        old = self.position_qty
        new = old + signed
        realized = 0.0

        if old == 0 or (old > 0) == (signed > 0):
            # Opening or adding: blend the average entry.
            total = abs(old) + qty
            self.position_entry = ((abs(old) * self.position_entry) + (qty * price)) / total if total else 0.0
        else:
            # Reducing, closing, or flipping.
            closing = min(abs(old), qty)
            direction = 1.0 if old > 0 else -1.0
            realized = (price - self.position_entry) * closing * direction
            self.realized_pnl += realized
            self.cash += realized
            if abs(signed) > abs(old):
                self.position_entry = price   # flipped through flat
            elif abs(new) < 1e-12:
                self.position_entry = 0.0

        self.position_qty = 0.0 if abs(new) < 1e-12 else new
        self.fill_log.append({
            "side": side, "price": price, "qty": qty,
            "fee": fee, "realized": realized, "position_after": self.position_qty,
        })


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass
class BacktestResult:
    config: dict = field(default_factory=dict)
    candles: int = 0
    starting_balance: float = 0.0
    ending_equity: float = 0.0
    gross_realized: float = 0.0
    fees_paid: float = 0.0
    net_pnl: float = 0.0
    engine_reported_pnl: float = 0.0
    fills: int = 0
    maker_fills: int = 0
    taker_fills: int = 0
    completed_cycles: int = 0
    winning_cycles: int = 0
    max_drawdown_pct: float = 0.0
    recenters: int = 0
    rejected_crossing: int = 0
    rejected_reduce_only: int = 0
    rejected_min_notional: int = 0
    stop_loss_hits: int = 0
    time_in_position_pct: float = 0.0
    time_paused_pct: float = 0.0
    equity_curve: list[float] = field(default_factory=list, repr=False)

    @property
    def return_pct(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return (self.ending_equity - self.starting_balance) / self.starting_balance

    @property
    def fee_ratio(self) -> float:
        """Fees as a multiple of gross realized profit. Above 1.0 means the exchange
        earned more from the strategy than the strategy did."""
        if self.gross_realized <= 0:
            return float("inf") if self.fees_paid > 0 else 0.0
        return self.fees_paid / self.gross_realized

    @property
    def win_rate(self) -> float:
        return self.winning_cycles / self.completed_cycles if self.completed_cycles else 0.0

    def summary(self) -> str:
        lines = [
            f"  candles              {self.candles}",
            f"  starting balance     {self.starting_balance:,.2f}",
            f"  ending equity        {self.ending_equity:,.2f}",
            f"  return               {self.return_pct:+.2%}",
            "",
            f"  gross realized       {self.gross_realized:+,.2f}",
            f"  fees paid            {-self.fees_paid:,.2f}",
            f"  net PnL              {self.net_pnl:+,.2f}",
            f"  fees / gross         {self.fee_ratio:.2f}x",
            "",
            f"  fills                {self.fills}  (maker {self.maker_fills}, taker {self.taker_fills})",
            f"  completed cycles     {self.completed_cycles}",
            f"  win rate             {self.win_rate:.1%}",
            f"  max drawdown         {self.max_drawdown_pct:.2%}",
            f"  time in position     {self.time_in_position_pct:.1%}",
            f"  time paused (trend)  {self.time_paused_pct:.1%}",
            "",
            f"  recenters            {self.recenters}",
            f"  stop-loss hits       {self.stop_loss_hits}",
            f"  crossing refused     {self.rejected_crossing}",
            f"  reduceOnly rejected  {self.rejected_reduce_only}",
            f"  min-notional skipped {self.rejected_min_notional}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_backtest(
    ohlcv: pd.DataFrame,
    *,
    symbol: str = "DOGEUSDT",
    starting_balance: float = 5000.0,
    grid_count: int = 10,
    capital_per_grid_pct: float = 0.018,
    max_position_pct: float = 0.12,
    max_exposure_pct: float = 0.50,
    range_atr_multiplier: float = 1.5,
    min_profit_multiplier: float = 3.0,
    maker_fee: float = 0.0002,
    taker_fee: float = 0.0004,
    stop_loss_pct: float = 0.03,
    trailing_sl_trigger_pct: float = 0.05,
    recenter_cooldown: int = 180,
    replacement_cooldown: int = 20,
    recenter_margin_pct: float = 0.008,
    candle_seconds: int = 3600,
    warmup: int = 50,
    price_decimals: int = 5,
    amount_decimals: int = 0,
    use_trend_filter: bool = False,
    adx_trend_threshold: float = 30.0,
    adx_range_threshold: float = 20.0,
    ema_fast: int = 20,
    ema_slow: int = 50,
    adx_period: int = 14,
    quiet: bool = True,
) -> BacktestResult:
    """Replay `ohlcv` through a real GridEngine and report what it would have done.

    `ohlcv` needs columns open/high/low/close. `warmup` candles are consumed to seed
    the ATR-based range before trading starts.
    """
    if quiet:
        logger.disable("grid")
        logger.disable("exchange")

    try:
        return _run(
            ohlcv, symbol=symbol, starting_balance=starting_balance, grid_count=grid_count,
            capital_per_grid_pct=capital_per_grid_pct, max_position_pct=max_position_pct,
            max_exposure_pct=max_exposure_pct, range_atr_multiplier=range_atr_multiplier,
            min_profit_multiplier=min_profit_multiplier, maker_fee=maker_fee, taker_fee=taker_fee,
            stop_loss_pct=stop_loss_pct, trailing_sl_trigger_pct=trailing_sl_trigger_pct,
            recenter_cooldown=recenter_cooldown, replacement_cooldown=replacement_cooldown,
            recenter_margin_pct=recenter_margin_pct, candle_seconds=candle_seconds,
            warmup=warmup, price_decimals=price_decimals, amount_decimals=amount_decimals,
            use_trend_filter=use_trend_filter, adx_trend_threshold=adx_trend_threshold,
            adx_range_threshold=adx_range_threshold, ema_fast=ema_fast, ema_slow=ema_slow,
            adx_period=adx_period,
        )
    finally:
        if quiet:
            logger.enable("grid")
            logger.enable("exchange")


def _regime_series(ohlcv, ema_fast, ema_slow, adx_period, trend_threshold, range_threshold):
    """Classify every candle using TrendFilter's own rule, vectorised.

    TrendFilter._evaluate_timeframe() recomputes EMA and ADX over a window on each
    call. Replaying it per candle would be O(n^2) for no benefit -- the indicators are
    running series, so computing them once and applying the same thresholds gives
    identical classifications far faster. The rule mirrors _evaluate_timeframe exactly:
    ADX above trend_threshold means trending (direction from the EMA cross), ADX below
    range_threshold means ranging, in between is uncertain.
    """
    import numpy as np

    from trend_filter import adx as calc_adx, ema as calc_ema

    fast = calc_ema(ohlcv["close"], ema_fast)
    slow = calc_ema(ohlcv["close"], ema_slow)
    adx_vals = calc_adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], adx_period)

    regimes = []
    for i in range(len(ohlcv)):
        a = adx_vals.iloc[i]
        a = 0.0 if (a is None or np.isnan(a)) else float(a)
        if i < ema_slow + adx_period:
            regimes.append("uncertain")
        elif a >= trend_threshold:
            regimes.append("uptrend" if fast.iloc[i] > slow.iloc[i] else "downtrend")
        elif a <= range_threshold:
            regimes.append("ranging")
        else:
            regimes.append("uncertain")
    return regimes


def _run(ohlcv, *, symbol, starting_balance, grid_count, capital_per_grid_pct,
         max_position_pct, max_exposure_pct, range_atr_multiplier, min_profit_multiplier,
         maker_fee, taker_fee, stop_loss_pct, trailing_sl_trigger_pct, recenter_cooldown,
         replacement_cooldown, recenter_margin_pct, candle_seconds, warmup,
         price_decimals, amount_decimals, use_trend_filter, adx_trend_threshold,
         adx_range_threshold, ema_fast, ema_slow, adx_period) -> BacktestResult:
    from trend_filter import atr as calc_atr

    if len(ohlcv) <= warmup + 10:
        raise ValueError(f"need more than {warmup + 10} candles, got {len(ohlcv)}")

    ex = SimulatedExchange(
        symbol=symbol, starting_balance=starting_balance,
        maker_fee=maker_fee, taker_fee=taker_fee,
        price_decimals=price_decimals, amount_decimals=amount_decimals,
    )
    clock = VirtualClock(start=0.0)

    atr_series = calc_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
    regimes = (
        _regime_series(ohlcv, ema_fast, ema_slow, adx_period,
                       adx_trend_threshold, adx_range_threshold)
        if use_trend_filter else None
    )

    seed = ohlcv.iloc[warmup]
    price = float(seed["close"])
    ex.price = price
    atr = float(atr_series.iloc[warmup])
    if pd.isna(atr) or atr <= 0:
        atr = price * 0.02

    result = BacktestResult(
        config={
            "grid_count": grid_count, "capital_per_grid_pct": capital_per_grid_pct,
            "max_position_pct": max_position_pct, "min_profit_multiplier": min_profit_multiplier,
            "range_atr_multiplier": range_atr_multiplier, "maker_fee": maker_fee,
            "stop_loss_pct": stop_loss_pct,
        },
        starting_balance=starting_balance,
    )

    with _clock_installed(clock):
        engine = GridEngine(
            ex, symbol,
            grid_lower=price - atr * range_atr_multiplier,
            grid_upper=price + atr * range_atr_multiplier,
            grid_count=grid_count,
            capital_per_grid_pct=capital_per_grid_pct,
            stop_loss_pct=stop_loss_pct,
            maker_fee_pct=maker_fee,
            taker_fee_pct=taker_fee,
            recenter_cooldown=recenter_cooldown,
            replacement_cooldown=replacement_cooldown,
            order_pacing_seconds=0.0,
            leverage=1,
            trailing_sl_trigger_pct=trailing_sl_trigger_pct,
            max_exposure_pct=max_exposure_pct,
            min_profit_multiplier=min_profit_multiplier,
        )
        engine.initialize(price, balance=starting_balance)
        engine.place_initial_orders(starting_balance)
        engine.active = True

        peak_equity = ex.equity
        in_position_candles = 0
        paused_candles = 0

        for i in range(warmup + 1, len(ohlcv)):
            row = ohlcv.iloc[i]
            o, h, l, c = (float(row["open"]), float(row["high"]),
                          float(row["low"]), float(row["close"]))

            clock.advance(candle_seconds)
            ex.step(o, h, l, c)

            balance = ex.equity

            # Mirror main.py's per-iteration bookkeeping.
            atr_now = float(atr_series.iloc[i])
            if not pd.isna(atr_now) and c > 0:
                engine.update_volatility(atr_now / c)

            long_qty = max(0.0, ex.position_qty)
            short_qty = max(0.0, -ex.position_qty)
            max_qty = (balance * max_position_pct) / c if c > 0 else 0.0
            engine.set_position_limit(
                long_position=long_qty, short_position=short_qty, max_position_qty=max_qty,
            )

            fills = engine.check_fills(balance)
            for f in fills:
                if f and f.get("completed_cycle"):
                    result.completed_cycles += 1
                    if f.get("profit", 0.0) > 0:
                        result.winning_cycles += 1

            # Stop-loss enforcement.
            if ex.position_qty > 0:
                engine.update_trailing_sl(c)
                sl = engine.get_stop_loss_price()
                if sl and l <= sl:
                    ex.price = sl
                    ex.close_position(symbol)
                    engine.reset_trailing()
                    result.stop_loss_hits += 1
            elif ex.position_qty < 0:
                engine.update_trailing_sl_short(c)
                sl = engine.get_short_stop_loss_price()
                if sl and h >= sl:
                    ex.price = sl
                    ex.close_position(symbol)
                    engine.reset_trailing()
                    result.stop_loss_hits += 1

            # Trend gating, mirroring main.py: a confirmed trend pauses the grid
            # (cancelling resting orders) but does NOT close the open position -- only
            # the stop-loss protects it from there.
            if regimes is not None:
                trending = regimes[i] in ("uptrend", "downtrend")
                if trending and engine.active:
                    engine.pause()
                elif not trending and not engine.active:
                    engine.active = True
                if not engine.active:
                    paused_candles += 1

            if engine.active:
                if engine.recenter(c, balance, margin_pct=recenter_margin_pct):
                    result.recenters += 1
                # Re-arm any level left unplaced (crossing, cooldown, min-notional).
                engine.place_initial_orders(balance)

            if ex.position_qty != 0:
                in_position_candles += 1
            eq = ex.equity
            result.equity_curve.append(eq)
            peak_equity = max(peak_equity, eq)
            if peak_equity > 0:
                dd = (peak_equity - eq) / peak_equity
                result.max_drawdown_pct = max(result.max_drawdown_pct, dd)

        # Flatten at the end so the result is realised, not marked-to-market.
        ex.price = float(ohlcv.iloc[-1]["close"])
        ex.close_position(symbol)

        result.candles = len(ohlcv) - warmup - 1
        result.ending_equity = ex.equity
        result.gross_realized = ex.realized_pnl
        result.fees_paid = ex.fees_paid
        result.net_pnl = ex.realized_pnl - ex.fees_paid
        result.engine_reported_pnl = engine.total_pnl - engine.total_fees
        result.fills = len(ex.fill_log)
        result.maker_fills = ex.maker_fills
        result.taker_fills = ex.taker_fills
        result.rejected_crossing = ex.rejected_crossing
        result.rejected_reduce_only = ex.rejected_reduce_only
        result.rejected_min_notional = ex.rejected_min_notional
        result.time_in_position_pct = in_position_candles / max(1, result.candles)
        result.time_paused_pct = paused_candles / max(1, result.candles)

    return result


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_ohlcv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def fetch_ohlcv(symbol: str = "DOGEUSDT", timeframe: str = "1h", days: int = 90,
                cache_dir: str | Path = "data") -> pd.DataFrame:
    """Download candles from Binance's public klines endpoint and cache them.

    Klines are public, so this needs no API credentials -- deliberately, so backtests
    can run without touching the trading account at all.
    """
    import ccxt

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{symbol.lower()}_{timeframe}_{days}d.csv"
    if cache.exists():
        logger.info("BACKTEST DATA | using cached {}", cache)
        return load_ohlcv(cache)

    client = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = client.milliseconds() - days * 86400 * 1000
    rows: list[list] = []
    cursor = since
    while True:
        batch = client.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        nxt = batch[-1][0] + 1
        if nxt <= cursor:
            break
        cursor = nxt
        _real_time.sleep(client.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(cache, index=False)
    logger.info("BACKTEST DATA | fetched {} candles -> {}", len(df), cache)
    return df
