from __future__ import annotations

import signal
import sys
import time
from datetime import datetime, timezone

import numpy as np
from loguru import logger

from config import settings
from logger import setup_logging
from exchange import Exchange
from grid import GridEngine, calculate_grid_range, calculate_dynamic_grid_count, validate_grid_spacing
from trend_filter import TrendFilter, atr as calc_atr
from risk import RiskManager
from state import StateManager
from telegram_notifier import TelegramNotifier
from trade_journal import TradeJournal
from event_journal import EventJournal
from pnl_tracker import PnLReconciler
from router import StrategyRouter
from signals import SignalGenerator
from trend_follower import TrendFollower


def trail_stop_fired(order: dict | None) -> bool:
    """Did the trailing stop actually trigger, or was it merely cancelled?

    A missing stop order is ambiguous: we cancel stops ourselves on every refresh, on
    pause(), and inside recenter(). Only the exchange's own status settles it. Anything
    that is not a completed fill -- cancelled, expired, unknown, unreachable -- must
    read as "did not fire", because wrongly latching the scale-out permanently strips
    the trailing leg from an open position (AUDIT #26).
    """
    return (order or {}).get("status") in ("closed", "filled")


def _install_strategy(engine: GridEngine, exchange: Exchange, events, notifier):
    """Return what the trading loop should drive.

    In the default 'grid' mode this is the engine itself, so behaviour is byte-identical
    to before the router existed. In 'router' mode the engine is wrapped alongside a
    trend follower and the router picks between them by regime -- it satisfies the same
    Strategy protocol and delegates everything else, so the loop below is unchanged
    either way.
    """
    if settings.strategy_mode != "router":
        return engine

    trend = TrendFollower(
        exchange, settings.symbol,
        capital_pct=settings.trend_capital_pct,
        stop_loss_pct=settings.stop_loss_pct,
        trailing_sl_trigger_pct=settings.trailing_sl_trigger_pct,
        atr_stop_multiplier=settings.trend_atr_stop_multiplier,
        leverage=settings.leverage,
        max_exposure_pct=settings.max_exposure_pct,
        min_hold_seconds=settings.trend_min_hold_seconds,
        event_journal=events,
        notifier=notifier,
    )
    logger.info(
        "STRATEGY ROUTER ENABLED | grid <-> trend | min_regime_hold={}s | handoff_grace={}s",
        settings.router_min_regime_seconds,
        settings.router_handoff_grace_seconds,
    )
    return StrategyRouter(
        strategies={"grid": engine, "trend": trend},
        default="grid",
        min_regime_seconds=settings.router_min_regime_seconds,
        handoff_grace_seconds=settings.router_handoff_grace_seconds,
        exchange=exchange,
        symbol=settings.symbol,
        notifier=notifier,
        event_journal=events,
    )


TIMEFRAME_MULTIPLIER = {
    "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
    "1h": 24, "4h": 6, "1d": 1,
}

def candles_for_lookback(timeframe: str, days: int) -> int:
    mult = TIMEFRAME_MULTIPLIER.get(timeframe, 24)
    return days * mult + 100


def build_scale_out_orders(
    side: str,
    qty: float,
    scale_out_pct: float,
    trail_price: float,
    hard_price: float,
    rounder=None,
    scale_out_done: bool = False,
    startup_trail_price: float = None,
) -> list[tuple[str, float, float]]:
    """Compute the stop-market orders for scale-out stop-loss protection.

    Returns a list of (kind, qty, price) tuples:
      - ("trail", ...) covers `scale_out_pct` of the position at the trailing level
      - ("hard", ...) covers the remainder at the static hard-stop level
    When the trailing level equals the hard level (no trailing protection yet) or
    the trailing leg has already fired (scale_out_done), a single full-position
    ("hard", ...) stop is returned instead. If the trailing level has not armed
    yet, `startup_trail_price` (an above-hard anchor, e.g. peak*(1-stop_loss_pct))
    may be passed to arm the split immediately.
    """
    scale = min(max(scale_out_pct, 0.0), 0.95)
    def _round(v: float) -> float:
        return float(rounder(v)) if rounder else float(v)
    qty = _round(qty)
    if qty <= 0:
        return []
    same_level = abs(trail_price - hard_price) <= 1e-9
    if same_level and startup_trail_price is not None and abs(startup_trail_price - hard_price) > 1e-9:
        trail_price = startup_trail_price
        same_level = False
    if scale_out_done or same_level:
        return [("hard", qty, hard_price)]
    trail_qty = _round(qty * scale)
    hard_qty = _round(qty - trail_qty)
    orders = []
    if trail_qty > 0:
        orders.append(("trail", trail_qty, trail_price))
    if hard_qty > 0:
        orders.append(("hard", hard_qty, hard_price))
    if not orders:
        orders.append(("hard", qty, hard_price))
    return orders


def get_total_position(exchange: Exchange, symbol: str) -> float:
    """Return total long position size (contracts) for the symbol."""
    try:
        positions = exchange.get_positions(symbol)
        total = 0.0
        for pos in positions:
            if pos.get("side") == "long":
                total += float(pos.get("contracts", 0) or 0)
        return total
    except Exception as e:
        logger.debug("Failed to fetch positions: {}", e)
        return 0.0


def get_position_breakdown(exchange: Exchange, symbol: str) -> tuple[float, float]:
    """Return (long_qty, short_qty) of open positions, normalizing the
    negative-contracts encoding of a short in one-way mode."""
    try:
        positions = exchange.get_positions(symbol)
        long_qty = 0.0
        short_qty = 0.0
        for pos in positions:
            side = pos.get("side", "")
            qty = float(pos.get("contracts", 0) or 0)
            if side == "long" and qty < 0:
                side, qty = "short", abs(qty)
            elif side == "short" and qty < 0:
                side, qty = "long", abs(qty)
            if side == "long":
                long_qty += qty
            elif side == "short":
                short_qty += qty
        return long_qty, short_qty
    except Exception as e:
        logger.debug("Failed to fetch positions: {}", e)
        return 0.0, 0.0


def get_short_position(exchange: Exchange, symbol: str) -> tuple[float, float]:
    """Return (short_qty, weighted_avg_entry_price) or (0.0, 0.0) when flat.
    Handles both one-way (side='short', qty>0) and negative-contracts encodings."""
    try:
        positions = exchange.get_positions(symbol)
        qty = 0.0
        notional = 0.0
        for pos in positions:
            side = pos.get("side", "")
            amt = float(pos.get("contracts", 0) or 0)
            entry = float(pos.get("entryPrice", 0) or 0)
            if side == "long" and amt < 0:
                side, amt = "short", abs(amt)
            if side == "short" and amt > 0:
                qty += amt
                notional += amt * entry
        if qty > 0:
            return qty, notional / qty
        return 0.0, 0.0
    except Exception as e:
        logger.debug("Failed to fetch short position: {}", e)
        return 0.0, 0.0


def get_net_position(exchange: Exchange, symbol: str) -> tuple[str, float]:
    """Return (side, qty) of the net open position: ('long', qty), ('short', qty),
    or ('', 0.0) when flat. Handles both one-way (side='short') and
    negative-contracts encodings of a short position.
    """
    long_qty, short_qty = get_position_breakdown(exchange, symbol)
    if long_qty > short_qty:
        return "long", long_qty - short_qty
    if short_qty > long_qty:
        return "short", short_qty - long_qty
    return "", 0.0


def get_position_details(exchange: Exchange, symbol: str) -> list[dict]:
    """Return list of position dicts with entry_price, qty, side, unrealized_pnl.

    Normalizes both standard one-way short encoding (side='short', qty>0) and
    negative-contracts encoding (side='long', qty<0) so the bot can consistently
    report and protect positions in either direction.

    unrealized_pnl passes through ccxt's unified 'unrealizedPnl' field -- Binance's
    own mark-price-based figure -- when the exchange provides it, else None (some
    mock/test exchanges omit it; see _position_unrealized_pnl for the fallback).
    """
    try:
        positions = exchange.get_positions(symbol)
        result = []
        for pos in positions:
            amt = float(pos.get("contracts", 0) or 0)
            entry = float(pos.get("entryPrice", 0) or 0)
            if amt == 0 or entry <= 0:
                continue
            side = pos.get("side", "")
            if side == "long" and amt < 0:
                side = "short"
                amt = abs(amt)
            elif side == "short" and amt < 0:
                side = "long"
                amt = abs(amt)
            if side not in {"long", "short"}:
                continue
            raw_upnl = pos.get("unrealizedPnl")
            try:
                upnl = float(raw_upnl) if raw_upnl is not None else None
            except (TypeError, ValueError):
                upnl = None
            result.append({
                "side": side,
                "entry_price": entry,
                "qty": amt,
                "unrealized_pnl": upnl,
            })
        return result
    except Exception:
        return []


def _position_unrealized_pnl(pos: dict, current_price: float) -> float:
    """Prefer the exchange's own mark-price-based unrealized_pnl (see
    get_position_details); only fall back to a last-price estimate when the
    exchange didn't provide one. Keeping the exchange's figure as the default
    avoids re-deriving numbers it already gives us (AUDIT.md follow-up to
    issues #7/#8) and sidesteps a latent sign bug the previous inline fallback
    had: it applied the long-side formula unconditionally regardless of side,
    which silently inverted the sign for any short position.
    """
    upnl = pos.get("unrealized_pnl")
    if upnl is not None:
        return upnl
    if pos["side"] == "short":
        return (pos["entry_price"] - current_price) * pos["qty"]
    return (current_price - pos["entry_price"]) * pos["qty"]


def _notify_status(
    notifier: TelegramNotifier, exchange: Exchange, symbol: str, price: float,
    pnl_reconciler: PnLReconciler | None = None,
) -> None:
    """Send position + balance to Telegram. Call only on significant events."""
    pos_details = get_position_details(exchange, symbol)
    for pos in pos_details:
        unrealized = _position_unrealized_pnl(pos, price)
        notifier.on_position_update(symbol, pos["side"], pos["entry_price"], pos["qty"], price, unrealized)
    balance_info = exchange.get_balance_info()
    equity = exchange.get_total_equity()
    exposure_pct = 0.0
    if equity > 0:
        exposure_usdt = sum(p["qty"] * price for p in pos_details)
        exposure_pct = exposure_usdt / equity
    total_pnl_verified = pnl_reconciler.net_realized_pnl if pnl_reconciler is not None else None
    notifier.on_balance_update(balance_info["free"], balance_info["used"], equity, exposure_pct, total_pnl_verified)


def daily_reset_check(
    risk: RiskManager, notifier: TelegramNotifier, exchange: Exchange, symbol: str,
    events: EventJournal = None, pnl_reconciler: PnLReconciler | None = None,
) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Roll the reconciler's daily bucket over first (before risk.reset_daily()) so
    # `completed_daily_pnl` below is yesterday's exchange-verified total, not the
    # grid's own drifting estimate -- see AUDIT.md "Daily PnL is still unreconciled".
    # Also a no-op-safe call on every non-rollover day.
    completed_daily_pnl = pnl_reconciler.rollover_daily(today) if pnl_reconciler is not None else risk.state.daily_realized_pnl
    if risk.state.last_reset_date != today:
        if risk.state.last_reset_date:
            balance = exchange.get_balance()
            notifier.on_daily_summary(
                completed_daily_pnl,
                risk.state.trades_today,
                balance,
            )
            if events:
                events.daily_reset(completed_daily_pnl, risk.state.trades_today, balance)
        risk.reset_daily()


def run_bot() -> None:
    setup_logging(settings.log_dir, "INFO")
    mode = "DEMO (testnet)" if settings.demo_mode else "LIVE"
    logger.info("=" * 50)
    logger.info("GRID BOT STARTING | mode={} | symbol={}", mode, settings.symbol)
    logger.info("=" * 50)

    try:
        settings.validate()
    except ValueError as e:
        logger.error("Invalid configuration: {}", e)
        return

    notifier = TelegramNotifier(
        settings.telegram_bot_token,
        settings.telegram_chat_id,
        settings.telegram_enabled,
    )

    events = EventJournal(settings.log_dir)

    try:
        config = settings.exchange_config
        exchange = Exchange(config, demo=settings.demo_mode)
    except Exception as e:
        logger.error("Failed to connect to exchange: {}", e)
        return

    exchange.set_leverage(settings.symbol, settings.leverage)

    logger.info("STARTUP CLEANUP | cancelling all orders and closing orphan positions...")
    cancelled = exchange.cancel_everything(settings.symbol)
    if cancelled:
        logger.warning("Cancelled {} leftover orders (limits + stops) from previous sessions", cancelled)
    closed = exchange.close_all_positions(settings.symbol)
    if closed:
        logger.warning("Closed {} orphan positions from previous sessions", closed)

    try:
        leftover = exchange.get_open_orders(settings.symbol)
    except Exception as e:
        logger.error("Cleanup verification failed ({}): cannot start on an uncertain book. Restart once the exchange write path recovers.", e)
        return
    if leftover:
        logger.error(
            "{} orders are still open after cleanup — the exchange write path is unreachable. "
            "NOT starting on a dirty book (avoids duplicate grid levels). Restart the bot once the exchange recovers.",
            len(leftover),
        )
        notifier.send(f"&#x1f6a8; <b>STARTUP ABORTED</b>\n{len(leftover)} stale orders could not be cancelled (exchange write path down). Book was left untouched.")
        return

    trend = TrendFilter(
        ema_fast=settings.ema_fast,
        ema_slow=settings.ema_slow,
        adx_period=settings.adx_period,
        trend_threshold=settings.adx_trend_threshold,
        range_threshold=settings.adx_range_threshold,
        check_interval=settings.trend_check_interval,
        confirmation_seconds=settings.trend_confirmation_seconds,
        flat_range_window=settings.flat_range_window,
        flat_range_pct=settings.flat_range_pct,
    )

    # Deliberately given no exchange handle -- it reports on regime changes and cannot
    # act on them. See signals.py.
    signal_gen = (
        SignalGenerator(
            symbol=settings.symbol,
            log_dir=settings.log_dir,
            notifier=notifier,
            event_journal=events,
            notify=settings.signals_notify,
        )
        if settings.signals_enabled
        else None
    )

    risk = RiskManager(
        stop_loss_pct=settings.stop_loss_pct,
        daily_loss_limit_pct=settings.daily_loss_limit_pct,
        max_drawdown_pct=settings.max_drawdown_pct,
        cooldown_seconds=settings.cooldown_seconds,
        max_exposure_pct=settings.max_exposure_pct,
        max_consecutive_losses=settings.max_consecutive_losses,
        max_recovery_count=settings.max_recovery_count,
        event_journal=events,
    )

    grid = None
    state_mgr = None

    state_mgr = StateManager(settings.state_dir, settings.symbol)
    journal = TradeJournal(settings.log_dir)
    saved_state = state_mgr.load()

    # PnL reconciler: reports cumulative PnL sourced from Binance's own income
    # ledger (realized PnL + commission + funding) rather than the grid engine's
    # internal per-level bookkeeping, so the number shown to the user always
    # agrees with the real account equity trajectory. Read-only — does not
    # affect order placement or fill handling.
    pnl_reconciler = PnLReconciler.from_dict(saved_state.get("pnl_reconciler") if saved_state else None)
    pnl_reconciler.sync(exchange, settings.symbol)
    logger.info(
        "PNL RECONCILER READY | net_realized_pnl={:.4f} USDT (realized={:.4f} commission={:.4f} funding={:.4f})",
        pnl_reconciler.net_realized_pnl, pnl_reconciler.realized_pnl,
        pnl_reconciler.commission, pnl_reconciler.funding_fee,
    )

    if saved_state and "grid" in saved_state:
        logger.info("Restoring saved grid state")
        balance = exchange.get_balance()
        risk.initialize(balance)

        grid = GridEngine(
            exchange, settings.symbol,
            grid_lower=saved_state["grid"]["grid_lower"],
            grid_upper=saved_state["grid"]["grid_upper"],
            grid_count=saved_state["grid"]["grid_count"],
            capital_per_grid_pct=settings.capital_per_grid_pct,
            stop_loss_pct=settings.stop_loss_pct,
            maker_fee_pct=settings.maker_fee_pct / 100,
            taker_fee_pct=settings.taker_fee_pct / 100,
            recenter_cooldown=settings.recenter_cooldown,
            replacement_cooldown=settings.replacement_cooldown,
            order_pacing_seconds=settings.order_pacing_seconds,
            capital_per_grid_usdt=settings.capital_per_grid_usdt,
            leverage=settings.leverage,
            trailing_sl_trigger_pct=settings.trailing_sl_trigger_pct,
            max_exposure_pct=settings.max_exposure_pct,
            min_profit_multiplier=settings.min_profit_multiplier,
            event_journal=events,
            notifier=notifier,
        )
        grid = _install_strategy(grid, exchange, events, notifier)
        grid.load_from_dict(saved_state["grid"], current_price=exchange.get_price(settings.symbol))

        if grid.state_corrupted:
            logger.warning("Deleting corrupt state file so next startup recalculates fresh bounds")
            state_mgr.delete()

        has_exchange_positions = any(
            float(p.get("contracts", 0) or 0) != 0
            for p in exchange.get_positions(settings.symbol)
        )

        if has_exchange_positions:
            grid.reconcile_state()
            grid.reconcile_positions()
        else:
            logger.info("No exchange positions — resetting stale grid levels to pending")
            for level in grid.levels:
                level.order_id = None
                if level.side == "sell" and level.status == "replaced":
                    level.side = "buy"
                    level.price = level.entry_price if level.entry_price else level.price
                    level.status = "pending"
                elif level.status != "pending":
                    level.status = "pending"
            logger.info("Placing fresh grid orders after cleanup")
            grid.place_initial_orders(exchange.get_balance())

        risk.load_from_dict(saved_state.get("risk", {}))
    else:
        logger.info("Calculating grid range from recent price action")
        ohlcv = exchange.get_ohlcv(settings.symbol, settings.grid_timeframe, limit=candles_for_lookback(settings.grid_timeframe, settings.range_lookback_days))
        current_price = exchange.get_price(settings.symbol)

        grid_lower, grid_upper = calculate_grid_range(
            ohlcv, current_price,
            lookback_days=settings.range_lookback_days,
            atr_multiplier=settings.range_atr_multiplier,
            timeframe=settings.grid_timeframe,
        )

        atr_series = calc_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
        current_atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else current_price * 0.02
        atr_pct = current_atr / current_price
        dynamic_count = calculate_dynamic_grid_count(atr_pct, settings.grid_count)
        if dynamic_count != settings.grid_count:
            logger.info("DYNAMIC GRID COUNT | ATR%={:.3f} | {} -> {} levels", atr_pct * 100, settings.grid_count, dynamic_count)

        dynamic_allocation = dynamic_count * settings.capital_per_grid_pct
        if dynamic_allocation > 0.5:
            logger.warning(
                "Dynamic grid allocation {:.0%} exceeds 50% limit — clamping grid_count",
                dynamic_allocation,
            )
            dynamic_count = min(int(0.5 / settings.capital_per_grid_pct), settings.grid_count)

        min_total_range = current_price * settings.range_min_spacing_pct * dynamic_count
        current_range = grid_upper - grid_lower
        if current_range < min_total_range:
            needed_half = min_total_range / 2 * 1.01
            grid_lower = current_price - needed_half
            grid_upper = current_price + needed_half
            logger.info(
                "GRID RANGE WIDENED for min spacing | new lower={} new upper={} (was {})",
                round(grid_lower, 8), round(grid_upper, 8), round(current_range, 8),
            )

        if not validate_grid_spacing(grid_lower, grid_upper, dynamic_count, settings.range_min_spacing_pct, current_price):
            logger.error("Grid spacing validation failed. Adjust grid_count or range parameters.")
            return

        balance = exchange.get_balance()
        risk.initialize(balance)

        grid = GridEngine(
            exchange, settings.symbol,
            grid_lower=grid_lower,
            grid_upper=grid_upper,
            grid_count=dynamic_count,
            capital_per_grid_pct=settings.capital_per_grid_pct,
            stop_loss_pct=settings.stop_loss_pct,
            maker_fee_pct=settings.maker_fee_pct / 100,
            taker_fee_pct=settings.taker_fee_pct / 100,
            recenter_cooldown=settings.recenter_cooldown,
            replacement_cooldown=settings.replacement_cooldown,
            order_pacing_seconds=settings.order_pacing_seconds,
            capital_per_grid_usdt=settings.capital_per_grid_usdt,
            leverage=settings.leverage,
            trailing_sl_trigger_pct=settings.trailing_sl_trigger_pct,
            max_exposure_pct=settings.max_exposure_pct,
            min_profit_multiplier=settings.min_profit_multiplier,
            event_journal=events,
            notifier=notifier,
        )
        grid = _install_strategy(grid, exchange, events, notifier)
        grid.initialize(current_price, balance)

    sl_orders: dict[str, dict] = {}
    _scale_out_done = False

    def _reset_sl():
        nonlocal sl_orders, _scale_out_done
        sl_orders = {}
        _scale_out_done = False

    def _desired_sl_orders(side: str, qty: float) -> list[tuple[str, float, float]]:
        """Return list of (kind, qty, price) stop-market orders for the open position.

        Scale-out design: the trailing stop covers sl_scale_out_pct of the position at the
        trailing level; the remainder is covered by a hard stop at the static stop-loss level.
        When the trailing level equals the hard level (no trailing protection yet), the
        startup anchor (peak*(1-stop_loss_pct)) is used so the split arms immediately.
        """
        if side == "long":
            trail_price = grid.get_stop_loss_price()
            # Ratcheted, not recomputed from grid_lower: a recenter must not push the
            # hard stop away from an open position (AUDIT #25).
            hard_price = grid.get_hard_stop_loss_price()
        else:
            trail_price = grid.get_short_stop_loss_price()
            hard_price = grid.get_short_hard_stop_loss_price()
        return build_scale_out_orders(
            side, qty, settings.sl_scale_out_pct, trail_price, hard_price,
            rounder=lambda q: float(exchange.exchange.amount_to_precision(settings.symbol, q)),
            scale_out_done=_scale_out_done,
            startup_trail_price=grid.get_scale_out_trail_price(side),
        )

    def _refresh_sl_stops(side: str, qty: float) -> None:
        nonlocal sl_orders
        exchange.cancel_all_stop_orders(settings.symbol)
        sl_orders = {}
        close_side = "sell" if side == "long" else "buy"
        for kind, oqty, oprice in _desired_sl_orders(side, qty):
            try:
                order = exchange.place_stop_market(settings.symbol, close_side, oqty, oprice)
                sl_orders[kind] = {"id": order["id"], "side": close_side, "qty": oqty, "price": oprice}
                logger.info(
                    "STOP-LOSS ORDER PLACED | kind={} side={} qty={} @ {}",
                    kind, close_side, oqty, oprice,
                )
            except Exception as e:
                logger.error("Failed to place {} stop-loss: {}", kind, e)

    def _detect_trail_fill() -> None:
        """Mark the scale-out done only when the trailing stop actually triggered.

        Order absence alone does not mean it fired -- we cancel stops ourselves on every
        refresh, on pause(), and inside recenter(). Observed live on 2026-08-12 at
        14:06: a recenter cancelled both legs, this read the missing id as a fire, and
        _scale_out_done latched permanently. The trailing leg was never re-placed and a
        7108 DOGE long spent the rest of its life on the hard stop alone.

        So confirm with the exchange: only a genuinely filled/closed order counts. A
        cancelled or unreadable one leaves the flag alone, and _refresh_sl_stops
        re-places the leg on the next pass (AUDIT #26).
        """
        nonlocal _scale_out_done
        if _scale_out_done or "trail" not in sl_orders:
            return
        if exchange.demo and not exchange.has_credentials:
            return
        trail_id = sl_orders["trail"]["id"]
        open_ids = {o.get("id") for o in exchange.get_stop_orders(settings.symbol)}
        if trail_id in open_ids:
            return

        try:
            order = exchange.fetch_order(trail_id, settings.symbol)
        except Exception as e:
            logger.debug("SCALE-OUT CHECK | could not fetch {} ({}) — assuming not fired", trail_id, e)
            order = None

        status = (order or {}).get("status")
        if trail_stop_fired(order):
            _scale_out_done = True
            logger.warning(
                "SCALE-OUT STOP FIRED | trailing leg filled at {} — remainder on hard stop only",
                sl_orders["trail"]["price"],
            )
            return

        logger.debug(
            "SCALE-OUT CHECK | trail stop {} gone with status={} (cancelled, not fired) "
            "— leg will be re-placed",
            trail_id, status,
        )
        sl_orders.pop("trail", None)

    # Refreshing a stop means cancel-then-place, which leaves the position unprotected
    # for the round trip. Rebuilding on every quantity change made that gap recur on
    # literally every partial fill. Stops are reduceOnly, so a stop LARGER than the
    # position is harmless (it closes whatever remains) -- only under-coverage is a
    # real exposure. So: always refresh when the stop no longer covers the position or
    # the trigger price moved; tolerate over-coverage until it drifts materially.
    SL_OVER_COVERAGE_TOLERANCE = 0.10

    def _sl_needs_update(side: str, qty: float) -> bool:
        if not sl_orders:
            return True
        desired = _desired_sl_orders(side, qty)
        if len(desired) != len(sl_orders):
            return True
        for kind, oqty, oprice in desired:
            cur = sl_orders.get(kind)
            if cur is None:
                return True
            if abs(cur["price"] - oprice) > max(oprice * 0.001, 1e-8):
                return True
            if cur["qty"] < oqty - max(1e-8, oqty * 1e-6):
                return True  # under-covered: the position outgrew its stop
            if cur["qty"] > oqty * (1 + SL_OVER_COVERAGE_TOLERANCE) + 1e-8:
                return True  # stale oversized stop, resize to keep sizing honest
        return False

    position_side, position_qty = get_net_position(exchange, settings.symbol)
    _last_side = position_side
    if position_side in ("long", "short"):
        try:
            start_price = exchange.get_price(settings.symbol)
            if position_side == "long":
                grid.update_trailing_sl(start_price)
            else:
                grid.update_trailing_sl_short(start_price)
            _refresh_sl_stops(position_side, position_qty)
            grid.log_sl_status(position_side)
        except Exception as e:
            logger.error("Failed to place stop-loss on startup: {}", e)

    price_now = exchange.get_price(settings.symbol)
    balance_info = exchange.get_balance_info()
    logger.info("Current price: {}", price_now)
    logger.info(
        "Balance: free={:.2f} USDT total={:.2f} USDT used={:.2f} USDT",
        balance_info["free"], balance_info["total"], balance_info["used"],
    )
    logger.info("Grid range: [{} - {}] | levels: {}", round(grid.grid_lower, 8), round(grid.grid_upper, 8), grid.grid_count)

    ohlcv_tf = exchange.get_ohlcv(settings.symbol, settings.trend_timeframe, limit=100)
    trend.update(ohlcv_tf, settings.trend_timeframe)
    try:
        ohlcv_fast = exchange.get_ohlcv(settings.symbol, settings.trend_timeframe_fast, limit=100)
        trend.add_timeframe(ohlcv_fast, settings.trend_timeframe_fast)
    except Exception:
        pass
    try:
        ohlcv_1d = exchange.get_ohlcv(settings.symbol, "1d", limit=100)
        trend.add_timeframe(ohlcv_1d, "1d")
    except Exception:
        pass

    if settings.force_trade_now or trend.is_ranging():
        grid.activate(exchange.get_balance())
        notifier.on_grid_start(settings.symbol, grid.grid_lower, grid.grid_upper, grid.grid_count)
        if settings.force_trade_now:
            logger.warning("Force trade mode enabled; bypassing trend gate and activating grid immediately.")
    else:
        if grid.active:
            grid.pause()
            _reset_sl()
        logger.info("Market is trending ({}), grid will wait for ranging conditions", trend.regime.value)

    logger.info("Bot running in {} mode. Press Ctrl+C to stop.", mode)
    logger.info("Poll interval: {}s | Trend check: {}s", settings.poll_interval, settings.trend_check_interval)

    notifier.on_startup_summary(
        symbol=settings.symbol,
        mode=mode,
        price=price_now,
        grid_lower=grid.grid_lower,
        grid_upper=grid.grid_upper,
        grid_count=grid.grid_count,
        balance_free=balance_info["free"],
        equity=exchange.get_total_equity(),
        leverage=settings.leverage,
        regime=trend.regime.value,
        adx=trend.adx_value,
        grid_active=grid.active,
        total_pnl_verified=pnl_reconciler.net_realized_pnl,
    )
    _notify_status(notifier, exchange, settings.symbol, price_now, pnl_reconciler)

    loop_count = 0
    consecutive_errors = 0
    last_analytics_fill_count = 0
    try:
        while True:
            try:
                loop_count += 1
                exchange.maybe_resync_time()
                daily_reset_check(risk, notifier, exchange, settings.symbol, events, pnl_reconciler)
                price = exchange.get_price(settings.symbol)

                if risk.is_in_recovery():
                    remaining = risk.recovery_cooldown_remaining()
                    if remaining > 0:
                        if loop_count % 10 == 0:
                            logger.info("RECOVERY COOLDOWN | {}s remaining", remaining)
                            events.recovery_event("cooldown", risk.state.recovery_count, cooldown_remaining=remaining)
                        time.sleep(settings.poll_interval)
                        continue
                    else:
                        logger.info("RECOVERY READY | recalculating grid around current price {}", price)
                        try:
                            ohlcv = exchange.get_ohlcv(settings.symbol, settings.grid_timeframe, limit=candles_for_lookback(settings.grid_timeframe, settings.range_lookback_days))
                            grid_lower, grid_upper = calculate_grid_range(
                                ohlcv, price,
                                lookback_days=settings.range_lookback_days,
                                atr_multiplier=settings.range_atr_multiplier,
                                timeframe=settings.grid_timeframe,
                            )

                            atr_series = calc_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
                            current_atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else price * 0.02
                            atr_pct = current_atr / price
                            dynamic_count = calculate_dynamic_grid_count(atr_pct, settings.grid_count)

                            min_total_range = price * settings.range_min_spacing_pct * dynamic_count
                            current_range = grid_upper - grid_lower
                            if current_range < min_total_range:
                                needed_half = min_total_range / 2 * 1.01
                                grid_lower = price - needed_half
                                grid_upper = price + needed_half

                            if not validate_grid_spacing(grid_lower, grid_upper, dynamic_count, settings.range_min_spacing_pct, price):
                                logger.error("Grid spacing validation failed during recovery — retrying next cycle")
                                time.sleep(settings.poll_interval)
                                continue

                            recovery_mult = risk.get_recovery_size_multiplier()
                            if recovery_mult == 0.0:
                                logger.error(
                                    "MAX RECOVERY REACHED ({}) — shutting down bot",
                                    risk.state.recovery_count,
                                )
                                notifier.on_kill_switch("Max recovery attempts reached — bot shutting down")
                                state_data = {
                                    "grid": grid.to_dict(),
                                    "risk": risk.to_dict(),
                                    "pnl_reconciler": pnl_reconciler.to_dict(),
                                    "last_update": datetime.now().isoformat(),
                                }
                                state_mgr.save(state_data)
                                return
                            adjusted_capital_pct = settings.capital_per_grid_pct * recovery_mult
                            logger.info(
                                "RECOVERY GRID | range=[{}-{}] | levels={} | sizing={:.0%}",
                                round(grid_lower, 8), round(grid_upper, 8), dynamic_count, recovery_mult,
                            )

                            old_fills = grid.total_fills
                            old_pnl = grid.total_pnl
                            old_fees = grid.total_fees
                            old_cycles = grid.total_completed_cycles
                            old_peak = grid._peak_price
                            old_grid_lower, old_grid_upper = grid.grid_lower, grid.grid_upper

                            grid = GridEngine(
                                exchange, settings.symbol,
                                grid_lower=grid_lower,
                                grid_upper=grid_upper,
                                grid_count=dynamic_count,
                                capital_per_grid_pct=adjusted_capital_pct,
                                stop_loss_pct=settings.stop_loss_pct,
                                maker_fee_pct=settings.maker_fee_pct / 100,
                                taker_fee_pct=settings.taker_fee_pct / 100,
                                recenter_cooldown=settings.recenter_cooldown,
                                replacement_cooldown=settings.replacement_cooldown,
                                order_pacing_seconds=settings.order_pacing_seconds,
                                capital_per_grid_usdt=0,
                                leverage=settings.leverage,
                                trailing_sl_trigger_pct=settings.trailing_sl_trigger_pct,
                                max_exposure_pct=settings.max_exposure_pct,
                                min_profit_multiplier=settings.min_profit_multiplier,
                                event_journal=events,
                                notifier=notifier,
                            )
                            grid = _install_strategy(grid, exchange, events, notifier)
                            grid.initialize(price, exchange.get_balance())
                            grid.total_fills = old_fills
                            grid.total_pnl = old_pnl
                            grid.total_fees = old_fees
                            grid.total_completed_cycles = old_cycles
                            grid._peak_price = old_peak
                            risk.exit_recovery()
                            grid.activate(exchange.get_balance())
                            notifier.on_grid_start(settings.symbol, grid.grid_lower, grid.grid_upper, grid.grid_count)
                            notifier.on_recovery_resume(recovery_mult)
                            notifier.on_grid_recalculated(settings.symbol, grid_lower, grid_upper, dynamic_count, "recovery")
                            events.grid_recalculated(
                                settings.symbol, old_grid_lower, old_grid_upper, grid_lower, grid_upper,
                                dynamic_count, "recovery", current_atr,
                            )
                            events.recovery_event("resume", risk.state.recovery_count, sizing_pct=recovery_mult)
                            _notify_status(notifier, exchange, settings.symbol, price, pnl_reconciler)
                            logger.info("Grid recovered and activated with {}% sizing", int(recovery_mult * 100))
                        except Exception as e:
                            logger.error("Recovery failed: {} — will retry next cycle", e)
                            time.sleep(settings.poll_interval)
                            continue

                if trend.time_to_check():
                    old_regime = trend.regime.value
                    ohlcv_tf = exchange.get_ohlcv(settings.symbol, settings.trend_timeframe, limit=100)
                    trend.update(ohlcv_tf, settings.trend_timeframe)
                    try:
                        ohlcv_fast = exchange.get_ohlcv(settings.symbol, settings.trend_timeframe_fast, limit=100)
                        trend.add_timeframe(ohlcv_fast, settings.trend_timeframe_fast)
                    except Exception:
                        pass
                    try:
                        ohlcv_1d = exchange.get_ohlcv(settings.symbol, "1d", limit=100)
                        trend.add_timeframe(ohlcv_1d, "1d")
                    except Exception:
                        pass

                    if trend.regime.value != old_regime:
                        events.trend_change(old_regime, trend.regime.value, trend.adx_value, settings.trend_timeframe)
                        notifier.on_trend_change(old_regime, trend.regime.value, trend.adx_value)

                    # Read-only: records the call and scores the previous one. Called
                    # unconditionally -- it does its own change detection, and the first
                    # call establishes a baseline rather than emitting a phantom signal.
                    if signal_gen is not None:
                        try:
                            signal_gen.observe(trend.regime.value, price, trend.adx_value)
                        except Exception as e:
                            logger.debug("SIGNAL | observe failed: {}", e)

                    atr_series = calc_atr(ohlcv_tf["high"], ohlcv_tf["low"], ohlcv_tf["close"], period=14)
                    current_atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else price * 0.02
                    grid.update_volatility(current_atr / price)
                    # In router mode this is what drives strategy selection; a bare
                    # GridEngine records it and carries on unchanged.
                    grid.update_regime(trend.regime.value)

                    if settings.strategy_mode == "router":
                        # The router owns pause/activate: a trend does not stop
                        # trading, it hands over to the trend follower.
                        logger.debug("Router mode: regime {} routed internally", trend.regime.value)
                    elif not settings.force_trade_now:
                        if trend.is_trending() and grid.active:
                            grid.pause()
                            _reset_sl()
                            notifier.on_trend_pause(trend.regime.value, trend.adx_value)
                        elif trend.is_ranging() and not grid.active:
                            grid.activate(exchange.get_balance())
                            notifier.on_grid_start(settings.symbol, grid.grid_lower, grid.grid_upper, grid.grid_count)
                            notifier.on_grid_resume()
                            _notify_status(notifier, exchange, settings.symbol, price, pnl_reconciler)
                    else:
                        logger.debug("Force trade mode active; grid remains enabled regardless of trend.")

                if (settings.strategy_mode != "router"
                        and not grid.active
                        and (settings.force_trade_now or trend.is_ranging())):
                    grid.activate(exchange.get_balance())

                if grid.active:
                    grid.update_orderbook()

                    if settings.recenter_enabled:
                        old_lower, old_upper = grid.grid_lower, grid.grid_upper
                        recentered = grid.recenter(price, exchange.get_balance(), settings.recenter_margin_pct)
                        if recentered:
                            logger.info("Grid recentered around current price")
                            notifier.on_recenter(old_lower, old_upper, grid.grid_lower, grid.grid_upper)
                            events.grid_recentered(settings.symbol, old_lower, old_upper, grid.grid_lower, grid.grid_upper)
                            _notify_status(notifier, exchange, settings.symbol, price, pnl_reconciler)

                    if price < grid.grid_lower:
                        total_pos = get_total_position(exchange, settings.symbol)
                        has_sells = any(l.side == "sell" and l.order_id for l in grid.levels)
                        if not has_sells and total_pos > 0:
                            logger.warning("GRID EXIT: price {} below lowest grid {} with no sell orders — entering recovery", price, grid.grid_lower)
                            events.grid_exit(settings.symbol, "price_below_grid", price, total_pos)
                            notifier.on_grid_exit(settings.symbol, "price below grid", price, total_pos)
                            grid.pause()
                            _reset_sl()
                            risk.trigger_kill_switch()
                        elif not has_sells:
                            short_qty, short_entry = get_short_position(exchange, settings.symbol)
                            if short_qty > 0 and short_entry > 0 and (short_entry < grid.grid_lower or short_entry > grid.grid_upper):
                                old_lower, old_upper = grid.grid_lower, grid.grid_upper
                                rebuilt = grid.recenter(short_entry, exchange.get_balance(), settings.recenter_margin_pct)
                                if rebuilt:
                                    logger.info(
                                        "GRID REBUILT (short): price {} below grid {} with short {} @ {} — grid recentered around short entry [{}-{}], SL stays active",
                                        price, old_lower, short_qty, short_entry, grid.grid_lower, grid.grid_upper,
                                    )
                                    events.grid_recentered(settings.symbol, old_lower, old_upper, grid.grid_lower, grid.grid_upper)
                                    notifier.on_recenter(old_lower, old_upper, grid.grid_lower, grid.grid_upper)
                                    _notify_status(notifier, exchange, settings.symbol, price, pnl_reconciler)

                    if price > grid.grid_upper:
                        total_pos = get_total_position(exchange, settings.symbol)
                        has_buys = any(l.side == "buy" and l.order_id for l in grid.levels)
                        if not has_buys and total_pos == 0:
                            logger.info("GRID EXIT: price {} above grid {} with no buy orders and no position — recentering", price, grid.grid_upper)
                            events.grid_exit(settings.symbol, "price_above_grid_no_position", price, total_pos)

                    balance = exchange.get_balance_cached()
                    equity = exchange.get_total_equity_cached()
                    exposure = grid.get_exposure_pct(equity)
                    # Sourced from the exchange's own positions (unrealized_pnl is
                    # Binance's real mark-price-based figure when available -- see
                    # get_position_details/_position_unrealized_pnl) rather than the
                    # grid's internal per-level guess, which can drift the same way
                    # the realized-PnL bookkeeping did (AUDIT.md issues #7/#8).
                    pos_details = get_position_details(exchange, settings.symbol)
                    unrealized = sum(_position_unrealized_pnl(p, price) for p in pos_details)
                    exchange.enforce_order_limit(settings.symbol, keep_count=grid.grid_count + 2, tracked_ids=grid.get_tracked_order_ids())
                    fills = grid.check_fills(balance)
                    if fills:
                        # Pull Binance's actual income ledger once per batch of fills so the
                        # PnL figures below reflect the exchange's own accounting rather than
                        # the grid engine's internal per-level estimate.
                        pnl_reconciler.sync(exchange, settings.symbol)
                    for fill in fills:
                        profit = fill["profit"]
                        fee = fill["fee"]
                        if fill["completed_cycle"]:
                            risk.record_trade(profit)
                        notifier.on_fill(
                            fill["side"], fill["price"], profit, grid.total_fills, pnl_reconciler.daily_net_pnl,
                            total_pnl_verified=pnl_reconciler.net_realized_pnl,
                        )
                        events.fill(
                            symbol=settings.symbol,
                            side=fill["side"],
                            price=fill["price"],
                            qty=fill["quantity"],
                            fee=fee,
                            cycle_pnl=profit,
                            completed_cycle=fill["completed_cycle"],
                            fill_number=grid.total_fills,
                            balance=balance,
                            equity=equity,
                            exposure_pct=exposure,
                            daily_pnl=pnl_reconciler.daily_net_pnl,
                            regime=trend.regime.value,
                        )
                        journal.record(
                            symbol=settings.symbol,
                            side=fill["side"],
                            price=fill["price"],
                            quantity=fill["quantity"],
                            grid_spacing=grid.grid_spacing,
                            fill_number=grid.total_fills,
                            cumulative_pnl=pnl_reconciler.net_realized_pnl,
                            daily_pnl=pnl_reconciler.daily_net_pnl,
                            trades_today=risk.state.trades_today,
                            regime=trend.regime.value,
                            fee=fee,
                            completed_cycle=fill["completed_cycle"],
                            cycle_pnl=profit,
                            regime_adx=trend.adx_value,
                            balance=balance,
                            equity=equity,
                            exposure_pct=exposure,
                            unrealized_pnl=unrealized,
                        )
                    if fills:
                        _notify_status(notifier, exchange, settings.symbol, price, pnl_reconciler)

                    risk.update_unrealized(unrealized)

                    for pos in pos_details:
                        pos_unrealized = _position_unrealized_pnl(pos, price)
                        events.position_snapshot(
                            symbol=settings.symbol, side=pos["side"],
                            entry_price=pos["entry_price"], qty=pos["qty"],
                            current_price=price, unrealized_pnl=pos_unrealized,
                        )

                    balance_info = exchange.get_balance_info_cached()
                    events.balance_snapshot(
                        free=balance_info["free"], used=balance_info["used"],
                        total_equity=equity, exposure_pct=exposure,
                    )

                    events.exposure_update(exposure_pct=exposure, equity=equity)

                    long_pos, short_pos = get_position_breakdown(exchange, settings.symbol)
                    if long_pos > short_pos:
                        position_side, position_qty = "long", long_pos - short_pos
                    elif short_pos > long_pos:
                        position_side, position_qty = "short", short_pos - long_pos
                    else:
                        position_side, position_qty = "", 0.0
                    max_pos_qty = equity * settings.max_position_pct / price if price > 0 else 0
                    grid.set_position_limit(long_pos, short_pos, max_pos_qty)

                    if position_side != _last_side:
                        _last_side = position_side
                        grid.reset_trailing()
                        if _scale_out_done:
                            _scale_out_done = False
                            sl_orders = {}

                    if position_side == "long":
                        grid.update_trailing_sl(price)
                        _detect_trail_fill()
                        if _sl_needs_update("long", position_qty):
                            try:
                                _refresh_sl_stops("long", position_qty)
                                grid.log_sl_status("long")
                            except Exception as e:
                                logger.error("Failed to place/update stop-loss: {}", e)
                    elif position_side == "short":
                        grid.update_trailing_sl_short(price)
                        _detect_trail_fill()
                        if _sl_needs_update("short", position_qty):
                            try:
                                _refresh_sl_stops("short", position_qty)
                                grid.log_sl_status("short")
                            except Exception as e:
                                logger.error("Failed to place/update stop-loss: {}", e)
                    else:
                        if sl_orders:
                            exchange.cancel_all_stop_orders(settings.symbol)
                            _reset_sl()

                    was_in_recovery = risk.is_in_recovery()
                    # grid_stop_loss_price must match position_side: get_stop_loss_price()
                    # is the long-side floor, get_short_stop_loss_price() the short-side
                    # ceiling. Passing the long floor unconditionally meant this backstop
                    # was silently blind whenever the grid held a short (price rising into
                    # danger never breaches a floor computed for the opposite direction).
                    # When flat there is no position to protect, so skip the check (0
                    # short-circuits it in risk.py) rather than risk a false kill switch
                    # off a stale/irrelevant long-side band.
                    if position_side == "short":
                        grid_sl_price = grid.get_short_stop_loss_price()
                    elif position_side == "long":
                        grid_sl_price = grid.get_stop_loss_price()
                    else:
                        grid_sl_price = 0.0
                    # daily_realized_pnl uses the reconciler's exchange-verified figure so
                    # the kill switch evaluates against the account's real daily P&L, not
                    # the grid's per-level estimate (see AUDIT.md "Daily PnL is still
                    # unreconciled" -- risk.state.daily_realized_pnl itself is left alone,
                    # it still drives consecutive_losses via record_trade()).
                    is_safe, is_fatal = risk.check_all(
                        equity, grid_sl_price, price, exposure, side=position_side or "long",
                        daily_realized_pnl=pnl_reconciler.daily_net_pnl,
                    )
                    if not is_safe and is_fatal:
                        if not was_in_recovery:
                            grid.emergency_stop()
                            _reset_sl()
                            notifier.on_kill_switch("Risk limit breached")
                            events.recovery_event("start", risk.state.recovery_count)
                            logger.warning("Entering recovery mode — will wait {}s then recalculate grid", settings.cooldown_seconds)
                            notifier.on_recovery_start(settings.cooldown_seconds, risk.state.recovery_count)
                            _notify_status(notifier, exchange, settings.symbol, price, pnl_reconciler)
                        state_data = {
                            "grid": grid.to_dict(),
                            "risk": risk.to_dict(),
                            "pnl_reconciler": pnl_reconciler.to_dict(),
                            "last_update": datetime.now().isoformat(),
                        }
                        state_mgr.save(state_data)
                        time.sleep(settings.poll_interval)
                        continue
                    elif not is_safe:
                        logger.warning("EXPOSURE WARNING | grid continues but new orders blocked")

                if not grid.active:
                    equity = exchange.get_total_equity()
                    balance = exchange.get_balance()

                # Periodic fallback sync so funding-fee settlements (which happen on a
                # schedule, independent of any grid fill) still get picked up promptly.
                if loop_count % 60 == 0:
                    pnl_reconciler.sync(exchange, settings.symbol)

                state_data = {
                    "grid": grid.to_dict(),
                    "risk": risk.to_dict(),
                    "pnl_reconciler": pnl_reconciler.to_dict(),
                    "last_update": datetime.now().isoformat(),
                }
                state_mgr.save(state_data)

                logger.info(
                    "PRICE={} | fills={} | gross={:.2f} fees={:.2f} net={:.2f} | verified_net={:.2f} verified_daily={:.2f} | balance_free={:.2f} total_equity={:.2f} | grid={} | regime={} | spread={:.4f}%",
                    price, grid.total_fills, grid.total_pnl, grid.total_fees,
                    grid.total_pnl - grid.total_fees,
                    pnl_reconciler.net_realized_pnl, pnl_reconciler.daily_net_pnl,
                    balance, equity,
                    "ON" if grid.active else "OFF", trend.regime.value,
                    grid._last_orderbook.get("spread_pct", 0) * 100,
                )

                if grid.active and grid.total_fills > 0 and grid.total_fills % 10 == 0 and grid.total_fills != last_analytics_fill_count:
                    grid.log_analytics(balance)
                    last_analytics_fill_count = grid.total_fills

                consecutive_errors = 0
                time.sleep(settings.poll_interval)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                consecutive_errors += 1
                logger.error("Loop error (consecutive={}): {}", consecutive_errors, e)
                if "-1021" in str(e):
                    exchange._sync_time()
                    time.sleep(2)
                elif consecutive_errors >= 3:
                    logger.warning("Multiple consecutive errors — attempting reconnection")
                    if exchange.reconnect():
                        consecutive_errors = 0
                        try:
                            grid.reconcile_state()
                            grid.reconcile_positions()
                            logger.info("State reconciled after reconnection")
                        except Exception as reconcile_err:
                            logger.error("Reconciliation failed after reconnect: {}", reconcile_err)
                    else:
                        logger.error("Reconnection failed — saving state and shutting down")
                        try:
                            state_data = {
                                "grid": grid.to_dict() if grid is not None else {},
                                "risk": risk.to_dict() if risk is not None else {},
                                "pnl_reconciler": pnl_reconciler.to_dict(),
                                "last_update": datetime.now().isoformat(),
                            }
                            state_mgr.save(state_data)
                        except Exception:
                            pass
                        return
                else:
                    time.sleep(10)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        try:
            if grid is not None:
                grid.emergency_stop()
                _reset_sl()
                if settings.close_on_exit:
                    closed = exchange.close_all_positions(settings.symbol)
                    if closed:
                        logger.warning("Closed {} open positions on shutdown", closed)
        except Exception as e:
            logger.error("Error during shutdown cleanup: {}", e)
        try:
            state_data = {
                "grid": grid.to_dict() if grid is not None else {},
                "risk": risk.to_dict() if risk is not None else {},
                "pnl_reconciler": pnl_reconciler.to_dict(),
                "last_update": datetime.now().isoformat(),
            }
            if state_mgr is not None:
                state_mgr.save(state_data)
        except Exception as e:
            logger.error("Failed to save state on shutdown: {}", e)
        if signal_gen is not None:
            try:
                signal_gen.log_accuracy()
            except Exception as e:
                logger.debug("SIGNAL | accuracy report failed: {}", e)
        notifier.close()
        logger.info("Bot stopped. State saved.")


def _signal_handler(signum: int, frame) -> None:
    logger.info("Received signal {} — shutting down...", signum)
    raise KeyboardInterrupt()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _signal_handler)
    run_bot()
