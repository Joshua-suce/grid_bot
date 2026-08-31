from __future__ import annotations

import signal
import sys
import time
from datetime import datetime, timezone

from typing import NoReturn
import ccxt
import numpy as np
from loguru import logger

from config import settings
from logger import setup_logging
from exchange import Exchange
from grid import (GridEngine, MIN_NOTIONAL_USDT, calculate_grid_range,
                  calculate_dynamic_grid_count, validate_grid_spacing)
from trend_filter import TrendFilter, atr as calc_atr
from exposure_registry import ExposureRegistry
from risk import RiskManager
from state import StateManager
from telegram_notifier import TelegramNotifier
from trade_journal import TradeJournal
from event_journal import EventJournal
from pnl_tracker import BOOTSTRAP_LOOKBACK_DAYS, PnLReconciler
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
        capital_usdt=settings.trend_capital_usdt,
        stop_loss_pct=settings.stop_loss_pct,
        trailing_sl_trigger_pct=settings.trailing_sl_trigger_pct,
        atr_stop_multiplier=settings.trend_atr_stop_multiplier,
        trail_atr_multiplier=settings.trend_trail_atr_multiplier,
        take_profit_r=settings.trend_take_profit_r,
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


# Exceptions that mean "this program is wrong", as opposed to "the exchange or the
# network misbehaved". The loop's recovery path (reconnect, resync time, retry) can do
# nothing about these, and treating them as connection trouble is how a one-line defect
# hid behind 27 minutes of RECONNECTED messages -- see AUDIT #31.
BUG_ERRORS = (
    AttributeError, TypeError, NameError, IndexError,
    UnboundLocalError, ZeroDivisionError, AssertionError,
)

TIMEFRAME_MULTIPLIER = {
    "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
    "1h": 24, "4h": 6, "1d": 1,
}

# How old the income ledger may get before the daily-loss kill switch is treated as
# flying blind. Generous on purpose: the periodic fallback sync runs every 60 loops, so
# this only trips on a genuinely broken feed, not a slow one. It self-clears on the next
# successful sync, because set_position_limit rebuilds the side blocks each iteration.
PNL_STALE_SECONDS = 1800.0

def candles_for_lookback(timeframe: str, days: int) -> int:
    mult = TIMEFRAME_MULTIPLIER.get(timeframe, 24)
    return days * mult + 100


def sleep_until_next_poll(started: float, interval: float) -> float:
    """Sleep the REMAINDER of the poll interval, not the whole of it. Returns seconds slept.

    `time.sleep(poll_interval)` at the end of the iteration makes the real cadence
    `work + interval`, so the configured number is a floor nobody ever hits and the drift
    grows silently with every call added to the loop. Measured 2026-08-17: seven REST
    round trips at ~400ms each, and a configured 10s poll running at 13-17s between
    PRICE= lines -- the ladder swept about two thirds as often as configured.

    This does not by itself win fills; a fill is detected within one sweep either way, and
    the crossings a grid can capture are set by price path over spacing, not by polling.
    What it buys is reaction time on the things that are latency-sensitive: the stop-loss
    refresh, the risk kill switches, and the awaiting-counter release check all run once
    per iteration.
    """
    elapsed = time.monotonic() - started
    remaining = max(0.0, interval - elapsed)
    if remaining <= 0.0:
        logger.debug(
            "POLL OVERRUN | iteration took {:.1f}s against a {}s interval — the loop is "
            "the bottleneck, not the sleep", elapsed, interval,
        )
    time.sleep(remaining)
    return remaining


def _stop_price_of(order: dict) -> float:
    """Trigger price of a stop order, however this ccxt version chose to spell it."""
    for key in ("triggerPrice", "stopPrice"):
        value = order.get(key)
        if value:
            return float(value)
    info = order.get("info") or {}
    for key in ("stopPrice", "triggerPrice"):
        value = info.get(key)
        if value:
            return float(value)
    return 0.0


def _stop_qty_of(order: dict) -> float:
    """Quantity a stop order would close. `remaining` is preferred over `amount` so a
    partially-filled stop is not counted as still covering the whole position."""
    for key in ("remaining", "amount"):
        value = order.get(key)
        if value:
            return float(value)
    info = order.get("info") or {}
    for key in ("origQty", "quantity"):
        value = info.get(key)
        if value:
            return float(value)
    return 0.0


def _is_immediately_triggering_rejection(e: Exception) -> bool:
    """True if the exchange refused a stop because its trigger was already crossed
    by the current price (Binance -2021), not for any other reason.

    Distinguishing this matters because it is the one placement failure retrying
    the SAME request cannot fix -- the price that made it invalid does not change
    between attempts, only between polls. Every other rejection (a bad quantity,
    a connectivity blip) is at least plausibly retryable as-is; this one needs a
    different price before a retry is worth making at all.
    """
    return "-2021" in str(e)


def _reprice_past_current(exchange, symbol: str, close_side: str, current_price: float) -> float:
    """A trigger on the correct side of `current_price`, close to it but not through
    it -- for a stop the exchange just rejected as already crossed (AUDIT #167).

    `close_side` is the CLOSE order's side, not the position's: "sell" closes a
    long with a stop that fires as price falls, so the reprice must land strictly
    BELOW current price; "buy" closes a short and must land strictly ABOVE. The
    0.1% buffer is arbitrary but small relative to a 2%-class stop_loss_pct -- it
    exists only to survive the round trip back to the exchange, not to change how
    far the stop sits from the position.
    """
    nudge = current_price * 0.001
    direction = -1 if close_side == "sell" else 1
    raw = current_price + direction * nudge
    return float(exchange.exchange.price_to_precision(symbol, raw))


def reconcile_stop_orders(
    exchange,
    symbol: str,
    close_side: str,
    desired: list[tuple[str, float, float]],
    live: list[dict],
    over_coverage_tolerance: float = float("inf"),
    price_tolerance_pct: float = 1e-4,
) -> tuple[dict[str, dict], float, float]:
    """Bring the exchange's stop book in line with `desired`, touching only what differs.

    Returns `(kept, covered_qty, desired_qty)` where `kept` maps leg kind -> order record.

    This replaces a cancel-everything-then-place-everything refresh. That version tore
    down an unchanged hard stop every time the trailing leg ratcheted, opening two extra
    unprotected round trips on the exact code path behind the -50.49 day. Binance has no
    atomic replace for stop orders, so the only way to shrink that window is to stop
    opening it when nothing changed (AUDIT #54).

    Live orders are matched to desired legs by trigger price and quantity. Whatever the
    exchange holds that no desired leg claims is cancelled -- that covers both stale legs
    and strays from an earlier crash.

    `over_coverage_tolerance` bounds how much LARGER than the desired leg a live order
    may be and still count as that leg. Unbounded (the default, and the only behaviour
    this had) it livelocks against the caller: _sl_needs_update asks for a refresh once a
    leg exceeds its desired size by more than SL_OVER_COVERAGE_TOLERANCE, this function
    then matches the oversized leg to the smaller desire and keeps it, and the next poll
    asks again. Observed for the whole of 2026-08-17: stops armed at 4445/4446 for an
    8891 short at 06:49 were still 4445/4446 at 10:31 with the position down to 5331 --
    a "50%" scale-out that was really 83/83, re-examined every two minutes for four hours
    and never once resized.

    `price_tolerance_pct` bounds how far a live order's trigger may sit from the desired
    one and still count as that leg. It must be kept at least as wide as the drift the
    caller tolerates, or the pair churns: _sl_needs_update leaves a trail leg alone until
    its trigger drifts 0.1%, but the 120-second trust-but-verify pass calls this function
    regardless of belief, and at the old hardcoded 0.01% every ratchet larger than a
    hundredth of a percent failed to match and was cancelled and re-placed. That, not the
    quantity tolerance, is what put ~14 stop cancels an hour in the 2026-08-17 12:55
    session: eight of the eleven trail re-placements between 13:07 and 15:51 moved the
    trigger less than 0.1%, and three of those moved it less than 0.03%.

    A kept leg records the trigger price the exchange ACTUALLY holds, not the one that was
    desired when it was matched. Recording the desire made the drift invisible to
    _sl_needs_update -- its comparison was then desired-against-desired, always zero -- so
    the only thing that ever re-placed a drifting leg was the periodic verify. With the
    real price recorded, drift accumulates against the caller's 0.1% test and the leg is
    re-placed when it has genuinely gone stale, not on a timer.
    """
    unmatched = list(live)
    kept: dict[str, dict] = {}
    for kind, oqty, oprice in desired:
        oversize_cap = (
            float("inf") if over_coverage_tolerance == float("inf")
            else oqty * (1.0 + over_coverage_tolerance) + max(1e-8, oqty * 1e-6)
        )
        for order in unmatched:
            if (abs(_stop_price_of(order) - oprice) <= max(oprice * price_tolerance_pct, 1e-9)
                    and oqty - max(1e-8, oqty * 1e-6) <= _stop_qty_of(order) <= oversize_cap):
                unmatched.remove(order)
                kept[kind] = {"id": order.get("id"), "side": close_side,
                              "qty": _stop_qty_of(order), "price": _stop_price_of(order)}
                break

    for order in unmatched:
        order_id = order.get("id")
        # cancel_STOP_order, not cancel_order: these are algo orders, and the ordinary
        # cancel endpoint answers "unknown order" for an algo id, which ccxt raises as
        # OrderNotFound and cancel_order reports as a successful cancel. Every stale leg
        # this loop retired therefore survived, silently (AUDIT #74).
        if order_id and not exchange.cancel_stop_order(order_id, symbol):
            logger.warning("STOP REFRESH | stale stop {} not confirmed cancelled", order_id)

    for kind, oqty, oprice in desired:
        if kind in kept:
            continue
        try:
            placed = exchange.place_stop_market(
                symbol, close_side, oqty, oprice,
                purpose="stop_trail" if kind == "trail" else "stop_hard",
            )
            kept[kind] = {"id": placed["id"], "side": close_side, "qty": oqty, "price": oprice}
            logger.info(
                "STOP-LOSS ORDER PLACED | kind={} side={} qty={} @ {}",
                kind, close_side, oqty, oprice,
            )
        except Exception as e:
            if not _is_immediately_triggering_rejection(e):
                logger.error("Failed to place {} stop-loss: {}", kind, e)
                continue
            # AUDIT #167. Observed live on 2026-08-19 16:30:26: a 6307-unit short
            # went from a working stop to none at all because the desired trigger
            # had gone stale (poll latency, a fast move) by the time it reached
            # the exchange. That left it at 0% coverage for 77 seconds -- caught
            # by #54's under-protected check and self-healed on the NEXT refresh,
            # but only because nothing worse happened in that window. Retrying the
            # identical request here would fail identically; repricing off a fresh
            # read and retrying once, right now, is the difference between a real
            # gap and no gap at all in the case where it does matter.
            try:
                current_price = exchange.get_price(symbol)
                repriced = _reprice_past_current(exchange, symbol, close_side, current_price)
                placed = exchange.place_stop_market(
                    symbol, close_side, oqty, repriced,
                    purpose="stop_trail" if kind == "trail" else "stop_hard",
                )
                kept[kind] = {"id": placed["id"], "side": close_side, "qty": oqty, "price": repriced}
                logger.warning(
                    "STOP-LOSS REPRICED AND PLACED | kind={} side={} qty={} @ {} "
                    "(desired {} had already been crossed)",
                    kind, close_side, oqty, repriced, oprice,
                )
            except Exception as e2:
                logger.error(
                    "Failed to place {} stop-loss even after repricing past current "
                    "price: {}", kind, e2,
                )

    # Coverage is a QUANTITY question, not a boolean one. The scale-out splits the
    # position across a trail leg and a hard leg, so the previous `bool(sl_orders)` test
    # called a position covered when one leg placed and the other did not -- half the
    # position naked, reported as protected (AUDIT #54).
    return kept, sum(o["qty"] for o in kept.values()), sum(q for _, q, _ in desired)


# build_scale_out_orders is reached from _sl_needs_update's predicate path, so it runs
# once per poll for as long as a position is open. Anything it logs unconditionally is a
# heartbeat rather than an event: a 7 DOGE position on 2026-08-17 put 64 identical
# STOP-LOSS NOT SPLIT lines into the log between 13:50:45 and 14:06:21, which is how a
# genuine one-off gets buried. Report the state when it CHANGES, and stay quiet while it
# simply goes on being true.
_last_stop_sizing_note: str | None = None


def _note_stop_sizing(level: str, message: str, *args) -> None:
    """Log at `level` when this note differs from the last one, at DEBUG while it repeats."""
    global _last_stop_sizing_note
    rendered = message.format(*args)
    if rendered == _last_stop_sizing_note:
        logger.debug(rendered)
        return
    _last_stop_sizing_note = rendered
    logger.log(level, rendered)


def _clear_stop_sizing_note() -> None:
    """A normal refresh re-arms the notes so the next abnormal one is heard at full volume."""
    global _last_stop_sizing_note
    _last_stop_sizing_note = None


def build_scale_out_orders(
    side: str,
    qty: float,
    scale_out_pct: float,
    trail_price: float | None,
    hard_price: float | None,
    rounder=None,
    scale_out_done: bool = False,
    startup_trail_price: float = None,
    min_notional: float = 0.0,
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
    if trail_price is None or hard_price is None:
        # The strategy has no stop to offer for this side -- it holds nothing there.
        # Reachable when the exchange still reports a position the live strategy has
        # already closed, or right after a handoff: asking a flat trend follower for a
        # short stop returns None, and the arithmetic below raised TypeError in the
        # loop (AUDIT #31). No stop is better than a crash; the next iteration re-reads
        # the position and places one if it is really there.
        _note_stop_sizing(
            "WARNING",
            "STOP-LOSS | no {} stop available from the live strategy (trail={} hard={}) "
            "-- skipping this refresh", side, trail_price, hard_price,
        )
        return []

    scale = min(max(scale_out_pct, 0.0), 0.95)
    def _round(v: float) -> float:
        """Never hand the rounder a non-positive amount.

        The live rounder is ccxt's amount_to_precision, which RAISES on anything that
        does not survive the amount step -- "amount of DOGE/USDT:USDT must be greater
        than minimum amount precision of 1" -- rather than returning 0. Every test here
        passed a plain rounding lambda that returned 0 happily, so a zero-sized leg was
        harmless in the suite and fatal in production (AUDIT #109).
        """
        if v <= 0:
            return 0.0
        try:
            return float(rounder(v)) if rounder else float(v)
        except Exception:
            # Below the exchange's amount step. That is "no order", not a crash: the
            # min-notional guards below turn it into an honest empty/partial answer,
            # and _refresh_sl_stops reports the position uncovered.
            return 0.0
    qty = _round(qty)
    if qty <= 0:
        return []

    # A position can be too small to protect AT ALL, not merely too small to split. Every
    # remaining branch places at most one full-size stop at hard_price, so if that order
    # is under the exchange's floor then no stop this function could return would be
    # accepted -- the split guard below would just hand -4164 a differently-shaped order.
    #
    # AUDIT #89 stopped the split from deadlocking the grid and the same deadlock simply
    # moved one step down. 2026-08-17 13:50:37, fill #46 left LONG 7.0 DOGE and 13:50:48
    # placed SELL 7.0 @ 0.06713568 -- 0.47 USDT of notional. Testnet took it; the live
    # venue answers -4164, coverage comes back short of desired, _refresh_sl_stops
    # returns False and the caller blocks new exposure. The bot then sits idle on a
    # position worth less than half a dollar, unable to trade the very remainder that
    # would clear the condition.
    #
    # So say so and return nothing: `desired` empty is the one answer _refresh_sl_stops
    # already treats as "covered", which keeps the ladder working. The exposure this
    # gives up on is bounded by the floor itself -- under 5 USDT of notional against a
    # ~4900 USDT account -- and the alternative is not "protected", it is "blocked AND
    # unprotected". Stops already on the book are deliberately left alone: they are
    # reduceOnly, so an oversized survivor from the larger position still closes this
    # remainder, and the next fill that lifts the position back over the floor rebuilds
    # the legs properly through reconcile_stop_orders.
    if min_notional > 0 and qty * hard_price < min_notional:
        _note_stop_sizing(
            "WARNING",
            "STOP-LOSS UNPROTECTABLE | {} {} is {:.2f} USDT of notional, under the {} USDT "
            "exchange minimum — no stop of any size would be accepted for it. Leaving the "
            "grid free to trade the remainder out rather than blocking on a position too "
            "small to protect.",
            side, qty, qty * hard_price, min_notional,
        )
        return []

    same_level = abs(trail_price - hard_price) <= 1e-9
    if same_level and startup_trail_price is not None and abs(startup_trail_price - hard_price) > 1e-9:
        trail_price = startup_trail_price
        same_level = False
    # scale == 0 means "no trailing leg", which is the same shape as a leg that has
    # already fired. Falling through instead computed _round(qty * 0) and handed the
    # exchange rounder a zero -- the 2026-08-18 06:26 failure, which left a SHORT 1785
    # position with no stop for 21 minutes while the sell side sat blocked.
    if scale_out_done or same_level or scale <= 0:
        return [("hard", qty, hard_price)]
    trail_qty = _round(qty * scale)
    if trail_qty <= 0:
        # The split rounded the trailing leg out of existence. One full-size stop is
        # strictly better protection than one leg plus nothing.
        return [("hard", qty, hard_price)]
    hard_qty = _round(qty - trail_qty)

    # Splitting a small position produces two legs the exchange will not accept.
    # Binance rejects anything under 5 USDT of notional (-4164), and grid.py already
    # enforces that on every limit order it places -- three separate checks. The
    # protective orders ignored the same rule, and they are the ones that matter: a
    # rejected stop is not a missed opportunity, it is an unprotected position.
    #
    # Worse, the failure is self-sustaining. reconcile_stop_orders reports coverage
    # short of desired, _refresh_sl_stops returns False, and the caller blocks new
    # exposure -- so the grid stops trading and can no longer work the position down
    # to nothing. It waits, blocked, on a position too small to protect. Observed on
    # 2026-08-16 20:52, where 3 DOGE became legs of 1.0 and 2.0: 0.07 and 0.14 USDT.
    # Testnet accepted them; the live venue would not have.
    #
    # One full-size stop clears the floor wherever two halves do not, and full
    # coverage at the hard level is strictly safer than no coverage at all.
    if min_notional > 0 and (trail_qty * trail_price < min_notional
                             or hard_qty * hard_price < min_notional):
        _note_stop_sizing(
            "INFO",
            "STOP-LOSS NOT SPLIT | {} {} would give legs of {}/{} — under the {} USDT "
            "minimum, so both would be rejected. Placing one full-size hard stop.",
            side, qty, trail_qty, hard_qty, min_notional,
        )
        return [("hard", qty, hard_price)]

    _clear_stop_sizing_note()
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


def get_position_details(exchange: Exchange, symbol: str) -> list[dict] | None:
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
    except Exception as e:
        # [] means FLAT. An unreadable account must not borrow that meaning: the
        # loop's flat branch sweeps stops, unblocks the position cap and zeroes the
        # unrealised figure, and the dormancy clock RESETS -- so one failed poll
        # during a network blip erased however long the book had been empty, and a
        # flaky venue would stop the watchdog ever accumulating. Exactly when the
        # exchange is unreliable is when that detector matters most (AUDIT #128).
        logger.warning("POSITIONS UNREADABLE | {} -- treating the position as UNKNOWN", e)
        return None


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
) -> None:
    """Send position + balance to Telegram. Call only on significant events.

    No pnl_reconciler here on purpose (AUDIT #147): this feeds on_balance_update,
    and a BALANCE message carrying the account's cumulative PnL read as the balance
    itself being negative -- see the docstring on on_balance_update. The fill and
    startup-summary notifications still carry that figure, correctly labelled, under
    headers that are already about PnL.
    """
    pos_details = get_position_details(exchange, symbol) or []
    for pos in pos_details:
        unrealized = _position_unrealized_pnl(pos, price)
        notifier.on_position_update(symbol, pos["side"], pos["entry_price"], pos["qty"], price, unrealized)
    balance_info = exchange.get_balance_info()
    equity = exchange.get_total_equity()
    exposure_pct = 0.0
    if equity > 0:
        exposure_usdt = sum(p["qty"] * price for p in pos_details)
        exposure_pct = exposure_usdt / equity
    notifier.on_balance_update(balance_info["free"], balance_info["used"], equity, exposure_pct)


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
                events.daily_reset(completed_daily_pnl, risk.state.trades_today, balance, risk.state.fills_today)
        risk.reset_daily()


# Two very different things wear the same "problem" hat, and they need opposite handling.
# "The exchange says 5x and your config says 25x" is a real misconfiguration: restarting
# changes nothing and a human has to fix it, so the process stops. "I could not reach the
# endpoint to ask" is an OUTAGE, and stopping over it means a backend timeout takes the
# bot down until somebody notices (AUDIT #104). Naming the second one lets the caller
# tell them apart.
ACCOUNT_UNREADABLE = (
    "account configuration could not be read, so leverage, margin mode and "
    "position mode are all unverified — refusing to size orders against "
    "assumptions nothing confirmed"
)


def verify_account_config(exchange: Exchange, cfg, balance: float) -> list[str]:
    """Check the EXCHANGE agrees with the assumptions the sizing and stop math make.

    Everything the bot computes about money -- notional per order, margin consumed,
    how far liquidation sits from the stop -- is derived from config values that the
    exchange is free to disagree with. Nothing verified that they matched. Measured on
    this account while writing the check: .env said LEVERAGE=25, the exchange said 5.
    On demo that difference is a number in a log; on a live account it is a 5x error
    in every margin figure the bot believes.

    Returns a list of blocking problems -- empty means the account is safe to trade.
    Read-only: it reports, it does not reconfigure the account (AUDIT #69).
    """
    problems: list[str] = []
    acct = exchange.get_account_config(cfg.symbol)
    if acct is None:
        return [ACCOUNT_UNREADABLE]

    logger.info(
        "ACCOUNT CONFIG | leverage={}x | margin={} | position mode={} | max notional={:,.0f}",
        acct["leverage"], acct["margin_mode"],
        "HEDGE (dual-side)" if acct["dual_side"] else "one-way",
        acct["max_notional"],
    )

    if acct["leverage"] != cfg.leverage:
        problems.append(
            f"leverage mismatch: config says {cfg.leverage}x, the exchange is on "
            f"{acct['leverage']}x. Order notional is CAPITAL_PER_GRID_USDT x LEVERAGE, so "
            f"every margin figure would be out by {cfg.leverage / max(1, acct['leverage']):.2g}x"
        )

    if acct["dual_side"]:
        problems.append(
            "the account is in HEDGE (dual-side) position mode. This bot sends no "
            "positionSide, which Binance rejects outright in hedge mode (-4061), and its "
            "reduce-only stops assume one net position. Switch the account to One-way mode"
        )

    # Isolated margin puts liquidation a fixed distance from entry. The stop has to be
    # comfortably nearer than that, or the position is closed by the exchange at a
    # liquidation fee instead of by the stop at a maker/taker fee.
    per_order = cfg.capital_per_grid_usdt * cfg.leverage if cfg.capital_per_grid_usdt > 0 else 0.0
    one_side = per_order * (cfg.grid_count / 2)
    if acct["isolated"]:
        mmr = exchange.get_maint_margin_ratio(cfg.symbol, one_side or 1.0)
        if mmr is None:
            problems.append(
                "the account is on ISOLATED margin and the maintenance-margin rate could "
                "not be read, so the distance from the stop to liquidation is unknown"
            )
        else:
            liq_distance = 1.0 / cfg.leverage - mmr
            logger.info(
                "ISOLATED MARGIN | liquidation ~{:.2%} from entry (1/{}x - {:.2%} maint) "
                "vs a {:.2%} stop",
                liq_distance, cfg.leverage, mmr, cfg.stop_loss_pct,
            )
            # Liquidation is priced off the MARK price, which wanders from last trade,
            # so "the stop is 0.1% nearer" is not clearance. Ask for a third.
            if cfg.stop_loss_pct > liq_distance * 0.75:
                problems.append(
                    f"on ISOLATED margin at {cfg.leverage}x, liquidation sits about "
                    f"{liq_distance:.2%} from entry while STOP_LOSS_PCT is "
                    f"{cfg.stop_loss_pct:.2%}. The stop needs real daylight beneath it "
                    f"(mark price differs from last), so either switch the symbol to CROSS "
                    f"margin, drop LEVERAGE, or tighten STOP_LOSS_PCT below "
                    f"{liq_distance * 0.75:.2%}"
                )
    else:
        logger.info(
            "CROSS MARGIN | the whole {:.2f} USDT wallet backs the position, so "
            "liquidation is far from the {:.2%} stop", balance, cfg.stop_loss_pct,
        )

    # Resting orders reserve initial margin before they fill. In one-way mode the two
    # sides cannot both increase the position, so Binance reserves for the LARGER side
    # only -- measured on this account: 8 rungs at 5 USDT reserved 19.997, not 40. The
    # binding figure is one side, and asking for the full ladder would refuse an account
    # that can genuinely fund it.
    if cfg.capital_per_grid_usdt > 0:
        needed = cfg.capital_per_grid_usdt * (cfg.grid_count / 2)
        if balance < needed:
            problems.append(
                f"free balance {balance:.2f} USDT cannot fund the ladder: {cfg.grid_count // 2} "
                f"rungs a side reserve {cfg.capital_per_grid_usdt:.2f} each, so "
                f"{needed:.2f} is the minimum before fees, stops, or any adverse move. "
                f"Fund the account, lower GRID_COUNT, or lower CAPITAL_PER_GRID_USDT"
            )
        elif balance < needed * 2:
            logger.warning(
                "THIN MARGIN | one side of the ladder reserves {:.2f} of {:.2f} free USDT "
                "— once rungs fill, the position holds margin too and a drawdown could "
                "stop new rungs being placed", needed, balance,
            )

    if acct["max_notional"] and one_side > acct["max_notional"]:
        problems.append(
            f"one side of the ladder is {one_side:.2f} USDT but {cfg.leverage}x is only "
            f"allowed up to {acct['max_notional']:,.0f} on this symbol"
        )

    # The minimum profitable spacing is built from the CONFIGURED fee rates. If the
    # account actually pays more, rungs go closer together than a cycle can pay for and
    # every completed cycle loses the difference -- silently, because the arithmetic all
    # agrees with itself. Demo and live are not on the same fee schedule.
    # MIN_NOTIONAL_USDT is hardcoded to 5.0 -- DOGE's number -- and every order the grid
    # or the trend follower declines to place below the floor is declined against it.
    # Nothing ever asked the exchange whether it was true. If the real minimum is HIGHER,
    # orders sized at the floor come back -4164 and the ladder simply never fills, with
    # the bot's arithmetic agreeing with itself the whole way down (AUDIT #107).
    exchange_min = exchange.get_min_notional(cfg.symbol)
    if exchange_min is None:
        logger.warning(
            "MIN NOTIONAL UNVERIFIED | sizing against the built-in {:.2f} USDT floor "
            "without confirming it against the exchange", MIN_NOTIONAL_USDT,
        )
    elif exchange_min > MIN_NOTIONAL_USDT:
        problems.append(
            f"the exchange requires {exchange_min:.2f} USDT per order on {cfg.symbol} but "
            f"the bot's floor is {MIN_NOTIONAL_USDT:.2f}. Every order sized between the "
            f"two would be rejected -4164, so rungs would silently fail to place while "
            f"the sizing arithmetic still agreed with itself"
        )
    elif exchange_min < MIN_NOTIONAL_USDT:
        logger.info(
            "MIN NOTIONAL | exchange allows {:.2f} USDT on {}, bot floors at {:.2f} — "
            "conservative, nothing is rejected by it",
            exchange_min, cfg.symbol, MIN_NOTIONAL_USDT,
        )
    else:
        logger.info("MIN NOTIONAL | {:.2f} USDT, matches the bot's floor", exchange_min)

    fees = exchange.get_commission_rates(cfg.symbol)
    if fees is None:
        logger.warning(
            "FEE RATES UNVERIFIED | trading on the configured {:.4f}%/{:.4f}% without "
            "confirming them against the account", cfg.maker_fee_pct, cfg.taker_fee_pct,
        )
    else:
        actual_rt = fees["maker_pct"] * 2
        config_rt = cfg.maker_fee_pct * 2
        logger.info(
            "FEE RATES | account maker={:.4f}% taker={:.4f}% | config maker={:.4f}% "
            "taker={:.4f}%",
            fees["maker_pct"], fees["taker_pct"], cfg.maker_fee_pct, cfg.taker_fee_pct,
        )
        # Grid cycles are pure maker, so the maker rate is what the spacing floor rides
        # on. 5% relative tolerance: a rounding difference is not a defect, a fee tier is.
        if actual_rt > config_rt * 1.05:
            floor = config_rt * cfg.min_profit_multiplier
            true_floor = actual_rt * cfg.min_profit_multiplier
            problems.append(
                f"the account pays {fees['maker_pct']:.4f}% maker but MAKER_FEE_PCT says "
                f"{cfg.maker_fee_pct:.4f}%. The minimum profitable spacing is built from "
                f"that number, so the grid would place rungs {floor:.4f}% apart when they "
                f"need {true_floor:.4f}%, and every cycle would lose the difference. Set "
                f"MAKER_FEE_PCT={fees['maker_pct']:.4f} and TAKER_FEE_PCT="
                f"{fees['taker_pct']:.4f}"
            )
        elif fees["taker_pct"] > cfg.taker_fee_pct * 1.05:
            # Taker only prices forced exits, which no spacing decision depends on -- but
            # it is the dominant cost in the measured history, so a stale figure makes
            # every PnL estimate optimistic.
            logger.warning(
                "TAKER FEE UNDERSTATED | the account pays {:.4f}% but TAKER_FEE_PCT says "
                "{:.4f}% — forced exits cost {:.0%} more than the bot's estimates. Set "
                "TAKER_FEE_PCT={:.4f}",
                fees["taker_pct"], cfg.taker_fee_pct,
                fees["taker_pct"] / max(1e-9, cfg.taker_fee_pct) - 1, fees["taker_pct"],
            )
        elif actual_rt < config_rt * 0.95:
            logger.info(
                "FEES OVERSTATED | the account pays less than configured, so spacing is "
                "wider than it needs to be. Harmless, but MAKER_FEE_PCT={:.4f} would be "
                "accurate", fees["maker_pct"],
            )

    return problems


def _is_auth_failure(exc: BaseException) -> bool:
    """Bad credentials, as opposed to a venue we simply cannot reach right now.

    The distinction decides whether the supervisor should try again: rotated or
    wrong-mode keys will fail identically forever, an unreachable endpoint will not.
    """
    if isinstance(exc, ccxt.AuthenticationError):
        return True
    text = str(exc).lower()
    # -2014 API-key format invalid, -2015 invalid key / IP / permissions.
    return any(s in text for s in
               ("api-key", "apikey", "signature", "unauthorized", "-2014", "-2015"))


def abort_startup(transient: bool) -> NoReturn:
    """End startup, telling supervise.py whether it should try again.

    RestartPolicy.should_restart treats a clean exit as a deliberate stop and stays
    down (supervise.py:63-70). The process exit code is therefore the ONLY channel
    this function has for saying "retry me" versus "a human must fix this first".

    Every startup guard used to end in a bare `return` -- which falls out of run_bot,
    ends __main__, and exits 0. For a misconfiguration that is correct: restarting
    into the same bad config forever helps nobody. For an unreachable endpoint it is
    backwards, and it is the reason a network outage turns into an indefinite outage:
    the bot stops, the supervisor honours the stop, and the position sits on the
    exchange with only its stop-loss until a human notices. The log lines at those
    sites even said "Restart the bot once the exchange recovers" -- addressed to a
    person, because nothing else was listening.

    AUDIT #104 drew this line at the account probe. These are the sites that never
    got it (AUDIT #126).
    """
    raise SystemExit(1 if transient else 0)
def pnl_divergence(engine_delta: float, account_delta: float,
                   tolerance: float) -> float:
    """Gap between what the engine thinks it earned and what the account actually
    earned, over the SAME interval. Returns 0.0 while within tolerance.

    Deltas, not totals, on purpose. The engine's total_pnl is restored from saved
    state and spans every session the ladder has run; the reconciler's session figure
    resets on restart. Comparing totals would report a divergence that is only a
    difference of window. Comparing what each ACCRUED between two checks is
    window-independent, and is the thing that can actually be wrong.

    It is wrong, structurally. Forced closes -- the hard stop-market leg, reconcile
    closes, emergency_stop -- go through exchange.close_position() and never reach
    _handle_fill, so the engine ledger never books them. 2026-08-20: the -74.84 stop
    produced no journal row at all, cycle_pnl showed 47 wins and 1 loss for +10.96,
    and the account was down ~71. Both numbers were already printed in every status
    line, ten seconds apart, for days, and nothing compared them (AUDIT #139).
    """
    if tolerance <= 0:
        return 0.0
    gap = engine_delta - account_delta
    return 0.0 if abs(gap) <= tolerance else gap


def account_recheck_due(last_checked: float, now: float, interval: float) -> bool:
    """Is another account-config check owed? interval <= 0 disables it entirely.

    verify_account_config runs once, at startup, and its own docstring says every
    money figure the bot computes depends on what it checks. Account settings can
    move under a running bot -- someone changes leverage in the Binance UI, or
    another process on the same account does -- and from that moment every notional,
    margin and stop figure is wrong with nothing saying so. This account went 25x ->
    5x between sessions on 2026-08-19/20; mid-session that would have been silent
    until a restart (AUDIT #137).
    """
    if interval <= 0:
        return False
    return (now - last_checked) >= interval


def dormancy_clock(has_position: bool, working_orders: int, strategy_active: bool,
                   dormant_since: float | None, now: float,
                   position_known: bool = True) -> tuple[float | None, float]:
    """How long has the bot held exposure with nothing working to resolve it?

    2026-08-19, 16:31:42 -> 19:44:58: a 6,307 ADA short, an empty order book, and a
    loop that polled 4,770 times without raising once. Every recovery path in this
    program hangs off an exception, and a bot calmly doing nothing throws none. The
    only liveness test that existed -- supervise.log_is_stale -- watches log mtime,
    and the deadlock wrote a PRICE= line every ten seconds, so it stayed satisfied
    the whole time. Unrealised went -7.36 -> -36 across those three hours.

    A deliberately paused strategy (recovery cooldown, post-kill-switch) is NOT
    dormant: having no orders is the entire point. Hence strategy_active.

    Returns (dormant_since, seconds_dormant); (None, 0.0) whenever healthy.
    (AUDIT #127)
    """
    if not position_known:
        # Freeze, never reset. An unreadable account is not evidence of health, and
        # resetting here would let intermittent read failures hold the clock at zero
        # forever. It cannot START the clock either -- that would be guessing at
        # exposure we cannot see (AUDIT #128).
        if dormant_since is None:
            return None, 0.0
        return dormant_since, max(0.0, now - dormant_since)
    if not (has_position and strategy_active) or working_orders > 0:
        return None, 0.0
    started = now if dormant_since is None else dormant_since
    return started, max(0.0, now - started)


def dormancy_action(seconds_dormant: float, seconds_since_alert: float,
                    alert_after: float, restart_after: float) -> str:
    """"none" | "alert" | "restart". Either threshold at 0 disables that rung.

    Re-alerts on the same interval rather than once. POSITION UNDER-PROTECTED fired
    exactly once on 2026-08-19 (16:30:26) and never again, which is how a correct
    detection still went unheard for three hours.
    """
    if restart_after > 0 and seconds_dormant >= restart_after:
        return "restart"
    if alert_after > 0 and seconds_dormant >= alert_after and seconds_since_alert >= alert_after:
        return "alert"
    return "none"
def ladder_cap_room(one_side_notional: float, cap: float, held_notional: float
                    ) -> tuple[bool, float]:
    """Does one side of the ladder fit in the part of the cap that is still free?

    A position already open eats the SAME cap the ladder is measured against, so the
    ladder's real headroom is the cap MINUS what is already held. #66 compared the
    ladder against the whole cap, which is right only from flat.

    2026-08-19 10:01:24, restarting with SHORT 4284 ADA already open:

        LADDER FITS THE CAP | one side commits 750.00 of 982.39 (77%), 1.9 rung(s) spare

    The held short was ~749 USDT at 0.1749. 750 + 749 = 1499 against a 982 cap -- 53%
    OVER before a single order was placed, reported as fitting with room to spare. The
    sell side then filled its way to 6307 ADA, set_position_limit hard-blocked it, every
    buy sat below break-even and was skipped, and the grid stood with an empty book for
    three hours while price ran 3.6% away from it (AUDIT #120).

    Returns (fits, room). A non-positive cap is not a verdict -- max_position_pct of 0
    means unconfigured, not 'nothing fits' -- so it reports fitting and leaves the
    decision to the cap enforcement itself.
    """
    room = cap - held_notional
    if cap <= 0:
        return True, room
    return one_side_notional <= room, room


def seed_position_limit(exchange, grid, symbol: str, cfg, daily_realized_pnl: float = 0.0) -> None:
    """Teach the grid its position cap BEFORE it places anything. AUDIT #125.

    set_position_limit is what computes _block_buys/_block_sells and the size taper, and
    its first call lived inside the main loop -- roughly 730 lines after startup
    placement. So every start laid a full ladder with the cap unknown: neither side
    blocked, no taper, and whatever inventory was already open ignored entirely.

    ADAUSDT 2026-08-19 10:01, restarting with SHORT 4284 already held:

        10:01:12-23  six buy rungs and three sell rungs placed
        10:01:24     LADDER FITS THE CAP | one side commits 750.00 of 982.39 (77%)
        ...          sells keep filling, short walks 4284 -> 5597 -> 6307
        14:56:27     POSITION LIMIT | short 6307.0 >= 5608.77 -- sell orders blocked

    Real room at 10:01 was 982.39 minus ~749 held = 233 USDT, about 1.9 rungs. The bot
    placed six, because nothing had told it otherwise yet. That is the shape of the whole
    day: every session ended holding a position, every restart adopted it and stacked a
    fresh full ladder on top, and the position ratcheted 0 -> 4284 -> 6307 without ever
    being able to come back.

    Seeded, that same restart taper the sell side to roughly half size and blocks it near
    the cap instead of overshooting it, and AUDIT #122's stuck-ladder exit then has a
    blocked side to react to -- together they let the position converge rather than
    ratchet.

    Costs one position read and one equity read, the same two the loop makes every
    iteration. Non-fatal: if either fails the opening ladder is sized exactly as badly as
    it was before, which is the status quo, not a new risk.
    """
    # Probe the book explicitly first. get_position_breakdown swallows a failed read
    # and returns (0.0, 0.0), which is indistinguishable from genuinely flat -- and
    # seeding 'flat' hands the ladder the entire cap, the exact assumption this
    # function exists to remove. Refuse to guess instead.
    try:
        exchange.get_positions(symbol)
    except Exception as e:
        logger.warning(
            "POSITION CAP NOT SEEDED | positions unreadable ({}) -- refusing to size "
            "the opening ladder as if flat", e,
        )
        return
    try:
        long_pos, short_pos = get_position_breakdown(exchange, symbol)
        price = exchange.get_price(symbol)
        equity = exchange.get_total_equity()
        max_pos_qty = equity * cfg.max_position_pct / price if price > 0 else 0.0
        grid.set_position_limit(long_pos, short_pos, max_pos_qty)
        grid.apply_open_loss_guard(price)
        # Sibling guard, same seed-before-placing discipline: re-arms the profit lock
        # before the initial ladder gets laid, so a restart mid-lockout can't re-open
        # exposure into a day that already hit its profit budget (see
        # apply_profit_lock_guard's persistence note).
        grid.apply_profit_lock_guard(daily_realized_pnl)
        if long_pos or short_pos:
            logger.info(
                "POSITION CAP SEEDED | long={:.0f} short={:.0f} against a cap of {:.0f} "
                "-- the opening ladder is sized against what is already held",
                long_pos, short_pos, max_pos_qty,
            )
    except Exception as e:
        logger.warning(
            "POSITION CAP NOT SEEDED | {} -- the opening ladder will be sized without it", e,
        )


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
        abort_startup(transient=False)

    notifier = TelegramNotifier(
        settings.telegram_bot_token,
        settings.telegram_chat_id,
        settings.telegram_enabled,
    )

    events = EventJournal(settings.log_dir, demo=settings.demo_mode)

    try:
        config = settings.exchange_config
        exchange = Exchange(config, demo=settings.demo_mode)
    except Exception as e:
        logger.error("Failed to connect to exchange: {}", e)
        # Demo Trading and live are separate accounts with separate keys, so the first
        # start after flipping DEMO_MODE fails here if the keys were not swapped too.
        # That reads as a network problem unless someone says otherwise.
        logger.error(
            "If DEMO_MODE was just changed: {} mode needs {} API keys — demo keys come "
            "from demo.binance.com and do not work against the live API (or the reverse). "
            "A live key also needs Futures trading enabled and this IP whitelisted",
            mode, "DEMO" if settings.demo_mode else "LIVE",
        )
        abort_startup(transient=not _is_auth_failure(e))

    # Every money figure the bot computes is config x exchange-reality. Confirm they
    # agree before any of it is spent (AUDIT #69).
    leverage_ok = exchange.set_leverage(settings.symbol, settings.leverage)
    try:
        startup_balance = exchange.get_balance()
    except Exception as e:
        # SystemExit(1), not `return`. A bare return leaves run_bot normally and the
        # process exits 0, which supervise.py reads as "a clean exit is a decision
        # someone made; honour it" -- and stays down. That is right for a deliberate
        # shutdown and wrong for a backend timeout: on 2026-08-18 01:24 one -1007 on
        # this very call put the bot down for the night, twenty seconds after starting.
        #
        # The read is retried three times before it can get here (AUDIT #102), so
        # reaching this point means the exchange is genuinely unreachable. That is
        # exactly what the supervisor's backoff is for, and its breaker still stops a
        # crash loop after the configured number of attempts.
        logger.error("Could not read balance to verify account configuration: {}", e)
        raise SystemExit(1)

    account_problems = verify_account_config(exchange, settings, startup_balance)
    if not leverage_ok and not account_problems:
        # set_leverage failed but the account was already on the right leverage. Nothing
        # is mis-sized, so this is a note, not a stop.
        logger.warning(
            "LEVERAGE CALL FAILED | the account was already on {}x, so sizing is correct, "
            "but the write path to account settings is not working", settings.leverage,
        )
    if account_problems:
        logger.error("=" * 50)
        logger.error("ACCOUNT NOT SAFE TO TRADE | {} problem(s) found", len(account_problems))
        for problem in account_problems:
            logger.error("  - {}", problem)
        logger.error("=" * 50)
        notifier.send(
            "&#x1f6a8; <b>STARTUP ABORTED</b>\nAccount configuration does not match the "
            "bot's sizing assumptions:\n"
            + "\n".join(f"• {p}" for p in account_problems)
        )
        # Unreadable is not the same as wrong. A real mismatch needs a human, so exiting
        # 0 is right -- supervise.py honours it and stays down instead of restarting into
        # the same misconfiguration forever. But an unreachable endpoint is temporary,
        # and exiting 0 there hands the supervisor a "someone decided to stop" it has no
        # way to question. 2026-08-18 02:04, with every signed Binance endpoint answering
        # HTTP 408, that is precisely what happened (AUDIT #104).
        if ACCOUNT_UNREADABLE in account_problems:
            abort_startup(transient=True)
        abort_startup(transient=False)

    # Read the state file BEFORE deciding what to do with any open position.
    #
    # This used to cancel orders and market-close every position unconditionally, then
    # load state 60 lines later -- so a position the bot had deliberately left open at
    # shutdown (CLOSE_ON_EXIT defaults to false, precisely so the grid can unwind it
    # through its own levels) was dumped at market by housekeeping before anything
    # asked whether it was an orphan. Measured on the 2026-08-12 23:18 restart: 6274
    # DOGE closed at market, verified PnL -1.83 -> -3.26, so that one restart cost more
    # than the session that preceded it. It also made the restore-with-position branch
    # below unreachable, because cleanup had always just flattened.
    #
    # Orders are still cancelled unconditionally -- untracked resting orders from a dead
    # session are genuinely dangerous and the grid re-places its own. A POSITION is
    # different: with saved state it is inventory with a ladder to unwind through, and
    # the same rule as AUDIT #32 applies -- do not book a loss to tidy up (AUDIT #37).
    state_mgr = StateManager(settings.state_dir, settings.symbol, demo=settings.demo_mode)
    # max_exposure_pct is a limit on the ACCOUNT, but every part of the check that
    # enforces it is per-process and per-symbol. Two instances on the same account each
    # permit the full cap unless they can see one another (AUDIT #97).
    exposure_registry = ExposureRegistry(
        settings.state_dir, demo=settings.demo_mode, symbol=settings.symbol,
    )
    saved_state = state_mgr.load()
    has_saved_grid = bool(saved_state and "grid" in saved_state)

    # Keep the stop book if anything is open. emergency_stop leaves stops armed on
    # purpose so an inherited position stays protected while the bot is down; cancelling
    # them here and re-placing an identical pair 26 seconds later (05:00:22 -> 05:00:48
    # on 2026-08-17) threw that away and opened the one window the shutdown path had
    # deliberately closed. The stop refresh in the loop reconciles them properly.
    #
    # An unreadable position counts as "open": a stray stop is reduceOnly and harmless
    # when flat -- the flat branch of the loop sweeps it -- whereas cancelling one that
    # was protecting something is not recoverable.
    try:
        _held_side, _held_qty = get_net_position(exchange, settings.symbol)
        _keep_stops = _held_side in ("long", "short") and _held_qty > 0
    except Exception as e:
        logger.warning("STARTUP CLEANUP | could not read the position ({}) — keeping stops", e)
        _keep_stops = True

    logger.info(
        "STARTUP CLEANUP | cancelling all orders{}...",
        " (stops kept: a position is open)" if _keep_stops else "",
    )
    cancelled = exchange.cancel_everything(settings.symbol, keep_stops=_keep_stops)
    if cancelled:
        logger.warning(
            "Cancelled {} leftover orders ({}) from previous sessions",
            cancelled, "limits only" if _keep_stops else "limits + stops",
        )
    if has_saved_grid:
        logger.info(
            "STARTUP | saved grid state found — keeping any open position for the "
            "restored grid to unwind rather than closing it at market"
        )
    elif not state_mgr.has_history():
        # Closing an "orphan" is right after a crash and wrong on a first run. This is
        # the first time the bot has been pointed at this account and symbol, so any
        # position here was opened by someone else -- most likely the human, right after
        # flipping DEMO_MODE. Market-closing it would be the bot's opening act on a live
        # account: a trade nobody asked for, at whatever the book offers (AUDIT #72).
        try:
            pre_existing = [
                p for p in exchange.get_positions(settings.symbol)
                if abs(float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)) > 0
            ]
        except Exception as e:
            logger.error("Could not check for pre-existing positions ({}) — not starting", e)
            abort_startup(transient=True)
        if pre_existing:
            logger.error("=" * 50)
            logger.error(
                "PRE-EXISTING POSITION | this is the first {} run for {} and a position is "
                "already open. The bot did not open it, so it will not close it.",
                state_mgr.mode.upper(), settings.symbol,
            )
            for p in pre_existing:
                logger.error(
                    "  - {} {} @ {}", p.get("side"), p.get("contracts"), p.get("entryPrice"),
                )
            logger.error(
                "Close it yourself (py cleanup.py) or let it run, then start the bot."
            )
            logger.error("=" * 50)
            notifier.send(
                "&#x1f6a8; <b>STARTUP ABORTED</b>\nA position was already open on the first "
                f"{state_mgr.mode.upper()} run for {settings.symbol}. The bot did not open "
                "it and will not close it."
            )
            abort_startup(transient=False)
        logger.info(
            "STARTUP | first {} run for {} — no prior state, book is clean",
            state_mgr.mode, settings.symbol,
        )
    else:
        closed = exchange.close_all_positions(settings.symbol)
        if closed:
            logger.warning(
                "Closed {} orphan positions from previous sessions (no saved grid state "
                "to unwind them with)", closed,
            )

    try:
        leftover = exchange.get_open_orders(settings.symbol)
    except Exception as e:
        logger.error("Cleanup verification failed ({}): cannot start on an uncertain book. Restart once the exchange write path recovers.", e)
        abort_startup(transient=True)
    if leftover:
        logger.error(
            "{} orders are still open after cleanup — the exchange write path is unreachable. "
            "NOT starting on a dirty book (avoids duplicate grid levels). Restart the bot once the exchange recovers.",
            len(leftover),
        )
        notifier.send(f"&#x1f6a8; <b>STARTUP ABORTED</b>\n{len(leftover)} stale orders could not be cancelled (exchange write path down). Book was left untouched.")
        abort_startup(transient=True)

    grid = None
    try:
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
            trend_min_votes=settings.regime_trend_min_votes,
        )
        # Restores the confirmed regime and both confirmation clocks if the saved snapshot
        # is fresh enough to trust -- see TrendFilter.STALE_AFTER_SECONDS. A no-op on a
        # missing/stale/first-ever state file: trend stays at its just-constructed default
        # (AUDIT #155/#156).
        trend.load_from_dict((saved_state or {}).get("trend", {}))

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
            daily_profit_lock_usdt=settings.daily_profit_lock_usdt,
            event_journal=events,
        )

        grid = None
        journal = TradeJournal(settings.log_dir, demo=settings.demo_mode)

        # PnL reconciler: reports cumulative PnL sourced from Binance's own income
        # ledger (realized PnL + commission + funding) rather than the grid engine's
        # internal per-level bookkeeping, so the number shown to the user always
        # agrees with the real account equity trajectory. Read-only — does not
        # affect order placement or fill handling.
        pnl_reconciler = PnLReconciler.from_dict(saved_state.get("pnl_reconciler") if saved_state else None)
        # Before the sync: a changed PNL_EPOCH has to clear the accumulated totals, or sync()
        # simply resumes from the old cursor and the operator reads a number they believe
        # they changed (AUDIT #60).
        pnl_reconciler.reset_for_epoch(settings.pnl_epoch_ms)
        pnl_reconciler.sync(exchange, settings.symbol)
        # Anchor session PnL before the first order. Everything the bot reported was either
        # the 89-day account lifetime or today, so a fresh start opened by announcing
        # "Total PnL (verified): -30.20" -- accurate, but it is account history (including a
        # -50.49 day from defects since fixed), not this run, and it reads as starting in
        # the red (AUDIT #59).
        pnl_reconciler.begin_session()
        # From the reconciler, never from settings: the label has to describe the value that
        # actually produced the number, or the two can disagree (AUDIT #60).
        pnl_window = pnl_reconciler.window_label
        logger.info(
            "PNL RECONCILER READY | session starts at 0.00 | account ({}) net={:+.4f} USDT "
            "(realized={:+.4f} commission={:+.4f} funding={:+.4f})",
            pnl_window,
            pnl_reconciler.net_realized_pnl, pnl_reconciler.realized_pnl,
            pnl_reconciler.commission, pnl_reconciler.funding_fee,
        )

        if saved_state and "grid" in saved_state:
            logger.info("Restoring saved grid state")
            balance = exchange.get_balance()
            risk.initialize(balance)
            # Restored here, before anything below decides whether to place an order. This
            # used to run one step later, after place_initial_orders() -- a restart landing
            # inside an active kill-switch recovery cooldown re-armed a full ladder into the
            # very conditions that tripped the switch, silently, because in_recovery was
            # still the freshly-constructed False at the moment that decision was made. The
            # loop's own is_in_recovery() check (below) only ever sees state AFTER the first
            # ladder was already resting on the exchange (AUDIT #155).
            risk.load_from_dict(saved_state.get("risk", {}))

            # GRID_COUNT comes from CONFIG, not from the saved state. Taking it from state
            # meant a config change was silently inert for as long as a state file existed:
            # GRID_COUNT 8 -> 14 was edited, the bot restarted, and it ran 8 rungs all night
            # while MAX_POSITION_PCT 0.12 -> 0.20 -- read from settings like everything else
            # -- took effect immediately. Half a geometry change applied is worse than none:
            # the cap doubled while the ladder stayed the same size, so the bot could carry
            # twice the one-sided inventory before anything stopped it (AUDIT #76).
            #
            # The BOUNDS still come from state on purpose. They are where the live orders
            # and the open position actually sit; recalculating them here would orphan the
            # book. load_from_dict rebuilds the levels across those bounds when the count
            # disagrees, and a recenter recomputes bounds from config soon enough.
            saved_count = int(saved_state["grid"].get("grid_count") or settings.grid_count)
            if saved_count != settings.grid_count:
                logger.warning(
                    "GRID COUNT CHANGED | saved state has {} rungs, config says {} — "
                    "rebuilding the ladder at {} across the saved bounds",
                    saved_count, settings.grid_count, settings.grid_count,
                )

            grid = GridEngine(
                exchange, settings.symbol,
                grid_lower=saved_state["grid"]["grid_lower"],
                grid_upper=saved_state["grid"]["grid_upper"],
                grid_count=settings.grid_count,
                capital_per_grid_pct=settings.capital_per_grid_pct,
                stop_loss_pct=settings.stop_loss_pct,
                maker_fee_pct=settings.maker_fee_pct / 100,
                taker_fee_pct=settings.taker_fee_pct / 100,
                taker_fill_share=settings.taker_fill_share_pct / 100,
                recenter_cooldown=settings.recenter_cooldown,
                replacement_cooldown=settings.replacement_cooldown,
                order_pacing_seconds=settings.order_pacing_seconds,
                capital_per_grid_usdt=settings.capital_per_grid_usdt,
                leverage=settings.leverage,
                trailing_sl_trigger_pct=settings.trailing_sl_trigger_pct,
                max_exposure_pct=settings.max_exposure_pct,
                min_profit_multiplier=settings.min_profit_multiplier,
                rung_loss_cap_pct=settings.rung_loss_cap_pct,
                max_open_loss_usdt=settings.max_open_loss_usdt,
                daily_profit_lock_usdt=settings.daily_profit_lock_usdt,
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

            seed_position_limit(exchange, grid, settings.symbol, settings, pnl_reconciler.daily_net_pnl)

            # Reconcile BEFORE deciding what to place, and whether the exchange holds a
            # position or not. Gating this on has_exchange_positions left one wedge: a
            # state file that still claimed a position while the exchange was flat. The
            # strategy's place_initial_orders no-ops while it believes it holds anything,
            # so nothing was ever placed again -- 2026-08-20 19:34 -> 21:31, the trend
            # follower sat "long" 647 ADA that an exchange-side stop had already closed,
            # polling in silence for two hours. The flat path of reconcile_positions is
            # exactly what clears that ghost; the holding path adopts real inventory as
            # before.
            grid.reconcile_state()
            grid.reconcile_positions()

            if not has_exchange_positions:
                if risk.is_in_recovery():
                    logger.warning(
                        "STARTUP DEFERRED | recovery cooldown {}s remaining -- leaving the "
                        "grid flat instead of re-arming a ladder into what tripped the kill "
                        "switch (AUDIT #155)",
                        risk.recovery_cooldown_remaining(),
                    )
                else:
                    logger.info("No exchange positions — resetting stale grid levels to pending")
                    grid.reset_levels_to_pending(exchange.get_price(settings.symbol))
                    logger.info("Placing fresh grid orders after cleanup")
                    grid.place_initial_orders(exchange.get_balance())
        else:
            logger.info("Calculating grid range from recent price action")
            ohlcv = exchange.get_ohlcv(settings.symbol, settings.grid_timeframe, limit=candles_for_lookback(settings.grid_timeframe, settings.range_lookback_days))
            current_price = exchange.get_price(settings.symbol)

            grid_lower, grid_upper = calculate_grid_range(
                ohlcv, current_price,
                lookback_days=settings.range_lookback_days,
                atr_multiplier=settings.range_atr_multiplier,
                timeframe=settings.grid_timeframe,
                mode=settings.range_mode,
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
                abort_startup(transient=False)

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
                taker_fill_share=settings.taker_fill_share_pct / 100,
                recenter_cooldown=settings.recenter_cooldown,
                replacement_cooldown=settings.replacement_cooldown,
                order_pacing_seconds=settings.order_pacing_seconds,
                capital_per_grid_usdt=settings.capital_per_grid_usdt,
                leverage=settings.leverage,
                trailing_sl_trigger_pct=settings.trailing_sl_trigger_pct,
                max_exposure_pct=settings.max_exposure_pct,
                min_profit_multiplier=settings.min_profit_multiplier,
                rung_loss_cap_pct=settings.rung_loss_cap_pct,
                max_open_loss_usdt=settings.max_open_loss_usdt,
                daily_profit_lock_usdt=settings.daily_profit_lock_usdt,
                event_journal=events,
                notifier=notifier,
            )
            grid = _install_strategy(grid, exchange, events, notifier)
            grid.initialize(current_price, balance)

        # Can one side of the ladder actually fill, or will the cap strand the outer rungs?
        #
        # config.validate() answers this for PERCENT sizing but cannot for the
        # CAPITAL_PER_GRID_USDT path: that size is absolute while the cap is a fraction of
        # equity, so the comparison needs a balance config time does not have. #63 correctly
        # stopped validating a figure that no longer decides anything and left nothing in its
        # place -- so the guard vanished exactly as the USDT path became the live one.
        #
        # The failure it guards is documented: a grid wider than its cap goes permanently
        # one-sided, the cap blocks that side partway through, and the bot lives in the
        # capped state that makes recentering destructive (89 recenters in one session).
        #
        # A warning, not an abort. The cap and the size taper keep this SAFE, only degraded,
        # and equity moves -- a restart after a drawdown should not refuse to start
        # (AUDIT #66).
        seed_position_limit(exchange, grid, settings.symbol, settings, pnl_reconciler.daily_net_pnl)

        try:
            _one_side = grid.one_side_notional(balance)
            _cap = balance * settings.max_position_pct
            # A position already open eats the SAME cap the ladder is measured against, so
            # the ladder's real headroom is the cap minus what is already held. Without that
            # term this compares the ladder against the whole cap and passes a book that is
            # already over it.
            #
            # 2026-08-19 10:01:24, restarting with SHORT 4284 ADA carried over:
            #
            #   LADDER FITS THE CAP | one side commits 750.00 of 982.39 (77%), 1.9 rung(s) spare
            #
            # The held short was ~749 USDT at 0.1749. 750 + 749 = 1499 against a 982 cap --
            # 53% OVER before a single order was placed, reported as fitting with room to
            # spare. The sell side then filled its way to 6307 ADA, set_position_limit hard-
            # blocked it, every buy sat below break-even and was skipped, and the grid stood
            # with an empty book for three hours while price ran 3.6% away (AUDIT #120).
            _held = abs(getattr(grid, '_pos_qty', 0.0) or 0.0) * (current_price or 0.0)
            _fits, _room = ladder_cap_room(_one_side, _cap, _held)
            if not _fits:
                _per_order = _one_side / max(1, settings.grid_count / 2)
                logger.warning(
                    "LADDER OUTGROWS THE CAP | {:.0f} rungs a side at {:.2f} USDT commits "
                    "{:.2f}, and {:.2f} is already held, against the {:.0%} position cap of "
                    "{:.2f} -- {:.2f} of room. The outer {:.1f} rung(s) can never fill and the "
                    "book will go one-sided. Lower GRID_COUNT, lower CAPITAL_PER_GRID_USDT or "
                    "LEVERAGE, or raise MAX_POSITION_PCT",
                    settings.grid_count / 2, _per_order, _one_side, _held,
                    settings.max_position_pct, _cap, _room,
                    (_one_side - _room) / _per_order,
                )
            else:
                logger.info(
                    "LADDER FITS THE CAP | one side commits {:.2f} of {:.2f} room ({:.2f} cap "
                    "less {:.2f} held), {:.1f} rung(s) spare",
                    _one_side, _room, _cap, _held,
                    (_room - _one_side) / max(1e-9, _one_side / max(1, settings.grid_count / 2)),
                )
        except Exception as e:
            logger.debug("Ladder/cap coherence check skipped: {}", e)

        sl_orders: dict[str, dict] = {}
        _scale_out_done = False
        # _sl_needs_update compares desired stops against `sl_orders`, which is a BELIEF.
        # If reality drifts from it -- a leg cancelled out of band, an exchange-side
        # expiry -- the belief still matches and the position silently stops being
        # covered. So verification is forced on a timer regardless of belief (AUDIT #54).
        _sl_last_verified = 0.0
        SL_VERIFY_INTERVAL_SECONDS = 120.0

        def _reset_sl():
            nonlocal sl_orders, _scale_out_done, _sl_last_verified
            sl_orders = {}
            _scale_out_done = False
            _sl_last_verified = 0.0

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
                min_notional=MIN_NOTIONAL_USDT,
            )

        def _refresh_sl_stops(side: str, qty: float) -> bool:
            """Re-arm the stop-loss legs. Returns True only if the position ends up covered.

            AUDIT #50. This cancels every stop FIRST and then places replacements inside a
            try/except that only logs. Any failure in between leaves an open position with
            no stop at all, and the caller could not tell -- it returned None either way.

            That is the 2026-08-08 sequence exactly: the cancel succeeded, four consecutive
            placements raised `Exchange.amount_to_precision() missing 1 required positional
            argument` (#47), each was logged and swallowed, and a position already 1.8x
            through its cap (#49) ran completely unprotected until EMERGENCY STOP. That day
            was -50.49, 60% of the fortnight's loss.

            The window cannot be closed entirely -- Binance has no atomic replace for stop
            orders -- but it can be made loud and it can be made to stop the bleeding: the
            caller blocks new exposure while uncovered, so an unprotected position can no
            longer also be a growing one.
            """
            nonlocal sl_orders, _sl_last_verified
            close_side = "sell" if side == "long" else "buy"
            desired = list(_desired_sl_orders(side, qty))
            if not desired:
                sl_orders = {}
                return True

            live = exchange.get_stop_orders(settings.symbol)
            if live is None:
                # Unknown state. Tearing down protection we cannot see is precisely how a
                # position ends up naked, so change NOTHING and report uncovered: the caller
                # stops adding exposure and the next pass tries again (AUDIT #54).
                logger.error(
                    "STOP REFRESH ABORTED | stop book unreadable — existing stops left in "
                    "place, new exposure blocked until it can be verified",
                )
                return False

            sl_orders, covered_qty, desired_qty = reconcile_stop_orders(
                exchange, settings.symbol, close_side, desired, live,
                over_coverage_tolerance=SL_OVER_COVERAGE_TOLERANCE,
                price_tolerance_pct=SL_PRICE_DRIFT_TOLERANCE,
            )
            _sl_last_verified = time.time()

            covered = covered_qty >= desired_qty - max(1e-8, desired_qty * 1e-6)
            if not covered:
                logger.error(
                    "POSITION UNDER-PROTECTED | {} {} — stops cover {:.8g} of {:.8g} "
                    "({:.0f}%). Blocking new exposure until fully covered (AUDIT #54)",
                    side, qty, covered_qty, desired_qty,
                    100.0 * covered_qty / desired_qty if desired_qty else 0.0,
                )
                events.risk_check("stop_loss_coverage", covered_qty, desired_qty, "UNPROTECTED")
            return covered

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
            live_stops = exchange.get_stop_orders(settings.symbol)
            if live_stops is None:
                # Unreadable book. "Absent" cannot be concluded from a failed read, and
                # concluding it here would latch _scale_out_done permanently (AUDIT #54).
                logger.debug("SCALE-OUT CHECK | stop book unreadable — deferring")
                return
            open_ids = {o.get("id") for o in live_stops}
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

        # How far a live trigger may drift from the desired one before the leg is re-placed.
        # This is the caller's half of a pair: reconcile_stop_orders must match at least this
        # loosely or the 120-second verify re-places legs this test would have left alone,
        # which is where ~14 stop cancels an hour came from on 2026-08-17. One constant, both
        # sides, so the two can no longer disagree.
        SL_PRICE_DRIFT_TOLERANCE = 0.001

        def _sl_needs_update(side: str, qty: float) -> bool:
            if not sl_orders:
                return True
            if time.time() - _sl_last_verified >= SL_VERIFY_INTERVAL_SECONDS:
                return True                      # periodic trust-but-verify (AUDIT #54)
            desired = _desired_sl_orders(side, qty)
            if len(desired) != len(sl_orders):
                return True
            for kind, oqty, oprice in desired:
                cur = sl_orders.get(kind)
                if cur is None:
                    return True
                if abs(cur["price"] - oprice) > max(oprice * SL_PRICE_DRIFT_TOLERANCE, 1e-8):
                    return True
                if cur["qty"] < oqty - max(1e-8, oqty * 1e-6):
                    return True  # under-covered: the position outgrew its stop
                if cur["qty"] > oqty * (1 + SL_OVER_COVERAGE_TOLERANCE) + 1e-8:
                    return True  # stale oversized stop, resize to keep sizing honest
            return False

        # get_net_position swallows a failed read and returns ("", 0.0) -- indistinguishable
        # from genuinely flat. Silently skipping the stop-loss check below on that value
        # would let grid.activate() add fresh exposure on top of a position that might be
        # real and completely unprotected. Probe explicitly first and refuse to guess,
        # mirroring seed_position_limit's AUDIT #125 pattern (AUDIT #159).
        try:
            exchange.get_positions(settings.symbol)
            positions_readable = True
        except Exception as e:
            positions_readable = False
            logger.error(
                "STARTUP STOP-LOSS CHECK SKIPPED | positions unreadable ({}) -- blocking "
                "both sides until the exchange is readable again",
                e,
            )
            grid.block_side("buy", "positions unreadable at startup")
            grid.block_side("sell", "positions unreadable at startup")

        position_side, position_qty = get_net_position(exchange, settings.symbol)
        _last_side = position_side
        if positions_readable and position_side in ("long", "short"):
            try:
                start_price = exchange.get_price(settings.symbol)
                if position_side == "long":
                    grid.update_trailing_sl(start_price)
                else:
                    grid.update_trailing_sl_short(start_price)
                # The third caller, and the one that matters most: this runs at STARTUP with
                # an inherited position, before the loop has done anything. Discarding the
                # answer here meant a restart that could not re-establish stops went on to
                # activate the grid and add exposure to a naked position (AUDIT #52).
                if not _refresh_sl_stops(position_side, position_qty):
                    grid.block_side("buy" if position_side == "long" else "sell",
                                    "stop-loss missing at startup")
                grid.log_sl_status(position_side)
            except Exception as e:
                logger.error("Failed to place stop-loss on startup: {}", e)
                grid.block_side("buy" if position_side == "long" else "sell",
                                "stop-loss placement raised at startup")

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
        except Exception as e:
            # AUDIT #163. This used to swallow with zero trace -- unlike every other
            # narrow-purpose catch in this file, which logs at least at debug. A
            # sustained outage on this feed silently degraded regime confirmation for
            # the whole session with nothing in the log to explain why.
            logger.debug("Fast timeframe ({}) unavailable at startup: {}", settings.trend_timeframe_fast, e)
        try:
            ohlcv_1d = exchange.get_ohlcv(settings.symbol, "1d", limit=100)
            trend.add_timeframe(ohlcv_1d, "1d")
        except Exception as e:
            logger.debug("1d timeframe unavailable at startup: {}", e)

        if risk.is_in_recovery():
            # Second, redundant check -- the branch above already refuses to place a fresh
            # ladder during a recovery cooldown, but this is also where an inherited grid
            # would otherwise get switched on regardless (AUDIT #155).
            if grid.active:
                grid.pause()
                _reset_sl()
            logger.warning(
                "STARTUP DEFERRED | recovery cooldown {}s remaining -- grid stays inactive",
                risk.recovery_cooldown_remaining(),
            )
        elif settings.force_trade_now or trend.is_ranging():
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
                            session_pnl=pnl_reconciler.session_pnl,
                            pnl_window=pnl_reconciler.window_label,
        )
        _notify_status(notifier, exchange, settings.symbol, price_now)
    except Exception as e:
        # The main loop already gets this via its own try/except/finally below --
        # this mirrors it for the startup stretch above, which used to have nothing
        # of its own. A crash here (a network blip mid-startup, an exchange 5xx)
        # used to propagate straight out of run_bot() uncaught, leaving whatever had
        # just been placed (a partial ladder, a lone entry order) completely
        # unmanaged until the next restart's blanket cleanup swept it up. Cancel/
        # close reads exchange state directly rather than trusting local tracking,
        # so it cleans up correctly regardless of how far startup got (AUDIT #157).
        logger.error("STARTUP FAILED | {} -- attempting cleanup before re-raising", e)
        try:
            if grid is not None:
                grid.emergency_stop(reason="startup_failure")
        except Exception as cleanup_err:
            logger.error("Cleanup after startup failure also failed: {}", cleanup_err)
        raise

    loop_count = 0
    consecutive_errors = 0
    dormant_since: float | None = None
    account_checked_at = time.time()   # verify_account_config ran during startup
    pnl_marks: tuple[float, float] | None = None   # (engine_net, account_session)
    dormant_alerted_at = 0.0
    # (timeframe, regime.value) already alerted this lone-trend episode -- fires once
    # per episode, not once per trend_check_interval for as long as it persists
    # (AUDIT #156).
    lone_trend_alerted_key: tuple[str, str] | None = None
    # Bound before the loop: the unrealised figure is now only recomputed when the
    # account was actually readable, so a failed first poll would leave it unset.
    unrealized = 0.0
    seen_bug_errors: set[str] = set()
    last_analytics_fill_count = 0
    try:
        while True:
            try:
                iteration_started = time.monotonic()
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
                        sleep_until_next_poll(iteration_started, settings.poll_interval)
                        continue
                    else:
                        logger.info("RECOVERY READY | recalculating grid around current price {}", price)
                        new_grid = None
                        try:
                            ohlcv = exchange.get_ohlcv(settings.symbol, settings.grid_timeframe, limit=candles_for_lookback(settings.grid_timeframe, settings.range_lookback_days))
                            grid_lower, grid_upper = calculate_grid_range(
                                ohlcv, price,
                                lookback_days=settings.range_lookback_days,
                                atr_multiplier=settings.range_atr_multiplier,
                                timeframe=settings.grid_timeframe,
                                mode=settings.range_mode,
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
                                sleep_until_next_poll(iteration_started, settings.poll_interval)
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
                                    "trend": trend.to_dict(),
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
                            old_peak = grid.peak_price
                            old_grid_lower, old_grid_upper = grid.grid_lower, grid.grid_upper

                            new_grid = GridEngine(
                                exchange, settings.symbol,
                                grid_lower=grid_lower,
                                grid_upper=grid_upper,
                                grid_count=dynamic_count,
                                capital_per_grid_pct=adjusted_capital_pct,
                                stop_loss_pct=settings.stop_loss_pct,
                                maker_fee_pct=settings.maker_fee_pct / 100,
                                taker_fee_pct=settings.taker_fee_pct / 100,
                                taker_fill_share=settings.taker_fill_share_pct / 100,
                                recenter_cooldown=settings.recenter_cooldown,
                                replacement_cooldown=settings.replacement_cooldown,
                                order_pacing_seconds=settings.order_pacing_seconds,
                                capital_per_grid_usdt=0,
                                leverage=settings.leverage,
                                trailing_sl_trigger_pct=settings.trailing_sl_trigger_pct,
                                max_exposure_pct=settings.max_exposure_pct,
                                min_profit_multiplier=settings.min_profit_multiplier,
                                rung_loss_cap_pct=settings.rung_loss_cap_pct,
                                max_open_loss_usdt=settings.max_open_loss_usdt,
                                daily_profit_lock_usdt=settings.daily_profit_lock_usdt,
                                event_journal=events,
                                notifier=notifier,
                            )
                            new_grid = _install_strategy(new_grid, exchange, events, notifier)
                            # The third placement site, and the one AUDIT #125 missed.
                            # The cooldown ends with whatever position triggered the
                            # kill switch still open, so rebuilding here without the
                            # cap hands a full-size ladder the entire budget on top of
                            # inherited inventory -- the exact arithmetic that ran the
                            # short to 6,307 on 2026-08-19. The ordering test asserted
                            # ">= 2 call sites" and passed while this one had none
                            # (AUDIT #130).
                            seed_position_limit(exchange, new_grid, settings.symbol, settings, pnl_reconciler.daily_net_pnl)
                            new_grid.initialize(price, exchange.get_balance())
                            new_grid.total_fills = old_fills
                            new_grid.total_pnl = old_pnl
                            new_grid.total_fees = old_fees
                            new_grid.total_completed_cycles = old_cycles
                            new_grid.peak_price = old_peak
                            new_grid.activate(exchange.get_balance())

                            # Everything throwable above succeeded -- only now does the
                            # outer `grid` and recovery state actually change (AUDIT
                            # #161). `grid` used to be reassigned to the fresh, zeroed
                            # engine BEFORE its stats were restored and BEFORE
                            # activate() ran; a throw anywhere in between left the outer
                            # `grid` pointing at a zeroed engine with risk.exit_recovery()
                            # sometimes already called too -- the bot's whole cumulative
                            # fill/PnL/fee history reading zero, and the retry path
                            # disabled since is_in_recovery() was already False. A
                            # mid-rebuild failure now leaves the OLD grid (real stats
                            # intact) and risk.is_in_recovery()==True completely
                            # untouched, so the next cycle retries cleanly.
                            grid = new_grid
                            risk.exit_recovery()
                            notifier.on_grid_start(settings.symbol, grid.grid_lower, grid.grid_upper, grid.grid_count)
                            notifier.on_recovery_resume(recovery_mult)
                            notifier.on_grid_recalculated(settings.symbol, grid_lower, grid_upper, dynamic_count, "recovery")
                            events.grid_recalculated(
                                settings.symbol, old_grid_lower, old_grid_upper, grid_lower, grid_upper,
                                dynamic_count, "recovery", current_atr,
                            )
                            events.recovery_event("resume", risk.state.recovery_count, sizing_pct=recovery_mult)
                            _notify_status(notifier, exchange, settings.symbol, price)
                            logger.info("Grid recovered and activated with {}% sizing", int(recovery_mult * 100))
                        except Exception as e:
                            if new_grid is not None:
                                # Whatever new_grid managed to place before the throw is
                                # real exchange state the OLD grid knows nothing about --
                                # sweep it via new_grid's own emergency_stop (reads the
                                # exchange directly) rather than leaving it untracked.
                                try:
                                    new_grid.emergency_stop(reason="recovery_rebuild_failed")
                                except Exception as cleanup_err:
                                    logger.error(
                                        "Cleanup after failed recovery rebuild also failed: {}",
                                        cleanup_err,
                                    )
                            if isinstance(e, BUG_ERRORS):
                                # A defect in this program, not a market/network problem
                                # -- same distinction and treatment as the outer loop
                                # handler (AUDIT #31), which this nested try/continue
                                # would otherwise hide from forever at poll_interval.
                                signature = f"{type(e).__name__}: {e}"
                                first_time = signature not in seen_bug_errors
                                logger.opt(exception=first_time).error(
                                    "BUG IN RECOVERY REBUILD | {} -- this is a code "
                                    "defect, not a connection problem", signature,
                                )
                                if first_time:
                                    seen_bug_errors.add(signature)
                                    try:
                                        notifier.send(
                                            f"<b>BOT DEFECT</b>\n{signature}\n"
                                            f"Recovery rebuild is raising every cycle."
                                        )
                                    except Exception:
                                        pass
                            else:
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
                    except Exception as e:
                        # AUDIT #163: was a silent pass -- see the startup fetch above.
                        logger.debug("Fast timeframe ({}) unavailable: {}", settings.trend_timeframe_fast, e)
                    try:
                        ohlcv_1d = exchange.get_ohlcv(settings.symbol, "1d", limit=100)
                        trend.add_timeframe(ohlcv_1d, "1d")
                    except Exception as e:
                        logger.debug("1d timeframe unavailable: {}", e)

                    logger.info("REGIME | {} -> {}", trend.explain(), trend.regime.value)

                    # Observational only -- does not pause the grid or change sizing,
                    # just surfaces a genuinely persistent lone-timeframe trend that
                    # would otherwise read identically to a fresh one (AUDIT #156).
                    lone = trend.lone_trend_duration() if settings.lone_trend_alert_seconds > 0 else None
                    if lone is not None:
                        lone_tf, lone_regime, lone_duration = lone
                        lone_key = (lone_tf, lone_regime.value)
                        if lone_duration >= settings.lone_trend_alert_seconds and lone_trend_alerted_key != lone_key:
                            lone_trend_alerted_key = lone_key
                            logger.warning(
                                "LONE TREND | {} has read {} for {:.0f}m without enough "
                                "agreement to act on it",
                                lone_tf, lone_regime.value, lone_duration / 60,
                            )
                            notifier.on_lone_trend(
                                lone_tf, lone_regime.value, lone_duration,
                                trend.adx_for(lone_tf),
                            )
                    elif lone_trend_alerted_key is not None:
                        lone_trend_alerted_key = None

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
                            _notify_status(notifier, exchange, settings.symbol, price)
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
                            _notify_status(notifier, exchange, settings.symbol, price)

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
                                    _notify_status(notifier, exchange, settings.symbol, price)

                    if price > grid.grid_upper:
                        total_pos = get_total_position(exchange, settings.symbol)
                        has_buys = any(l.side == "buy" and l.order_id for l in grid.levels)
                        if not has_buys and total_pos == 0:
                            logger.info("GRID EXIT: price {} above grid {} with no buy orders and no position — recentering", price, grid.grid_upper)
                            events.grid_exit(settings.symbol, "price_above_grid_no_position", price, total_pos)

                    balance = exchange.get_balance_cached()

                    # Alert, do NOT abort. Startup refuses to trade on a mismatched
                    # account because nothing is open yet; killing a running bot that
                    # holds a position is a bigger risk than the mis-sizing it would
                    # avoid. So this is the loudest thing short of acting (AUDIT #137).
                    # Does the engine's story match the account's? (AUDIT #139)
                    _engine_net = grid.total_pnl - grid.total_fees
                    _account_net = pnl_reconciler.session_pnl
                    _due = account_recheck_due(account_checked_at, time.time(),
                                               settings.account_recheck_seconds)
                    if pnl_marks is None:
                        pnl_marks = (_engine_net, _account_net)
                    elif _due:
                        _engine_moved = _engine_net - pnl_marks[0]
                        _account_moved = _account_net - pnl_marks[1]
                        pnl_marks = (_engine_net, _account_net)
                        _gap = pnl_divergence(_engine_moved, _account_moved,
                                              settings.pnl_divergence_alert_usdt)
                        if _gap:
                            logger.error(
                                "PNL DIVERGED | since the last check the engine says "
                                "it earned {:+.2f} and the account says {:+.2f} -- a "
                                "{:+.2f} gap. Forced closes never reach the engine "
                                "ledger, so trust account=, not net= (AUDIT #139)",
                                _engine_moved, _account_moved, _gap,
                            )
                            events.risk_check("pnl_divergence", abs(_gap),
                                              settings.pnl_divergence_alert_usdt,
                                              "DIVERGED")
                            try:
                                notifier.send(
                                    "&#x26a0; <b>PNL DIVERGED</b>\n"
                                    f"engine {_engine_moved:+.2f} vs account "
                                    f"{_account_moved:+.2f} ({_gap:+.2f} gap)\n"
                                    "The engine does not book forced closes. "
                                    "Trust the account figure."
                                )
                            except Exception:
                                pass

                    if _due:
                        account_checked_at = time.time()
                        try:
                            _drift = verify_account_config(exchange, settings, balance)
                        except Exception as _e:
                            logger.warning("ACCOUNT RECHECK FAILED | {}", _e)
                            _drift = []
                        if ACCOUNT_UNREADABLE in _drift:
                            # An outage is not drift, and the loop has its own
                            # machinery for outages. Say so quietly and wait.
                            logger.warning(
                                "ACCOUNT RECHECK | account unreadable -- no verdict on "
                                "whether settings still match",
                            )
                            _drift = []
                        if _drift:
                            logger.error(
                                "ACCOUNT DRIFTED | {} problem(s) appeared since startup "
                                "-- every notional and margin figure computed from here "
                                "is suspect (AUDIT #137)", len(_drift),
                            )
                            for _problem in _drift:
                                logger.error("  - {}", _problem)
                            events.risk_check("account_config", len(_drift), 0.0, "DRIFT")
                            try:
                                notifier.send(
                                    "&#x26a0; <b>ACCOUNT DRIFTED</b>\n"
                                    "Settings changed under the running bot:\n"
                                    + "\n".join(f"* {_p}" for _p in _drift)
                                )
                            except Exception:
                                pass
                    equity = exchange.get_total_equity_cached()
                    exposure = grid.get_exposure_pct(equity)
                    # `exposure` stays this symbol's own figure -- that is what the event
                    # records mean. The RISK CAP is an account limit, so it is evaluated
                    # against every instance's exposure combined (AUDIT #97).
                    exposure_registry.publish(exposure, exposure * equity)
                    account_exposure = exposure_registry.account_exposure_pct(exposure)
                    if account_exposure > exposure + 1e-9 and loop_count % 20 == 1:
                        logger.info(
                            "ACCOUNT EXPOSURE | {} {:.1%} + [{}] = {:.1%} against a {:.0%} cap",
                            settings.symbol, exposure, exposure_registry.describe_others(),
                            account_exposure, risk.max_exposure_pct,
                        )
                    # Sourced from the exchange's own positions (unrealized_pnl is
                    # Binance's real mark-price-based figure when available -- see
                    # get_position_details/_position_unrealized_pnl) rather than the
                    # grid's internal per-level guess, which can drift the same way
                    # the realized-PnL bookkeeping did (AUDIT.md issues #7/#8).
                    pos_details = get_position_details(exchange, settings.symbol)
                    _pos_known = pos_details is not None
                    if not _pos_known:
                        # Keep the previous unrealised figure rather than reporting a
                        # loss that vanished because a read failed.
                        pos_details = []
                    else:
                        unrealized = sum(_position_unrealized_pnl(p, price) for p in pos_details)
                    # One open-order read per iteration, shared by the limit check and the
                    # fill sweep. They ran back to back against the same book and each
                    # paid its own ~400ms round trip.
                    open_orders = exchange.get_open_orders(settings.symbol)
                    if exchange.enforce_order_limit(
                            settings.symbol, keep_count=grid.grid_count + 2,
                            tracked_ids=grid.get_tracked_order_ids(), orders=open_orders):
                        # It cancelled something, so that snapshot is now a lie about the
                        # book -- and check_fills infers fills from ABSENCE from this list.
                        open_orders = exchange.get_open_orders(settings.symbol)
                    # Count only orders the ladder is actually working. fetch_open_orders
                    # returns the conditional book too, so a lone stop-loss leg would read as a
                    # healthy book and mask the exact condition this watches for.
                    _tracked = grid.get_tracked_order_ids()
                    _working = sum(1 for o in open_orders if str(o.get("id")) in _tracked)
                    # Dust does not arm the clock. Below MIN_NOTIONAL_USDT the exchange
                    # refuses every order this program could place for the position, so
                    # no restart can ever re-lay anything for it -- and on 2026-08-20
                    # that futility ran as a restart every 45 minutes while the real
                    # defect (a strategy holding zero tracked orders) sat upstream. The
                    # stop legs still cover what is held, or STOP-LOSS UNPROTECTABLE
                    # has already said loudly why they cannot.
                    _pos_notional = sum(
                        abs(float(p.get("qty") or 0.0)) * float(p.get("entry_price") or 0.0)
                        for p in pos_details
                    )
                    dormant_since, _dormant_for = dormancy_clock(
                        has_position=_pos_notional >= MIN_NOTIONAL_USDT,
                        working_orders=_working,
                        strategy_active=bool(getattr(grid, "active", True)),
                        dormant_since=dormant_since, now=time.time(),
                        position_known=_pos_known,
                    )
                    if dormant_since is None:
                        dormant_alerted_at = 0.0
                    else:
                        _act = dormancy_action(
                            _dormant_for,
                            time.time() - dormant_alerted_at if dormant_alerted_at else 1e9,
                            settings.empty_book_alert_seconds,
                            settings.empty_book_restart_seconds,
                        )
                        if _act in ("alert", "restart"):
                            dormant_alerted_at = time.time()
                            logger.error(
                                "DORMANT WITH EXPOSURE | {:.0f}s holding a position with no working "
                                "ladder orders. Nothing is raising, so nothing else will notice "
                                "(AUDIT #127)", _dormant_for,
                            )
                            events.risk_check("empty_book", _dormant_for,
                                              settings.empty_book_alert_seconds, _act.upper())
                            try:
                                notifier.send(
                                    "&#x26a0; <b>DORMANT WITH EXPOSURE</b>\n"
                                    f"{_dormant_for/60:.0f} min holding a position with an empty "
                                    "ladder."
                                    + ("\nRestarting so the ladder is re-laid."
                                       if _act == "restart" else "")
                                )
                            except Exception:
                                pass
                        if _act == "restart":
                            # Exit non-zero so supervise.py restarts us (abort_startup's contract,
                            # AUDIT #126). A restart re-lays the ladder with the cap seeded -- the
                            # thing that actually broke this deadlock on 2026-08-19 at 16:29:58.
                            # SystemExit is a BaseException, so the loop's `except Exception`
                            # cannot swallow it, and the outer finally still runs emergency_stop,
                            # which keeps the stop legs armed while a position is open.
                            logger.error("DORMANT WITH EXPOSURE | restarting to re-lay the ladder")
                            raise SystemExit(1)
                    fills = grid.check_fills(balance, open_orders=open_orders)

                    # A stop leg that fired on the exchange never passes through
                    # check_fills, so the ladder keeps a position that no longer
                    # exists and the loss is never booked. On 2026-08-20 the -74.84
                    # that was 105% of the period's loss produced no journal row at
                    # all. Prepended so it is journalled ahead of anything the ladder
                    # did afterwards (AUDIT #143).
                    _closed = grid.detect_external_close(price)
                    if _closed:
                        fills = [_closed] + list(fills)
                        try:
                            notifier.send(
                                "&#x26a0; <b>POSITION CLOSED BY THE EXCHANGE</b>\n"
                                f"{_closed['quantity']:.1f} {settings.symbol} "
                                f"@ ~{_closed['price']}\n"
                                f"Estimated {_closed['profit']:+.2f} USDT. A stop leg "
                                "firing is the usual cause."
                            )
                        except Exception:
                            pass
                    if fills:
                        # Pull Binance's actual income ledger once per batch of fills so the
                        # PnL figures below reflect the exchange's own accounting rather than
                        # the grid engine's internal per-level estimate.
                        verified_before = pnl_reconciler.net_realized_pnl
                        pnl_reconciler.sync(exchange, settings.symbol)
                        # The account's own realized change across this batch. This, not
                        # the grid's per-level sum, is what the risk manager is told --
                        # the two diverge by ~20x in magnitude and can differ in SIGN
                        # when a falling market leaves the blended entry worse than the
                        # level a sell is paired against (AUDIT #43).
                        risk.record_cycles(
                            sum(1 for f in fills if f["completed_cycle"]),
                            pnl_reconciler.net_realized_pnl - verified_before,
                            sum(f["profit"] for f in fills if f["completed_cycle"]),
                        )
                    for fill in fills:
                        profit = fill["profit"]
                        fee = fill["fee"]
                        risk.record_fill()
                        notifier.on_fill(
                            fill["side"], fill["price"], profit, grid.total_fills, pnl_reconciler.daily_net_pnl,
                            total_pnl_verified=pnl_reconciler.net_realized_pnl,
                            session_pnl=pnl_reconciler.session_pnl,
                            pnl_window=pnl_reconciler.window_label,
                            daily_fill_count=risk.state.fills_today,
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
                            fills_today=risk.state.fills_today,
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
                            fills_today=risk.state.fills_today,
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
                        _notify_status(notifier, exchange, settings.symbol, price)

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
                    # The loss budget rides the same per-tick position refresh as the
                    # cap: both are recomputed from live data, neither is persisted.
                    grid.apply_open_loss_guard(price)
                    # Sibling guard: same cadence, called after daily_reset_check (above,
                    # this same iteration) has already rolled pnl_reconciler's daily
                    # bucket over, so a day that just rolled reads today's fresh pnl
                    # (~0), not yesterday's stale total that already tripped the lock.
                    grid.apply_profit_lock_guard(pnl_reconciler.daily_net_pnl)

                    if position_side != _last_side:
                        commit_flip = True
                        if position_side == "" and _last_side in ("long", "short"):
                            # The exchange just went flat under us -- a stop leg fired,
                            # a manual close, an ADL. Tell the live strategy BEFORE
                            # reset_trailing() wipes the ratchet it would have used to
                            # notice: on 2026-08-20 19:34 the trend follower's stop
                            # anchor was erased one tick before its software check
                            # could see the breach, and it then sat two hours believing
                            # it held 647 ADA that no longer existed, placing nothing.
                            # reconcile_positions is "the exchange is the truth" made
                            # callable; for a flat grid it is a harmless no-op.
                            #
                            # The fresh read guards against a transient positions
                            # failure: get_position_breakdown returns (0, 0) when the
                            # API hiccups, and clearing strategy state on that would
                            # orphan a position that is very much still there. If this
                            # read fails too, assume the position remains.
                            try:
                                still_held = any(
                                    abs(float(p.get("contracts", 0) or 0)) > 0
                                    for p in exchange.get_positions(settings.symbol)
                                )
                            except Exception as e:
                                logger.warning(
                                    "SIDE FLIP | could not re-verify flat ({}) — "
                                    "leaving strategy state alone this tick", e,
                                )
                                still_held = True
                            if still_held:
                                # AUDIT #160: don't commit a flat reading this branch
                                # just refused to trust -- retry the re-verify next tick.
                                commit_flip = False
                            else:
                                try:
                                    grid.reconcile_positions()
                                except Exception as e:
                                    logger.warning(
                                        "SIDE FLIP | reconcile after external close "
                                        "failed: {} — retrying next tick", e,
                                    )
                                    commit_flip = False
                        if commit_flip:
                            _last_side = position_side
                            grid.reset_trailing()
                            if _scale_out_done:
                                _scale_out_done = False
                                sl_orders = {}

                    # AUDIT #50. An unprotected position must not also be a growing one.
                    # On 08-08 the stop legs all failed to place and the grid kept
                    # adding to a position already through its cap. Whichever side would
                    # ADD exposure is blocked until a stop is live again; the exit side
                    # stays open so inventory can still unwind (#42).
                    sl_covered = True
                    if position_side == "long":
                        grid.update_trailing_sl(price)
                        _detect_trail_fill()
                        if _sl_needs_update("long", position_qty):
                            try:
                                sl_covered = _refresh_sl_stops("long", position_qty)
                                grid.log_sl_status("long")
                            except Exception as e:
                                sl_covered = False
                                logger.error("Failed to place/update stop-loss: {}", e)
                        if not sl_covered:
                            grid.block_side("buy", "stop-loss missing")
                    elif position_side == "short":
                        grid.update_trailing_sl_short(price)
                        _detect_trail_fill()
                        if _sl_needs_update("short", position_qty):
                            try:
                                sl_covered = _refresh_sl_stops("short", position_qty)
                                grid.log_sl_status("short")
                            except Exception as e:
                                sl_covered = False
                                logger.error("Failed to place/update stop-loss: {}", e)
                        if not sl_covered:
                            grid.block_side("sell", "stop-loss missing")
                    else:
                        if sl_orders:
                            # Flat: drop the stops. Only forget them if the sweep was
                            # actually confirmed -- None means the book was unreadable,
                            # and clearing belief there strands live stop orders that
                            # would arm against the NEXT position (AUDIT #54).
                            if exchange.cancel_all_stop_orders(settings.symbol) is None:
                                logger.warning(
                                    "STOP SWEEP UNVERIFIED | flat but the stop book could "
                                    "not be read — retrying next iteration",
                                )
                            else:
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
                    # A kill switch fed a frozen number is not a kill switch. If the
                    # income ledger has gone quiet, daily_net_pnl stops moving and the
                    # daily-loss limit can never trip, however badly the account is
                    # doing. Same rule as an unprotected position (#50): it may be
                    # closed and it may be held, but it may not grow (AUDIT #52).
                    if pnl_reconciler.is_stale(PNL_STALE_SECONDS):
                        age = pnl_reconciler.seconds_since_sync()
                        logger.error(
                            "PNL FEED STALE | no income data for {} — the daily-loss kill "
                            "switch is reading a frozen figure; blocking new exposure",
                            "ever" if age == float("inf") else f"{age / 60:.0f}m",
                        )
                        events.risk_check("pnl_feed_age", age, float(PNL_STALE_SECONDS), "STALE")
                        grid.block_side("buy", "PnL feed stale")
                        grid.block_side("sell", "PnL feed stale")

                    is_safe, is_fatal = risk.check_all(
                        equity, grid_sl_price, price, account_exposure,
                        side=position_side or "long",
                        daily_realized_pnl=pnl_reconciler.daily_net_pnl,
                    )
                    if not is_safe and is_fatal:
                        if not was_in_recovery:
                            grid.emergency_stop()
                            _reset_sl()
                            notifier.on_kill_switch("Risk limit breached")
                            events.recovery_event("start", risk.state.recovery_count)
                            # Use the actual backoff-scaled wait, not the raw config value --
                            # recovery_cooldown_remaining() already applies the same up-to-4x
                            # multiplier risk.py logs internally, and reading it here (right
                            # after trigger_kill_switch() set recovery_start_time) reflects the
                            # real wait instead of understating it on repeat recoveries.
                            effective_cooldown = risk.recovery_cooldown_remaining()
                            logger.warning("Entering recovery mode — will wait {}s then recalculate grid", effective_cooldown)
                            notifier.on_recovery_start(effective_cooldown, risk.state.recovery_count)
                            _notify_status(notifier, exchange, settings.symbol, price)
                        state_data = {
                            "grid": grid.to_dict(),
                            "risk": risk.to_dict(),
                            "trend": trend.to_dict(),
                            "pnl_reconciler": pnl_reconciler.to_dict(),
                            "last_update": datetime.now().isoformat(),
                        }
                        state_mgr.save(state_data)
                        sleep_until_next_poll(iteration_started, settings.poll_interval)
                        continue
                    elif not is_safe:
                        logger.warning("EXPOSURE WARNING | grid continues but new orders blocked")

                if not grid.active:
                    equity = exchange.get_total_equity()
                    balance = exchange.get_balance()

                    # A paused strategy is not a flat one. pause() deliberately keeps the
                    # position -- "the position stays under stop protection", per the
                    # Strategy protocol -- and the trend filter pauses on a confirmed
                    # TREND, which is precisely when a held position runs away from you.
                    #
                    # Everything below used to sit inside `if grid.active:`. So for the
                    # majority of this bot's life (measured at ~60% paused on DOGE, see
                    # AUDIT #52) the stops were never refreshed and NOT ONE kill switch
                    # was evaluated: no drawdown check, no daily-loss check, no stop-loss
                    # backstop, no consecutive-loss check, and daily_unrealized_pnl frozen
                    # at whatever it held when the grid went quiet.
                    #
                    # The risk layer was off duty exactly when it was needed (AUDIT #53).
                    held_side, held_qty = get_net_position(exchange, settings.symbol)
                    if held_side in ("long", "short") and held_qty > 0:
                        if held_side == "long":
                            grid.update_trailing_sl(price)
                        else:
                            grid.update_trailing_sl_short(price)

                        if _sl_needs_update(held_side, held_qty):
                            try:
                                if not _refresh_sl_stops(held_side, held_qty):
                                    logger.error(
                                        "PAUSED POSITION UNPROTECTED | {} {} has no "
                                        "stop-loss and the grid is paused",
                                        held_side, held_qty,
                                    )
                            except Exception as e:
                                logger.error("PAUSED POSITION | stop refresh failed: {}", e)

                        risk.update_unrealized(sum(
                            _position_unrealized_pnl(p, price)
                            for p in (get_position_details(exchange, settings.symbol) or [])
                        ))

                        held_sl = (grid.get_short_stop_loss_price() if held_side == "short"
                                   else grid.get_stop_loss_price())
                        was_paused_recovery = risk.is_in_recovery()
                        held_own = grid.get_exposure_pct(balance)
                        # A paused strategy still HOLDS its position, so it still spends
                        # account budget and still has to be seen by the other instances.
                        exposure_registry.publish(held_own, held_own * equity)
                        held_safe, held_fatal = risk.check_all(
                            equity, held_sl or 0.0, price,
                            exposure_registry.account_exposure_pct(held_own),
                            side=held_side,
                            daily_realized_pnl=pnl_reconciler.daily_net_pnl,
                        )
                        if not held_safe and held_fatal and not was_paused_recovery:
                            grid.emergency_stop("risk limit breached while paused")
                            _reset_sl()
                            notifier.on_kill_switch(
                                "Risk limit breached (grid paused, position still open)"
                            )
                            events.recovery_event("start", risk.state.recovery_count)

                # Periodic fallback sync so funding-fee settlements (which happen on a
                # schedule, independent of any grid fill) still get picked up promptly.
                if loop_count % 60 == 0:
                    pnl_reconciler.sync(exchange, settings.symbol)

                state_data = {
                    "grid": grid.to_dict(),
                    "risk": risk.to_dict(),
                    "trend": trend.to_dict(),
                    "pnl_reconciler": pnl_reconciler.to_dict(),
                    "last_update": datetime.now().isoformat(),
                }
                state_mgr.save(state_data)

                logger.info(
                    # "account(...)=-71.00" was read as a negative BALANCE. It is
                    # cumulative realised P&L since the PNL epoch -- ten days, of which
                    # -74.84 is one stop-out on 2026-08-20 -- sitting on the same line
                    # as an equity of 4,866. Say which is which (AUDIT #144).
                    "PRICE={} | fills={} | gross={:.2f} fees={:.2f} net={:.2f} | session_pnl={:+.2f} "
                    "pnl_since({})={:+.2f} today={:+.2f} | balance_free={:.2f} total_equity={:.2f} | "
                    "grid={} | regime={} | spread={:.4f}%",
                    price, grid.total_fills, grid.total_pnl, grid.total_fees,
                    grid.total_pnl - grid.total_fees,
                    pnl_reconciler.session_pnl, pnl_window,
                    pnl_reconciler.net_realized_pnl, pnl_reconciler.daily_net_pnl,
                    balance, equity,
                    "ON" if grid.active else "OFF",
                    f"{trend.regime.value}(adx={trend.adx_value:.1f})",
                    grid.get_spread_pct() * 100,
                )

                if grid.active and grid.total_fills > 0 and grid.total_fills % 10 == 0 and grid.total_fills != last_analytics_fill_count:
                    grid.log_analytics(balance)
                    last_analytics_fill_count = grid.total_fills

                consecutive_errors = 0
                sleep_until_next_poll(iteration_started, settings.poll_interval)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                consecutive_errors += 1
                signature = f"{type(e).__name__}: {e}"

                if isinstance(e, BUG_ERRORS):
                    # A defect in this program, not a market or network problem.
                    # Reconnecting cannot fix it and the old handler did exactly that
                    # every third iteration, burning the poll budget while logging only
                    # str(e) -- an AttributeError reads as the bare attribute name, so
                    # 27 minutes of log said "Loop error: _last_orderbook" and nothing
                    # else. Log the type and a traceback, alert once, and keep the loop
                    # running at full speed so whatever still works keeps working
                    # (AUDIT #31).
                    first_time = signature not in seen_bug_errors
                    if first_time or consecutive_errors % 10 == 0:
                        logger.opt(exception=True).error(
                            "BUG IN LOOP (consecutive={}) | {} | this is a code defect, "
                            "not a connection problem", consecutive_errors, signature,
                        )
                    else:
                        logger.error("BUG IN LOOP (consecutive={}) | {}", consecutive_errors, signature)
                    if first_time:
                        seen_bug_errors.add(signature)
                        try:
                            notifier.send(
                                f"<b>BOT DEFECT</b>\n{signature}\n"
                                f"The trading loop is raising every iteration. Orders "
                                f"already placed remain on the exchange."
                            )
                        except Exception:
                            pass
                    # Deliberately no auto-shutdown: emergency_stop cancels the resting
                    # stop-loss orders too, so killing the bot over a defect that may
                    # sit in the tail of the iteration would leave an open position
                    # with nothing protecting it. Loud and running beats silent and flat.
                    #
                    # A FULL interval here on purpose, not the deadline sleep the normal
                    # paths use. A deadline sleep shortens itself by the work already
                    # done, so an iteration that raises late in its run would retry almost
                    # immediately -- the loop would spin fastest on exactly the failures
                    # that take longest to reach.
                    time.sleep(settings.poll_interval)
                    continue

                logger.error("Loop error (consecutive={}): {}", consecutive_errors, signature)
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
                                "trend": trend.to_dict(),
                                "pnl_reconciler": pnl_reconciler.to_dict(),
                                "last_update": datetime.now().isoformat(),
                            }
                            state_mgr.save(state_data)
                        except Exception:
                            pass
                        abort_startup(transient=True)
                else:
                    time.sleep(10)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        try:
            if grid is not None:
                grid.emergency_stop(reason="shutdown")
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
                "trend": trend.to_dict(),
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
