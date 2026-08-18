"""Move the bot to a different symbol, safely. AUDIT #112.

A symbol change is three things that must happen in order, and getting the order wrong
strands money:

  1. the CURRENT symbol must be flat, with no resting orders. Switching away from an
     open position abandons it -- the stop stays on the exchange but nothing manages
     it, unwinds it, or reports it. This script REFUSES rather than doing that, and it
     does not close positions itself: that is a trade, and it is yours to place.
  2. the new symbol must actually suit the bot -- min notional under one rung, and a
     price tick fine enough that rungs do not collide at the fee floor.
  3. the saved state must go. It holds the OLD symbol's bounds, levels and inventory,
     and restoring it against a new symbol is meaningless. Archived, not deleted, so a
     switch back does not start blind.

    py switch_symbol.py ADAUSDT           # dry run: check and report, change nothing
    py switch_symbol.py ADAUSDT --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from loguru import logger

from config import settings
from exchange import Exchange
from grid import MIN_NOTIONAL_USDT, calculate_grid_range

# Below this the ladder has too little price resolution: rungs round onto each other and
# the fee-floor deformation check starts firing on a grid that was built correctly.
MIN_TICKS_PER_RUNG = 5


def preflight(exchange: Exchange, symbol: str) -> tuple[bool, list[str]]:
    """Does this symbol fit the bot's mechanical requirements?"""
    notes, ok = [], True
    try:
        market = exchange.exchange.market(symbol)
    except Exception as e:
        return False, [f"{symbol} is not tradable on this account: {e}"]

    price = exchange.get_price(symbol)
    per_order = settings.capital_per_grid_usdt * settings.leverage
    notes.append(f"price {price:.6f}   rung {per_order:.0f} USDT notional")

    min_notional = exchange.get_min_notional(symbol)
    if min_notional is not None and per_order < min_notional:
        ok = False
        notes.append(f"REJECT: exchange minimum is {min_notional:.2f} USDT, "
                     f"a rung is only {per_order:.2f}")
    else:
        notes.append(f"min notional {min_notional} — a rung clears it")

    tick = float(market["precision"]["price"])
    from regime_study import fetch_klines
    ohlcv = fetch_klines(symbol, settings.grid_timeframe)
    lower, upper = calculate_grid_range(
        ohlcv, price, lookback_days=settings.range_lookback_days,
        atr_multiplier=settings.range_atr_multiplier,
        timeframe=settings.grid_timeframe, mode=settings.range_mode)
    spacing = (upper - lower) / settings.grid_count
    ticks = spacing / tick if tick > 0 else float("inf")
    floor = 2 * settings.maker_fee_pct * settings.min_profit_multiplier / 100
    notes.append(f"range {(upper-lower)/price:.2%} over {settings.grid_count} rungs "
                 f"= {spacing/price:.3%} spacing")
    notes.append(f"        {ticks:.1f} ticks per rung, {spacing/price/floor:.1f}x the "
                 f"{floor:.3%} fee floor")
    if ticks < MIN_TICKS_PER_RUNG:
        ok = False
        notes.append(f"REJECT: only {ticks:.1f} ticks between rungs — they will round "
                     f"onto each other")
    if spacing / price < floor:
        ok = False
        notes.append("REJECT: spacing is under the fee floor; no cycle can profit")
    return ok, notes


def open_exposure(exchange: Exchange, symbol: str) -> tuple[float, int]:
    qty = 0.0
    for pos in exchange.get_positions(symbol) or []:
        try:
            qty += abs(float(pos.get("contracts") or pos.get("positionAmt") or 0))
        except (TypeError, ValueError):
            pass
    return qty, len(exchange.get_open_orders(symbol) or [])


def state_path(symbol: str) -> Path:
    suffix = "_demo" if settings.demo_mode else ""
    return Path(settings.state_dir) / f"grid_{symbol.lower()}{suffix}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("symbol", help="the symbol to move to, e.g. ADAUSDT")
    ap.add_argument("--apply", action="store_true",
                    help="actually switch (default is a dry run)")
    args = ap.parse_args(argv)
    new = args.symbol.upper()
    old = settings.symbol

    print("=" * 70)
    print(f"  {'DEMO (testnet)' if settings.demo_mode else 'LIVE — REAL MONEY'}")
    print(f"  {old}  ->  {new}")
    print(f"  {'APPLY' if args.apply else 'DRY RUN — nothing will change'}")
    print("=" * 70)
    if new == old:
        print("\nAlready on that symbol.")
        return 0

    exchange = Exchange(settings.exchange_config, demo=settings.demo_mode)

    print(f"\n1. is {old} flat?")
    qty, orders = open_exposure(exchange, old)
    if qty > 0 or orders:
        print(f"   NO — {qty:g} contracts open, {orders} resting order(s).")
        print(f"   Switching now would abandon that position: its stop stays on the")
        print(f"   exchange but nothing manages or unwinds it.")
        print(f"\n   Close it first. This script will not place that trade for you.")
        return 2
    print(f"   yes — flat, no resting orders")

    print(f"\n2. does {new} suit the bot?")
    ok, notes = preflight(exchange, new)
    for n in notes:
        print(f"   {n}")
    if not ok:
        print(f"\n   {new} is not a good fit. Nothing changed.")
        return 1

    print(f"\n3. saved state")
    old_state = state_path(old)
    if old_state.exists():
        print(f"   {old_state} holds {old}'s bounds and levels — will be archived")
    else:
        print(f"   no saved state for {old}")

    if not args.apply:
        print(f"\nDry run. Re-run with --apply to switch:")
        print(f"    py switch_symbol.py {new} --apply")
        return 0

    if old_state.exists():
        archived = old_state.with_suffix(f".json.switched-from-{old.lower()}")
        shutil.move(str(old_state), str(archived))
        print(f"   archived -> {archived}")

    env = Path(".env")
    raw = env.read_bytes()
    if f"SYMBOL={old}".encode() not in raw:
        logger.error("Could not find SYMBOL=%s in .env — change it by hand", old)
        return 1
    env.write_bytes(raw.replace(f"SYMBOL={old}".encode(), f"SYMBOL={new}".encode(), 1))
    print(f"   .env SYMBOL -> {new}")

    print(f"\nDone. Start the bot and it will build a fresh {new} ladder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
