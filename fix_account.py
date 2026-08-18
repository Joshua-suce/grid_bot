"""Put the account's leverage, margin mode and position mode back where config expects.

Binance testnet resets wipe account settings. On 2026-08-18 this one came back as 10x
ISOLATED with a 1,800,000 max notional, against the 25x CROSS / 600,000 the previous
session ran on -- so startup refused to trade, correctly:

    ACCOUNT CONFIG | leverage=10x | margin=isolated | position mode=one-way
    ACCOUNT NOT SAFE TO TRADE | 1 problem(s) found
      - leverage mismatch: config says 25x, the exchange is on 10x. Order notional is
        CAPITAL_PER_GRID_USDT x LEVERAGE, so every margin figure would be out by 2.5x

That refusal is the right behaviour and this script is the other half of it: the thing
a human runs to make the account match the config again.

    py fix_account.py            # DRY RUN -- reports what it would change, changes nothing
    py fix_account.py --apply    # actually changes the account

It changes ACCOUNT SETTINGS, not positions. It never opens, closes or sizes a trade,
and it refuses to touch anything while a position or a resting order exists -- Binance
rejects a margin-type change in that state anyway (-4047 / -4048), and doing it under
an open position would silently reprice liquidation.

WHICH ACCOUNT. It acts on whichever account DEMO_MODE selects, and says which one in
capitals before doing anything. There is no flag to override that: the account is
whatever .env points at.
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from config import settings
from exchange import Exchange

# Binance answers "you asked for what is already set" with its own error code rather
# than a success, on both of these. Idempotence is the point of the script, so they
# are outcomes, not failures.
ALREADY_SET = {"-4046", "no need to change margin type"}
POSITION_BLOCKS = {"-4047", "-4048", "existing position", "open orders"}


def _bare_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace(":USDT", "")


def describe(cfg: dict | None) -> str:
    if cfg is None:
        return "unreadable"
    return (f"leverage={cfg['leverage']}x | margin={cfg['margin_mode']} | "
            f"position mode={'HEDGE' if cfg['dual_side'] else 'one-way'} | "
            f"max notional={cfg['max_notional']:,.0f}")


def open_exposure(exchange: Exchange, symbol: str) -> tuple[float, int]:
    """(net position qty, resting order count). Both must be zero to change anything."""
    qty = 0.0
    for pos in exchange.get_positions(symbol) or []:
        try:
            qty += abs(float(pos.get("contracts") or pos.get("positionAmt") or 0))
        except (TypeError, ValueError):
            pass
    orders = exchange.get_open_orders(symbol) or []
    return qty, len(orders)


def set_margin_mode(exchange: Exchange, symbol: str, mode: str) -> bool:
    """`mode` is 'cross' or 'isolated'. Binance's own wire value is CROSSED."""
    wire = "CROSSED" if mode == "cross" else "ISOLATED"
    try:
        exchange.exchange.fapiPrivatePostMarginType(
            {"symbol": _bare_symbol(symbol), "marginType": wire})
        return True
    except Exception as e:
        text = str(e).lower()
        if any(marker in text for marker in ALREADY_SET):
            logger.info("MARGIN MODE | already {} — nothing to do", mode)
            return True
        if any(marker in text for marker in POSITION_BLOCKS):
            logger.error(
                "MARGIN MODE | refused because the symbol still has a position or a "
                "resting order: {}", e)
            return False
        logger.error("MARGIN MODE | {} rejected: {}", mode, e)
        return False


def set_position_mode(exchange: Exchange, one_way: bool = True) -> bool:
    """One-way, i.e. a single NET position. The bot sends no positionSide, which hedge
    mode rejects outright (-4061), and its reduce-only stops assume one net position."""
    try:
        exchange.exchange.fapiPrivatePostPositionSideDual(
            {"dualSidePosition": "false" if one_way else "true"})
        return True
    except Exception as e:
        text = str(e).lower()
        if "-4059" in text or "no need to change position side" in text:
            logger.info("POSITION MODE | already one-way — nothing to do")
            return True
        logger.error("POSITION MODE | could not set one-way: {}", e)
        return False


def plan(cfg: dict) -> list[tuple[str, str, str]]:
    """(what, from, to) for every setting that does not match config."""
    todo = []
    if cfg["leverage"] != settings.leverage:
        todo.append(("leverage", f"{cfg['leverage']}x", f"{settings.leverage}x"))
    if cfg["margin_mode"] != "cross":
        todo.append(("margin mode", cfg["margin_mode"], "cross"))
    if cfg["dual_side"]:
        todo.append(("position mode", "HEDGE", "one-way"))
    return todo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true",
                    help="actually change the account (default is a dry run)")
    args = ap.parse_args(argv)

    mode = "DEMO (testnet)" if settings.demo_mode else "LIVE — REAL MONEY"
    print("=" * 68)
    print(f"  ACCOUNT: {mode}")
    print(f"  SYMBOL:  {settings.symbol}")
    print(f"  MODE:    {'APPLY — this will change account settings' if args.apply else 'DRY RUN — nothing will change'}")
    print("=" * 68)

    exchange = Exchange(settings.exchange_config, demo=settings.demo_mode)

    before = exchange.get_account_config(settings.symbol)
    if before is None:
        logger.error("Could not read the account. Nothing changed.")
        return 1
    print(f"\ncurrent:  {describe(before)}")
    print(f"config:   leverage={settings.leverage}x | margin=cross | position mode=one-way")

    todo = plan(before)
    if not todo:
        print("\nAccount already matches config. Nothing to do.")
        return 0

    print("\nwould change:")
    for what, old, new in todo:
        print(f"  {what:<14} {old:>10}  ->  {new}")

    qty, orders = open_exposure(exchange, settings.symbol)
    if qty > 0 or orders:
        print(f"\nREFUSING: position={qty:g} contracts, {orders} resting order(s).")
        print("Binance rejects a margin-type change in that state, and changing leverage")
        print("under an open position reprices liquidation without telling you.")
        print("Close the position and cancel the orders first.")
        return 2

    if not args.apply:
        print("\nDry run. Re-run with --apply to make these changes:")
        print("    py fix_account.py --apply")
        return 0

    # Margin mode first: it is the one Binance refuses under any open interest, so a
    # failure there should stop the run before leverage has been touched.
    ok = True
    if before["margin_mode"] != "cross":
        ok = set_margin_mode(exchange, settings.symbol, "cross") and ok
    if before["dual_side"]:
        ok = set_position_mode(exchange, one_way=True) and ok
    if ok and before["leverage"] != settings.leverage:
        ok = exchange.set_leverage(settings.symbol, settings.leverage) and ok

    # Verify from the exchange rather than from the replies. set_leverage logged success
    # on 2026-08-18 while the account stayed on 10x, which is exactly the case a
    # confirming read exists to catch.
    after = exchange.get_account_config(settings.symbol)
    print(f"\nnow:      {describe(after)}")
    if after is None:
        logger.error("Could not read the account back. Verify manually before starting.")
        return 1

    remaining = plan(after)
    if remaining:
        print("\nSTILL WRONG:")
        for what, old, new in remaining:
            print(f"  {what:<14} {old:>10}  should be {new}")
        return 1

    print("\nAccount now matches config. The bot will pass its startup check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
