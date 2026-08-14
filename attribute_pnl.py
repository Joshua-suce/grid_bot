"""Attribute realized PnL to the mechanism that caused it.

    python attribute_pnl.py [days]

Thirty days of ledger said maker fills netted +98.74 and taker fills -132.04. The grid
is profitable; something else is taking the money back. But "taker" covers stop-outs,
reconcile closes, emergency closes and crossed unwinds all at once, so it does not say
WHICH -- and you cannot fix what you cannot name.

Every order the bot places now carries a two-character purpose tag in its clientOrderId
(exchange.PURPOSE_TAGS). This joins userTrades to their orders and groups the realized
PnL by that tag. Orders placed before the tagging landed show up as "untagged".

Read-only: fetches trades and orders, places nothing.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

from loguru import logger

from config import settings
from exchange import Exchange, purpose_of_client_order_id

DAY_MS = 24 * 3600 * 1000


def _paged(fetch, raw_symbol: str, start: int, end: int, window_ms: int) -> list[dict]:
    """Binance caps several of these endpoints at a 7-day span per query."""
    out: list[dict] = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + window_ms - 1, end)
        cursor = window_start
        while True:
            page = fetch({
                "symbol": raw_symbol, "startTime": cursor,
                "endTime": window_end, "limit": 1000,
            })
            if not page:
                break
            out.extend(page)
            last = int(page[-1]["time"])
            if len(page) < 1000 or last <= cursor:
                break
            cursor = last + 1
        window_start = window_end + 1
    return out


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    ex = Exchange(settings.exchange_config, demo=settings.demo_mode)
    raw_symbol = settings.symbol.replace("/", "").split(":")[0]

    end = int(time.time() * 1000)
    start = end - days * DAY_MS

    trades = _paged(ex.exchange.fapiPrivateGetUserTrades, raw_symbol, start, end, 7 * DAY_MS)
    orders = _paged(ex.exchange.fapiPrivateGetAllOrders, raw_symbol, start, end, 7 * DAY_MS)
    logger.info("{} executions, {} orders over {} days", len(trades), len(orders), days)

    purpose_by_order = {
        str(o.get("orderId")): purpose_of_client_order_id(o.get("clientOrderId"))
        for o in orders
    }

    buckets = defaultdict(lambda: {"pnl": 0.0, "comm": 0.0, "n": 0, "notional": 0.0,
                                   "wins": 0, "losses": 0, "worst": 0.0})
    for t in trades:
        purpose = purpose_by_order.get(str(t.get("orderId")), "untagged")
        pnl = float(t["realizedPnl"])
        b = buckets[purpose]
        b["pnl"] += pnl
        b["comm"] -= float(t["commission"])
        b["notional"] += float(t["quoteQty"])
        b["n"] += 1
        if pnl > 0:
            b["wins"] += 1
        elif pnl < 0:
            b["losses"] += 1
            b["worst"] = min(b["worst"], pnl)

    print(f"\n{'purpose':<12} {'execs':>6} {'notional':>12} {'realized':>10} "
          f"{'comm':>9} {'NET':>10} {'W/L':>10} {'worst':>9}")
    print("-" * 84)
    total = 0.0
    for purpose in sorted(buckets, key=lambda k: buckets[k]["pnl"] + buckets[k]["comm"]):
        b = buckets[purpose]
        net = b["pnl"] + b["comm"]
        total += net
        win_loss = f"{b['wins']}/{b['losses']}"
        print(f"{purpose:<12} {b['n']:>6} {b['notional']:>12,.0f} {b['pnl']:>10.2f} "
              f"{b['comm']:>9.2f} {net:>10.2f} {win_loss:>10} {b['worst']:>9.2f}")
    print("-" * 84)
    print(f"{'TOTAL':<12} {'':>6} {'':>12} {'':>10} {'':>9} {total:>10.2f}")

    untagged = buckets.get("untagged", {}).get("n", 0)
    if untagged:
        print(f"\n{untagged} executions predate purpose tagging (AUDIT #56) and cannot be "
              f"attributed. Re-run after the bot has traded for a while.")


if __name__ == "__main__":
    main()
