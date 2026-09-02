"""Attribute realized PnL to the mechanism that caused it.

    python attribute_pnl.py [days] [--json]

Thirty days of ledger said maker fills netted +98.74 and taker fills -132.04. The grid
is profitable; something else is taking the money back. But "taker" covers stop-outs,
reconcile closes, emergency closes and crossed unwinds all at once, so it does not say
WHICH -- and you cannot fix what you cannot name.

Every order the bot places now carries a two-character purpose tag in its clientOrderId
(exchange.PURPOSE_TAGS). This joins userTrades to their orders and groups the realized
PnL by that tag. Orders placed before the tagging landed show up as "untagged".

--json additionally writes logs/attribution_{demo,live}.json. dashboard.py reads that
cache rather than calling the exchange itself: the bot shares a rate limit with anything
using the same key, and its order placement is the latency-sensitive part. Caching keeps
the dashboard free of exchange calls and makes the staleness visible instead of hidden.

Read-only: fetches trades and orders, places nothing.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from config import settings
from exchange import Exchange, purpose_of_client_order_id

DAY_MS = 24 * 3600 * 1000


def _paged(fetch, raw_symbol: str, start: int, end: int, window_ms: int) -> list[dict]:
    """Binance caps several of these endpoints at a 7-day span per query.

    Resumes each inner page AT the last row's timestamp (inclusive), not
    last+1, and dedupes by the row's own unique id. pnl_audit.py's docstring
    documents why the obvious "+1" cursor is wrong: a full page ending
    mid-millisecond silently drops any sibling record stamped identically that
    did not fit -- there it cost 128 rows and the wrong SIGN on the headline.
    userTrades rows carry a unique `id`; allOrders rows don't have one but
    `orderId` is unique for them, so prefer `id` and fall back to `orderId`.
    """
    out: list[dict] = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + window_ms - 1, end)
        cursor = window_start
        seen: set = set()
        while True:
            page = fetch({
                "symbol": raw_symbol, "startTime": cursor,
                "endTime": window_end, "limit": 1000,
            })
            if not page:
                break
            added = 0
            for r in page:
                key = r.get("id") if r.get("id") is not None else r.get("orderId")
                if key in seen:
                    continue
                seen.add(key)
                out.append(r)
                added += 1
            last = int(page[-1]["time"])
            if len(page) < 1000:
                break                       # short page: this window is exhausted
            if added == 0:
                break                       # a full page we have already read: no progress
            cursor = last
        window_start = window_end + 1
    return out


def write_cache(buckets: dict, days: int, log_dir: str, demo: bool) -> Path:
    """Snapshot for dashboard.py. Named per account for the same reason the journals
    are (AUDIT #70): demo executions and live executions describe different money."""
    path = Path(log_dir) / f"attribution_{'demo' if demo else 'live'}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "buckets": {
            name: {
                "n": b["n"], "notional": round(b["notional"], 2),
                "pnl": round(b["pnl"], 4), "comm": round(b["comm"], 4),
                "net": round(b["pnl"] + b["comm"], 4),
                "wins": b["wins"], "losses": b["losses"], "worst": round(b["worst"], 4),
            }
            for name, b in buckets.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    argv = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv
    days = int(argv[0]) if argv else 30
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

    # The maker/taker split is the headline: 30 days of ledger said maker fills netted
    # +98.74 and taker fills -132.04. Purpose tags say WHICH taker path; this says how
    # much of the money moved through a forced exit at all.
    maker = {"pnl": 0.0, "comm": 0.0, "n": 0, "notional": 0.0}
    taker = {"pnl": 0.0, "comm": 0.0, "n": 0, "notional": 0.0}
    for t in trades:
        b = maker if t.get("maker") else taker
        b["pnl"] += float(t["realizedPnl"])
        b["comm"] -= float(t["commission"])
        b["notional"] += float(t["quoteQty"])
        b["n"] += 1
    print(f"\n{'':<12} {'execs':>6} {'notional':>12} {'realized':>10} {'comm':>9} {'NET':>10}")
    print("-" * 62)
    for label, b in (("maker (grid)", maker), ("TAKER (forced)", taker)):
        print(f"{label:<12} {b['n']:>6} {b['notional']:>12,.0f} {b['pnl']:>10.2f} "
              f"{b['comm']:>9.2f} {b['pnl'] + b['comm']:>10.2f}")
    if taker["n"] and maker["n"]:
        print(f"\ntaker is {taker['n'] / (taker['n'] + maker['n']):.1%} of executions but "
              f"{taker['notional'] / max(1e-9, maker['notional'] + taker['notional']):.1%} "
              f"of notional — forced exits move bigger size than grid cycles")

    untagged = buckets.get("untagged", {}).get("n", 0)
    if untagged:
        print(f"\n{untagged} executions predate purpose tagging (AUDIT #56) and cannot be "
              f"attributed. Re-run after the bot has traded for a while.")

    if as_json:
        combined = dict(buckets)
        combined["_maker"] = {**maker, "wins": 0, "losses": 0, "worst": 0.0}
        combined["_taker"] = {**taker, "wins": 0, "losses": 0, "worst": 0.0}
        path = write_cache(combined, days, settings.log_dir, settings.demo_mode)
        print(f"\nwrote {path} — dashboard.py reads this instead of calling the exchange")


if __name__ == "__main__":
    main()
