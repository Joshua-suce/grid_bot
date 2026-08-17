"""Measure the GRID-CYCLE taker share from the exchange, and say whether config is safe.

`taker_fill_share_pct` sets round_trip_fee_pct, which sets every break-even price and the
level-profitability floor. Understating it is the AUDIT #51 failure: exits clamped to
"break-even" book a real loss, and the floor lets through levels that do not clear their
own fees. So this is not a curiosity, it is the input that decides whether the bot's idea
of profit is the same as the exchange's.

The number cannot be read off the account total. Measured 2026-08-17 over 35 days:

    purpose        fills      notional   taker share
    untagged        3033     123115.3u        44.9%
    grid_entry       244       6900.4u         0.0%
    reconcile          3        469.6u       100.0%

Account-wide is 41.6% and grid CYCLES are 0.0% -- the taker is all forced exits, stops,
reconcile closes and crossed unwinds. config.py says as much and warns against using the
account figure; this joins userTrades -> orders -> clientOrderId purpose tag so the two
can actually be told apart (AUDIT #56 tagging).

Why it must be re-run on a live account: that 0.0% was measured on testnet, whose book is
thin and synthetic, so post-only orders rest trivially. On a real book they queue behind
real flow. If the live grid-cycle share is materially above the configured 5%, the fee
floor is too low and every break-even price is short -- BEFORE any decision to scale size.

Usage:
    py taker_share.py                 # whichever account .env points at
    py taker_share.py --days 7
    py taker_share.py --min-fills 200

Exit codes: 0 usable and config is safe, 1 config understates the share, 2 sample too
small to conclude anything.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

Z95 = 1.959964

# Cycle legs, as opposed to forced exits (stop_trail, stop_hard, reconcile, unwind,
# emergency). Only these pay the fee that round_trip_fee_pct is supposed to describe.
GRID_PURPOSES = ("grid_entry", "grid_exit")

# Below this the share is not an estimate, it is an anecdote. 200 fills puts the 95%
# interval at roughly +/-7 points at a 5% share, which is tight enough to act on.
DEFAULT_MIN_FILLS = 200


@dataclass
class TakerSample:
    """Grid-cycle fills over some window."""

    fills: int = 0
    taker_fills: int = 0
    notional: float = 0.0
    taker_notional: float = 0.0
    commission: float = 0.0

    @property
    def share_by_notional(self) -> float:
        """The share the fee model actually needs -- fees are charged on notional."""
        return self.taker_notional / self.notional if self.notional else 0.0

    @property
    def share_by_count(self) -> float:
        return self.taker_fills / self.fills if self.fills else 0.0

    @property
    def realised_rate(self) -> float:
        """Commission actually paid per unit notional, one side."""
        return self.commission / self.notional if self.notional else 0.0

    def interval(self) -> tuple[float, float]:
        """95% interval on the share, from the COUNT proportion.

        Notional-weighting is the right point estimate but has no clean interval; the
        count proportion does, and the two track each other closely enough to size the
        uncertainty. Wilson rather than normal-approximation because the share is near
        zero, where the normal interval famously produces negative lower bounds and, at
        exactly zero successes, a width of nothing at all -- which would read as
        certainty from a sample that has merely not seen a taker fill yet.
        """
        return wilson_interval(self.taker_fills, self.fills)


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fee_model(maker: float, taker: float, share: float,
              min_profit_multiplier: float) -> dict[str, float]:
    """Round trip and level floor implied by a taker share."""
    per_side = maker * (1.0 - share) + taker * share
    round_trip = 2.0 * per_side
    return {
        "per_side": per_side,
        "round_trip": round_trip,
        "floor": round_trip * min_profit_multiplier,
    }


def verdict(sample: TakerSample, configured: float,
            min_fills: int = DEFAULT_MIN_FILLS) -> tuple[str, str]:
    """(code, explanation). Codes: INSUFFICIENT, UNDERSTATED, ELEVATED, OK.

    The asymmetry is deliberate. Overstating the taker share costs a little trading --
    the floor is higher than it needs to be and some profitable levels are skipped.
    UNDERSTATING it means the bot computes break-even prices that are not break-even and
    books small real losses as flat, which is what AUDIT #51 found. So the alarm fires on
    evidence that the truth is ABOVE config, not merely different from it.
    """
    if sample.fills < min_fills:
        return ("INSUFFICIENT",
                f"only {sample.fills} grid-cycle fills; {min_fills} needed before this "
                f"share means anything. Nothing about the config is confirmed or refuted")

    lo, hi = sample.interval()
    measured = sample.share_by_notional
    if lo > configured:
        return ("UNDERSTATED",
                f"the share is above the configured {configured:.1%} with 95% confidence "
                f"(interval {lo:.1%}-{hi:.1%}). Break-even prices are short and the level "
                f"floor is too low — fix the config before trading on it")
    if measured > configured:
        return ("ELEVATED",
                f"point estimate {measured:.1%} is above the configured {configured:.1%}, "
                f"but the interval {lo:.1%}-{hi:.1%} still covers it. Not yet actionable; "
                f"re-run with more fills")
    return ("OK",
            f"measured {measured:.1%} against a configured {configured:.1%}, interval "
            f"{lo:.1%}-{hi:.1%}. The config is at or above the truth, which is the safe side")


# --------------------------------------------------------------------------------
# the runner (network; everything above is pure)
# --------------------------------------------------------------------------------

def tally(rows) -> tuple[TakerSample, dict[str, TakerSample]]:
    """Bucket (purpose, notional, commission, is_taker) rows into per-purpose samples.

    Returns (grid_cycles, by_purpose). Separated from the fetching because deciding WHICH
    fills count as a cycle fee is the whole judgement this tool makes, and it should not
    require a network round trip to test.
    """
    by_purpose: dict[str, TakerSample] = {}
    grid = TakerSample()
    for purpose, cost, fee, is_taker in rows:
        if cost <= 0:
            continue
        buckets = [by_purpose.setdefault(purpose, TakerSample())]
        if purpose in GRID_PURPOSES:
            buckets.append(grid)
        for bucket in buckets:
            bucket.fills += 1
            bucket.notional += cost
            bucket.commission += fee
            if is_taker:
                bucket.taker_fills += 1
                bucket.taker_notional += cost
    return grid, by_purpose


def collect(exchange, symbol: str, days: int) -> tuple[TakerSample, dict[str, TakerSample]]:
    """Join userTrades -> orders -> purpose tag over the window."""
    from exchange import purpose_of_client_order_id

    DAY = 86_400_000
    now = int(time.time() * 1000)
    trades: dict[str, dict] = {}
    orders: dict[str, str] = {}

    start = now - days * DAY
    while start < now:
        end = min(start + 6 * DAY, now)
        window = {"startTime": start, "endTime": end}
        try:
            for t in exchange.exchange.fetch_my_trades(symbol, params=window, limit=1000):
                trades[t["id"]] = t
        except Exception as e:
            print(f"  trades {time.strftime('%m-%d', time.localtime(start/1000))}: "
                  f"{type(e).__name__}")
        try:
            for o in exchange.exchange.fetch_orders(symbol, params=window, limit=1000):
                orders[str(o.get("id"))] = (
                    o.get("clientOrderId")
                    or (o.get("info") or {}).get("clientOrderId") or "")
        except Exception as e:
            print(f"  orders {time.strftime('%m-%d', time.localtime(start/1000))}: "
                  f"{type(e).__name__}")
        start = end + 1

    def rows():
        for t in trades.values():
            cid = orders.get(str(t.get("order")))
            yield (
                purpose_of_client_order_id(cid) if cid is not None else "UNMAPPED",
                float(t.get("cost") or 0.0),
                float((t.get("fee") or {}).get("cost") or 0.0),
                t.get("takerOrMaker") == "taker",
            )

    return tally(rows())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=35)
    ap.add_argument("--min-fills", type=int, default=DEFAULT_MIN_FILLS)
    a = ap.parse_args()

    from config import settings
    from exchange import Exchange

    account = "DEMO (testnet)" if settings.demo_mode else "LIVE"
    ex = Exchange(settings.exchange_config, demo=settings.demo_mode)
    print(f"\naccount: {account}   symbol: {settings.symbol}   window: {a.days}d\n")

    grid, by_purpose = collect(ex, settings.symbol, a.days)

    print(f"  {'purpose':<14} {'fills':>7} {'notional':>13} {'taker':>8} {'rate/side':>11}")
    print("  " + "-" * 58)
    for purpose in sorted(by_purpose, key=lambda p: -by_purpose[p].notional):
        s = by_purpose[purpose]
        mark = "  <- cycle" if purpose in GRID_PURPOSES else ""
        print(f"  {purpose:<14} {s.fills:>7} {s.notional:>12.1f}u "
              f"{s.share_by_notional:>7.1%} {s.realised_rate:>10.4%}{mark}")
    print("  " + "-" * 58)
    print(f"  {'GRID CYCLES':<14} {grid.fills:>7} {grid.notional:>12.1f}u "
          f"{grid.share_by_notional:>7.1%} {grid.realised_rate:>10.4%}")

    configured = settings.taker_fill_share_pct / 100
    maker, taker = settings.maker_fee_pct / 100, settings.taker_fee_pct / 100
    mult = settings.min_profit_multiplier
    lo, hi = grid.interval()

    print(f"\n  share by notional {grid.share_by_notional:>7.1%}   by count "
          f"{grid.share_by_count:>6.1%}   95% CI {lo:.1%}-{hi:.1%}")
    print(f"  configured        {configured:>7.1%}\n")

    print(f"  {'':<12} {'round trip':>11} {'floor':>10} {'spacing 0.200% is':>19}")
    for name, share in (("configured", configured),
                        ("measured", grid.share_by_notional),
                        ("CI upper", hi)):
        m = fee_model(maker, taker, share, mult)
        print(f"  {name:<12} {m['round_trip']:>10.4%} {m['floor']:>9.4%} "
              f"{0.002 / m['floor']:>18.2f}x")

    code, why = verdict(grid, configured, a.min_fills)
    print(f"\n  {code}: {why}")
    if not settings.demo_mode:
        print("\n  This is LIVE data — it supersedes any testnet measurement.")
    else:
        print("\n  This is TESTNET: a thin synthetic book rests post-only orders that a "
              "real\n  one would queue behind real flow.\n"
              "  Do NOT carry this share to live.")
    return {"OK": 0, "ELEVATED": 0, "UNDERSTATED": 1, "INSUFFICIENT": 2}[code]


if __name__ == "__main__":
    sys.exit(main())
