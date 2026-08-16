"""Ask the exchange what the account actually did. AUDIT #87.

Every other number in this repo is the bot's own bookkeeping, and the bot's own
bookkeeping has been wrong twice: it overstated P&L by 69% until #80, and the replay
harness has produced three different "best" spacings, none of which survived contact
with real money.

Binance keeps its own ledger -- REALIZED_PNL, COMMISSION, FUNDING_FEE -- and it is the
only account of what happened that does not pass through code written here. This reads
it. pnl_tracker.py already consumes the same endpoint to reconcile a running session;
this is the offline, whole-history view, for answering "is this thing making money"
rather than "what is my P&L right now".

What it found, 24.1 days to 2026-08-16 (verified twice, see the pagination note below):

    REALIZED_PNL    +18.6902      the trades themselves, 1737 entries
    COMMISSION      -23.1663      the fee bill, 2607 charged fills
    FUNDING_FEE      +3.0270
    NET              -1.4492      = -0.06 USDT/day

So the account is flat, not bleeding -- but the flatness is two large events nearly
cancelling, which is not the same thing as a strategy that breaks even:

    2026-08-08   -49.193   a short the grid had grown from 18,022 to 31,761 DOGE,
                           market-closed by startup cleanup on a restart (fixed 7f89b88)
    2026-07-26   +52.028

Strip both out and the structural picture appears:

    realized +12.8045, commission -20.1955, funding +3.1061  ->  net -4.2849

The fee bill is 158% of the gross edge. That is the single most important fact about
this bot: it is not short of opportunities, it is short of margin per opportunity. No
replay produces this -- a harness that fills on touch manufactures the fills that fees
then eat, which is why replay --sweep keeps preferring tight spacings (see replay.py)
while the live record prefers the opposite.

Corroboration from the far end of the range: 2026-08-13..16 traded 16-39 fills/day
instead of 350, and netted +7.15 on gross +7.96 against fees of just -0.79 -- fees 10%
of edge rather than 158%. Fewer, wider, better-paid fills is the only configuration in
the record that is consistently positive.

Hence outlier_days(): a mean is the wrong summary for a distribution containing a -49
and a +52, and "-0.06 USDT/day" alone would be true and misleading at the same time.

PAGINATION, and why the numbers above say "verified twice": Binance pages this endpoint
by startTime, and resuming at newest+1 -- the obvious thing, and what the first cut of
this module did -- steps over every entry that shares the boundary millisecond. That
silently dropped 128 rows worth +27.37 and produced a headline of -28.82 instead of
-1.45, i.e. the wrong SIGN on the question the module exists to answer. fetch_income
now resumes AT newest and leans on tranId dedup, and the totals were checked against an
independent day-anchored sweep (31 overlapping windows, no shared pagination path):
both return 4404 rows and -1.4492 exactly.

Usage:
    py pnl_audit.py                 # 90 days, or as far back as Binance serves
    py pnl_audit.py --days 7
    py pnl_audit.py --json          # machine-readable, for a cron
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone

INCOME_TYPES = ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE")


def fetch_income(exchange, symbol: str, days: int = 90,
                 max_requests: int = 100) -> list[dict]:
    """Every income entry for the symbol, paginated back `days`.

    Binance pages by startTime and caps each response, so this walks forward from the
    oldest entry. Entries are deduplicated on tranId because a page boundary that lands
    inside a millisecond will otherwise repeat rows, and a repeated COMMISSION row
    silently inflates the fee bill.
    """
    now = int(time.time() * 1000)
    since = now - days * 86400 * 1000
    rows: list[dict] = []
    seen: set = set()
    for _ in range(max_requests):
        batch = exchange.get_income_history(symbol, since_ms=since, limit=1000)
        if not batch:
            break
        added = 0
        for r in batch:
            key = r.get("tranId") or (r.get("time"), r.get("incomeType"), r.get("income"))
            if key not in seen:
                seen.add(key)
                rows.append(r)
                added += 1
        if len(batch) < 1000:
            break                       # short page: the ledger is exhausted
        if added == 0:
            break                       # a full page we have already read: no progress
        # Resume AT `newest`, not newest+1. Entries routinely share a millisecond -- a
        # fill's COMMISSION and REALIZED_PNL are stamped identically -- so +1 steps over
        # whichever of them did not fit in the page. Re-reading the boundary costs one
        # request; skipping it silently drops real rows, and a dropped COMMISSION
        # understates the fee bill this module exists to measure.
        since = max(int(r["time"]) for r in batch)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Totals by income type, plus the net that actually hit the balance."""
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        totals[r["incomeType"]] += float(r["income"])
        counts[r["incomeType"]] += 1
    span_days = 0.0
    if rows:
        t0 = min(int(r["time"]) for r in rows)
        t1 = max(int(r["time"]) for r in rows)
        span_days = (t1 - t0) / 86400000
    net = sum(totals.values())
    return {
        "totals": dict(totals),
        "counts": dict(counts),
        "net": net,
        "span_days": span_days,
        "per_day": net / span_days if span_days > 0 else 0.0,
        "fills": counts.get("COMMISSION", 0),
    }


def daily(rows: list[dict]) -> dict[str, dict]:
    """Per-UTC-day breakdown. Keyed YYYY-MM-DD so it sorts chronologically."""
    out: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        day = datetime.fromtimestamp(int(r["time"]) / 1000, timezone.utc).strftime("%Y-%m-%d")
        out[day][r["incomeType"]] += float(r["income"])
        if r["incomeType"] == "COMMISSION":
            out[day]["fills"] += 1
    for day, v in out.items():
        v["net"] = sum(v[t] for t in INCOME_TYPES)
    return {k: dict(v) for k, v in out.items()}


def outlier_days(by_day: dict[str, dict], threshold: float = 3.5) -> list[tuple[str, float, float]]:
    """Days whose net is a different animal from the rest, by modified z-score.

    Median and MAD rather than mean and standard deviation, because the outlier we are
    hunting is large enough to drag both of those toward itself and hide behind the
    result -- 2026-08-08's -50.49 sits inside one standard deviation of a sample it is
    itself 95% of.

    Returns (day, net, score) sorted worst first. Scaling constant 0.6745 makes the
    score comparable to a standard z; 3.5 is the conventional cut.
    """
    if len(by_day) < 3:
        return []
    nets = sorted(v["net"] for v in by_day.values())
    mid = len(nets) // 2
    median = nets[mid] if len(nets) % 2 else (nets[mid - 1] + nets[mid]) / 2
    devs = sorted(abs(n - median) for n in nets)
    mad = devs[mid] if len(devs) % 2 else (devs[mid - 1] + devs[mid]) / 2
    if mad > 0:
        scale, const = mad, 0.6745
    else:
        # MAD is zero whenever MORE THAN HALF the days are identical, and a steady
        # baseline plus one disaster is exactly that shape -- so the naive guard
        # ("mad <= 0: give up") went blind precisely on the case worth catching.
        # Mean absolute deviation is nonzero as soon as any day differs at all.
        scale, const = sum(abs(n - median) for n in nets) / len(nets), 0.7979
        if scale <= 0:
            return []                   # every day identical: nothing to flag
    scored = [(d, v["net"], const * (v["net"] - median) / scale) for d, v in by_day.items()]
    flagged = [s for s in scored if abs(s[2]) >= threshold]
    return sorted(flagged, key=lambda s: s[1])


def excluding(rows: list[dict], days: set[str]) -> list[dict]:
    """The same ledger with certain UTC days removed, to see what the rest looks like."""
    return [r for r in rows
            if datetime.fromtimestamp(int(r["time"]) / 1000, timezone.utc).strftime("%Y-%m-%d")
            not in days]


def _report(rows: list[dict], symbol: str) -> None:
    s = summarize(rows)
    by_day = daily(rows)
    print(f"\n{symbol} — Binance's own ledger")
    print(f"{len(rows)} entries over {s['span_days']:.1f} days, {s['fills']} charged fills\n")

    print(f"{'date':<12}{'fills':>7}{'realized':>11}{'commission':>12}{'funding':>9}"
          f"{'net':>10}{'cum':>10}")
    cum = 0.0
    for day in sorted(by_day):
        v = by_day[day]
        cum += v["net"]
        print(f"{day:<12}{int(v.get('fills', 0)):>7}{v.get('REALIZED_PNL', 0.0):>+11.3f}"
              f"{v.get('COMMISSION', 0.0):>+12.3f}{v.get('FUNDING_FEE', 0.0):>+9.3f}"
              f"{v['net']:>+10.3f}{cum:>+10.2f}")

    print(f"\n{'':<12}{'':>7}{'':>11}{'':>12}{'':>9}{'-'*9:>10}")
    for t in INCOME_TYPES:
        print(f"  {t:<28}{s['totals'].get(t, 0.0):>+12.4f}  ({s['counts'].get(t, 0)} entries)")
    print(f"  {'NET':<28}{s['net']:>+12.4f}   = {s['per_day']:+.4f} USDT/day")

    flagged = outlier_days(by_day)
    if not flagged:
        return

    # A mean over a distribution with a -50 in it is not a summary, it is a disguise.
    print(f"\n{len(flagged)} outlier day(s) — these are events, not trading:")
    for day, net, score in flagged:
        share = net / s["net"] * 100 if s["net"] else 0.0
        print(f"  {day}  {net:+.3f}  ({score:+.1f} MAD, {share:.0f}% of the total)")

    rest = summarize(excluding(rows, {d for d, _, _ in flagged}))
    print(f"\nWithout them — {rest['span_days']:.1f} days:")
    for t in INCOME_TYPES:
        print(f"  {t:<28}{rest['totals'].get(t, 0.0):>+12.4f}")
    print(f"  {'NET':<28}{rest['net']:>+12.4f}   = {rest['per_day']:+.4f} USDT/day")

    gross = rest["totals"].get("REALIZED_PNL", 0.0)
    fees = rest["totals"].get("COMMISSION", 0.0)
    if gross > 0 and fees < 0:
        print(f"\n  gross edge {gross:+.2f} vs fee bill {fees:+.2f} — "
              f"fees are {abs(fees) / gross * 100:.0f}% of the edge")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--json", action="store_true", help="machine-readable, for a cron")
    args = ap.parse_args()

    from config import settings
    from exchange import Exchange

    ex = Exchange(settings.exchange_config, demo=settings.demo_mode)
    rows = fetch_income(ex, settings.symbol, days=args.days)
    if not rows:
        raise SystemExit("Binance returned no income entries for that window.")

    if args.json:
        by_day = daily(rows)
        print(json.dumps({
            "symbol": settings.symbol,
            "summary": summarize(rows),
            "daily": by_day,
            "outliers": [{"day": d, "net": n, "score": s} for d, n, s in outlier_days(by_day)],
        }, indent=2, default=str))
    else:
        _report(rows, settings.symbol)


if __name__ == "__main__":
    main()
