"""Replay from candles instead of poll snapshots. AUDIT #86.

The bot writes one PRICE= line per poll, ~15s apart, and replay.py was reading those
as if they were the price path. They are not -- they are samples of it. Everything
between two samples is invisible, and that is where most fills happen.

Measured against two sessions that actually traded:

    2026-08-15 22:38    live 16 fills, 5 cycles     snapshot replay  6 fills, 2 cycles
    2026-08-16 03:56    live 10 fills, 3 cycles     snapshot replay  6 fills, 3 cycles

So the harness was missing roughly 60% of them, and I had it backwards: I said it ran
OPTIMISTIC because it fills on touch with no queue position. The sampling loss is much
larger than the queue-position gain, so it runs pessimistic, and every spacing number I
derived from it is understated -- worst for tight spacings, which need the smallest
moves and therefore lose the most to sampling.

A candle's HIGH and LOW are exactly what a grid needs: the extremes that reach resting
orders. This fetches them for a recorded session's time window and expands each candle
into the path price plausibly took.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

CACHE = Path("logs/klines")


class TradeHistoryUnavailable(RuntimeError):
    """The tape does not reach that far back. Binance serves ~2 days of aggTrades."""


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def session_window(log: Path, run: int = 0) -> tuple[int, int]:
    """First and last log timestamp of one run, as epoch milliseconds.

    Log timestamps are naive LOCAL time; Binance works in UTC. astimezone() on a naive
    datetime interprets it as local, which is what we want -- but it silently produces
    the wrong window if the machine's timezone changed since the log was written, so
    callers should sanity-check the prices they get back (see verify_against_log).
    """
    text = log.read_text(encoding="utf-8", errors="replace")
    chunks = text.split("GRID BOT STARTING")[1:]
    if not chunks:
        raise SystemExit(f"{log} contains no runs")
    if run >= len(chunks):
        raise SystemExit(f"{log} has {len(chunks)} run(s); asked for {run}")
    chunk = chunks[-1 - run]
    stamps = re.findall(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", chunk, re.M)
    if len(stamps) < 2:
        raise SystemExit("could not read a start and end timestamp from that run")

    def ms(s: str) -> int:
        naive = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return int(naive.astimezone().astimezone(timezone.utc).timestamp() * 1000)

    return ms(stamps[0]), ms(stamps[-1])


def fetch_klines(exchange, symbol: str, start_ms: int, end_ms: int,
                 timeframe: str = "1m", use_cache: bool = True) -> list[list]:
    """OHLCV covering the window, paginated and cached.

    Cached because a replay gets run over and over while iterating on the engine, and
    re-pulling the same candles each time is both slow and rude to the API.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"{symbol}_{timeframe}_{start_ms}_{end_ms}.json"
    if use_cache and key.exists():
        return json.loads(key.read_text())

    out: list[list] = []
    since = start_ms
    while since < end_ms:
        batch = exchange.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        fresh = [c for c in batch if c[0] <= end_ms]
        out.extend(fresh)
        if len(fresh) < len(batch) or batch[-1][0] <= since:
            break
        since = batch[-1][0] + 1
    key.write_text(json.dumps(out))
    return out


def path_from_klines(candles: list[list]) -> list[float]:
    """Expand candles into the path price plausibly took.

    OHLC does not record the order in which the high and low happened, so this uses the
    usual convention: an up bar is assumed to have dipped first (O -> L -> H -> C) and a
    down bar to have popped first (O -> H -> L -> C). It is a guess about SEQUENCE, but
    never about EXTENT -- the high and low are real, so every resting order the market
    actually reached still gets reached here.
    """
    path: list[float] = []
    for _, o, h, l, c, *_ in candles:
        o, h, l, c = float(o), float(h), float(l), float(c)
        path.extend([o, l, h, c] if c >= o else [o, h, l, c])
    return path


def verify_against_log(path: list[float], snapshots: list[float],
                       tolerance: float = 0.02) -> str | None:
    """Do the candles actually cover the session the log recorded?

    A wrong time window -- stale timezone, off-by-a-day, the wrong run index -- yields
    a perfectly plausible price path for the WRONG hours, and every conclusion drawn
    from it is quietly false. The snapshots are the ground truth for what this session
    saw, so the candle range has to contain them.
    """
    if not path or not snapshots:
        return "no data on one side"
    lo, hi = min(path), max(path)
    slo, shi = min(snapshots), max(snapshots)
    pad = (hi - lo) * tolerance + 1e-12
    if slo < lo - pad or shi > hi + pad:
        return (f"candles cover {lo}-{hi} but the log recorded {slo}-{shi} — "
                f"wrong window (timezone? run index?)")
    return None


def fetch_trade_path(exchange, symbol: str, start_ms: int, end_ms: int,
                     use_cache: bool = True, max_requests: int = 200) -> list[float]:
    """Every trade price in the window: the actual path, not a sample of it.

    1m candles were not enough either. Their high and low are real, but O->L->H->C
    crosses any given rung at most twice per minute, and a grid rung can fill, be
    replaced and fill again inside that minute. Measured on 2026-08-15 22:38 -- live
    16 fills, snapshots 6, 1m candles 7. Only the trade tape reproduces it.

    ~68 trades/min on DOGEUSDT, so a few thousand per hour: large enough to be worth
    caching, small enough to be practical.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"{symbol}_trades_{start_ms}_{end_ms}.json"
    if use_cache and key.exists():
        return json.loads(key.read_text())

    prices: list[float] = []
    since, seen = start_ms, 0
    for _ in range(max_requests):
        try:
            batch = exchange.exchange.fetch_trades(symbol, since=since, limit=1000)
        except Exception as e:
            # Binance serves aggTrades for the RECENT 2 DAYS only (-4166). Anything
            # older simply cannot be replayed from the tape, and the raw ccxt error
            # ("Search window is restricted...") does not make it obvious that this is
            # a data-availability wall rather than a bug in the caller.
            if "-4166" in str(e) or "restricted to recent" in str(e):
                raise TradeHistoryUnavailable(
                    f"{symbol} trade history covers only the last ~2 days; "
                    f"this window starts {(_now_ms() - start_ms)/86400000:.1f} days ago. "
                    f"Use --timeframe 1m for older sessions, accepting that candles "
                    f"cross a rung at most twice a minute."
                ) from e
            raise
        if not batch:
            break
        fresh = [t for t in batch if t["timestamp"] <= end_ms]
        prices.extend(float(t["price"]) for t in fresh)
        seen += len(fresh)
        if len(fresh) < len(batch):
            break                                  # ran past the end of the window
        last = batch[-1]["timestamp"]
        if last <= since:
            break                                  # no forward progress; stop rather than spin
        since = last + 1
    else:
        # Hit the request cap: say so rather than silently replaying a truncated
        # session, which would read as "the market went quiet" instead of "we stopped
        # asking".
        raise SystemExit(
            f"trade fetch hit the {max_requests}-request cap after {seen:,} trades; "
            f"replay the session in smaller pieces or raise max_requests")
    key.write_text(json.dumps(prices))
    return prices
