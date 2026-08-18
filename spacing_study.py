"""Rank grid spacings, or say why the data cannot. AUDIT #99.

replay.py --sweep prints a spacing table and its docstring tells you not to believe it.
That warning is correct and it is also a dead end: three sweeps named three winners, and
the reason was never sample size. From that docstring:

    session        hours   0.08   0.10   0.12   0.15   0.20   0.25   0.30
    2026-08-15 -0    2.3  -0.00  +1.66  -0.25  +0.89  +1.81  +2.27  +2.64
    2026-08-15 -2    8.2  +3.58  +2.74  +3.41  +2.67  +1.15  +0.65  -0.15
    2026-08-16 -2    2.6  +8.49  +3.95  +6.49  +2.84  +4.91  +7.36  +8.35

Session one rises with spacing, session two falls, session three is U-shaped. Read as
LEVELS these disagree completely, and averaging them averages over the price path -- the
one thing that moves the number.

But every spacing in a row was replayed against the SAME path. The rows are blocks, and
the quantity that carries information is the DIFFERENCE within a row, not the level. Take
session two: 0.08 beats 0.20 by +2.43 USDT/day. Session one: 0.08 loses to 0.20 by -1.81.
Those differences are directly comparable to each other in a way the raw numbers are not,
because each one has its own session's path divided out of it.

So this module does three things the sweep does not:

  * differences, not levels -- each spacing is scored against a baseline WITHIN a
    session, which cancels the between-session variance that swamped the pooled table
  * a confidence interval on those differences, so "0.08% wins" becomes "0.08% wins by
    X +/- Y USDT/day across N sessions" or, far more often, "these data cannot tell"
  * a refusal. When the interval spans zero it reports how many more sessions would be
    needed rather than naming the largest mean and calling it a winner.

It also checks its own resolution before believing anything. A 1m candle contributes four
path points, so intra-minute oscillation is invisible -- and that hurts TIGHT spacings
more than wide ones, which is a bias in exactly the direction the study is trying to
measure. --validate replays one session at both 1m and raw trades and compares fill
recovery per spacing; if recovery depends on spacing, candle data cannot rank spacings and
the study says so instead of producing a table.

Usage:
    py spacing_study.py --validate           # is the data good enough to rank at all?
    py spacing_study.py                      # run the study over recorded sessions
    py spacing_study.py --min-hours 2 --max-sessions 30
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Two-sided 95% Student's t. A table rather than scipy, which is not installed here and
# is a heavy dependency for eleven numbers. Values beyond the table interpolate toward
# the normal limit, which is what t converges to.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 50: 2.009,
    60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
}
# Imported rather than restated: this was 1.960 here and 1.959964 in
# taker_share.py, two values for one constant (AUDIT #107).
from taker_share import Z95  # noqa: E402
Z80 = 0.8416          # one-sided 80% power


def t_critical_95(df: int) -> float:
    """Two-sided 95% critical value for `df` degrees of freedom."""
    if df < 1:
        return float("inf")
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    if df > keys[-1]:
        return Z95
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    span = hi - lo
    return _T95[lo] + (_T95[hi] - _T95[lo]) * (df - lo) / span


@dataclass
class SessionResult:
    """One recorded session replayed at every spacing."""

    name: str
    hours: float
    net: dict[float, float] = field(default_factory=dict)      # USDT over the session
    fills: dict[float, int] = field(default_factory=dict)

    def per_day(self, spacing: float) -> float:
        """Net normalised to USDT/day.

        Sessions run from 20 minutes to 8 hours, so raw USDT compares a long session's
        result against a short one's and calls the difference a spacing effect.
        """
        if self.hours <= 0:
            raise ValueError(f"session {self.name} has no duration")
        return self.net[spacing] * 24.0 / self.hours


@dataclass
class Comparison:
    """One spacing scored against the baseline, across sessions."""

    spacing: float
    baseline: float
    diffs: list[float]

    @property
    def n(self) -> int:
        return len(self.diffs)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.diffs) if self.diffs else 0.0

    @property
    def sd(self) -> float:
        return statistics.stdev(self.diffs) if len(self.diffs) > 1 else float("nan")

    @property
    def stderr(self) -> float:
        return self.sd / math.sqrt(self.n) if self.n > 1 else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        if self.n < 2:
            return (float("-inf"), float("inf"))
        half = t_critical_95(self.n - 1) * self.stderr
        return (self.mean - half, self.mean + half)

    @property
    def decisive(self) -> bool:
        """True only when the interval excludes zero -- i.e. a direction is established."""
        lo, hi = self.ci
        return lo > 0.0 or hi < 0.0

    def sessions_needed(self) -> int:
        """Paired-design sample size to resolve an effect this size at 95%/80%.

        Returns the total sessions required, so the caller can say how far away an answer
        is rather than implying one exists. Infinite when the observed effect is zero:
        no sample size resolves a difference that is not there.
        """
        if self.n < 2 or not math.isfinite(self.sd) or self.mean == 0.0:
            return sys.maxsize
        return max(2, math.ceil(((Z95 + Z80) * self.sd / abs(self.mean)) ** 2))


def compare_spacings(results: list[SessionResult], baseline: float) -> list[Comparison]:
    """Paired within-session differences for every spacing against `baseline`.

    Pairing is the whole point. A session's price path sets the scale of every number in
    its row, so differencing within the row divides that scale out; pooling the levels
    instead leaves it in, and it is larger than the effect being measured.
    """
    spacings = sorted({s for r in results for s in r.net})
    out = []
    for spacing in spacings:
        if spacing == baseline:
            continue
        diffs = [r.per_day(spacing) - r.per_day(baseline)
                 for r in results
                 if spacing in r.net and baseline in r.net and r.hours > 0]
        out.append(Comparison(spacing=spacing, baseline=baseline, diffs=diffs))
    return out


def rank(results: list[SessionResult], baseline: float) -> tuple[Comparison | None, list[Comparison]]:
    """Return (winner, all comparisons). Winner is None unless one is decisive.

    Deliberately not `max(..., key=mean)`. The failure this module exists to prevent is
    naming the largest mean of a set of indistinguishable numbers, which is what three
    previous sweeps did.
    """
    comparisons = sorted(compare_spacings(results, baseline), key=lambda c: -c.mean)
    decisive = [c for c in comparisons if c.decisive and c.mean > 0]
    return (decisive[0] if decisive else None), comparisons


def resolution_bias(fine: dict[float, int], coarse: dict[float, int],
                    tolerance: float = 0.15, min_fills: int = 10) -> str | None:
    """Does the cheap price path recover fills evenly across spacings?

    A 1m candle contributes four points, so everything between them is invisible. That
    is survivable if it costs every spacing the same fraction of its fills -- the
    differences still rank. It is fatal if the loss depends on spacing, because tight
    ladders live on exactly the oscillation a candle hides, and the study would read a
    data artifact as a parameter effect.

    `min_fills` guards against the other way of being wrong here: a ratio of 3 fills to 2
    is not evidence of anything, and a handful of those looks reassuringly flat. A
    spacing with too few fills on the fine path is dropped rather than counted.

    Returns None when recovery is flat enough to rank on, or a description of the bias.
    """
    shared = sorted(s for s in fine if s in coarse and fine[s] >= min_fills)
    if len(shared) < 2:
        return (f"only {len(shared)} spacing(s) reached {min_fills} fills on the fine "
                f"path — too few to tell an even loss from a spacing-dependent one")
    ratios = {s: coarse[s] / fine[s] for s in shared}
    lo_s = min(ratios, key=lambda s: ratios[s])
    hi_s = max(ratios, key=lambda s: ratios[s])
    spread = ratios[hi_s] - ratios[lo_s]
    if spread <= tolerance:
        return None
    return (f"fill recovery varies {ratios[lo_s]:.0%}-{ratios[hi_s]:.0%} across spacings "
            f"(worst {lo_s:.2%}, best {hi_s:.2%}, spread {spread:.0%} > {tolerance:.0%}). "
            f"The coarse path costs tight spacings more than wide ones, so a spacing "
            f"ranking off it would be measuring the data source")


def level_floor(maker: float, taker: float, taker_share: float,
                min_profit_multiplier: float) -> float:
    """Spacing below which GridEngine places no level at all.

    _is_level_profitable requires a level's spacing to clear the round trip times
    MIN_PROFIT_MULTIPLIER. Below that the ladder is inert: no orders, no fills, and a
    paper_net that is just an untraded position drifting.

    This is not a detail. Three of the seven columns the old sweep printed -- 0.08%,
    0.10%, 0.12% against a 0.126% floor -- are inert in current settings, and 0.08% is
    the spacing two of the three previous sweeps named as the winner. Measured on
    2026-08-17 over 14,777 real trade ticks, those three columns produced exactly zero
    fills while 0.15% produced 14. A column where nothing trades is not a result.
    """
    round_trip = 2.0 * (maker * (1.0 - taker_share) + taker * taker_share)
    return round_trip * min_profit_multiplier


def split_by_floor(spacings, floor: float) -> tuple[list[float], list[float]]:
    """(rankable, inert). Inert spacings must be excluded before anything is compared."""
    rankable = [s for s in spacings if s >= floor]
    inert = [s for s in spacings if s < floor]
    return rankable, inert


def theoretical_optimum(maker: float, taker: float, taker_share: float) -> float:
    """Spacing that maximises profit for a driftless diffusion: twice the round trip.

    Crossings of a grid of spacing s scale as quadratic variation / s^2, so profit goes
    as (1/s^2)(s - f), which is maximised at s = 2f. Offered only as a reference point --
    real price paths trend, and a trend is precisely what punishes a tight ladder.
    """
    round_trip = 2.0 * (maker * (1.0 - taker_share) + taker * taker_share)
    return 2.0 * round_trip


def format_report(results: list[SessionResult], baseline: float) -> str:
    winner, comparisons = rank(results, baseline)
    n = len(results)
    hours = sum(r.hours for r in results)
    lines = [
        f"{n} sessions, {hours:.1f}h, baseline {baseline:.2%}",
        "",
        f"  {'spacing':>8} {'n':>3} {'mean d':>9} {'95% CI':>20} {'verdict':>12}",
        "  " + "-" * 58,
    ]
    for c in comparisons:
        lo, hi = c.ci
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if math.isfinite(lo) else "[insufficient]"
        verdict = ("BETTER" if c.decisive and c.mean > 0 else
                   "WORSE" if c.decisive else "indistinct")
        lines.append(f"  {c.spacing:>7.2%} {c.n:>3} {c.mean:>+9.2f} {ci:>20} {verdict:>12}")
    lines.append("")
    if winner:
        lo, hi = winner.ci
        lines.append(f"  WINNER {winner.spacing:.2%}: {winner.mean:+.2f} USDT/day vs "
                     f"{baseline:.2%}, 95% CI [{lo:+.2f}, {hi:+.2f}]")
    else:
        lines.append("  NO WINNER. Every interval spans zero, so no spacing is")
        lines.append("  distinguishable from the baseline on this evidence.")
        best = max(comparisons, key=lambda c: c.mean, default=None)
        if best is not None and best.n >= 2:
            need = best.sessions_needed()
            have = best.n
            lines.append(
                f"  Largest mean was {best.spacing:.2%} at {best.mean:+.2f} USDT/day; "
                f"resolving an effect that size needs "
                + (f"~{need} sessions ({have} available)." if need != sys.maxsize
                   else "an unbounded sample -- the observed effect is ~0.")
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------------
# the runner (network + disk; the logic above is deliberately free of both)
# --------------------------------------------------------------------------------

# Every one of these must clear the level floor, or its column is inert (see
# level_floor). At the current 0.126% floor the old sweep's 0.08/0.10/0.12 columns place
# nothing at all, so the range starts just above it and extends the other way instead.
SPACINGS = (0.0013, 0.0015, 0.00175, 0.0020, 0.0025, 0.0030, 0.0040)


def settings_floor() -> float:
    from config import settings
    return level_floor(settings.maker_fee_pct / 100, settings.taker_fee_pct / 100,
                       settings.taker_fill_share_pct / 100, settings.min_profit_multiplier)


def _quieten() -> None:
    """The engine logs a paragraph per fill; a sweep runs it hundreds of times."""
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")


def sweep_path(prices: list[float], spacings=SPACINGS) -> tuple[dict, dict]:
    """Replay one price path at every spacing. Returns (net, fills) keyed by spacing."""
    from replay import replay

    net, fills = {}, {}
    for s in spacings:
        r = replay(prices, spacing_pct=s, quiet=True)
        net[s], fills[s] = r["paper_net"], r["fills"]
    return net, fills


def discover_sessions(logs_dir: Path, newest_first: bool = True) -> list[tuple[Path, int]]:
    """Every (log, run index) pair on disk, newest run of the newest log first."""
    out = []
    logs = sorted(logs_dir.glob("grid_*.log"), key=lambda p: p.stat().st_mtime,
                  reverse=newest_first)
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        for run in range(text.count("GRID BOT STARTING")):
            out.append((log, run))
    return out


def validate_resolution(exchange, symbol: str, log: Path, run: int) -> int:
    """Replay one session at 1m and at the raw tape, and compare fills per spacing."""
    from klines import (TradeHistoryUnavailable, fetch_klines, fetch_trade_path,
                        path_from_klines, session_window, verify_against_log)
    from replay import load_prices

    _quieten()
    floor = settings_floor()
    rankable, _inert = split_by_floor(SPACINGS, floor)
    start, end = session_window(log, run)
    snapshots = load_prices(log, run)
    hours = (end - start) / 3_600_000
    print(f"validating on {log.name} run -{run}: {hours:.1f}h")
    print(f"level floor {floor:.3%}; sweeping "
          + ", ".join(f"{s:.3%}" for s in rankable) + "\n")

    try:
        tape = fetch_trade_path(exchange, symbol, start, end)
    except TradeHistoryUnavailable as e:
        print(f"REFUSING: {e}")
        return 2
    candles = fetch_klines(exchange, symbol, start, end, timeframe="1m")
    coarse = path_from_klines(candles)
    for name, path in (("tape", tape), ("1m", coarse)):
        problem = verify_against_log(path, snapshots)
        if problem:
            print(f"REFUSING: {name} path does not cover the session — {problem}")
            return 2

    print(f"  tape {len(tape):,} ticks   1m {len(candles)} candles "
          f"-> {len(coarse):,} path points\n")
    _, fine_fills = sweep_path(tape, rankable)
    _, coarse_fills = sweep_path(coarse, rankable)

    print(f"  {'spacing':>8} {'tape fills':>11} {'1m fills':>9} {'recovered':>10}")
    print("  " + "-" * 42)
    for s in rankable:
        rec = coarse_fills[s] / fine_fills[s] if fine_fills[s] else float("nan")
        print(f"  {s:>7.2%} {fine_fills[s]:>11} {coarse_fills[s]:>9} {rec:>9.0%}")

    problem = resolution_bias(fine_fills, coarse_fills)
    print()
    if problem:
        print(f"  UNUSABLE FOR RANKING: {problem}.")
        print("  Candle data can still verify ladder invariants; it cannot rank spacings.")
        return 1
    print("  Fill recovery is flat across spacings — 1m candles can rank on differences.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate", action="store_true",
                    help="check whether the cheap price path can rank spacings at all")
    ap.add_argument("--log", help="log file for --validate (default: newest)")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--baseline", type=float, default=0.0020)
    ap.add_argument("--min-hours", type=float, default=1.0)
    ap.add_argument("--max-sessions", type=int, default=40)
    ap.add_argument("--force", action="store_true",
                    help="rank even if the resolution check fails (the table will be "
                         "measuring the data source as much as the parameter)")
    a = ap.parse_args()

    from config import settings
    from exchange import Exchange

    _quieten()
    floor = settings_floor()
    rankable, inert = split_by_floor(SPACINGS, floor)
    print(f"level floor {floor:.3%} (round trip x {settings.min_profit_multiplier}); "
          f"sweeping {', '.join(f'{s:.3%}' for s in rankable)}")
    if inert:
        print(f"  excluded as inert (place no levels): "
              f"{', '.join(f'{s:.3%}' for s in inert)}")
    print()

    logs = Path("logs")
    if a.validate:
        log = Path(a.log) if a.log else max(logs.glob("grid_*.log"),
                                            key=lambda p: p.stat().st_mtime)
        return validate_resolution(Exchange(settings.exchange_config,
                                            demo=settings.demo_mode),
                                   settings.symbol, log, a.run)

    from klines import (fetch_klines, path_from_klines, session_window,
                        verify_against_log)
    from replay import load_prices

    ex = Exchange(settings.exchange_config, demo=settings.demo_mode)

    # Gate the study on its own resolution check. Producing a confident table off data
    # this module has already shown to be spacing-biased is precisely the failure it
    # exists to prevent -- and the previous harness's table was believed three times.
    if not a.force:
        newest = max(logs.glob("grid_*.log"), key=lambda p: p.stat().st_mtime)
        if validate_resolution(ex, settings.symbol, newest, 0) != 0:
            print("\nREFUSING TO RANK on this data. The only price source available for")
            print("older sessions is 1m candles, and they do not reach tight ladders.")
            print("Ranking spacings needs tick data collected FORWARD, one session at a")
            print("time, until the sample is large enough (see 'sessions needed' above).")
            print("Re-run with --force to see the table anyway.")
            return 1
        print()

    results: list[SessionResult] = []
    for log, run in discover_sessions(logs):
        if len(results) >= a.max_sessions:
            break
        try:
            start, end = session_window(log, run)
            hours = (end - start) / 3_600_000
            if hours < a.min_hours:
                continue
            snapshots = load_prices(log, run)
            if len(snapshots) < 10:
                continue
            candles = fetch_klines(ex, settings.symbol, start, end, timeframe="1m")
            path = path_from_klines(candles)
            if verify_against_log(path, snapshots):
                continue
            net, fills = sweep_path(path, rankable)
        except Exception as e:
            print(f"  skipped {log.name} -{run}: {type(e).__name__}: {e}")
            continue
        results.append(SessionResult(f"{log.stem} -{run}", hours, net, fills))
        print(f"  {log.stem} -{run}  {hours:5.1f}h  "
              + "  ".join(f"{net[s]:+6.2f}" for s in rankable))

    if len(results) < 2:
        print("\nnot enough replayable sessions to compare")
        return 2
    print("\n" + format_report(results, a.baseline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
