"""Performance analysis / tuning-decision report for the grid bot.

Reads the bot's own trade history (logs/trades_{demo,live}.csv) -- no exchange
connection or API keys required -- and reports win rate, net PnL, and fee
drag, broken down by market regime and by day. Flags regimes that are net
losers and estimates how much of the drag is concentrated in a handful of
outlier trades.

Defaults to the journal matching DEMO_MODE, because reporting demo fills as live
results is the one output nobody could use.

This is meant to be re-run periodically (e.g. weekly, or after switching a
configs/*.env profile) to make config tuning an evidence-based, repeatable
check instead of a one-off guess. It does not place trades or touch the
exchange.

Usage:
    python tools/analyze_performance.py
    python tools/analyze_performance.py --csv logs/trades_live.csv --top 10
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RegimeStats:
    n: int = 0
    wins: int = 0
    gross: float = 0.0
    fee: float = 0.0

    @property
    def net(self) -> float:
        return self.gross - self.fee

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def net_per_trade(self) -> float:
        return self.net / self.n if self.n else 0.0


@dataclass
class Report:
    total_fills: int = 0
    completed_cycles: int = 0
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    by_regime: dict[str, RegimeStats] = field(default_factory=lambda: defaultdict(RegimeStats))
    worst_trades: list[dict] = field(default_factory=list)
    span_days: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_fees

    @property
    def fee_pct_of_gross(self) -> float:
        return self.total_fees / self.gross_pnl if self.gross_pnl else 0.0

    @property
    def cycles_per_day(self) -> float:
        return self.completed_cycles / self.span_days if self.span_days else 0.0

    @property
    def net_per_day(self) -> float:
        return self.net_pnl / self.span_days if self.span_days else 0.0


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. The bot writes this file as it trades "
            "(see trade_journal.py) -- run the bot for a while first."
        )
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def build_report(rows: list[dict], top_n: int = 10) -> Report:
    report = Report()
    completed = [r for r in rows if r.get("completed_cycle") == "True"]

    report.total_fills = len(rows)
    report.completed_cycles = len(completed)

    for r in completed:
        gross = float(r["cycle_pnl"])
        fee = float(r["fee"])
        net = gross - fee
        report.gross_pnl += gross
        report.total_fees += fee

        stats = report.by_regime[r.get("regime", "unknown")]
        stats.n += 1
        stats.gross += gross
        stats.fee += fee
        if net > 0:
            stats.wins += 1

    report.worst_trades = sorted(
        completed,
        key=lambda r: float(r["cycle_pnl"]) - float(r["fee"]),
    )[:top_n]

    if rows:
        first = datetime.fromisoformat(rows[0]["timestamp"])
        last = datetime.fromisoformat(rows[-1]["timestamp"])
        report.span_days = max((last - first).total_seconds() / 86400, 1e-9)

    return report


def format_report(report: Report) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("GRID BOT PERFORMANCE REPORT")
    lines.append("=" * 72)
    lines.append(
        f"Span: {report.span_days:.1f} days | fills: {report.total_fills} | "
        f"completed cycles: {report.completed_cycles} ({report.cycles_per_day:.1f}/day)"
    )
    lines.append(
        f"Gross: {report.gross_pnl:.2f} | fees: {report.total_fees:.2f} "
        f"({report.fee_pct_of_gross:.1%} of gross) | net: {report.net_pnl:.2f} "
        f"({report.net_per_day:.2f}/day)"
    )
    lines.append("")
    lines.append("By regime (sorted by trade count):")
    lines.append(f"  {'regime':<12}{'n':>6}{'win%':>8}{'gross':>10}{'fee':>8}{'net':>10}{'net/trade':>12}")
    net_losers = []
    for regime, stats in sorted(report.by_regime.items(), key=lambda kv: -kv[1].n):
        lines.append(
            f"  {regime:<12}{stats.n:>6}{stats.win_rate:>7.1%} "
            f"{stats.gross:>9.2f}{stats.fee:>8.2f}{stats.net:>10.2f}{stats.net_per_trade:>12.4f}"
        )
        if stats.net < 0:
            net_losers.append((regime, stats))

    lines.append("")
    lines.append(f"Worst {len(report.worst_trades)} completed cycles (net):")
    for r in report.worst_trades:
        net = float(r["cycle_pnl"]) - float(r["fee"])
        lines.append(f"  {r['timestamp']}  {r.get('regime','?'):<10}  net={net:>9.4f}")

    lines.append("")
    lines.append("Findings:")
    if net_losers:
        for regime, stats in net_losers:
            lines.append(
                f"  - '{regime}' regime is net NEGATIVE: {stats.n} trades, net {stats.net:.2f}. "
                "If this is 'downtrend' or 'uptrend', consider lowering TREND_CHECK_INTERVAL "
                "and/or TREND_CONFIRMATION_SECONDS (see configs/high_frequency.env) so the "
                "grid pauses faster once a real trend forms."
            )
    else:
        lines.append("  - No regime is net negative over this window.")

    if report.fee_pct_of_gross > 0.15:
        lines.append(
            f"  - Fees are {report.fee_pct_of_gross:.1%} of gross profit (high). Check that "
            "grid orders are filling as maker, not taker -- stop-loss and recovery/unwind "
            "fills are taker by design and expected, but a high ratio outside of those "
            "usually means spacing is too tight for current volatility."
        )

    lines.append("=" * 72)
    return "\n".join(lines)


def default_csv(log_dir: str = "logs") -> Path:
    """Pick the journal for the mode the bot is configured for.

    The journals are per-account since AUDIT #70, so there is no single trades.csv to
    default to -- and defaulting to the wrong one would report demo results as live.
    """
    demo = os.environ.get("DEMO_MODE", "").strip().lower()
    if demo in ("true", "1", "yes", "on"):
        return Path(log_dir) / "trades_demo.csv"
    if demo in ("false", "0", "no", "off"):
        return Path(log_dir) / "trades_live.csv"
    try:
        from config import settings  # optional: only if the package imports cleanly
        return Path(log_dir) / f"trades_{'demo' if settings.demo_mode else 'live'}.csv"
    except Exception:
        return Path(log_dir) / "trades_demo.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None,
                        help="Path to a trades journal (default: the one matching DEMO_MODE)")
    parser.add_argument("--top", type=int, default=10, help="Number of worst trades to show")
    args = parser.parse_args()

    path = Path(args.csv) if args.csv else default_csv()
    print(f"reading {path}", file=sys.stderr)

    try:
        rows = load_rows(path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    if not rows:
        print("No trades recorded yet.", file=sys.stderr)
        return 1

    report = build_report(rows, top_n=args.top)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
