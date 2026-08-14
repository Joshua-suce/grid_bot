"""Health check for a running (or finished) session. Reads logs only -- trades nothing.

A session produces thousands of lines and the ones that matter are rare, so "is it
behaving?" turns into scrolling. This counts the signatures that mean something specific
and surfaces the latest analytics line.

    py check_run.py             # most recent log
    py check_run.py 2026-08-15  # a specific day

Every pattern below is pinned to a verbatim log string in tests/test_check_run.py. A
health check whose patterns have rotted reports "nothing flagged" forever, which is a
false all-clear -- worse than no check at all.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

LOG_DIR = Path("logs")

# (label, pattern, what a non-zero count means). Worst first.
SIGNALS: list[tuple[str, str, str]] = [
    ("shut down", r"MAX RECOVERY REACHED",
     "the bot gave up and stopped — read the errors below"),
    ("emergency stop", r"EMERGENCY STOP",
     "a risk limit tripped and the grid was pulled"),
    ("naked position", r"POSITION UNDER-PROTECTED|PAUSED POSITION UNPROTECTED|position unprotected",
     "a position was held with no stop covering it"),
    ("stop book unreadable", r"STOP REFRESH ABORTED|STOP SWEEP UNVERIFIED|POSITION CHECK FAILED",
     "stop state could not be verified — it fails safe, but look"),
    ("state not persisted", r"STATE NOT PERSISTED",
     "a restart would reload stale stop ratchets"),
    ("legacy state ignored", r"LEGACY STATE IGNORED",
     "an unattributable state file is sitting in state/"),
    ("state corrupt", r"GRID STATE CORRUPT|Corrupt state file",
     "state failed to restore and the grid rebuilt from scratch"),
    ("ladder outgrows cap", r"LADDER OUTGROWS THE CAP",
     "geometry too wide for MAX_POSITION_PCT — one side will jam"),
    ("pnl feed stale", r"PNL FEED STALE",
     "the daily-loss kill switch is flying blind"),
    ("recenter aborted", r"RECENTER ABORTED",
     "a recenter could not verify a clean book"),
    ("exposure blocked", r"EXPOSURE WARNING",
     "total exposure hit its cap, new orders blocked"),
    ("side blocked", r"POSITION LIMIT \| (long|short) ",
     "the position cap blocked one side — the grid is running one-legged"),
    ("recenter", r"RECENTERING GRID|GRID RECENTERED|Grid recentered",
     "grid rebuilt around price — each one is a TAKER exit, the expensive kind"),
    ("stops placed", r"STOP-LOSS ORDER PLACED",
     "hard/trail stop legs armed (informational)"),
    ("scale-out fired", r"SCALE-OUT STOP FIRED",
     "the trailing leg filled, remainder on the hard stop"),
    ("spacing too tight", r"GRID SPACING TOO TIGHT",
     "rungs closer than the fee floor — cycles that cannot pay for themselves"),
    ("level stranded", r"LEVEL STRANDED",
     "a rung duplicated an order already tracked"),
    ("grid count mismatch", r"GRID COUNT MISMATCH",
     "tick rounding collapsed rungs together"),
    ("post-only rejected", r"-2019",
     "a maker order would have crossed — harmless in small numbers"),
    ("reduce-only rejected", r"-2022",
     "a stop was placed against a position that had already gone"),
    ("min notional", r"-4164",
     "an order came in under 5 USDT — a sizing bug"),
    ("order not found", r"-2013",
     "an order vanished between reading it and acting on it"),
    ("timestamp drift", r"-1021",
     "clock skew against Binance"),
]

ANALYTICS = re.compile(r"GRID ANALYTICS \|.*")
LEVEL = re.compile(r"^\S+ \S+ \| (\w+)")

# "REGIME | <explanation> -> <regime>" is emitted on every evaluation, not on change, so
# counting the lines counts polls. The regime VALUE is what carries information: how the
# run was split, and how often it actually flipped.
REGIME = re.compile(r"REGIME \| .* -> (\S+)")


def regime_summary(text: str) -> tuple[Counter, int]:
    values = REGIME.findall(text)
    flips = sum(1 for a, b in zip(values, values[1:]) if a != b)
    return Counter(values), flips


def main() -> int:
    day = sys.argv[1] if len(sys.argv) > 1 else None
    if day:
        path = LOG_DIR / f"grid_{day}.log"
    else:
        logs = sorted(LOG_DIR.glob("grid_*.log"))
        if not logs:
            print(f"no logs in {LOG_DIR.resolve()}")
            return 1
        path = logs[-1]

    if not path.exists():
        print(f"no such log: {path}")
        return 1

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    levels = Counter(m.group(1) for m in (LEVEL.match(ln) for ln in lines) if m)

    print(f"log      {path}   ({len(lines):,} lines)")
    if lines:
        print(f"span     {lines[0][:19]}  ->  {lines[-1][:19]}")
    print("levels   " + "  ".join(f"{k}={v}" for k, v in levels.most_common()))

    regimes, flips = regime_summary(text)
    if regimes:
        total = sum(regimes.values())
        mix = "  ".join(f"{k} {v / total:.0%}" for k, v in regimes.most_common())
        print(f"regime   {mix}   ({flips} flip(s) over {total} reading(s))")

    analytics = ANALYTICS.findall(text)
    if analytics:
        print(f"latest   {analytics[-1]}")
    print()

    flagged = 0
    for label, pattern, meaning in SIGNALS:
        n = len(re.findall(pattern, text))
        if n:
            flagged += 1
            print(f"  {n:>6}  {label:<21} {meaning}")
    if not flagged:
        print("  nothing flagged")

    # ERROR lines are the ones worth reading verbatim -- but collapsed by shape, so one
    # retry loop does not print four hundred times.
    errors = [ln for ln in lines if "| ERROR" in ln or "| CRITICAL" in ln]
    if errors:
        shapes = Counter(
            re.sub(r"[\d.]{4,}", "N", ln.split("|", 3)[-1].strip())[:110] for ln in errors
        )
        print(f"\ndistinct error shapes ({len(errors)} total):")
        for shape, n in shapes.most_common(12):
            print(f"  {n:>5}x  {shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
