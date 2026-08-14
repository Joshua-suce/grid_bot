"""check_run.py must not report a false all-clear.

A health check whose patterns have rotted is worse than no health check: it turns
"I looked and it's fine" into "I looked at nothing." Every signal is pinned here against
the verbatim string the bot emits, so renaming a log line breaks a test instead of
silently blinding the check.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from check_run import SIGNALS, regime_summary  # noqa: E402

STAMP = "2026-08-15 06:43:15 | WARNING | main:loop:1 | "

# One verbatim line per signal, taken from the emitting call site (or from a real log).
REAL_LINES: dict[str, str] = {
    "shut down": "MAX RECOVERY REACHED (5) — shutting down bot",
    "emergency stop": "EMERGENCY STOP | cancelling grid orders (8)",
    "naked position": "POSITION UNDER-PROTECTED | long 1264.0 — stops cover 632 of 1264",
    "stop book unreadable": "STOP REFRESH ABORTED | stop book unreadable — existing stops left in place",
    "state not persisted": "STATE NOT PERSISTED | 5 consecutive save failures — a restart will reload STALE state",
    "legacy state ignored": "LEGACY STATE IGNORED | state/grid_dogeusdt.json predates per-account state files",
    "state corrupt": "GRID STATE CORRUPT | expected 8 levels but restored 5 — rebuilding",
    "ladder outgrows cap": "LADDER OUTGROWS THE CAP | 4 rungs a side at 125.00 USDT commits 500.00",
    "pnl feed stale": "PNL FEED STALE | no income data for 900s — the daily-loss kill switch is blind",
    "recenter aborted": "RECENTER ABORTED | 3 orders still open after pause (write path down)",
    "exposure blocked": "EXPOSURE WARNING | grid continues but new orders blocked",
    "side blocked": "POSITION LIMIT | long 7200.0 >= 7100.0 — buy orders blocked",
    "recenter": "RECENTERING GRID | price 0.06972 outside [0.06936-0.07116] (margin 2.0%)",
    "stops placed": "STOP-LOSS ORDER PLACED | kind=hard side=sell qty=1264.0 @ 0.0677",
    "scale-out fired": "SCALE-OUT STOP FIRED | trailing leg filled at 0.0712 — remainder on hard stop only",
    "spacing too tight": "GRID SPACING TOO TIGHT | spacing=0.00018 (0.0180%) < min (0.1300%)",
    "level stranded": "LEVEL STRANDED | buy @ 0.0691 duplicates order 12345 already tracked by level 3",
    "grid count mismatch": "GRID COUNT MISMATCH | expected=8 actual=7 after tick dedup — adjusting",
    "post-only rejected": 'binanceusdm {"code":-2019,"msg":"Margin is insufficient."}',
    "reduce-only rejected": 'Invalid order: sell 1264.0 DOGEUSDT @ 0.0702: binanceusdm {"code":-2022,"msg":"ReduceOnly Order is rejected."}',
    "min notional": 'binanceusdm {"code":-4164,"msg":"Order\'s notional must be no smaller than 5"}',
    "order not found": 'binanceusdm {"code":-2013,"msg":"Order does not exist."}',
    "timestamp drift": 'binanceusdm {"code":-1021,"msg":"Timestamp for this request is outside of the recvWindow."}',
}


def _run(workspace: Path, day: str) -> str:
    out = subprocess.run(
        [sys.executable, "check_run.py", day],
        cwd=workspace, capture_output=True, text=True, timeout=60,
    )
    return out.stdout + out.stderr


@pytest.fixture
def workspace(tmp_path):
    """check_run resolves logs/ relative to cwd, so give it a cwd of its own."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "check_run.py").write_text(
        (ROOT / "check_run.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return tmp_path


def test_every_signal_has_a_pinned_line():
    """A signal with no example is a pattern nobody has ever proved matches anything."""
    unpinned = [label for label, _, _ in SIGNALS if label not in REAL_LINES]
    assert not unpinned, f"signals with no verbatim example: {unpinned}"


@pytest.mark.parametrize("label", sorted(REAL_LINES))
def test_signal_matches_the_line_the_bot_writes(workspace, label):
    (workspace / "logs" / "grid_2026-08-15.log").write_text(
        STAMP + REAL_LINES[label] + "\n", encoding="utf-8"
    )
    out = _run(workspace, "2026-08-15")
    assert label in out, f"'{label}' no longer matches the line the bot emits"
    assert "nothing flagged" not in out


@pytest.mark.parametrize("label", sorted(REAL_LINES))
def test_signal_does_not_match_everything(workspace, label):
    """A pattern that fires on ordinary traffic is noise, not a signal."""
    (workspace / "logs" / "grid_2026-08-15.log").write_text(
        STAMP + "Grid healthy, 8 orders resting at 0.06979\n", encoding="utf-8"
    )
    out = _run(workspace, "2026-08-15")
    assert label not in out, f"'{label}' fired on a benign line"


def test_a_quiet_log_reports_nothing_flagged(workspace):
    (workspace / "logs" / "grid_2026-08-15.log").write_text(
        STAMP + "Grid healthy, 8 orders resting\n", encoding="utf-8"
    )
    assert "nothing flagged" in _run(workspace, "2026-08-15")


def test_a_missing_log_is_an_error_not_an_all_clear(workspace):
    out = _run(workspace, "2026-08-15")
    assert "nothing flagged" not in out, "a missing log read as a clean run"
    assert "no such log" in out


def test_the_analytics_line_surfaces(workspace):
    (workspace / "logs" / "grid_2026-08-15.log").write_text(
        STAMP + "GRID ANALYTICS | fills=1 cycles=0 | gross=0.1 fees=0.02 net=0.08\n"
        + STAMP + "GRID ANALYTICS | fills=9 cycles=4 | gross=1.4 fees=0.30 net=1.10\n",
        encoding="utf-8",
    )
    out = _run(workspace, "2026-08-15")
    assert "fills=9 cycles=4" in out, "the latest analytics line was not surfaced"
    assert "fills=1 cycles=0" not in out, "an earlier analytics line was shown instead"


def test_regime_lines_are_counted_as_readings_not_switches():
    """The bot logs REGIME on every evaluation. Counting the lines counts polls, which
    read as 107 regime changes in a run that never changed regime once."""
    text = "\n".join(
        STAMP + "REGIME | needs 2 of 3 to agree -> uncertain" for _ in range(107)
    )
    regimes, flips = regime_summary(text)

    assert flips == 0, "an unchanging regime was reported as flipping"
    assert regimes == {"uncertain": 107}


def test_regime_flips_are_counted_on_transitions_only():
    values = ["uncertain", "uncertain", "trend_up", "trend_up", "uncertain"]
    text = "\n".join(STAMP + f"REGIME | 2 of 3 agree -> {v}" for v in values)

    regimes, flips = regime_summary(text)

    assert flips == 2
    assert regimes == {"uncertain": 3, "trend_up": 2}


def test_regime_mix_is_reported(workspace):
    (workspace / "logs" / "grid_2026-08-15.log").write_text(
        "\n".join(
            STAMP + f"REGIME | 2 of 3 agree -> {v}"
            for v in ["uncertain", "uncertain", "uncertain", "trend_up"]
        ),
        encoding="utf-8",
    )
    out = _run(workspace, "2026-08-15")
    assert "uncertain 75%" in out
    assert "1 flip(s) over 4 reading(s)" in out


def test_labels_are_unique():
    labels = [label for label, _, _ in SIGNALS]
    assert len(labels) == len(set(labels))


def test_patterns_compile():
    for _, pattern, _ in SIGNALS:
        re.compile(pattern)
