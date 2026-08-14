import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from analyze_performance import build_report, default_csv  # noqa: E402


@pytest.mark.parametrize("env,expected", [
    ("true", "trades_demo.csv"),
    ("True", "trades_demo.csv"),
    ("1", "trades_demo.csv"),
    ("false", "trades_live.csv"),
    ("False", "trades_live.csv"),
    ("0", "trades_live.csv"),
])
def test_the_default_journal_follows_demo_mode(monkeypatch, env, expected):
    """The journals are per-account since #70, so there is no single trades.csv to
    default to -- and defaulting to the wrong one reports demo fills as live results."""
    monkeypatch.setenv("DEMO_MODE", env)
    assert default_csv().name == expected


def _row(ts, regime, cycle_pnl, fee, completed=True):
    return {
        "timestamp": ts,
        "regime": regime,
        "cycle_pnl": str(cycle_pnl),
        "fee": str(fee),
        "completed_cycle": "True" if completed else "False",
    }


def test_build_report_basic_totals():
    rows = [
        _row("2026-01-01T00:00:00", "ranging", 1.0, 0.1),
        _row("2026-01-01T01:00:00", "ranging", 2.0, 0.1),
        _row("2026-01-02T00:00:00", "downtrend", -1.0, 0.1),
    ]
    report = build_report(rows)

    assert report.total_fills == 3
    assert report.completed_cycles == 3
    assert report.gross_pnl == pytest.approx(2.0)
    assert report.total_fees == pytest.approx(0.3)
    assert report.net_pnl == pytest.approx(1.7)


def test_build_report_flags_net_negative_regime():
    rows = [
        _row("2026-01-01T00:00:00", "ranging", 5.0, 0.1),
        _row("2026-01-01T01:00:00", "downtrend", -3.0, 0.2),
        _row("2026-01-01T02:00:00", "downtrend", -3.0, 0.2),
    ]
    report = build_report(rows)

    assert report.by_regime["ranging"].net > 0
    assert report.by_regime["downtrend"].net < 0
    assert report.by_regime["downtrend"].n == 2


def test_build_report_ignores_open_fills_not_yet_completed():
    rows = [
        _row("2026-01-01T00:00:00", "ranging", 0.0, 0.0, completed=False),
        _row("2026-01-01T01:00:00", "ranging", 1.0, 0.1, completed=True),
    ]
    report = build_report(rows)

    assert report.total_fills == 2
    assert report.completed_cycles == 1
    assert report.gross_pnl == 1.0


def test_build_report_win_rate():
    rows = [
        _row("2026-01-01T00:00:00", "ranging", 1.0, 0.1),
        _row("2026-01-01T01:00:00", "ranging", -1.0, 0.1),
    ]
    report = build_report(rows)
    stats = report.by_regime["ranging"]
    assert stats.n == 2
    assert stats.wins == 1
    assert stats.win_rate == 0.5
