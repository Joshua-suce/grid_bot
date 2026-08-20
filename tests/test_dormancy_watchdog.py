"""A bot holding exposure with an empty ladder must not be able to sit there quietly.

2026-08-19, 16:31:42 -> 19:44:58. A 6,307 ADA short, zero working orders, and 4,770
clean polls. Unrealised went -7.36 -> -36. Nothing raised, so every exception-driven
recovery path in the program stayed asleep, and supervise.log_is_stale -- the only
liveness test that existed -- watches log mtime, which a healthy-looking loop keeps
fresh by definition. AUDIT #127.
"""
from __future__ import annotations

import inspect
import re

import pytest

import main as main_module
from main import dormancy_action, dormancy_clock

ALERT, RESTART = 900.0, 2700.0


# ------------------------------------------------------------------- the clock
def test_a_working_ladder_is_never_dormant():
    since, secs = dormancy_clock(True, working_orders=4, strategy_active=True,
                                 dormant_since=None, now=1000.0)
    assert (since, secs) == (None, 0.0)


def test_being_flat_is_never_dormant():
    """No position means an empty book is just an idle bot, which is fine."""
    since, secs = dormancy_clock(False, working_orders=0, strategy_active=True,
                                 dormant_since=None, now=1000.0)
    assert (since, secs) == (None, 0.0)


def test_a_deliberately_paused_strategy_is_not_dormant():
    """Recovery cooldown and the post-kill-switch pause both hold a position with no
    orders on purpose. Counting those would alert on correct behaviour."""
    since, secs = dormancy_clock(True, working_orders=0, strategy_active=False,
                                 dormant_since=None, now=1000.0)
    assert (since, secs) == (None, 0.0)


def test_the_clock_starts_on_the_first_dormant_poll():
    since, secs = dormancy_clock(True, 0, True, dormant_since=None, now=1000.0)
    assert since == 1000.0 and secs == 0.0


def test_the_clock_accumulates_across_polls():
    since, secs = dormancy_clock(True, 0, True, dormant_since=1000.0, now=1600.0)
    assert since == 1000.0 and secs == 600.0


def test_one_placed_order_resets_the_clock():
    """The condition is 'nothing is working', not 'things are bad'. A single live
    rung means the ladder can still resolve the position on its own."""
    since, secs = dormancy_clock(True, working_orders=1, strategy_active=True,
                                 dormant_since=1000.0, now=9999.0)
    assert (since, secs) == (None, 0.0)


def test_a_clock_that_went_backwards_never_reports_negative():
    since, secs = dormancy_clock(True, 0, True, dormant_since=2000.0, now=1000.0)
    assert secs == 0.0


def test_the_real_incident_is_measured_correctly():
    """16:31:42 -> 19:44:58 is 11,596 seconds."""
    _, secs = dormancy_clock(True, 0, True, dormant_since=0.0, now=11596.0)
    assert secs == pytest.approx(11596.0)
    assert dormancy_action(secs, 1e9, ALERT, RESTART) == "restart"


# ------------------------------------------------------------------ escalation
@pytest.mark.parametrize("secs,expected", [
    (0.0, "none"), (899.0, "none"), (900.0, "alert"),
    (2699.0, "alert"), (2700.0, "restart"), (11596.0, "restart"),
])
def test_the_escalation_ladder(secs, expected):
    assert dormancy_action(secs, 1e9, ALERT, RESTART) == expected


def test_the_alert_repeats_rather_than_firing_once():
    """POSITION UNDER-PROTECTED fired exactly once on 2026-08-19 (16:30:26) and never
    again. A correct detection that speaks once is a detection nobody hears."""
    assert dormancy_action(1800.0, seconds_since_alert=901.0,
                           alert_after=ALERT, restart_after=RESTART) == "alert"


def test_it_does_not_re_alert_before_the_interval_elapses():
    assert dormancy_action(1800.0, seconds_since_alert=10.0,
                           alert_after=ALERT, restart_after=RESTART) == "none"


def test_a_restart_is_never_suppressed_by_the_alert_throttle():
    """The throttle governs chatter, not the escape hatch."""
    assert dormancy_action(3000.0, seconds_since_alert=0.0,
                           alert_after=ALERT, restart_after=RESTART) == "restart"


def test_zero_disables_each_rung_independently():
    assert dormancy_action(99999.0, 1e9, alert_after=0.0, restart_after=0.0) == "none"
    assert dormancy_action(99999.0, 1e9, alert_after=0.0, restart_after=RESTART) == "restart"
    assert dormancy_action(99999.0, 1e9, alert_after=ALERT, restart_after=0.0) == "alert"


def test_alert_fires_before_restart_at_the_shipped_defaults():
    """A restart threshold at or below the alert threshold would kill the process
    with no warning ever sent."""
    from config import Settings
    d = Settings.model_fields
    assert (d["empty_book_restart_seconds"].default
            > d["empty_book_alert_seconds"].default > 0)


# --------------------------------------------------------- wired into the loop
def _loop_source() -> str:
    src = inspect.getsource(main_module.run_bot)
    return src[src.index("while True:"):]


def test_the_watchdog_actually_runs_in_the_trading_loop():
    """A helper nothing calls is not a watchdog. This is the wiring, and it is the
    part that has silently gone missing before -- ladder_cap_room needed the same
    pin."""
    loop = _loop_source()
    assert "dormancy_clock(" in loop, "dormancy_clock is never called from the loop"
    assert "dormancy_action(" in loop, "dormancy_action is never called from the loop"


def test_the_loop_slice_is_not_empty():
    """Guard on the guard: if `while True:` moves, the slice above could go empty and
    every assertion on it would pass vacuously."""
    assert len(_loop_source()) > 5000


def test_the_count_excludes_the_conditional_book():
    """fetch_open_orders returns stop orders too. Counting them would mean a lone
    stop-loss leg reads as a healthy ladder -- masking the exact state being watched.

    Asserts on the COUNTING EXPRESSION, not on the surrounding lines. The first
    version of this test looked for get_tracked_order_ids() anywhere in the loop and
    passed against a mutant that had replaced the count with len(open_orders) -- the
    lookup line was still there, one line above, doing nothing.
    """
    loop = _loop_source()
    m = re.search(r"^\s*_working\s*=\s*(.+)$", loop, re.M)
    assert m, "no _working assignment found in the loop"
    expr = m.group(1)
    assert "_tracked" in expr, (
        f"the dormancy count must be filtered against the grid's tracked ids, "
        f"but it is computed as: {expr.strip()}"
    )


def test_the_tracked_id_lookup_feeds_that_expression():
    """The filter is worthless if _tracked is never populated from the grid."""
    loop = _loop_source()
    m = re.search(r"^\s*_tracked\s*=\s*(.+)$", loop, re.M)
    assert m and "get_tracked_order_ids()" in m.group(1)


def test_the_restart_rung_exits_non_zero():
    """Exiting 0 would tell supervise.py a human chose to stop, and the bot would
    stay down holding the position -- strictly worse than the deadlock it replaces
    (AUDIT #126)."""
    loop = _loop_source()
    assert "SystemExit(1)" in loop, "the restart rung must exit non-zero"


def test_the_restart_cannot_be_swallowed_by_the_loops_error_handler():
    """The loop wraps each iteration in `except Exception`. SystemExit derives from
    BaseException precisely so this escape hatch cannot be caught by it -- pin that
    the code relies on a BaseException and not, say, RuntimeError."""
    assert not issubclass(SystemExit, Exception)
    loop = _loop_source()
    idx = loop.index("SystemExit(1)")
    assert "raise" in loop[max(0, idx - 40):idx]
