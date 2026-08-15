"""The supervisor's decisions. AUDIT #84.

The trading loop survives almost everything; the PROCESS dying is what nothing covers,
and it dies quietly -- position and stops stay on the exchange with nothing tending
them. These cover the two judgements that can strand you at 3am: when to put the bot
back, and when to stop trying.
"""

import time

from supervise import RestartPolicy, log_is_stale


def policy(**kw):
    kw.setdefault("max_restarts", 3)
    kw.setdefault("window_seconds", 600.0)
    return RestartPolicy(**kw)


# --- a clean exit is a decision someone made -------------------------------------

def test_a_clean_exit_is_not_restarted():
    """Ctrl-C, or an operator stopping the bot, must not be undone."""
    assert policy().should_restart(0) is False


def test_a_crash_is_restarted():
    assert policy().should_restart(1) is True


def test_a_killed_process_is_restarted():
    assert policy().should_restart(-9) is True


def test_no_exit_code_is_treated_as_a_crash():
    assert policy().should_restart(None) is True


# --- the breaker ------------------------------------------------------------------

def test_it_keeps_restarting_up_to_the_limit():
    p = policy(max_restarts=3)
    now = time.time()
    for i in range(3):
        p.record_crash(now + i)
        assert p.should_restart(1) is True, f"gave up after {i+1} crash(es)"
    assert p.tripped() is False


def test_it_stops_once_the_limit_is_passed():
    """Restarting churns the book -- startup cancels every resting order. A bot that
    keeps dying should stay down predictably rather than loop."""
    p = policy(max_restarts=3)
    now = time.time()
    for i in range(4):
        p.record_crash(now + i)

    assert p.tripped() is True
    assert p.should_restart(1) is False


def test_a_clean_exit_still_wins_after_the_breaker_trips():
    p = policy(max_restarts=1)
    now = time.time()
    p.record_crash(now)
    p.record_crash(now + 1)
    assert p.should_restart(0) is False


def test_old_crashes_fall_out_of_the_window():
    """A bot that dies once a week is not crash-looping."""
    p = policy(max_restarts=2, window_seconds=600)
    now = time.time()
    p.record_crash(now - 5000)
    p.record_crash(now - 4000)
    p.record_crash(now)

    assert p.crashes_in_window() == 1
    assert p.should_restart(1) is True


def test_the_window_is_measured_from_the_newest_crash():
    p = policy(window_seconds=100)
    p.record_crash(1000.0)
    p.record_crash(1150.0)
    p.record_crash(1200.0)          # cutoff becomes 1100, so 1000.0 falls out
    assert p.crashes_in_window() == 2


# --- backoff ----------------------------------------------------------------------

def test_the_first_restart_is_quick():
    p = policy(base_backoff=10)
    p.record_crash(time.time())
    assert p.backoff() == 10


def test_backoff_doubles_per_consecutive_crash():
    p = policy(base_backoff=10, max_backoff=1000)
    now = time.time()
    seen = []
    for i in range(4):
        p.record_crash(now + i)
        seen.append(p.backoff())
    assert seen == [10, 20, 40, 80]


def test_backoff_is_capped():
    p = policy(base_backoff=10, max_backoff=60)
    now = time.time()
    for i in range(10):
        p.record_crash(now + i)
    assert p.backoff() == 60


def test_backoff_never_returns_a_negative_wait():
    assert policy().backoff() >= 0


# --- the hang detector ------------------------------------------------------------
#
# A hung bot is worse than a dead one: it holds a position, places nothing, and the
# supervisor cannot see it. The loop writes a line every poll, so the log is the pulse.

def test_a_log_that_stopped_advancing_reads_as_hung(tmp_path):
    log = tmp_path / "grid_2026-08-15.log"
    log.write_text("x", encoding="utf-8")
    import os
    os.utime(log, (1000, 1000))

    assert log_is_stale(tmp_path, limit_seconds=600, now=2000) is True


def test_a_log_being_written_reads_as_alive(tmp_path):
    log = tmp_path / "grid_2026-08-15.log"
    log.write_text("x", encoding="utf-8")
    import os
    os.utime(log, (1900, 1900))

    assert log_is_stale(tmp_path, limit_seconds=600, now=2000) is False


def test_no_log_at_all_is_not_a_hang(tmp_path):
    """A bot that has not started writing yet must not be killed for it."""
    assert log_is_stale(tmp_path, limit_seconds=600, now=2000) is False


def test_the_newest_log_is_the_one_that_counts(tmp_path):
    """Yesterday's rotated log must not make a live bot look hung."""
    import os
    old = tmp_path / "grid_2026-08-14.log"
    new = tmp_path / "grid_2026-08-15.log"
    old.write_text("x", encoding="utf-8")
    new.write_text("x", encoding="utf-8")
    os.utime(old, (1000, 1000))
    os.utime(new, (1950, 1950))

    assert log_is_stale(tmp_path, limit_seconds=600, now=2000) is False


def test_an_unreadable_log_dir_is_not_a_hang(tmp_path):
    assert log_is_stale(tmp_path / "does-not-exist", limit_seconds=1, now=2000) is False
