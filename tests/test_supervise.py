"""The supervisor's decisions. AUDIT #84.

The trading loop survives almost everything; the PROCESS dying is what nothing covers,
and it dies quietly -- position and stops stay on the exchange with nothing tending
them. These cover the two judgements that can strand you at 3am: when to put the bot
back, and when to stop trying.
"""

import os
import time

import config
import telegram_notifier
from supervise import RestartPolicy, _alert, log_is_stale


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


# --- consecutive hangs: a slow, silent gap the wall-clock window cannot see -------
#
# A hang-kill costs at least stale_after seconds to even happen, once per attempt. A
# bot that hangs identically every time therefore produces crashes spaced FURTHER
# apart than window_seconds can hold with the module's own matching 600s defaults, and
# each one evicts the last from crashes_in_window() before a second can ever join it.
# That is invisible to the wall-clock breaker above no matter how many times it
# repeats, so it is tracked as a simple consecutive count instead.

def test_the_wall_clock_window_alone_cannot_see_a_hang_that_recurs_slower_than_it():
    """Documents the actual gap: with default constants, a hang-detect cycle
    (stale_after + the ~30s poll + backoff) already exceeds window_seconds, so
    crashes_in_window() never accumulates past 1 no matter how many times the
    identical hang repeats -- the ordinary breaker alone would never trip."""
    p = policy(max_restarts=5, window_seconds=600.0, base_backoff=10.0)
    t = 0.0
    for _ in range(8):
        t += 640.0                      # stale_after(600) + poll(~30) + backoff(10)
        p.record_crash(t)
    assert p.crashes_in_window() == 1, "this test's own premise is wrong if this fails"
    assert p.tripped() is False, "the wall-clock breaker alone never sees this pattern"


def test_repeated_hangs_trip_even_when_far_apart_in_time():
    p = policy(max_restarts=3)
    for _ in range(4):
        p.record_hang(True)
    assert p.hang_tripped() is True


def test_hangs_at_or_below_the_limit_do_not_trip():
    p = policy(max_restarts=3)
    for _ in range(3):
        p.record_hang(True)
    assert p.hang_tripped() is False


def test_a_non_hang_crash_resets_the_hang_streak():
    """A genuinely different failure in between is not "the same hang repeating" --
    the streak must not silently carry through it."""
    p = policy(max_restarts=3)
    for _ in range(3):
        p.record_hang(True)
    assert p.hang_tripped() is False

    p.record_hang(False)                # an ordinary crash, not a hang
    assert p.consecutive_hangs() == 0

    for _ in range(3):
        p.record_hang(True)
    assert p.hang_tripped() is False, "the earlier streak leaked through the reset"


def test_a_clean_or_successful_run_also_resets_the_hang_streak():
    p = policy(max_restarts=3)
    for _ in range(3):
        p.record_hang(True)
    p.record_hang(False)                # e.g. a clean exit
    assert p.consecutive_hangs() == 0


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


def test_a_stale_log_from_before_this_attempt_does_not_condemn_a_fresh_start(tmp_path):
    """The bug actually hit in production: the machine was off, so the log's last
    write is long in the past, and the very next restart attempt inherited that stale
    clock from the moment it started -- getting killed as "hung" on the first 30s
    poll, before it had written a single line, let alone had a fair chance to. `since`
    (this attempt's own start time) floors the staleness clock so real downtime from
    BEFORE the attempt began cannot poison it."""
    log = tmp_path / "grid_2026-08-28.log"
    log.write_text("x", encoding="utf-8")
    os.utime(log, (1000, 1000))          # last written long before this attempt began

    # "now" is only 10s after this attempt's own start (`since`) -- nowhere near the
    # 600s hang threshold from THIS attempt's perspective, even though the log file
    # itself is ancient.
    assert log_is_stale(tmp_path, limit_seconds=600, now=1910, since=1900) is False


def test_a_genuinely_hung_fresh_attempt_is_still_caught(tmp_path):
    """`since` must not become a free pass -- an attempt that never writes anything is
    still hung once ITS OWN limit_seconds has elapsed, exactly as before."""
    log = tmp_path / "grid_2026-08-28.log"
    log.write_text("x", encoding="utf-8")
    os.utime(log, (1000, 1000))

    assert log_is_stale(tmp_path, limit_seconds=600, now=2600, since=1900) is True


# --- the supervisor's own alert -----------------------------------------------------
#
# Exercised only when the breaker trips -- exactly the failure it exists to report,
# and exactly why a bug here can sit unnoticed until the one time it matters.

class _FakeNotifier:
    """Records what it was built and asked to do, but sends nothing for real."""

    instances: list["_FakeNotifier"] = []

    def __init__(self, bot_token, chat_id, enabled=False):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.sent: list[str] = []
        self.closed = False
        _FakeNotifier.instances.append(self)

    def send(self, message):
        self.sent.append(message)
        return True

    def close(self):
        self.closed = True


def test_alert_builds_the_notifier_with_token_and_chat_id(monkeypatch):
    """TelegramNotifier(settings) used to pass the whole settings object as bot_token
    and never supply chat_id at all -- a guaranteed TypeError, on the one call this
    whole breaker exists to make."""
    _FakeNotifier.instances.clear()
    monkeypatch.setattr(telegram_notifier, "TelegramNotifier", _FakeNotifier)
    monkeypatch.setattr(config.settings, "telegram_bot_token", "test-token")
    monkeypatch.setattr(config.settings, "telegram_chat_id", "test-chat")
    monkeypatch.setattr(config.settings, "telegram_enabled", True)

    _alert("bot supervisor stopped")

    assert len(_FakeNotifier.instances) == 1
    n = _FakeNotifier.instances[0]
    assert (n.bot_token, n.chat_id, n.enabled) == ("test-token", "test-chat", True)
    assert n.sent == ["<b>SUPERVISOR</b>\nbot supervisor stopped"]


def test_alert_closes_the_notifier_so_the_background_worker_gets_to_send(monkeypatch):
    """send() only queues the message for a background daemon thread; it does not send
    it. _alert() returns straight into a process exit, so without an explicit, bounded
    close() the message is still sitting in the queue when the daemon thread is killed
    with the process -- an alert that was "sent" successfully and never arrived."""
    _FakeNotifier.instances.clear()
    monkeypatch.setattr(telegram_notifier, "TelegramNotifier", _FakeNotifier)
    monkeypatch.setattr(config.settings, "telegram_bot_token", "test-token")
    monkeypatch.setattr(config.settings, "telegram_chat_id", "test-chat")
    monkeypatch.setattr(config.settings, "telegram_enabled", True)

    _alert("bot supervisor stopped")

    assert _FakeNotifier.instances[0].closed is True


def test_alert_is_best_effort_and_never_raises(monkeypatch):
    """A supervisor that dies trying to report its own death is worse than useless."""
    class ExplodingNotifier:
        def __init__(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(telegram_notifier, "TelegramNotifier", ExplodingNotifier)

    _alert("this must not raise")  # no exception == pass
