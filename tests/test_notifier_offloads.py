"""Notifications must never be on the trading loop's critical path. AUDIT #91.

TelegramNotifier.send used to do the HTTP round trip inline, so a slow Telegram stopped
the bot. Four measured stalls in the 2026-08-17 05:00 session, 16-18 seconds each, with
the next log line arriving only once the send returned:

    06:54:39 ORDER PLACED -> 06:54:55 TG OK
    08:20:48 FILL #41     -> 08:21:06 TG OK
    08:50:18 ORDER PLACED -> 08:50:34 TG OK
    09:10:41 ORDER PLACED -> 09:10:57 TG OK

That is why a 10-second poll interval ran at a 15-16 second cadence and stretched to
22-28 seconds in places, and why restoring a 14-order ladder took 26 seconds. Every
second spent inside an HTTP call is a second the ladder is not being checked for fills.
"""

import queue
import threading
import time

from telegram_notifier import TelegramNotifier

# Generous enough that a loaded CI box will not trip it, tight enough that it can only
# pass if the send did NOT wait on the network: one inline send would already blow it.
IMMEDIATE_SECONDS = 0.5
SLOW_SEND_SECONDS = 0.25


class FakeClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class RecordingNotifier(TelegramNotifier):
    """Real queue, real worker thread, fake transport."""

    def __init__(self, delay=0.0, limit=None, explode=False):
        self.enabled = True
        self._client = FakeClient()
        self.max_retries = 1
        self._last_event_time = {}
        self.order_event_cooldown = 30.0
        self._queue = queue.Queue(maxsize=limit or self.QUEUE_LIMIT)
        self._worker = None
        self._worker_lock = threading.Lock()
        self._dropped = 0
        self._closed = False
        self.delay = delay
        self.explode = explode
        self.received = []
        self.gate = threading.Event()

    def _send_now(self, message):
        if self.delay:
            self.gate.wait(timeout=self.delay)
        if self.explode:
            raise RuntimeError("telegram exploded")
        self.received.append(message)
        return True


def drain(n, timeout=5.0):
    """Wait for the worker to finish everything queued so far."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if n._queue.unfinished_tasks == 0:
            return True
        time.sleep(0.01)
    return False


# --- the stall itself ----------------------------------------------------------------

def test_send_returns_without_waiting_for_the_network():
    """Twenty sends against a transport that takes a quarter second each. Inline that is
    five seconds of a stopped trading loop; queued it is not measurable."""
    n = RecordingNotifier(delay=SLOW_SEND_SECONDS)
    try:
        started = time.monotonic()
        for i in range(20):
            n.send(f"m{i}")
        elapsed = time.monotonic() - started

        assert elapsed < IMMEDIATE_SECONDS, f"send blocked the caller for {elapsed:.2f}s"
    finally:
        n.gate.set()
        n.close()


def test_a_fill_notification_still_reaches_telegram():
    """Off the hot path, not dropped on the floor."""
    n = RecordingNotifier()
    try:
        n.on_fill("buy", 0.07012, 0.2310, 41, daily_pnl=2.61)
        assert drain(n), "message never sent"
        assert len(n.received) == 1
        assert "FILL #41" in n.received[0]
    finally:
        n.close()


def test_messages_keep_their_order():
    """One worker, FIFO queue. Out-of-order fills would misreport the sequence of a
    session to the one place the operator actually watches."""
    n = RecordingNotifier()
    try:
        for i in range(25):
            n.send(f"m{i}")
        assert drain(n)
        assert n.received == [f"m{i}" for i in range(25)]
    finally:
        n.close()


# --- failure modes that must not reach the trading loop -------------------------------

def test_a_wedged_telegram_drops_messages_instead_of_blocking():
    """The backlog is bounded on purpose. An unbounded queue inside a process that has
    to keep trading is just a slower way to fail."""
    n = RecordingNotifier(delay=2.0, limit=4)
    try:
        started = time.monotonic()
        results = [n.send(f"m{i}") for i in range(20)]
        elapsed = time.monotonic() - started

        assert elapsed < IMMEDIATE_SECONDS, f"a wedged Telegram blocked for {elapsed:.2f}s"
        assert any(r is False for r in results), "queue grew without bound"
        assert n._dropped > 0
        assert n._queue.qsize() <= 4
    finally:
        n.gate.set()
        n.close()


def test_a_raising_transport_does_not_kill_the_worker():
    """If one send throws and the thread dies, every later notification is silently lost
    while send() keeps cheerfully returning True."""
    n = RecordingNotifier(explode=True)
    try:
        n.send("boom")
        assert drain(n)

        n.explode = False
        n.send("after")
        assert drain(n)
        assert n.received == ["after"], "worker died on the first exception"
    finally:
        n.close()


def test_close_shuts_the_worker_down():
    n = RecordingNotifier()
    n.send("last")
    n.close()

    assert n._worker is None
    assert n.send("after close") is False, "kept queueing after close"


def test_close_does_not_hang_on_a_wedged_send():
    """Shutdown has orders to cancel and stops to leave armed; it cannot wait on
    Telegram. close() is bounded by DRAIN_TIMEOUT_SECONDS."""
    n = RecordingNotifier(delay=30.0)
    n.DRAIN_TIMEOUT_SECONDS = 0.2
    n.send("stuck")
    time.sleep(0.05)

    started = time.monotonic()
    n.close()
    elapsed = time.monotonic() - started
    n.gate.set()

    assert elapsed < 2.0, f"close() waited {elapsed:.2f}s on a wedged send"


def test_the_real_constructor_bounds_the_queue():
    """The drop behaviour above is exercised through a queue the fixture builds, so it
    says nothing about what run_bot actually gets. Assert the wiring directly: an
    unbounded production queue turns a wedged Telegram into unbounded memory growth
    inside the trading process, and every test above would still pass."""
    n = TelegramNotifier("", "", enabled=False)

    assert n._queue.maxsize == TelegramNotifier.QUEUE_LIMIT > 0


def test_a_disabled_notifier_starts_no_thread():
    """The common test/CI configuration. Spawning a worker for messages that can never
    be sent leaks a thread per notifier."""
    n = TelegramNotifier("", "", enabled=False)

    assert n.send("anything") is False
    assert n._worker is None
