"""One waiting handoff must not read as forty-one failing ones. AUDIT #121.

_begin_handoff is called from update_regime on every regime check once the regime has
held for min_regime_seconds -- so on a stable regime it is called every cycle, forever.
It is idempotent by design: the second call onward returns early, deliberately, because
restarting the clock would push the grace deadline out every iteration and it would never
expire.

The "handoff starting" line sat ABOVE that guard, so it printed on every call.

ADAUSDT, 2026-08-19, grid holding SHORT 6307 it could not exit:

    16:35:37  REGIME CHANGE (confirmed after 309s) | uncertain -> uptrend | ADX=28.5
    16:51:09  ROUTER | handoff grid -> trend starting (waiting for flat, grace 21600s)
    16:56:22  ROUTER | handoff grid -> trend starting (waiting for flat, grace 21600s)
    17:01:24  ROUTER | handoff grid -> trend starting (waiting for flat, grace 21600s)
    ... 41 identical lines through 19:40 ...

Read straight, that is a handoff restarting or failing every five minutes. It was one
handoff, waiting exactly as designed, 3h07m short of the 6h grace that would have forced
it. Nothing in the log said so: not the elapsed time, not the deadline, not that the
count was one rather than forty-one.

The clock guard itself is the load-bearing part and is pinned first -- if it ever goes,
the grace becomes unreachable and the force-close never fires at all.
"""

import re
import time
from unittest.mock import MagicMock

from loguru import logger

from router import StrategyRouter
from strategy import Strategy


def make(position=6307.0, handoff_grace_seconds=21600):
    """A router mid-flight: grid active and holding, trend waiting to take over.

    Self-contained rather than importing tests/test_router.py -- nothing else in this
    suite cross-imports test modules, and tests/ is not on sys.path.
    """
    grid = MagicMock(spec=Strategy); grid.name = "grid"; grid.active = True; grid.levels = []
    trend = MagicMock(spec=Strategy); trend.name = "trend"; trend.active = False; trend.levels = []
    ex = MagicMock()
    ex.get_positions.return_value = (
        [] if position == 0 else
        [{"side": "long" if position > 0 else "short", "contracts": abs(position)}]
    )
    ex.get_price.return_value = 0.1806
    r = StrategyRouter(
        strategies={"grid": grid, "trend": trend}, min_regime_seconds=0,
        handoff_grace_seconds=handoff_grace_seconds, exchange=ex, symbol="ADAUSDT",
    )
    return r, grid, trend, ex


def capture(fn):
    """Run fn, return the INFO+ lines it logged."""
    sink = []
    handle = logger.add(lambda m: sink.append(str(m)), level="INFO")
    try:
        fn()
    finally:
        logger.remove(handle)
    return sink


def lines(sink, needle):
    return [l for l in sink if needle in l]


# --- the load-bearing guard -----------------------------------------------------------

def test_repeat_calls_do_not_restart_the_grace_clock():
    """If this breaks, the deadline moves every cycle and the force-close never fires --
    the handoff waits forever instead of six hours."""
    r, grid, trend, ex = make(position=6307.0, handoff_grace_seconds=21600)

    r._begin_handoff("trend")
    started = r._handoff_started

    time.sleep(0.02)
    r._begin_handoff("trend")
    r._begin_handoff("trend")

    assert r._handoff_started == started, "grace clock was pushed out by a repeat call"


def test_a_different_target_does_restart_it():
    r, grid, trend, ex = make(position=6307.0, handoff_grace_seconds=21600)

    r._begin_handoff("trend")
    first = r._handoff_started
    time.sleep(0.02)
    r._begin_handoff("grid")

    assert r._handoff_target == "grid"
    assert r._handoff_started > first


# --- what the log actually says ---------------------------------------------------------

def test_starting_is_announced_once_not_once_per_cycle():
    """The 41-line case."""
    r, grid, trend, ex = make(position=6307.0, handoff_grace_seconds=21600)

    sink = capture(lambda: [r._begin_handoff("trend") for _ in range(41)])

    assert len(lines(sink, "handoff grid -> trend starting")) == 1


def test_the_repeats_report_the_wait_instead():
    r, grid, trend, ex = make(position=6307.0, handoff_grace_seconds=21600)
    r._begin_handoff("trend")

    sink = capture(lambda: r._begin_handoff("trend"))

    waiting = lines(sink, "still waiting for flat")
    assert len(waiting) == 1
    assert "elapsed" in waiting[0]
    assert "force-closed" in waiting[0]


def test_the_wait_line_counts_down_toward_the_deadline():
    """The number a reader needs: is this progressing, and when does it give up."""
    r, grid, trend, ex = make(position=6307.0, handoff_grace_seconds=600)
    r._begin_handoff("trend")
    r._handoff_started = time.time() - 540.0        # 9 minutes in on a 10 minute grace

    sink = capture(lambda: r._begin_handoff("trend"))

    line = lines(sink, "still waiting for flat")[0]
    assert "540s elapsed" in line
    assert "60s until" in line


def test_an_expired_grace_never_reports_negative_time_remaining():
    r, grid, trend, ex = make(position=6307.0, handoff_grace_seconds=600)
    r._begin_handoff("trend")
    r._handoff_started = time.time() - 5000.0

    sink = capture(lambda: r._begin_handoff("trend"))

    line = lines(sink, "still waiting for flat")[0]
    assert "0s until" in line
    # Scoped to the countdown field only. A bare "-" test trips on "force-closed", and
    # a minus-before-digit test run over the WHOLE line trips on the log timestamp
    # (2026-08-19 contains "-0"). Both were tried; both were wrong.
    countdown = line.split("elapsed,")[1]
    assert re.search(r"-\d", countdown) is None, f"negative time remaining: {line}"


def test_the_first_call_still_says_starting_and_arms_the_clock():
    r, grid, trend, ex = make(position=6307.0, handoff_grace_seconds=21600)

    sink = capture(lambda: r._begin_handoff("trend"))

    assert len(lines(sink, "starting (waiting for flat, grace 21600s)")) == 1
    assert lines(sink, "still waiting") == []
    assert r._handoff_target == "trend"
    assert r._handoff_started > 0
