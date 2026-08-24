"""The handoff grace deadline is wall-clock absolute, so it must survive a restart.

load_from_dict restored `handoff_target` but re-stamped `_handoff_started = time.time()`,
so every restart pushed the force-close deadline out by the full grace period. A handoff
waiting on a position that can never reach flat could therefore be deferred forever by
anything that merely restarts the bot.

That was latent before. It is not now: AUDIT #127 made the bot restart ITSELF after 45
dormant minutes, and dormancy is precisely the state a stuck handoff produces. The
mechanism added to break the deadlock would have kept resetting the deadline meant to
break it. AUDIT #129.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from router import StrategyRouter
from strategy import Strategy

GRACE = 1800.0


def make(handoff_grace_seconds=GRACE):
    grid = MagicMock(spec=Strategy); grid.name = "grid"; grid.active = True; grid.levels = []
    trend = MagicMock(spec=Strategy); trend.name = "trend"; trend.active = False; trend.levels = []
    ex = MagicMock()
    ex.get_positions.return_value = [{"side": "short", "contracts": 6307.0}]
    ex.get_price.return_value = 0.1806
    return StrategyRouter(
        strategies={"grid": grid, "trend": trend}, min_regime_seconds=0,
        handoff_grace_seconds=handoff_grace_seconds, exchange=ex, symbol="ADAUSDT",
    )


def _restore(saved_started, grace=GRACE):
    r = make(grace)
    r.load_from_dict({"router": {
        "active_name": "grid", "regime": "uptrend",
        "handoff_target": "trend", "handoff_started": saved_started,
    }}, current_price=0.1806)
    return r


def test_the_started_stamp_is_persisted_at_all():
    r = make()
    r._handoff_target = "trend"
    r._handoff_started = 12345.0
    assert r.to_dict()["router"]["handoff_started"] == 12345.0


def test_a_restart_does_not_restart_the_countdown():
    """The whole defect in one assertion."""
    started = time.time() - 1200.0          # 20 minutes already elapsed
    r = _restore(started)
    assert r._handoff_started == started, "the restart reset the grace clock"


def test_elapsed_time_survives_the_round_trip():
    started = time.time() - 900.0
    r = make()
    r._handoff_target = "trend"
    r._handoff_started = started

    revived = make()
    revived.load_from_dict(r.to_dict(), current_price=0.1806)

    assert abs((time.time() - revived._handoff_started) - 900.0) < 5.0


def test_a_deadline_already_passed_is_not_pushed_back_out():
    """A handoff that has outlived its grace must force-close on the next tick, not
    be granted a fresh window by the restart that found it."""
    started = time.time() - (GRACE + 600.0)
    r = _restore(started)
    assert time.time() - r._handoff_started >= GRACE, (
        "an expired grace was renewed by the restart"
    )


def test_no_handoff_in_flight_leaves_the_clock_alone():
    r = make()
    r.load_from_dict({"router": {"active_name": "grid", "handoff_target": None}},
                     current_price=0.1806)
    assert r._handoff_target is None
    assert r._handoff_started == 0.0


# --- corrupt or hand-edited files ------------------------------------------------------
def test_a_missing_stamp_falls_back_to_now():
    """Old state files predate the field. Starting the clock now is the safe read:
    it defers the force-close, it does not trigger one on a number we do not have."""
    before = time.time()
    r = _restore(None)
    assert before <= r._handoff_started <= time.time() + 1.0


def test_a_stamp_from_the_future_is_rejected():
    r = _restore(time.time() + 99999.0)
    assert r._handoff_started <= time.time() + 1.0


def test_an_absurdly_old_stamp_does_not_force_an_instant_close():
    """A 1970 timestamp from a truncated or hand-edited file reads as 'expired by 56
    years' and would market-close the position on the first tick after startup.
    Refuse the number rather than act on it.

    Note the boundary this must NOT cross: an ORDINARY expired deadline has to be
    honoured (see the test above), so the rule rejects absurdity, not expiry."""
    for absurd in (0.0, 1000.0, -5.0):
        r = _restore(absurd)
        assert r._handoff_started > time.time() - GRACE, f"{absurd} was acted on"


def test_a_non_numeric_stamp_does_not_raise():
    r = _restore("not-a-number")
    assert isinstance(r._handoff_started, float)
    assert r._handoff_started > 0.0


def test_the_clock_is_still_reset_when_a_handoff_is_cancelled():
    """Persistence must not make a finished handoff sticky."""
    r = _restore(time.time() - 600.0)
    r._handoff_target = None
    r._handoff_started = 0.0
    assert r.to_dict()["router"]["handoff_started"] == 0.0
