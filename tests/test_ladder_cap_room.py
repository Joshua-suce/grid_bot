"""The cap the ladder is measured against is the part that is still FREE. AUDIT #120.

#66 added the ladder-vs-cap preflight and compared one side of the ladder against the
whole position cap. That is right only from flat. A position already open eats the same
cap, so a restart that inherits inventory measures the ladder against room it does not
have.

ADAUSDT, 2026-08-19 10:01:24, restarting with SHORT 4284 ADA carried over:

    LADDER FITS THE CAP | one side commits 750.00 of 982.39 (77%), 1.9 rung(s) spare

The held short was ~749 USDT at 0.1749. 750 + 749 = 1499 against a 982 cap -- 53% OVER
before a single order was placed, reported as fitting with 1.9 rungs to spare.

What followed, in order:

    12:40-14:56  sells fill into a rising market, short walks 2143 -> 6307
    14:54:59     SELL SCALE | short=5597.0/5629.9 | scale=0.01   (32 units of headroom)
    14:56:17     FILL #28 | SELL @ 0.176 | qty=710               (a 710 sell was resting)
    14:56:27     POSITION LIMIT | short 6307.0 >= 5608.77 -- sell orders blocked
    15:06:28     KILL SWITCH: price 0.1776 above stop loss 0.17753
    16:30:22+    SKIP BUY @ 0.1752/0.1756/0.176/0.1765 | below break-even 0.17475982
    16:31:42     POSITION LIMIT | short 6307.0 >= 5433.6 -- sell orders blocked

Sells capped, buys under break-even: every rung blocked, an empty book, and three hours
of nothing while price ran 3.6% away. The preflight is the one place that could have said
so before the first order went out, and it said the opposite.

This does not PREVENT the deadlock -- it is a warning, deliberately (AUDIT #66: a restart
after a drawdown must not refuse to start). It makes the condition visible at 10:01
instead of inferrable from a three-hour silence.
"""

import pytest

from main import ladder_cap_room

# The live numbers, 2026-08-19 10:01:24.
ONE_SIDE = 750.00      # 6 rungs a side x 125 USDT notional
CAP = 982.39           # 20% of 4911.95 equity
HELD = 4284 * 0.1749   # SHORT 4284 ADA carried in from the previous session


def test_the_live_restart_did_not_fit():
    """The whole point. It reported fitting with 1.9 rungs spare."""
    fits, room = ladder_cap_room(ONE_SIDE, CAP, HELD)

    assert fits is False
    assert room == pytest.approx(233.4, abs=1.0)
    assert ONE_SIDE > room


def test_the_same_ladder_fits_from_flat():
    """Guards the premise: nothing is wrong with the ladder itself. It is the inherited
    position that makes it not fit, which is exactly the term #66 was missing."""
    fits, room = ladder_cap_room(ONE_SIDE, CAP, 0.0)

    assert fits is True
    assert room == pytest.approx(CAP)


def test_held_inventory_shrinks_the_room_one_for_one():
    for held in (0.0, 100.0, 500.0, 900.0):
        _, room = ladder_cap_room(ONE_SIDE, CAP, held)
        assert room == pytest.approx(CAP - held)


def test_the_boundary_is_exact_fit():
    fits, _ = ladder_cap_room(ONE_SIDE, CAP, CAP - ONE_SIDE)
    assert fits is True, "exactly filling the room is not outgrowing it"

    fits, _ = ladder_cap_room(ONE_SIDE, CAP, CAP - ONE_SIDE + 0.01)
    assert fits is False


def test_held_beyond_the_cap_leaves_negative_room():
    """6307 ADA at 0.1806 is 1139 USDT against a 982 cap -- the state the bot sat in for
    three hours. Room is negative and nothing fits, which is the truth."""
    fits, room = ladder_cap_room(ONE_SIDE, CAP, 6307 * 0.1806)

    assert fits is False
    assert room < 0


def test_an_unconfigured_cap_is_not_a_verdict():
    """max_position_pct of 0 means the cap is not set, not that nothing fits. Reporting
    'outgrows' there would fire the warning on every start for a config that never
    intended a cap."""
    fits, _ = ladder_cap_room(ONE_SIDE, 0.0, 0.0)
    assert fits is True

    fits, _ = ladder_cap_room(ONE_SIDE, -1.0, 0.0)
    assert fits is True


def test_flat_and_capped_still_catches_an_oversized_ladder():
    """#66's original case must keep working: no position, ladder simply too big."""
    fits, room = ladder_cap_room(1200.0, CAP, 0.0)

    assert fits is False
    assert room == pytest.approx(CAP)


# --- and it is actually wired into startup ------------------------------------------------

def test_the_startup_check_calls_it_with_the_held_position():
    """A helper nothing calls is not a guard. Pins that run_bot routes the preflight
    through it AND passes a held-notional term derived from the open position."""
    from pathlib import Path
    import main

    src = Path(main.__file__).read_text(encoding="utf-8")

    assert "ladder_cap_room(_one_side, _cap, _held)" in src, "preflight bypasses the helper"
    assert "_pos_qty" in src.split("ladder_cap_room(_one_side")[0][-400:], (
        "the held term is not derived from the open position"
    )
    assert "LADDER OUTGROWS THE CAP" in src, "the warning is gone"
