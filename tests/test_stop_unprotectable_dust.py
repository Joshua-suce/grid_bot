"""A position too small to protect must not also stop the bot. AUDIT #94.

AUDIT #89 stopped `build_scale_out_orders` from splitting a small position into two legs
the exchange rejects. It checked the two legs and never the total, so the same deadlock
simply moved one step down: a position too small to SPLIT got one full-size stop, and a
position too small to protect AT ALL got one full-size stop that is itself under the
floor.

Live on 2026-08-17. Fill #46 at 13:50:37 sold 1778 of a 1785 long and left 7 DOGE:

    13:50:45  STOP-LOSS NOT SPLIT | long 7.0 would give legs of 3.0/4.0 ...
    13:50:48  STOP-MARKET PLACED | SELL 7.0 DOGEUSDT @ stop=0.06713567705399999

7 x 0.06713568 is 0.47 USDT. Testnet accepted it. The live venue answers -4164, so
reconcile_stop_orders reports coverage short of desired, _refresh_sl_stops returns False,
and the caller blocks new exposure -- leaving the bot idle on a position worth less than
half a dollar, unable to trade out the very remainder that would clear the condition.
That is the self-sustaining failure #89 was written to prevent, one size class lower.

The fix asks for no stop at all in that case. `desired` empty is the one answer
_refresh_sl_stops already treats as covered, so the ladder keeps working; the exposure
given up is bounded by the exchange floor itself, and the alternative on the live venue
was never "protected" -- it was "blocked AND unprotected".
"""

import re
from pathlib import Path

import pytest

import main
from main import build_scale_out_orders

FLOOR = 5.0

# The 13:50 book, to the digit.
HARD = 0.06713567705399999
TRAIL = 0.0681619
DUST = 7.0


def kinds(orders):
    return [o[0] for o in orders]


def total_qty(orders):
    return sum(o[1] for o in orders)


@pytest.fixture(autouse=True)
def quiet_notes():
    """_note_stop_sizing keeps module state; each test starts from a clean slate."""
    main._clear_stop_sizing_note()
    yield
    main._clear_stop_sizing_note()


# --- the position that deadlocked ----------------------------------------------------

def test_the_live_dust_stop_was_under_the_exchange_floor():
    """The arithmetic the testnet hid. Stated first because every assertion below is
    only interesting if this order really would have been rejected."""
    assert DUST * HARD == pytest.approx(0.4699, abs=0.001)
    assert DUST * HARD < FLOOR


def test_a_position_too_small_for_any_stop_asks_for_none():
    orders = build_scale_out_orders("long", DUST, 0.5, TRAIL, HARD, min_notional=FLOOR)

    assert orders == [], "asked for a stop the exchange would reject with -4164"


def test_no_returned_stop_is_ever_under_the_floor():
    """The property that matters, swept across the whole size range rather than asserted
    at one point: whatever this function returns, the exchange will accept. A guard that
    fires at the wrong size shows up here and nowhere else."""
    for qty in range(1, 400):
        for scale in (0.05, 0.5, 0.95):
            orders = build_scale_out_orders("long", float(qty), scale, TRAIL, HARD,
                                            min_notional=FLOOR)
            for kind, q, price in orders:
                assert q * price >= FLOOR, (
                    f"qty={qty} scale={scale} -> {kind} leg of {q} @ {price} "
                    f"is {q * price:.3f} USDT, under the {FLOOR} floor"
                )


def test_coverage_is_all_or_nothing():
    """Either the whole position is covered or none of it is. A partial answer would be
    the worst of both: the caller sees coverage short of desired and blocks, which is
    the deadlock again."""
    for qty in range(1, 400):
        orders = build_scale_out_orders("long", float(qty), 0.5, TRAIL, HARD,
                                        min_notional=FLOOR)
        assert total_qty(orders) in (0.0, float(qty)), f"partial coverage at qty={qty}"


def test_a_short_dust_position_is_handled_the_same_way():
    short_hard, short_trail = 0.07332360065400001, 0.0726562

    assert build_scale_out_orders("short", DUST, 0.5, short_trail, short_hard,
                                  min_notional=FLOOR) == []


# --- the branches that could route around the check ----------------------------------

def test_an_already_fired_scale_out_does_not_bypass_the_check():
    """scale_out_done returns one full-size hard stop directly. Placing the dust check
    after that branch passes every fixture above while still emitting the rejected
    order on the exact path a scaled-out position takes."""
    assert build_scale_out_orders("long", DUST, 0.5, TRAIL, HARD,
                                  scale_out_done=True, min_notional=FLOOR) == []


def test_an_unarmed_trailing_level_does_not_bypass_the_check():
    """trail == hard with no startup anchor is the other early return of a full-size
    hard stop -- the shape a freshly opened position has before the trail ratchets."""
    assert build_scale_out_orders("long", DUST, 0.5, HARD, HARD,
                                  min_notional=FLOOR) == []


def test_dust_is_measured_after_rounding():
    """amount_to_precision rounds down, and the rounded size is what the exchange prices.
    75 DOGE clears the floor; the same order rounded down to 70 does not."""
    step = lambda q: float(int(q // 10) * 10)

    assert 75.0 * HARD > FLOOR, "unrounded size clears the floor"
    assert step(75.0) * HARD < FLOOR, "rounded size does not"
    assert build_scale_out_orders("long", 75.0, 0.5, TRAIL, HARD,
                                  rounder=step, min_notional=FLOOR) == []


def test_the_boundary_lands_where_the_arithmetic_says():
    just_under = (FLOOR / HARD) * 0.98
    just_over = (FLOOR / HARD) * 1.02

    assert build_scale_out_orders("long", just_under, 0.5, TRAIL, HARD,
                                  min_notional=FLOOR) == []
    assert kinds(build_scale_out_orders("long", just_over, 0.5, TRAIL, HARD,
                                        min_notional=FLOOR)) == ["hard"]


# --- and it must not touch positions that can be protected ---------------------------

def test_a_position_too_small_to_split_is_still_covered_whole():
    """AUDIT #89's case must survive intact: 100 DOGE is 6.7 USDT whole and 3.4 per
    half, so it gets one full-size stop -- not nothing."""
    orders = build_scale_out_orders("long", 100.0, 0.5, TRAIL, HARD, min_notional=FLOOR)

    assert kinds(orders) == ["hard"]
    assert total_qty(orders) == 100.0


def test_a_position_that_splits_cleanly_still_splits():
    orders = build_scale_out_orders("long", 8000.0, 0.5, TRAIL, HARD, min_notional=FLOOR)

    assert kinds(orders) == ["trail", "hard"]
    assert total_qty(orders) == 8000.0


def test_the_floor_is_still_opt_in():
    """Default 0.0 keeps every caller that does not pass a floor on the old path."""
    assert kinds(build_scale_out_orders("long", DUST, 0.5, TRAIL, HARD)) == ["trail", "hard"]


# --- the caller contract the fix depends on ------------------------------------------

def test_an_empty_desired_list_leaves_the_grid_unblocked():
    """Returning [] only helps because _refresh_sl_stops treats an empty `desired` as
    covered. That coupling is invisible from here and load-bearing: flip it to `return
    False` and dust deadlocks the grid again with every test above still green."""
    src = Path(main.__file__).read_text(encoding="utf-8")
    body = src[src.index("def _refresh_sl_stops("):]
    branch = re.search(r"if not desired:\n(.*?\n)*?\s*return (True|False)\n", body)

    assert branch is not None, "_refresh_sl_stops no longer short-circuits on empty desired"
    assert branch.group(2) == "True", (
        "_refresh_sl_stops reports an unprotectable position as uncovered — the caller "
        "blocks new exposure and the grid can never trade the dust away"
    )


# --- the log stops being a heartbeat --------------------------------------------------

def capture(level="DEBUG"):
    """Collect rendered records at `level` and above."""
    from loguru import logger

    seen = []
    sink_id = logger.add(lambda m: seen.append(m.record), level=level, format="{message}")
    return seen, sink_id


def test_a_persisting_condition_is_reported_once_not_every_poll():
    """13:50:45 to 14:06:21: 64 identical STOP-LOSS NOT SPLIT lines, one per poll, for
    one 7 DOGE position. That is how a genuine one-off gets buried."""
    from loguru import logger

    seen, sink_id = capture()
    try:
        for _ in range(64):
            build_scale_out_orders("long", DUST, 0.5, TRAIL, HARD, min_notional=FLOOR)
    finally:
        logger.remove(sink_id)

    loud = [r for r in seen if r["level"].name in ("INFO", "WARNING")]
    assert len(loud) == 1, f"{len(loud)} loud lines for one unchanging condition"
    assert "UNPROTECTABLE" in loud[0]["message"]
    assert len(seen) == 64, "the repeats were dropped rather than demoted"


def test_a_changed_condition_is_reported_again():
    """Quiet must mean 'nothing new', not 'stopped listening'."""
    from loguru import logger

    seen, sink_id = capture(level="INFO")
    try:
        build_scale_out_orders("long", DUST, 0.5, TRAIL, HARD, min_notional=FLOOR)
        build_scale_out_orders("long", 20.0, 0.5, TRAIL, HARD, min_notional=FLOOR)
    finally:
        logger.remove(sink_id)

    assert len(seen) == 2, "a different position size was swallowed as a repeat"


def test_a_normal_refresh_re_arms_the_note():
    """After the position recovers, the next dust event must be heard at full volume."""
    from loguru import logger

    build_scale_out_orders("long", DUST, 0.5, TRAIL, HARD, min_notional=FLOOR)
    build_scale_out_orders("long", 8000.0, 0.5, TRAIL, HARD, min_notional=FLOOR)

    seen, sink_id = capture(level="INFO")
    try:
        build_scale_out_orders("long", DUST, 0.5, TRAIL, HARD, min_notional=FLOOR)
    finally:
        logger.remove(sink_id)

    assert len(seen) == 1, "the note stayed suppressed across a healthy refresh"
