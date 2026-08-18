"""Stop legs must clear the exchange's notional floor. AUDIT #89.

grid.py enforces MIN_NOTIONAL_USDT on every limit order it places -- three separate
checks -- because Binance rejects anything under 5 USDT with -4164. The protective
stops did not, and they are the orders that matter: a rejected limit order is a missed
fill, a rejected stop is an unprotected position.

The failure is self-sustaining, which is what makes it worth a module. Coverage comes
back short, _refresh_sl_stops returns False, the caller blocks new exposure, and the
grid can no longer trade the position down to nothing -- it waits, blocked, on a
position too small to protect. Observed 2026-08-16 20:52: 3 DOGE split into legs of
1.0 and 2.0, i.e. 0.07 and 0.14 USDT. Testnet took them; the live venue would not.
"""

from main import build_scale_out_orders

FLOOR = 5.0
TRAIL, HARD = 0.0678, 0.0667


def kinds(orders):
    return [o[0] for o in orders]


def total_qty(orders):
    return sum(o[1] for o in orders)


# --- the case that deadlocks ---------------------------------------------------------

def test_a_position_too_small_to_split_gets_one_full_stop():
    """100 DOGE at ~0.067 is 6.7 USDT whole and 3.4 per half. Split it and both halves
    are rejected; the position ends up with no stop at all and the grid blocks itself.

    This fixture used to say 3 DOGE, which was a poor exemplar of its own case: 0.20 USDT
    is too small to place ANY stop, so the full-size fallback it asserts here would have
    been rejected too. That is AUDIT #94 and it has its own module -- what belongs here
    is a position the fallback genuinely rescues."""
    orders = build_scale_out_orders("long", 100.0, 0.5, TRAIL, HARD, min_notional=FLOOR)

    assert kinds(orders) == ["hard"]
    assert total_qty(orders) == 100.0


def test_the_whole_position_is_still_covered_when_not_split():
    """Refusing to split must not mean refusing to protect. Full size at the hard
    level is the safer of the two available outcomes, not a reduction in coverage.

    Every size here clears the floor whole -- below FLOOR/HARD there is no stop to place
    and coverage is correctly zero (AUDIT #94)."""
    for qty in (80.0, 100.0, 140.0):
        assert qty * HARD > FLOOR, f"qty={qty} cannot be protected at all"
        orders = build_scale_out_orders("long", qty, 0.5, TRAIL, HARD, min_notional=FLOOR)
        assert total_qty(orders) == qty, f"lost coverage at qty={qty}"


def test_a_single_full_stop_can_clear_the_floor_where_two_halves_cannot():
    """This is the whole point: 100 DOGE is 6.7 USDT whole and 3.3 USDT per half."""
    qty = 100.0
    assert qty * HARD > FLOOR
    assert (qty * 0.5) * HARD < FLOOR

    orders = build_scale_out_orders("long", qty, 0.5, TRAIL, HARD, min_notional=FLOOR)
    assert kinds(orders) == ["hard"]
    assert all(q * p >= FLOOR for _, q, p in orders)


def test_an_undersized_trail_leg_refuses_the_split():
    """Asymmetric scale-out, small side on the trail leg."""
    orders = build_scale_out_orders("long", 200.0, 0.05, TRAIL, HARD, min_notional=FLOOR)

    assert (200.0 * 0.05) * TRAIL < FLOOR
    assert (200.0 * 0.95) * HARD > FLOOR      # the other leg is fine on its own
    assert kinds(orders) == ["hard"]


def test_an_undersized_hard_leg_also_refuses_the_split():
    """The mirror case, and it needs its own test: a guard that only measured the
    trail leg passes every trail-side fixture while still placing a doomed hard leg.
    That mutant survived until this existed."""
    orders = build_scale_out_orders("long", 200.0, 0.95, TRAIL, HARD, min_notional=FLOOR)

    assert (200.0 * 0.95) * TRAIL > FLOOR     # trail leg is comfortably fine
    assert (200.0 * 0.05) * HARD < FLOOR      # hard leg is not
    assert kinds(orders) == ["hard"]
    assert total_qty(orders) == 200.0


# --- and must not disturb the normal path -------------------------------------------

def test_a_position_that_splits_cleanly_still_splits():
    orders = build_scale_out_orders("long", 8000.0, 0.5, TRAIL, HARD, min_notional=FLOOR)

    assert kinds(orders) == ["trail", "hard"]
    assert total_qty(orders) == 8000.0
    assert all(q * p >= FLOOR for _, q, p in orders)


def test_the_split_boundary_lands_where_the_arithmetic_says():
    """Just above the threshold splits, just below does not. A guard that fired at the
    wrong size would either deadlock or place rejects, with no visible difference."""
    below = FLOOR / min(TRAIL, HARD) / 0.5 * 0.98
    above = FLOOR / min(TRAIL, HARD) / 0.5 * 1.02

    assert kinds(build_scale_out_orders("long", below, 0.5, TRAIL, HARD, min_notional=FLOOR)) == ["hard"]
    assert kinds(build_scale_out_orders("long", above, 0.5, TRAIL, HARD, min_notional=FLOOR)) == ["trail", "hard"]


def test_the_floor_is_opt_in():
    """Default 0.0 keeps every existing caller and test on the old path."""
    orders = build_scale_out_orders("long", 3.0, 0.5, TRAIL, HARD)
    assert kinds(orders) == ["trail", "hard"]


def test_a_short_position_is_handled_the_same_way():
    orders = build_scale_out_orders("short", 100.0, 0.5, TRAIL, HARD, min_notional=FLOOR)
    assert kinds(orders) == ["hard"]
    assert total_qty(orders) == 100.0


# --- interactions with the existing branches -----------------------------------------

def test_an_already_fired_scale_out_still_returns_one_hard_stop():
    orders = build_scale_out_orders("long", 8000.0, 0.5, TRAIL, HARD,
                                    scale_out_done=True, min_notional=FLOOR)
    assert kinds(orders) == ["hard"]
    assert total_qty(orders) == 8000.0


def test_no_stop_available_still_returns_nothing():
    """The #31 branch runs before the notional check and must keep doing so -- there
    is no price to measure notional against."""
    assert build_scale_out_orders("long", 8000.0, 0.5, None, HARD, min_notional=FLOOR) == []
    assert build_scale_out_orders("long", 8000.0, 0.5, TRAIL, None, min_notional=FLOOR) == []


def test_rounding_is_applied_before_the_notional_test():
    """amount_to_precision rounds a leg DOWN, and the rounded size is what the exchange
    sees. Sizes here are chosen so the unrounded halves clear the floor and the rounded
    ones do not -- the only shape that tells the two orderings apart, and the reason
    the earlier fixture let a check-before-rounding mutant live.
    """
    step = lambda q: float(int(q // 10) * 10)          # coarse, always rounds down
    orders = build_scale_out_orders("long", 158.0, 0.5, TRAIL, HARD,
                                    rounder=step, min_notional=FLOOR)

    assert (150.0 * 0.5) * TRAIL > FLOOR, "unrounded half clears the floor"
    assert step(150.0 * 0.5) * TRAIL < FLOOR, "rounded half does not"
    assert kinds(orders) == ["hard"], "measured the unrounded size"
    assert total_qty(orders) == 150.0


# --- disabling the split entirely -------------------------------------------------------

def test_a_zero_scale_out_gives_one_full_size_hard_stop():
    """SL_SCALE_OUT_PCT=0 means "no trailing leg, one hard stop", and it is the only safe
    setting when STOP_LOSS_PCT is tight.

    The trailing leg is anchored at peak*(1-STOP_LOSS_PCT). At the old 3% that sits well
    under the ladder; at 0.5% it sits 0.5% under the peak, which is two grid steps and
    INSIDE the grid -- it would fire on ordinary movement and force taker exits all day.

    build_scale_out_orders has always handled scale=0; the config field was bounded gt=0,
    so the one configuration that makes a tight stop safe could not be expressed.

    The rounder is NOT optional here. Passing none made this test pass against code that
    failed in production for 21 minutes: the live rounder is ccxt's amount_to_precision,
    which raises on a zero amount instead of returning it, and scale=0 computed exactly
    that (AUDIT #109). See tests/test_stop_rounder_zero.py."""
    from tests.test_stop_rounder_zero import ccxt_like

    orders = build_scale_out_orders("long", 1777.0, 0.0, TRAIL, HARD, rounder=ccxt_like,
                                    min_notional=FLOOR)

    assert kinds(orders) == ["hard"]
    assert orders[0][1] == 1777.0
    assert total_qty(orders) == 1777.0


def test_the_config_permits_a_zero_scale_out():
    """A pydantic bound of gt=0 would reject it before the engine ever saw it."""
    from config import Settings

    field = Settings.model_fields["sl_scale_out_pct"]
    lows = [m for m in field.metadata if hasattr(m, "ge") or hasattr(m, "gt")]
    assert any(getattr(m, "ge", None) == 0 for m in lows), (
        f"sl_scale_out_pct still cannot be set to 0: {lows}")
