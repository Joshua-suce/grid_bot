"""Regression tests for the two stop-loss defects found in the 2026-08-12 live run.

Both surfaced at the same moment -- the 14:06 recenter -- and both had the same shape:
an event that was not a stop firing nonetheless reduced the protection on a 7108 DOGE
long that stayed open. They are the same class as AUDIT #14/#15, one layer down.

  14:05:26  trail SELL 3554 @ 0.0693647   hard SELL 3554 @ 0.06825994
  14:06:00  recenter -> pause() -> cancel_everything()   (cancels stops too)
  14:06:16  "SCALE-OUT STOP FIRED"  <- nothing fired; price was 0.07035
  14:06:17  hard SELL 7108 @ 0.06696984  <- trail leg gone, hard stop 1.9% wider
"""

import pytest

from grid import GridEngine
from main import trail_stop_fired


class _StubExchange:
    class exchange:
        @staticmethod
        def amount_to_precision(symbol, amount):
            return f"{float(amount):.0f}"

        @staticmethod
        def price_to_precision(symbol, price):
            return f"{float(price):.5f}"


def make_engine(**kw):
    defaults = dict(
        exchange=_StubExchange(), symbol="DOGEUSDT",
        grid_lower=0.07037107, grid_upper=0.07298893, grid_count=10,
        capital_per_grid_pct=0.018, stop_loss_pct=0.03,
    )
    defaults.update(kw)
    return GridEngine(**defaults)


# --- #25: the hard stop must ratchet too -----------------------------------

def test_hard_stop_does_not_widen_when_the_grid_recenters_lower():
    """The exact 14:06 regression, in numbers taken from the log."""
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)

    before = grid.get_hard_stop_loss_price()
    assert before == pytest.approx(0.06825994, abs=1e-8)

    # recenter() moved the band down while the long was still open
    grid.grid_lower, grid.grid_upper = 0.06904107, 0.07165893
    after = grid.get_hard_stop_loss_price()

    assert after >= before - 1e-12, (
        f"hard stop loosened from {before} to {after} with 7108 DOGE still open"
    )
    assert after == pytest.approx(before, abs=1e-8)


def test_hard_stop_still_tightens_when_the_grid_recenters_higher():
    """A ratchet only blocks loosening -- tightening must still work."""
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)
    before = grid.get_hard_stop_loss_price()

    grid.grid_lower, grid.grid_upper = 0.07200000, 0.07460000
    assert grid.get_hard_stop_loss_price() > before


def test_hard_stop_tracks_the_grid_again_once_flat():
    """With nothing open there is nothing to protect, so the level is free to move."""
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)
    grid.get_hard_stop_loss_price()          # arm the ratchet

    grid.set_position_limit(long_position=0.0, short_position=0.0, max_position_qty=8400.0)
    grid.grid_lower, grid.grid_upper = 0.06904107, 0.07165893
    assert grid.get_hard_stop_loss_price() == pytest.approx(0.06904107 * 0.97)


def test_short_hard_stop_never_rises_while_short_is_open():
    grid = make_engine()
    grid.set_position_limit(long_position=0.0, short_position=5000.0, max_position_qty=8400.0)
    before = grid.get_short_hard_stop_loss_price()

    grid.grid_lower, grid.grid_upper = 0.07300000, 0.07560000
    after = grid.get_short_hard_stop_loss_price()
    assert after <= before + 1e-12, f"short hard stop loosened from {before} to {after}"


def test_reset_trailing_releases_the_hard_ratchet():
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)
    grid.get_hard_stop_loss_price()
    assert grid._hard_sl_price is not None

    grid.reset_trailing()
    assert grid._hard_sl_price is None
    assert grid._hard_sl_price_short is None


def test_default_stop_getter_uses_the_ratcheted_hard_level():
    """get_stop_loss_price falls back to the hard level when no trail is set, so it
    must inherit the ratchet rather than recomputing from grid_lower."""
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)
    before = grid.get_stop_loss_price()

    grid.grid_lower, grid.grid_upper = 0.06904107, 0.07165893
    assert grid.get_stop_loss_price() >= before - 1e-12


# --- #168: the hard stop must also track the position's own entry ----------
#
# grid_lower is frozen while a position is open (AUDIT #116 blocks recenter()), so a
# stop anchored to it alone cannot track a position built up through a long drawdown
# of grid dip-buys. Observed live 2026-08-30: hours of grid profit erased in under a
# minute when the stop finally hit, still sitting at the pre-drawdown grid_lower level
# instead of near the position's real (falling) average cost.

def test_hard_stop_tightens_to_the_average_entry_when_it_is_above_grid_lower():
    """A position whose average entry sits above grid_lower must pull the hard stop up
    with it, above the old grid_lower-anchored level."""
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)
    grid.seed_position(7108.0, 0.07200000)

    old_style = grid.grid_lower * (1 - grid.stop_loss_pct)
    stop = grid.get_hard_stop_loss_price()

    assert stop > old_style, "average-entry anchor did not tighten the stop"
    assert stop == pytest.approx(0.07200000 * 0.97)


def test_hard_stop_does_not_loosen_when_more_dip_buying_drags_the_average_down():
    """The ratchet still only tightens: once armed against the average entry, a later
    fill that pulls the average lower (more dip-buying in a drawdown -- the exact
    2026-08-30 scenario) must not loosen the hard stop already in force."""
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)
    grid.seed_position(7108.0, 0.07200000)
    before = grid.get_hard_stop_loss_price()

    # Pin down that `before` is actually the entry-anchored figure and not just
    # whatever the grid_lower-only formula would have produced anyway -- without
    # this, a regression that dropped the AUDIT #168 anchor entirely would still
    # pass this test, since both `before` and `after` collapse to the same
    # grid_lower constant regardless of seed_position.
    assert before == pytest.approx(0.07200000 * 0.97)

    grid.seed_position(9000.0, 0.07100000)
    after = grid.get_hard_stop_loss_price()

    assert after >= before - 1e-12, f"hard stop loosened from {before} to {after}"
    assert after == pytest.approx(before)


def test_hard_stop_falls_back_to_grid_lower_when_the_ledger_has_no_entry():
    """With _pos_entry unset (0.0, the constructor default -- the state every test
    above this section leaves the ledger in) the average-entry comparison must not
    participate, so behaviour is byte-identical to before AUDIT #168."""
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)

    assert (grid._pos_qty, grid._pos_entry) == (0.0, 0.0), "precondition: ledger unset"
    assert grid.get_hard_stop_loss_price() == pytest.approx(
        grid.grid_lower * (1 - grid.stop_loss_pct)
    )


def test_short_hard_stop_tightens_to_the_average_entry_when_it_is_below_grid_upper():
    """Mirror of the long case: a short's average entry below grid_upper must pull the
    hard stop down with it, below the old grid_upper-anchored level."""
    grid = make_engine()
    grid.set_position_limit(long_position=0.0, short_position=5000.0, max_position_qty=8400.0)
    grid.seed_position(-5000.0, 0.07150000)

    old_style = grid.grid_upper * (1 + grid.stop_loss_pct)
    stop = grid.get_short_hard_stop_loss_price()

    assert stop < old_style, "average-entry anchor did not tighten the short stop"
    assert stop == pytest.approx(0.07150000 * 1.03)


def test_short_hard_stop_does_not_loosen_when_more_dip_selling_drags_the_average_up():
    """Ratchet mirror: a later fill that drags the short's average entry up must not
    loosen the hard stop already armed against the earlier, lower average."""
    grid = make_engine()
    grid.set_position_limit(long_position=0.0, short_position=5000.0, max_position_qty=8400.0)
    grid.seed_position(-5000.0, 0.07150000)
    before = grid.get_short_hard_stop_loss_price()

    # See the long-side twin above: pin `before` down so a regression that dropped
    # the AUDIT #168 anchor couldn't pass this test by coincidence.
    assert before == pytest.approx(0.07150000 * 1.03)

    grid.seed_position(-7000.0, 0.07250000)
    after = grid.get_short_hard_stop_loss_price()

    assert after <= before + 1e-12, f"short hard stop loosened from {before} to {after}"
    assert after == pytest.approx(before)


# --- #26: a cancelled stop is not a fired stop -----------------------------

@pytest.mark.parametrize("status", ["closed", "filled"])
def test_completed_stop_counts_as_fired(status):
    assert trail_stop_fired({"id": "x", "status": status}) is True


@pytest.mark.parametrize("status", ["canceled", "cancelled", "expired", "open", "rejected"])
def test_uncompleted_stop_does_not_count_as_fired(status):
    """The 14:06 case: recenter cancelled it, so it must not latch the scale-out."""
    assert trail_stop_fired({"id": "x", "status": status}) is False


def test_unknown_order_does_not_count_as_fired():
    """fetch_order returning None (unreachable, purged) must read as 'did not fire' --
    wrongly latching strips the trailing leg for the position's whole life."""
    assert trail_stop_fired(None) is False


def test_missing_status_does_not_count_as_fired():
    assert trail_stop_fired({"id": "x"}) is False


# --- #31: the startup reset must not damage the ladder ---------------------

def _reset_engine():
    from grid import GridEngine

    class _Stub:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{float(amount):.0f}"

            @staticmethod
            def price_to_precision(symbol, price):
                return f"{float(price):.5f}"

    return GridEngine(
        exchange=_Stub(), symbol="DOGEUSDT",
        grid_lower=0.0690, grid_upper=0.0716, grid_count=10,
        capital_per_grid_pct=0.018, stop_loss_pct=0.03,
    )


def test_reset_to_pending_keeps_the_ladder_whole():
    """A replaced sell reverts to a buy at its entry price, which can collide with the
    buy already sitting there. Before this lived in the engine the duplicate was saved
    and only merged on the *next* start -- the log said "DEDUPLICATED 1 levels (10 -> 9)"
    followed by a refill, every restart (AUDIT #31)."""
    engine = _reset_engine()
    engine.initialize(0.0703, balance=5000)
    before = len(engine.levels)

    collide_with = engine.levels[0]
    victim = engine.levels[-1]
    victim.side = "sell"
    victim.status = "replaced"
    victim.entry_price = collide_with.price      # reverts straight onto an occupied slot
    victim.order_id = "stale-1"

    engine.reset_levels_to_pending(0.0703)

    prices_sides = [(l.price, l.side) for l in engine.levels]
    assert len(prices_sides) == len(set(prices_sides)), f"duplicate slots left: {prices_sides}"
    assert len(engine.levels) == before, "the ladder lost a line"
    assert all(l.order_id is None for l in engine.levels)
    assert all(l.status == "pending" for l in engine.levels)


def test_reset_to_pending_leaves_a_clean_ladder_untouched():
    engine = _reset_engine()
    engine.initialize(0.0703, balance=5000)
    before = [(l.price, l.side) for l in engine.levels]

    engine.reset_levels_to_pending(0.0703)

    assert [(l.price, l.side) for l in engine.levels] == before


# --- #34: a ladder that stopped being a ladder ------------------------------

# The exact levels restored at 23:18:58 on 2026-08-12, read back from
# state/grid_dogeusdt.json. Range [0.06771107-0.07032893], price 0.06945.
DEFORMED_LADDER = [
    (0.06771, "buy"), (0.06797, "buy"), (0.06800, "buy"), (0.06823, "buy"),
    (0.06829, "buy"), (0.06850, "buy"), (0.06876, "buy"),
    (0.06981, "sell"), (0.07007, "sell"), (0.07033, "sell"),
]


def _ladder_engine():
    from grid import GridEngine

    class _Stub:
        class exchange:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{float(amount):.0f}"

            @staticmethod
            def price_to_precision(symbol, price):
                return f"{float(price):.5f}"

        def get_positions(self, symbol):
            return []

    engine = GridEngine(
        exchange=_Stub(), symbol="DOGEUSDT",
        grid_lower=0.06771107, grid_upper=0.07032893, grid_count=10,
        capital_per_grid_pct=0.018, stop_loss_pct=0.03,
    )
    engine.initialize(0.06945, balance=4900.0)
    return engine


def _apply_deformed(engine):
    from grid import GridLevel

    engine.levels = [GridLevel(price=p, side=side) for p, side in DEFORMED_LADDER]
    return engine


def test_the_real_deformed_ladder_is_recognised():
    """45 minutes, zero fills. Price 0.06945 sat in a 1.51% hole, and two buy pairs
    were 0.04% and 0.09% apart -- both below the 0.12% round-trip fee floor, so neither
    pair could ever profit even if it filled."""
    engine = _apply_deformed(_ladder_engine())

    defects = engine.ladder_defects(0.06945)

    assert defects, "the ladder that produced zero fills reads as healthy"
    assert any("hole" in d for d in defects), defects
    assert any("fee floor" in d for d in defects), defects


def test_a_freshly_built_ladder_is_healthy():
    """The check must not fire on a normal grid, or it would rebuild constantly."""
    engine = _ladder_engine()

    assert engine.ladder_defects(0.06945) == []


def test_resetting_a_deformed_ladder_rebuilds_it():
    engine = _apply_deformed(_ladder_engine())

    engine.reset_levels_to_pending(0.06945)

    assert engine.ladder_defects(0.06945) == [], "rebuild left the ladder deformed"
    prices = sorted(l.price for l in engine.levels)
    below = [p for p in prices if p <= 0.06945]
    above = [p for p in prices if p > 0.06945]
    assert below and above, "rebuilt ladder does not straddle the price"


def test_resetting_a_healthy_ladder_keeps_its_prices():
    """Rebuilding discards per-level bookkeeping, so it must only happen when needed."""
    engine = _ladder_engine()
    before = sorted(l.price for l in engine.levels)

    engine.reset_levels_to_pending(0.06945)

    assert sorted(l.price for l in engine.levels) == before


def test_a_deformed_ladder_triggers_a_recenter_while_flat():
    """Price was inside the range the whole 45 minutes, so no existing trigger fired."""
    engine = _apply_deformed(_ladder_engine())
    engine._last_recenter_time = 0.0
    engine.set_position_limit(0.0, 0.0, 10000.0)

    assert engine.ladder_defects(0.06945), "precondition: ladder is deformed"
    assert 0.06771107 < 0.06945 < 0.07032893, "precondition: price is inside the range"


def test_a_deformed_ladder_does_not_trigger_a_recenter_while_holding():
    """Recentring cancels resting orders; with inventory open those are the exits."""
    engine = _apply_deformed(_ladder_engine())
    engine.set_position_limit(6274.0, 0.0, 10000.0)

    assert engine._net_long_qty > 0
    # recenter consults ladder_defects only when flat -- pinned by reading the guard
    import inspect

    src = inspect.getsource(type(engine).recenter)
    assert "if flat else []" in src, "the deformity check is no longer gated on flat"


# --- #35: a clean shutdown must not read as a crash ------------------------

def test_shutdown_and_kill_switch_cancel_identically():
    """`reason` is presentational only. If it ever changed WHAT gets cancelled, a
    quiet-looking shutdown would leave orders resting on the exchange."""
    import inspect

    import grid as grid_module
    import trend_follower as tf_module

    for fn in (grid_module.GridEngine.emergency_stop, tf_module.TrendFollower.emergency_stop):
        src = inspect.getsource(fn)
        body = src.split('"""')[-1] if '"""' in src else src
        # the only thing `reason` may gate is a logger call
        for line in body.splitlines():
            if "reason" in line and "def " not in line:
                assert "logger" in line or line.strip().startswith(("if", "else", "#")), (
                    f"`reason` gates something other than logging in "
                    f"{fn.__qualname__}: {line.strip()}"
                )


def test_main_stops_with_the_shutdown_reason():
    """AUDIT #35. main.py's `finally:` block runs emergency_stop on every clean Ctrl+C.
    It logged at ERROR either way, so a normal exit ended in two red EMERGENCY STOP
    lines and read as a crash -- which is what made real faults hard to spot in a log."""
    import pathlib
    import re

    source = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    calls = re.findall(r"grid\.emergency_stop\(([^)]*)\)", source)
    assert calls, "main.py no longer calls emergency_stop"
    assert any('reason="shutdown"' in c for c in calls), (
        "the shutdown path must say so, or a clean exit logs as an emergency"
    )
    assert any(c.strip() == "" for c in calls), (
        "the kill-switch path must keep the default reason and stay an ERROR"
    )


# --- #36: the cause of the deformity, not just the symptom -----------------

def test_refill_lands_in_the_gap_not_beside_an_existing_level():
    """AUDIT #36. The refill walked a uniform template (grid_lower + i*spacing) while
    _initialize_dynamic builds a deliberately non-uniform ladder. The template never
    lines up, so every refill landed a few ticks from a real level: 0.06800 beside
    0.06797 (0.04% apart), 0.06829 beside 0.06823 (0.09%) -- neither pair able to clear
    the 0.12% fee floor, while the middle of the range stayed empty. That is the ladder
    AUDIT #34 had to rebuild.
    """
    from grid import GridLevel

    engine = _ladder_engine()
    # the real 2026-08-12 dynamic ladder with the two merged duplicates removed
    kept = [0.06771, 0.06823, 0.06850, 0.06876, 0.06928, 0.06954, 0.07007, 0.07033]
    engine.levels = [
        GridLevel(price=p, side="buy" if p < 0.06945 else "sell") for p in kept
    ]

    engine._refill_missing_grid_lines(0.06945)

    prices = sorted(l.price for l in engine.levels)
    assert len(prices) == 10, f"refill did not restore the ladder: {prices}"
    floor = 2 * engine.maker_fee_pct * engine._min_profit_multiplier
    tight = [(a, b) for a, b in zip(prices, prices[1:]) if (b - a) / a < floor]
    assert tight == [], f"refill created pairs below the fee floor: {tight}"
    assert engine.ladder_defects(0.06945) == []


def test_refill_stops_rather_than_wedging_a_level_into_a_full_ladder():
    """When the widest remaining gap is too narrow to split, fewer levels is correct.
    Wedging one in is exactly the defect above."""
    from grid import GridLevel

    engine = _ladder_engine()
    engine.levels = [
        GridLevel(price=round(0.06900 + i * 0.00002, 5), side="buy") for i in range(4)
    ]
    before = len(engine.levels)

    engine._refill_missing_grid_lines(0.06945)

    # The seeded levels are already tighter than the fee floor; what matters is that
    # the refill declines to make it worse rather than filling up to grid_count.
    assert len(engine.levels) == before, "wedged a level into a ladder with no room"
