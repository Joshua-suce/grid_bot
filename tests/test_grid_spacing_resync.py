"""grid_spacing must describe the ladder that exists, not the one that used to. AUDIT #114.

load_from_dict restores grid_spacing verbatim (grid.py) while the levels beside it are
deduped, refilled and re-sided. Nothing reconciled the two: _rebuild_levels recomputes it
but only fires when the level count disagrees with config, and _refill_missing_grid_lines
deliberately inserts into the ladder's ACTUAL gaps without touching it. A state file that
survives a GRID_COUNT change therefore carries the OLD count's spacing forever.

The live file on 2026-08-18 held 0.0006586412 == (upper-lower)/3, which is exactly what
_rebuild_levels writes for grid_count=4 -- against a restored 8-rung ladder whose real
mean gap was 0.0002828571. The four rungs that ladder was built from were all still in
it: 0.06895, 0.06961, 0.07027, 0.07093.

The cost is not cosmetic. Replacements are posted at level.price +/- grid_spacing, so the
exit for the 0.07027 fill went to 0.06961 -- two rungs down, 0.94% away -- rather than the
adjacent 0.06994. Price ranged 0.07022-0.07051 for the next 2h28m and touched nothing:

    17:57:20  FILL #65 | SELL @ 0.07027
    17:57:20  REPLACEMENT SLOT TAKEN | BUY @ 0.06961 is ... this fill's exit
    20:25:09  Shutting down...          (zero fills in between)
"""

from unittest.mock import MagicMock

import pytest

from grid import GridEngine

# The ladder as it actually sat in state/grid_dogeusdt_demo.json at 20:25:14.
LIVE_LOWER = 0.0689520382
LIVE_UPPER = 0.07092796180000001
LIVE_STALE_SPACING = 0.0006586412        # == (upper-lower)/3, the grid_count=4 value
LIVE_PRICES = [(0.06895, "buy"), (0.06928, "buy"), (0.06945, "buy"), (0.06961, "buy"),
               (0.06994, "sell"), (0.07027, "sell"), (0.07060, "sell"), (0.07093, "sell")]
LIVE_MEAN_GAP = (0.07093 - 0.06895) / 7  # 0.0002828571


def level(price, side):
    return {"price": price, "side": side, "order_id": None, "status": "pending",
            "fill_count": 0, "total_pnl": 0.0, "quantity": 1800.0,
            "entry_price": price if side == "buy" else 0.0,
            "awaiting_side": None, "awaiting_price": None}


def state(prices=None, spacing=LIVE_STALE_SPACING, lower=LIVE_LOWER, upper=LIVE_UPPER):
    return {"grid_lower": lower, "grid_upper": upper, "grid_count": len(prices or LIVE_PRICES),
            "grid_spacing": spacing, "active": True,
            "levels": [level(p, s) for p, s in (prices or LIVE_PRICES)]}


def engine(count=8, lower=LIVE_LOWER, upper=LIVE_UPPER):
    ex = MagicMock()
    ex.exchange.amount_to_precision.side_effect = lambda s, a: str(int(float(a)))
    ex.exchange.price_to_precision.side_effect = lambda s, p: f"{float(p):.5f}"
    return GridEngine(ex, "DOGEUSDT", grid_lower=lower, grid_upper=upper,
                      grid_count=count, capital_per_grid_pct=0.018,
                      stop_loss_pct=0.005, capital_per_grid_usdt=5.0, leverage=25)


# --- the live failure ------------------------------------------------------------------

def test_the_stale_spacing_is_replaced_by_the_measured_one():
    """The exact 2026-08-18 file."""
    g = engine()

    g.load_from_dict(state(), 0.07027)

    assert g.grid_spacing == pytest.approx(LIVE_MEAN_GAP, rel=1e-6)


def test_the_restored_value_really_was_off_by_that_much():
    """Guards the premise. If these two ever coincide the test above proves nothing."""
    assert LIVE_STALE_SPACING / LIVE_MEAN_GAP == pytest.approx(2.33, abs=0.01)
    assert LIVE_STALE_SPACING == pytest.approx((LIVE_UPPER - LIVE_LOWER) / 3, rel=1e-9)


def test_the_replacement_now_lands_nearer_the_adjacent_rung():
    """The consequence that cost 2h28m. A fill at 0.07027 posts its exit at
    price - grid_spacing; with the stale value that was 0.06961, two rungs down.

    Note it still does not land exactly on 0.06994: the ladder is non-uniform, so a mean
    cannot index it. Replacements are computed from grid_spacing rather than snapped to
    the neighbouring level, which is a separate design question -- this only removes the
    2.33x error."""
    g = engine()
    g.load_from_dict(state(), 0.07027)

    exit_price = 0.07027 - g.grid_spacing

    assert abs(exit_price - 0.06994) < abs(exit_price - 0.06961)
    assert exit_price > 0.06961, "still skipping the adjacent rung entirely"


def test_the_stale_value_would_have_skipped_a_rung():
    """The same arithmetic before the fix, so the regression is pinned from both ends."""
    assert 0.07027 - LIVE_STALE_SPACING == pytest.approx(0.06961, abs=1e-5)


# --- it is the mean, because the ladder is not uniform ----------------------------------

def test_spacing_is_the_mean_gap_not_the_smallest():
    """_initialize_dynamic concentrates rungs near price on purpose, so the ladder has
    both 0.00016 and 0.00033 gaps. min() would misreport it as half its real width."""
    g = engine()
    g.load_from_dict(state(), 0.07027)

    gaps = [round(b - a, 8) for (a, _), (b, _) in zip(LIVE_PRICES, LIVE_PRICES[1:])]

    assert min(gaps) < g.grid_spacing < max(gaps)


def test_a_uniform_ladder_measures_its_own_step():
    prices = [(round(0.069 + i * 0.0004, 5), "buy" if i < 3 else "sell") for i in range(6)]
    g = engine(count=6)

    g.load_from_dict(state(prices=prices, spacing=0.09), 0.0701)

    assert g.grid_spacing == pytest.approx(0.0004, rel=1e-6)


# --- the healthy case must not move -----------------------------------------------------

def test_a_correct_spacing_survives_untouched():
    """A file whose scalar already agrees must come back bit-identical, or every restart
    quietly perturbs a ladder that was fine."""
    g = engine()

    g.load_from_dict(state(spacing=LIVE_MEAN_GAP), 0.07027)

    assert g.grid_spacing == pytest.approx(LIVE_MEAN_GAP, rel=1e-9)


def test_the_rebuild_path_still_sets_its_own_spacing():
    """When the level count disagrees with config, _rebuild_levels recomputes both the
    levels and the spacing. The resync must not fight it."""
    g = engine(count=4)

    g.load_from_dict(state(), 0.07027)     # 8 saved levels, config says 4

    assert len(g.levels) == 4
    assert g.grid_spacing == pytest.approx((LIVE_UPPER - LIVE_LOWER) / 3, rel=1e-6)


# --- degenerate input --------------------------------------------------------------------

def test_a_single_level_leaves_the_spacing_alone():
    """No gap to measure. Dividing by len(levels)-1 would raise."""
    g = engine(count=1)

    g.load_from_dict(state(prices=[(0.07027, "sell")], spacing=0.0004), 0.07027)

    assert g.grid_spacing > 0


def test_an_empty_ladder_does_not_crash():
    g = engine(count=8)
    g.load_from_dict(state(prices=[], spacing=0.0004), 0.07027)

    assert g.grid_spacing > 0


# --- and the file heals itself -----------------------------------------------------------

def test_the_corrected_value_is_what_gets_saved():
    """Otherwise the stale scalar is rewritten on every shutdown and the next start
    inherits it again."""
    g = engine()
    g.load_from_dict(state(), 0.07027)

    assert g.to_dict()["grid_spacing"] == pytest.approx(LIVE_MEAN_GAP, rel=1e-6)


def test_the_invariant_holds_after_any_restore():
    """The property the whole module exists to protect: whatever path load_from_dict
    takes, grid_spacing describes the levels it left behind."""
    for count, price in ((8, 0.07027), (4, 0.07027), (8, 0.06900), (6, 0.07100)):
        g = engine(count=count)
        g.load_from_dict(state(), price)

        gaps = [g.levels[i + 1].price - g.levels[i].price for i in range(len(g.levels) - 1)]
        if len(gaps) < 1:
            continue
        assert g.grid_spacing == pytest.approx(sum(gaps) / len(gaps), rel=0.02), (
            f"count={count} price={price} left spacing {g.grid_spacing} against "
            f"{len(g.levels)} levels averaging {sum(gaps)/len(gaps)}")
