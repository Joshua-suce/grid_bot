"""GRID_COUNT comes from config, and has to survive the restore. AUDIT #108.

AUDIT #76 established the rule: taking the rung count from the saved state made a config
change silently inert for as long as a state file existed, while MAX_POSITION_PCT -- read
from settings like everything else -- took effect immediately. Half a geometry change
applied is worse than none, because the cap moves while the ladder does not.

main.py duly constructs the engine with settings.grid_count and logs the change. Then
load_from_dict opened with

    self.grid_count = data["grid_count"]

and put the saved value straight back, one layer below the fix. Observed live on
2026-08-18 06:22, with GRID_COUNT=4 in .env:

    06:22:46  GRID COUNT CHANGED | saved state has 14 rungs, config says 4 —
              rebuilding the ladder at 4 across the saved bounds
    06:23:04  Grid range: [0.06921204 - 0.07118796] | levels: 14
    06:23:53  PLACED 14 initial grid orders

A 14-rung ladder commits 875 USDT a side at 5 USDT x 25x -- 89% of the position cap --
against the 250 the config asked for. 3.5x the intended one-sided inventory, with the
log stating the opposite.

The bounds are a different matter and still come from state: that is where the live
orders and the open position actually sit, and recomputing them would orphan the book.
"""

from unittest.mock import MagicMock

import pytest

from grid import GridEngine


def exchange():
    ex = MagicMock()
    ex.exchange.amount_to_precision.side_effect = lambda s, a: str(int(float(a)))
    ex.exchange.price_to_precision.side_effect = lambda s, p: f"{float(p):.8f}"
    ex.get_positions.return_value = []
    return ex


LOWER, UPPER = 0.0689520382, 0.0709279618


def saved_state(count=14):
    step = (UPPER - LOWER) / count
    return {
        "grid_lower": LOWER, "grid_upper": UPPER, "grid_count": count,
        "grid_spacing": step, "active": True,
        "levels": [{"price": round(LOWER + i * step, 8), "side": "buy",
                    "order_id": None, "filled": False, "quantity": 100.0}
                   for i in range(count)],
    }


def engine(grid_count):
    return GridEngine(exchange(), "DOGEUSDT", grid_lower=LOWER, grid_upper=UPPER,
                      grid_count=grid_count, capital_per_grid_pct=0.018,
                      stop_loss_pct=0.005, capital_per_grid_usdt=5.0, leverage=25)


# --- the config wins ------------------------------------------------------------------

def test_the_saved_count_does_not_override_the_configured_one():
    """The live failure, exactly: state saved at 14, config says 4."""
    g = engine(4)
    g.load_from_dict(saved_state(14), current_price=0.0699)

    assert g.grid_count == 4


def test_the_ladder_is_rebuilt_at_the_configured_count():
    """Holding the number while running the old ladder would be the same bug wearing
    a correct-looking attribute."""
    g = engine(4)
    g.load_from_dict(saved_state(14), current_price=0.0699)

    assert len(g.levels) == 4


def test_growing_the_count_works_too():
    g = engine(20)
    g.load_from_dict(saved_state(14), current_price=0.0699)

    assert g.grid_count == 20
    assert len(g.levels) == 20


def test_a_matching_count_restores_the_saved_levels_untouched():
    """When nothing changed, the restored ladder must be the one the live orders sit on
    -- rebuilding it would orphan the book."""
    state = saved_state(14)
    g = engine(14)
    g.load_from_dict(state, current_price=0.0699)

    assert g.grid_count == 14
    assert len(g.levels) == 14
    assert g.state_corrupted is False


# --- the bounds still come from state -------------------------------------------------

def test_the_bounds_are_restored_not_recomputed():
    """Where the live orders and the open position actually are."""
    g = engine(4)
    g.load_from_dict(saved_state(14), current_price=0.0699)

    assert g.grid_lower == pytest.approx(LOWER)
    assert g.grid_upper == pytest.approx(UPPER)


def test_the_rebuilt_ladder_stays_inside_the_saved_bounds():
    g = engine(4)
    g.load_from_dict(saved_state(14), current_price=0.0699)

    for level in g.levels:
        assert LOWER <= level.price <= UPPER


# --- a config change is not corruption ------------------------------------------------

def test_a_deliberate_count_change_is_not_flagged_corrupt():
    """state_corrupted makes main.py delete the state file, which throws away the PnL
    and risk history alongside a ladder that is doing what it was told."""
    g = engine(4)
    g.load_from_dict(saved_state(14), current_price=0.0699)

    assert g.state_corrupted is False


def test_a_genuinely_short_level_list_is_still_flagged_corrupt():
    """The check still has to catch real damage: the count agrees, the levels do not."""
    state = saved_state(14)
    state["levels"] = state["levels"][:5]
    g = engine(14)
    g.load_from_dict(state, current_price=0.0699)

    assert g.state_corrupted is True
    assert len(g.levels) == 14


# --- what it was costing --------------------------------------------------------------

def test_the_one_sided_commitment_follows_the_configured_count():
    """5 USDT x 25x per rung. At 14 rungs one side is 875 USDT against a ~987 cap; at 4
    it is 250. That gap is the whole point of the fix."""
    per_order = 5.0 * 25
    assert per_order * (14 / 2) == pytest.approx(875.0)
    assert per_order * (4 / 2) == pytest.approx(250.0)

    g = engine(4)
    g.load_from_dict(saved_state(14), current_price=0.0699)

    assert per_order * (g.grid_count / 2) == pytest.approx(250.0)


def test_a_state_file_without_a_count_does_not_crash():
    """Older state files predate the key."""
    state = saved_state(14)
    del state["grid_count"]
    g = engine(4)
    g.load_from_dict(state, current_price=0.0699)

    assert g.grid_count == 4


def test_the_restore_does_not_read_the_count_back_from_the_file():
    """Pinned at source: the assignment that caused this must not return."""
    from pathlib import Path

    import grid

    src = Path(grid.__file__).read_text(encoding="utf-8")
    at = src.index("    def load_from_dict")
    body = src[at:at + 3000]

    assert 'self.grid_count = data["grid_count"]' not in body
