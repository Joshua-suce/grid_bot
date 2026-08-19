"""The grid must know its position cap before it places anything. AUDIT #125.

set_position_limit computes _block_buys/_block_sells and the order-size taper. Its first
call lived inside the main loop, roughly 730 lines after startup placement, so every start
laid a full ladder with the cap unknown: neither side blocked, no taper, and any inventory
already open ignored.

ADAUSDT 2026-08-19 10:01, restarting with SHORT 4284 held:

    10:01:12-23  six buy rungs and three sell rungs placed
    10:01:24     LADDER FITS THE CAP | one side commits 750.00 of 982.39 (77%)
    ...          sells keep filling, short walks 4284 -> 5597 -> 6307
    14:56:27     POSITION LIMIT | short 6307.0 >= 5608.77 -- sell orders blocked

Real room was 982.39 minus ~749 held = 233 USDT, about 1.9 rungs. Six were placed.

That is the shape of the whole day, and it is what the operator meant by "this keeps
happening": every session ended holding a position, every restart adopted it and stacked
a fresh full ladder on top, and the short ratcheted 0 -> 4284 -> 6307 with no way back.

    08:07  shutdown  -> SHORT 4284 left open
    10:01  restart   -> adopts it, full ladder on top -> 6307
    15:25  crash     -> SHORT 6307 left open
    16:30  restart   -> adopts it, deadlocks, 0 fills in 3h13m
    19:44  shutdown  -> SHORT 6307 still open

Seeded, that restart tapers the sell side to roughly half size and blocks it near the cap
instead of overshooting, and AUDIT #122's stuck-ladder exit then has a blocked side to
react to. Together the position converges instead of ratcheting.
"""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import main
from main import seed_position_limit

# The live numbers, 2026-08-19 10:01.
EQUITY = 4911.95
PRICE = 0.1749
HELD_SHORT = 4284.0
CFG = SimpleNamespace(max_position_pct=0.20)
CAP_QTY = EQUITY * 0.20 / PRICE          # 5616.7 ADA


def fake_exchange(short=0.0, long=0.0, price=PRICE, equity=EQUITY, raises=False):
    ex = MagicMock()
    if raises:
        ex.get_positions.side_effect = RuntimeError("book unreadable")
    else:
        rows = []
        if short:
            rows.append({"side": "short", "contracts": short, "entryPrice": price})
        if long:
            rows.append({"side": "long", "contracts": long, "entryPrice": price})
        ex.get_positions.return_value = rows
    ex.get_price.return_value = price
    ex.get_total_equity.return_value = equity
    return ex


# --- the live restart ------------------------------------------------------------------

def test_an_inherited_short_tapers_the_sell_side_before_a_single_order():
    """4284 against a 5617 cap is 76% of it -- the taper should be well under full size."""
    ex = fake_exchange(short=HELD_SHORT)
    grid = MagicMock()

    seed_position_limit(ex, grid, "ADAUSDT", CFG)

    long_pos, short_pos, max_qty = grid.set_position_limit.call_args[0]
    assert short_pos == pytest.approx(HELD_SHORT)
    assert long_pos == 0
    assert max_qty == pytest.approx(CAP_QTY, rel=1e-6)


def test_the_taper_a_real_engine_derives_from_it():
    """Not a mock: the real _position_limit_state, fed the live numbers. The opening
    ladder goes on at roughly half size instead of full."""
    from grid import GridEngine

    blocked, scale = GridEngine._position_limit_state(HELD_SHORT, CAP_QTY)

    assert blocked is False
    assert scale == pytest.approx(0.475, abs=0.01), "sell side should be tapered, not full"


def test_a_position_already_past_the_cap_blocks_outright():
    """The 16:30 restart: 6307 held against a cap of ~5434."""
    from grid import GridEngine

    blocked, scale = GridEngine._position_limit_state(6307.0, 5433.6)

    assert blocked is True
    assert scale == 0.0


def test_starting_flat_leaves_the_ladder_at_full_size():
    """The seed must not quietly shrink a normal start."""
    from grid import GridEngine

    blocked, scale = GridEngine._position_limit_state(0.0, CAP_QTY)

    assert blocked is False
    assert scale == 1.0


# --- it is safe -------------------------------------------------------------------------

def test_an_unreadable_book_refuses_to_seed_rather_than_claiming_flat():
    """get_position_breakdown swallows a failed read and returns (0.0, 0.0), which is
    indistinguishable from genuinely flat -- and seeding flat hands the ladder the whole
    cap, the exact assumption this function exists to remove. It must not guess.

    Startup still proceeds: the ladder is then sized as badly as it was before, which is
    the status quo, not a new risk. Aborting startup would be worse than the bug."""
    ex = fake_exchange(raises=True)
    grid = MagicMock()

    seed_position_limit(ex, grid, "ADAUSDT", CFG)          # must not raise

    assert not grid.set_position_limit.called, "seeded a cap from a book it could not read"


def test_a_zero_price_does_not_divide_by_zero():
    ex = fake_exchange(short=HELD_SHORT, price=0.0)
    grid = MagicMock()

    seed_position_limit(ex, grid, "ADAUSDT", CFG)

    _, _, max_qty = grid.set_position_limit.call_args[0]
    assert max_qty == 0.0


def test_it_says_so_when_it_seeds_against_real_inventory():
    from loguru import logger

    ex = fake_exchange(short=HELD_SHORT)
    grid = MagicMock()
    sink = []
    h = logger.add(lambda m: sink.append(str(m)), level="INFO")
    try:
        seed_position_limit(ex, grid, "ADAUSDT", CFG)
    finally:
        logger.remove(h)

    assert any("POSITION CAP SEEDED" in line for line in sink)


# --- ordering is the whole fix ------------------------------------------------------------

def _seed_call_sites(src: str) -> list[int]:
    """Offsets of CALLS to seed_position_limit, excluding its definition.

    The first version of these two tests matched the bare string
    "seed_position_limit(exchange, grid" -- which the `def` line contains. So index()
    returned the definition's offset, far above every placement call, and the ordering
    assertion below could not fail; and count() >= 2 counted the definition as one of the
    two, so deleting a real call still passed. A mutant that removed the restored-state
    seed survived both.

    Calls are indented; the definition is at column zero. That is the whole distinction.
    """
    return [m.start() for m in re.finditer(
        r"^[ 	]+seed_position_limit\(exchange, grid", src, re.M)]


def test_the_call_site_scan_excludes_the_definition():
    """A guard on the guard, because the first version of it did not."""
    src = Path(main.__file__).read_text(encoding="utf-8")
    definition = src.index("def seed_position_limit(")

    sites = _seed_call_sites(src)

    assert sites, "no call sites found -- the scan is broken"
    assert all(o > definition for o in sites), "the definition is being counted as a call"


def test_seeding_happens_before_every_placement_path():
    """A seed that runs after the ladder is on the book is worth nothing. This is the
    assertion that actually prevents the recurrence."""
    src = Path(main.__file__).read_text(encoding="utf-8")

    first_seed = _seed_call_sites(src)[0]

    for call in ("grid.reconcile_state()", "grid.place_initial_orders(", "grid.activate("):
        assert src.index(call) > first_seed, (
            f"{call} runs before the position cap is seeded -- the ladder is sized "
            f"against a cap it does not know yet (AUDIT #125)"
        )


def test_both_startup_routes_are_covered():
    """Restored-state and fresh-start reach placement by different paths; one seed on
    only one of them leaves the other exactly as it was."""
    src = Path(main.__file__).read_text(encoding="utf-8")

    assert len(_seed_call_sites(src)) >= 2, "only one startup route is seeded"
