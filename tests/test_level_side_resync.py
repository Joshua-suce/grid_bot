"""A level stranded on the wrong side of spot can never be placed. AUDIT #115.

reset_levels_to_pending already receives current_price and never used it to decide sides:
it cleared order ids, reverted replaced sells, and left level.side exactly as the state
file had it. The saved side was decided against whatever spot was when the ladder was
built, so once price drifts past a rung that rung is tagged for the wrong side of the
market -- and post-only rejects a sell below the bid or a buy above the ask with -2019,
every attempt, forever.

Observed 2026-08-18. The restored ladder's centre was 0.069940 and spot opened at 0.07027,
so 0.06994 was still tagged "sell" while sitting BELOW the market:

    17:43:10  PLACED 7 initial grid orders (1 failed, 0 awaiting counter)
    17:43:15  PLACED 0 initial grid orders (1 failed, 0 awaiting counter)

Same level, both times. That left three sell rungs above spot instead of four, and when
the innermost (0.07027) filled at 17:57 the book above price was 0.07060 and 0.07093 only.
Price then ran to 0.07051 -- nine ticks short -- and filled nothing for 2h28m.

WHY THIS IS SAFE HERE AND NOWHERE ELSE: main.py calls this only when the exchange reports
no position. Flipping a side while inventory is open would turn a reduce-only exit into an
order that ADDS exposure, which is why the reconcile path beside it does not do this.
test_the_reset_is_gated_on_a_flat_account pins that precondition at source.
"""

from unittest.mock import MagicMock

import pytest

from grid import GridEngine, GridLevel

SPOT = 0.07027
LADDER = [0.06895, 0.06928, 0.06945, 0.06961, 0.06994, 0.07027, 0.07060, 0.07093]


def engine(prices=None, sides=None):
    ex = MagicMock()
    ex.exchange.amount_to_precision.side_effect = lambda s, a: str(int(float(a)))
    ex.exchange.price_to_precision.side_effect = lambda s, p: f"{float(p):.5f}"
    g = GridEngine(ex, "DOGEUSDT", grid_lower=0.0689520382, grid_upper=0.0709279618,
                   grid_count=8, capital_per_grid_pct=0.018, stop_loss_pct=0.005,
                   capital_per_grid_usdt=5.0, leverage=25)
    prices = prices if prices is not None else LADDER
    # The sides as the 2026-08-18 file held them: decided against the ladder's centre of
    # 0.069940, not against spot.
    sides = sides if sides is not None else ["buy" if p < 0.069940 else "sell" for p in prices]
    levels = []
    for p, s in zip(prices, sides):
        lvl = GridLevel(price=p, side=s)
        lvl.status = "pending"
        if s == "buy":
            lvl.entry_price = p
        levels.append(lvl)
    g.levels = levels
    return g


def side_of(g, price):
    return next(l.side for l in g.levels if l.price == pytest.approx(price))


# --- the live failure -------------------------------------------------------------------

def test_the_sell_below_spot_becomes_a_buy():
    """0.06994 tagged sell with spot at 0.07027 — the level that failed twice."""
    g = engine()
    assert side_of(g, 0.06994) == "sell", "premise: the file really had it as a sell"

    g.reset_levels_to_pending(SPOT)

    assert side_of(g, 0.06994) == "buy"


def test_every_level_ends_up_on_the_side_spot_says():
    """The invariant. Anything else is an order the exchange will not accept."""
    g = engine()

    g.reset_levels_to_pending(SPOT)

    for l in g.levels:
        expected = "buy" if l.price < SPOT else "sell"
        assert l.side == expected, f"{l.price} tagged {l.side} with spot at {SPOT}"


def test_a_buy_stranded_above_spot_flips_too():
    """The mirror case: price falls through the ladder instead of rising through it.
    Sides decided against a centre at the top of the range leave every rung tagged buy,
    so with spot at 0.07000 the upper three are the unplaceable ones."""
    g = engine(sides=["buy"] * len(LADDER))

    g.reset_levels_to_pending(0.07000)

    assert side_of(g, 0.06961) == "buy"
    assert side_of(g, 0.07027) == "sell"
    assert side_of(g, 0.07093) == "sell"


def test_levels_already_correct_are_left_alone():
    """A ladder that agrees with spot must come back unchanged, or every restart
    reshuffles a book that was fine."""
    g = engine(sides=["buy" if p < SPOT else "sell" for p in LADDER])
    before = [(l.price, l.side) for l in g.levels]

    g.reset_levels_to_pending(SPOT)

    assert [(l.price, l.side) for l in g.levels] == before


def test_a_deformed_ladder_is_rebuilt_and_still_correctly_sided():
    """reset_levels_to_pending has a second route: a ladder with a hole around the price
    is rebuilt wholesale rather than traded (AUDIT #34/#36), which re-sides it via
    _initialize_dynamic instead of the loop above. A sparse two-rung ladder takes that
    route — 2.14% hole, 5.3x the spacing — and must land on the same invariant, or the
    fix only holds for ladders that happen to skip the rebuild."""
    g = engine(prices=[0.06900, 0.07100], sides=["buy", "buy"])

    g.reset_levels_to_pending(0.07000)

    assert len(g.levels) >= 2
    for l in g.levels:
        assert l.side == ("buy" if l.price < 0.07000 else "sell"), (
            f"{l.price} tagged {l.side} after the deformation rebuild")


# --- the flip must leave a coherent level ------------------------------------------------

def test_flipping_to_buy_sets_its_entry_price():
    """_initialize_uniform and _rebuild_levels both set entry_price on buy levels; a
    flipped level that skips it prices its cycle off 0.0."""
    g = engine()

    g.reset_levels_to_pending(SPOT)

    flipped = next(l for l in g.levels if l.price == pytest.approx(0.06994))
    assert flipped.entry_price == pytest.approx(0.06994)


def test_flipping_clears_a_stale_counter_obligation():
    """The live 0.06994 carried awaiting_side=buy awaiting_price=0.06961 from a cycle
    that the flat account proves is over. Carrying it onto a re-sided level would queue
    an exit for inventory that does not exist."""
    g = engine()
    lvl = next(l for l in g.levels if l.price == pytest.approx(0.06994))
    lvl.awaiting_side, lvl.awaiting_price = "buy", 0.06961

    g.reset_levels_to_pending(SPOT)

    assert lvl.awaiting_side is None
    assert lvl.awaiting_price is None


# --- degenerate input --------------------------------------------------------------------

def test_no_price_means_no_residing():
    """current_price is optional. Without it there is nothing to decide sides against,
    and guessing is worse than leaving them."""
    g = engine()
    before = [(l.price, l.side) for l in g.levels]

    g.reset_levels_to_pending(None)

    assert [(l.price, l.side) for l in g.levels] == before


def test_a_zero_price_is_not_treated_as_a_market():
    """A failed price read must not re-tag the whole ladder as sells."""
    g = engine()
    before = [(l.price, l.side) for l in g.levels]

    g.reset_levels_to_pending(0.0)

    assert [(l.price, l.side) for l in g.levels] == before


# --- the safety precondition -------------------------------------------------------------

def test_the_reset_is_gated_on_a_flat_account():
    """Re-siding is only safe with no position open: with inventory, flipping a level
    turns a reduce-only exit into an order that ADDS exposure. main.py must call this
    only under `if not has_exchange_positions:` -- with reconcile_positions run
    unconditionally BEFORE that gate, so a state file still claiming a position the
    exchange has already closed is cleared against the exchange's truth first."""
    from pathlib import Path

    import main

    src = Path(main.__file__).read_text(encoding="utf-8")
    reconciled = src.index("grid.reconcile_positions()")
    at = src.index("if not has_exchange_positions:")
    block = src[at:at + 900]

    assert reconciled < at, (
        "reconcile_positions was gated behind the positions check again — a stale "
        "state-file position then blocks place_initial_orders forever")
    assert "reset_levels_to_pending" in block, (
        "reset_levels_to_pending left the flat branch — re-siding beside held "
        "inventory can convert an exit into an entry")
    assert src.count("reset_levels_to_pending") == 1, (
        "a second call site appeared; each one must be checked for which side of "
        "the flat-account gate it sits on")
