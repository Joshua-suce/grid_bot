"""The recurring `1 failed` at startup. AUDIT #118.

AUDIT #115 re-sided levels in reset_levels_to_pending, which main.py calls ONLY when the
exchange reports no position. With a position open reconcile_state runs instead, and it
reconciles order ids and re-places orphans without ever re-deriving level.side. The same
defect therefore survived on the holding path, which is the path every restart with an
open position takes.

The 2026-08-19 state file, saved with SHORT 1787 open:

    0.06994  sell  awaiting_counter      <- holds the short, exit queued as a buy
    0.0703   sell  pending
    0.0706   sell  pending
    0.07093  sell  pending

Start with spot at 0.07004 and 0.06994 is a "sell" below the market. post-only rejects it
with -2019, every attempt, and startup reports:

    00:19:39  PLACED 2 initial grid orders (1 failed, 0 awaiting counter)

Two things were wrong. The rung was never re-sided, and a rung the exchange physically
cannot accept was counted as a failure -- so a structural, self-clearing state read as a
broken order path on every single start.

THE ASYMMETRY that stopped this reusing the flat-path rule: with inventory open, a flip is
only safe in the direction that does not ADD exposure. Short: flip TO buy reduces (allow),
flip TO sell adds (refuse). Long: the mirror. Flat: nothing to add to, so anything goes.
"""

from unittest.mock import MagicMock

import pytest

from grid import GridEngine, GridLevel

SPOT = 0.07004
LOWER, UPPER = 0.0689520382, 0.0709279618

# The ladder exactly as state/grid_dogeusdt_demo.json held it at 03:00:20.
SAVED = [(0.06895, "buy"), (0.06928, "buy"), (0.06945, "buy"), (0.06961, "buy"),
         (0.06994, "sell"), (0.0703, "sell"), (0.0706, "sell"), (0.07093, "sell")]


def engine(levels=None, pos_qty=0.0, holding_level=None):
    ex = MagicMock()
    ex.exchange.amount_to_precision.side_effect = lambda s, a: str(int(float(a)))
    ex.exchange.price_to_precision.side_effect = lambda s, p: f"{float(p):.5f}"
    g = GridEngine(ex, "DOGEUSDT", grid_lower=LOWER, grid_upper=UPPER, grid_count=8,
                   capital_per_grid_pct=0.018, stop_loss_pct=0.005,
                   capital_per_grid_usdt=5.0, leverage=25)
    built = []
    for price, side in (levels if levels is not None else SAVED):
        lvl = GridLevel(price=price, side=side)
        lvl.status = "awaiting_counter" if price == holding_level else "pending"
        if price == holding_level:
            lvl.awaiting_side, lvl.awaiting_price = "buy", price - 0.0003
        built.append(lvl)
    g.levels = built
    g._pos_qty = pos_qty
    return g


def side_of(g, price):
    return next(l.side for l in g.levels if l.price == pytest.approx(price))


# --- the live case ------------------------------------------------------------------------

def test_a_sell_below_spot_is_resided_while_short():
    """0.06994 tagged sell with spot 0.07004 and SHORT 1787 open. A buy there reduces the
    short, so the flip is safe and the rung becomes placeable."""
    g = engine(pos_qty=-1787.0)

    assert g._reside_safe_levels(SPOT) >= 1
    assert side_of(g, 0.06994) == "buy"


def test_it_refuses_a_flip_that_would_add_to_the_short():
    """The asymmetry. A buy stranded ABOVE spot would have to become a sell, and another
    sell while already short adds exposure. Left alone deliberately."""
    g = engine(levels=[(0.07050, "buy"), (0.06900, "buy")], pos_qty=-1787.0)

    g._reside_safe_levels(SPOT)

    assert side_of(g, 0.07050) == "buy", "flipped to sell and added to the short"


def test_it_refuses_a_flip_that_would_add_to_the_long():
    g = engine(levels=[(0.06900, "sell"), (0.07050, "sell")], pos_qty=+1787.0)

    g._reside_safe_levels(SPOT)

    assert side_of(g, 0.06900) == "sell", "flipped to buy and added to the long"


def test_flat_allows_both_directions():
    """With no inventory no flip can add anything, so the safety test is vacuous."""
    g = engine(levels=[(0.07050, "buy"), (0.06900, "sell")], pos_qty=0.0)

    g._reside_safe_levels(SPOT)

    assert side_of(g, 0.07050) == "sell"
    assert side_of(g, 0.06900) == "buy"


def test_a_held_rung_is_never_resided():
    """awaiting_counter holds inventory whose exit is already queued. Re-siding it would
    strand that exit."""
    g = engine(pos_qty=-1787.0, holding_level=0.06994)

    g._reside_safe_levels(SPOT)

    held = next(l for l in g.levels if l.price == pytest.approx(0.06994))
    assert held.side == "sell"
    assert held.awaiting_side == "buy"


def test_a_resting_order_is_never_resided():
    """Relocating a live order needs a cancel; this runs where none is issued."""
    g = engine(levels=[(0.06994, "sell")], pos_qty=-1787.0)
    g.levels[0].order_id = "live"

    assert g._reside_safe_levels(SPOT) == 0
    assert side_of(g, 0.06994) == "sell"


def test_correct_levels_are_left_alone():
    g = engine(levels=[(0.06900, "buy"), (0.07050, "sell")], pos_qty=-1787.0)

    assert g._reside_safe_levels(SPOT) == 0


# --- what counts as unplaceable --------------------------------------------------------------

def test_a_sell_below_spot_is_wrong_side():
    g = engine()
    assert g._wrong_side_of(GridLevel(price=0.06994, side="sell"), SPOT) is True


def test_a_buy_above_spot_is_wrong_side():
    g = engine()
    assert g._wrong_side_of(GridLevel(price=0.07050, side="buy"), SPOT) is True


def test_correctly_sided_rungs_are_placeable():
    g = engine()
    assert g._wrong_side_of(GridLevel(price=0.06900, side="buy"), SPOT) is False
    assert g._wrong_side_of(GridLevel(price=0.07050, side="sell"), SPOT) is False


def test_an_unknown_price_blocks_nothing():
    """Without a price there is nothing to judge against, and refusing to place would be
    worse than trying."""
    g = engine()
    assert g._wrong_side_of(GridLevel(price=0.06994, side="sell"), None) is False
    assert g._wrong_side_of(GridLevel(price=0.06994, side="sell"), 0.0) is False


def test_no_price_means_no_residing():
    g = engine(pos_qty=-1787.0)
    assert g._reside_safe_levels(None) == 0
    assert g._reside_safe_levels(0.0) == 0


# --- and it is actually wired into the placement loop -------------------------------------------

def attempts(g, balance=4900.0):
    """Run place_initial_orders with the exchange calls stubbed, recording which rungs it
    actually tried to post."""
    tried = []
    g._current_price_or_none = lambda: SPOT
    g._release_awaiting_levels = lambda p: None
    g._repair_ladder = lambda p: None
    g._place_order_for_level = lambda lvl, bal: (tried.append(lvl.price), True)[1]
    g.place_initial_orders(balance)
    return tried


def test_a_resided_rung_is_attempted():
    """The whole point: 0.06994 was a sell under spot, becomes a buy, and gets posted
    instead of counted as a failure."""
    g = engine(levels=[(0.06994, "sell")], pos_qty=-1787.0)

    assert 0.06994 in attempts(g)


def test_a_stranded_rung_is_never_attempted():
    """The other half. 0.07050 is a buy above spot and cannot be flipped to sell without
    adding to the short, so post-only would reject it on every try. Skipping it is what
    takes the count to zero -- attempting it is what produced `1 failed` every start."""
    g = engine(levels=[(0.07050, "buy"), (0.06900, "buy")], pos_qty=-1787.0)

    tried = attempts(g)

    assert 0.06900 in tried
    assert 0.07050 not in tried


# --- the flip leaves a coherent level ---------------------------------------------------------

def test_flipping_to_buy_sets_entry_price():
    g = engine(levels=[(0.06994, "sell")], pos_qty=-1787.0)

    g._reside_safe_levels(SPOT)

    assert g.levels[0].entry_price == pytest.approx(0.06994)


def test_flipping_clears_a_stale_counter():
    g = engine(levels=[(0.06994, "sell")], pos_qty=-1787.0)
    g.levels[0].awaiting_side, g.levels[0].awaiting_price = "buy", 0.06961

    g._reside_safe_levels(SPOT)

    assert g.levels[0].awaiting_side is None
    assert g.levels[0].awaiting_price is None
