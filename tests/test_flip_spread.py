"""A flipped rung must capture a spread, not just avoid a loss. AUDIT #119.

A grid earns one spacing per cycle, and it earns it because the counter sits a spacing
away from the fill. _release_awaiting_levels has a third way out of the hold: when the
counter slot is busy and price has moved a full spacing clear, the rung comes back on its
OWN line, flipped to whichever side can rest there. That flip exists for a real reason --
without it a sell that price has climbed past can never re-arm, and the book drains
(AUDIT #77).

But on its own line the two legs of the cycle are the same price. The rung closes
inventory for whatever the average entry has drifted, and pays a full round trip of fees
to do it.

Measured on ADAUSDT, 2026-08-19 04:52-08:07:

    05:03:58  FILL #1 | SELL @ 0.1743 | profit=0.000000 fees=0.024995
    05:10:17  RUNG FLIPPED | SELL 0.1743 -> BUY
    05:10:18  ORDER PLACED | BUY 717.0 ADAUSDT @ 0.1743

Sold at 0.1743, bought back at 0.1743. Four flips like it in three hours. Session totals:
8 fills, 7 of them profit=0.000000, gross 0.05 against fees 0.20, net -0.15. The single
cycle that earned anything made 0.053 -- the drift between the average short entry
0.174874 and the close 0.1748, not a rung -- against 0.05 of fees. Net 0.003, where the
0.2507% spacing should have paid 0.263.

_books_a_loss was already in the path and did not stop any of it: it only asks whether a
leg loses money, and a leg that beats break-even by a hundredth of a rung passes while
still paying the exchange more than it takes. The gate has to be the FEE FLOOR, which is
the bar every other cycle in the ladder clears.

THE TRADE-OFF, stated plainly: blocked flips stay held, so in a one-way market the book
thins instead of churning. That is the shape of AUDIT #77's complaint, and it is the
intended behaviour here -- an idle rung costs nothing, a zero-spread rung costs a fee
every time. Same-side re-arms are untouched, so the classic release valve still works.
"""

from unittest.mock import MagicMock

import pytest

from grid import GridEngine, GridLevel

# Fee floor with the default fee model: 2 x blended per-side at an 11.8% taker share
# = 0.04472%, x3.0 profit multiplier = 0.13416%.
FLOOR = 0.0013416

# The live numbers. Short average entry 0.17454965 at 05:10, break-even a hair below.
LIVE_BREAK_EVEN = 0.174480
LIVE_FLIP_PRICE = 0.1743


def engine(spacing=0.00043636):
    ex = MagicMock()
    ex.exchange.amount_to_precision.side_effect = lambda s, a: str(int(float(a)))
    ex.exchange.price_to_precision.side_effect = lambda s, p: f"{float(p):.4f}"
    g = GridEngine(ex, "ADAUSDT", grid_lower=0.17145296, grid_upper=0.17674704,
                   grid_count=12, capital_per_grid_pct=0.018, stop_loss_pct=0.005,
                   capital_per_grid_usdt=5.0, leverage=25)
    g.grid_spacing = spacing
    return g


def held(price, side, counter_price, counter_side):
    lvl = GridLevel(price=price, side=side)
    lvl.status = "awaiting_counter"
    lvl.awaiting_price = counter_price
    lvl.awaiting_side = counter_side
    return lvl


def occupied(price, side):
    lvl = GridLevel(price=price, side=side)
    lvl.status = "pending"
    lvl.order_id = f"live-{price}-{side}"
    return lvl


# --- the premise -----------------------------------------------------------------------

def test_the_fee_floor_is_what_it_should_be():
    """If the fee model changes, every threshold below moves with it."""
    g = engine()

    assert g.round_trip_fee_pct * g._min_profit_multiplier == pytest.approx(FLOOR, rel=0.01)


def test_the_live_flip_really_was_under_the_floor():
    """Guards the premise: 0.103% of spread against a 0.134% floor. If these ever cross,
    the tests below pass vacuously."""
    captured = (LIVE_BREAK_EVEN - LIVE_FLIP_PRICE) / LIVE_BREAK_EVEN

    assert captured < FLOOR
    assert captured == pytest.approx(0.00103, abs=0.0001)


# --- the gate itself ---------------------------------------------------------------------

def test_the_live_flip_is_refused():
    """BUY 0.1743 against a short break-even of 0.174480 — the 05:10:17 flip."""
    g = engine()
    g._position_break_even = lambda: ("short", LIVE_BREAK_EVEN)

    assert g._flip_captures_a_spread(LIVE_FLIP_PRICE, "buy") is False


def test_a_flip_with_real_room_is_allowed():
    """The flip is AUDIT #77's release valve and must still work when it earns."""
    g = engine()
    g._position_break_even = lambda: ("short", 0.1760)

    assert g._flip_captures_a_spread(0.1750, "buy") is True


def test_the_boundary_is_the_floor_not_zero():
    """_books_a_loss already allowed anything beating break-even by a hair. That is the
    behaviour this replaces, so the bar has to sit at the floor."""
    g = engine()
    be = 0.1760
    g._position_break_even = lambda: ("short", be)

    just_under = be * (1 - FLOOR * 0.9)
    just_over = be * (1 - FLOOR * 1.1)

    assert g._flip_captures_a_spread(just_under, "buy") is False
    assert g._flip_captures_a_spread(just_over, "buy") is True


def test_a_long_is_the_mirror():
    g = engine()
    be = 0.1760
    g._position_break_even = lambda: ("long", be)

    assert g._flip_captures_a_spread(be * (1 + FLOOR * 0.9), "sell") is False
    assert g._flip_captures_a_spread(be * (1 + FLOOR * 1.1), "sell") is True


def test_flat_allows_the_flip():
    """No inventory means nothing to close and no spread to undercut — the flip is only
    a re-siting."""
    g = engine()
    g._position_break_even = lambda: None

    assert g._flip_captures_a_spread(0.1743, "buy") is True


def test_a_flip_that_adds_exposure_is_not_this_gate_s_business():
    """Selling while already short opens more, it does not close anything. The position
    cap governs that, not the spread floor."""
    g = engine()
    g._position_break_even = lambda: ("short", 0.1750)

    assert g._flip_captures_a_spread(0.1760, "sell") is True


def test_an_unreadable_break_even_does_not_block_everything():
    g = engine()
    g._position_break_even = lambda: ("short", 0.0)

    assert g._flip_captures_a_spread(0.1743, "buy") is True


# --- wired into the release path ------------------------------------------------------------

def release(g, levels, price):
    g.levels = levels
    g._ladder_holes = lambda: []          # isolate from the hole path (AUDIT #79)
    g._release_awaiting_levels(price)


def test_a_zero_spread_flip_leaves_the_rung_held():
    """The whole point. Counter busy, price cleared, flip would earn less than the floor
    — so the rung waits instead of buying back what it just sold."""
    g = engine()
    g._position_break_even = lambda: ("short", LIVE_BREAK_EVEN)
    rung = held(0.1743, "sell", 0.1739, "buy")

    release(g, [rung, occupied(0.1739, "buy")], 0.1748)

    assert rung.status == "awaiting_counter"
    assert rung.side == "sell"
    assert rung.order_id is None


def test_a_profitable_flip_still_releases():
    g = engine()
    g._position_break_even = lambda: ("short", 0.1760)
    rung = held(0.1743, "sell", 0.1739, "buy")

    release(g, [rung, occupied(0.1739, "buy")], 0.1748)

    assert rung.status == "pending"
    assert rung.side == "buy"
    assert rung.price == pytest.approx(0.1743)


def test_a_same_side_rearm_is_not_gated():
    """Price fell back below the rung, so it comes back as the SELL it already was. No
    inventory is closed on its own line, so the floor does not apply — and this is the
    release valve AUDIT #77 added."""
    g = engine()
    g._position_break_even = lambda: ("short", LIVE_BREAK_EVEN)   # would block a flip
    rung = held(0.1752, "sell", 0.1748, "buy")

    release(g, [rung, occupied(0.1748, "buy")], 0.1747)

    assert rung.status == "pending"
    assert rung.side == "sell"
    assert rung.price == pytest.approx(0.1752)


def test_a_same_side_rearm_is_not_gated_even_when_the_gate_would_refuse():
    """The discriminating version. The test above cannot fail if `not flipping` is
    removed: for a SHORT position the gate returns True on (short, sell) anyway, so it
    passes either way. A mutant that gated same-side re-arms survived it.

    This one puts the gate in a state where it WOULD refuse -- net LONG, break-even
    0.1759, the rung re-arming as a sell at 0.1760, which is 0.057% of spread against a
    0.134% floor. One-way netting makes that reachable: buys can leave the book net long
    while a sell rung sits held. Only `not flipping` lets it through, which is correct,
    because re-arming on its own line as the side it already is closes nothing.
    """
    g = engine()
    g._position_break_even = lambda: ("long", 0.1759)
    rung = held(0.1760, "sell", 0.1756, "buy")

    assert g._flip_captures_a_spread(0.1760, "sell") is False, "premise: gate would refuse"

    release(g, [rung, occupied(0.1756, "buy")], 0.1750)

    assert rung.status == "pending"
    assert rung.side == "sell"
    assert rung.price == pytest.approx(0.1760)


def test_the_counter_path_is_untouched():
    """When the counter slot IS free the rung moves there — a full spacing away, which is
    where the spread comes from. This path was always correct."""
    g = engine()
    g._position_break_even = lambda: ("short", LIVE_BREAK_EVEN)
    rung = held(0.1743, "sell", 0.1739, "buy")

    release(g, [rung], 0.1748)

    assert rung.status == "pending"
    assert rung.side == "buy"
    assert rung.price == pytest.approx(0.1739)
