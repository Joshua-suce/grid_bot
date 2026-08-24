"""The position cap must bound what the ladder has COMMITTED, not what has filled.

Enforcing it against the filled position alone made resting orders invisible: six
sell rungs of ~125 USDT could sit under a 243 USDT cap while the gate reported "not
blocked", right up to the moment they all filled.

2026-08-20, from the Binance execution ledger (`py attribute_pnl.py 9`):

    grid_entry    179 maker execs   +2.08 net   57 wins / 0 losses
    stop_hard       6 taker execs  -76.98       worst single -74.84
    TOTAL                          -71.17

The short book reached ~1,103 USDT against a configured max_position_pct of 0.05 =
243 USDT, 4.5x over. One uncapped position gave back roughly thirty years of what
the ladder earns. AUDIT #138.

Cancelling the resting side once blocked is a backstop, not a bound -- the overshoot
happens between the cap being reached and the next poll, and a run through six rungs
takes less than one 10s interval.
"""
from __future__ import annotations

import pytest

from grid import GridEngine, GridLevel


def _grid(levels=()):
    g = GridEngine.__new__(GridEngine)
    g.levels = list(levels)
    g._block_buys = False
    g._block_sells = False
    g._buy_scale = 1.0
    g._sell_scale = 1.0
    g._net_long_qty = 0.0
    g._net_short_qty = 0.0
    g._event_journal = None
    g._notifier = None
    g._cancel_resting_orders = lambda side, reason: None
    return g


def _resting(side, qty, price=0.19, oid="x"):
    return GridLevel(price=price, side=side, order_id=oid, quantity=qty)


# --------------------------------------------------------- the exposure sum
def test_resting_orders_count_toward_the_side_they_would_open():
    g = _grid([_resting("sell", 400, oid="a"), _resting("sell", 400, oid="b")])

    _, committed_short = g._committed_exposure(0.0, 200.0)

    assert committed_short == pytest.approx(1000.0), (
        "resting sells were invisible to the cap -- this is the 1,103-vs-243 breach"
    )


def test_unplaced_levels_do_not_count():
    """A level with no live order is an intention, not a commitment."""
    g = _grid([GridLevel(price=0.19, side="sell", order_id=None, quantity=400)])

    _, committed_short = g._committed_exposure(0.0, 200.0)

    assert committed_short == pytest.approx(200.0)


def test_a_sell_against_an_open_long_reduces_rather_than_shorts():
    """One-way netting. Counting a reducing order as new short would block the exits
    and weld the position in place -- the failure the break-even rule already causes
    on its own (AUDIT #32)."""
    g = _grid([_resting("sell", 300)])

    committed_long, committed_short = g._committed_exposure(1000.0, 0.0)

    assert committed_short == 0.0, "an exit was counted as new exposure"
    assert committed_long == pytest.approx(1000.0)


def test_the_exit_side_is_not_gated_at_all_while_holding():
    """Reversed deliberately, after this welded a live grid on 2026-08-24.

    This used to assert that resting sells beyond the long counted as committed short
    (1500 resting - 1000 long = 500). That is arithmetically true and operationally
    fatal: the gate then BLOCKS the sell side, and sells are how a long gets closed.

    router.set_position_limit clamps the cap to the open size during a handoff so the
    outgoing strategy "can still close through its own levels but cannot open anything
    new". With long 114 against a clamped cap of 114 and 338 of resting sells, the old
    rule computed 224 >= 114 and blocked the exits -- the grid could not work itself
    flat, and the grace expires into the forced market dump AUDIT #29 measured at
    -46.16 across 18 handoffs.

    Only the side that ADDS is gated now. Overshoot past flat is bounded by the next
    poll, when the position has flipped and that side becomes the adding one.
    """
    g = _grid([_resting("sell", 1500)])

    _, committed_short = g._committed_exposure(1000.0, 0.0)

    assert committed_short == 0.0, "the exit side is gated, which welds the position"


def test_the_live_handoff_weld_does_not_recur():
    """The exact 2026-08-24 state: long 114, cap clamped to 114 by the handoff, three
    sell rungs of 113/113/112 resting."""
    g = _grid([_resting("sell", 113, oid="a"), _resting("sell", 113, oid="b"),
               _resting("sell", 112, oid="c")])

    g.set_position_limit(114.0, 0.0, 114.0)

    assert g._block_sells is False, (
        "the grid cannot place the sells that would get it flat -- the handoff then "
        "expires into a forced market close"
    )
    assert g._block_buys is True, "it must still refuse to grow the long"


def test_resting_buys_add_to_the_long_side():
    g = _grid([_resting("buy", 600)])

    committed_long, _ = g._committed_exposure(400.0, 0.0)

    assert committed_long == pytest.approx(1000.0)


def test_a_buy_against_an_open_short_reduces_it():
    g = _grid([_resting("buy", 300)])

    committed_long, committed_short = g._committed_exposure(0.0, 1000.0)

    assert committed_long == 0.0
    assert committed_short == pytest.approx(1000.0)


# ------------------------------------------------------------- the gate itself
def test_the_cap_blocks_before_the_resting_orders_fill():
    """The whole fix. Filled short is well under the cap; committed is over it."""
    g = _grid([_resting("sell", 400, oid="a"), _resting("sell", 400, oid="b"),
               _resting("sell", 400, oid="c")])

    g.set_position_limit(0.0, 200.0, 1000.0)

    assert g._block_sells is True, (
        "filled 200 of a 1000 cap looks safe, but 1200 more is already committed"
    )


def test_it_does_not_block_a_side_that_is_genuinely_clear():
    g = _grid([_resting("sell", 100)])

    g.set_position_limit(0.0, 100.0, 1000.0)

    assert g._block_sells is False
    assert g._sell_scale == 1.0


def test_the_exit_side_stays_open_while_holding_a_position():
    """Blocking the reducing side is how a position becomes unexitable."""
    g = _grid([_resting("sell", 500)])

    g.set_position_limit(2000.0, 0.0, 1000.0)

    assert g._block_sells is False, "the ladder cannot exit its own long"


def test_scaling_kicks_in_on_committed_exposure_too():
    """The taper past 50% of the cap must see commitments as well, or size only
    shrinks after the position is already large."""
    g = _grid([_resting("sell", 600)])

    g.set_position_limit(0.0, 100.0, 1000.0)

    assert 0.0 < g._sell_scale < 1.0, g._sell_scale


def test_the_incident_would_have_been_blocked():
    """Reconstructed: cap 243 USDT at ~0.19 = ~1279 ADA. Six rungs of ~125 USDT =
    ~658 ADA each side. With 2000 ADA already short, the next sell must not place."""
    cap_qty = 243.0 / 0.19
    rung = 125.0 / 0.19
    g = _grid([_resting("sell", rung, oid=str(i)) for i in range(3)])

    g.set_position_limit(0.0, 2000.0, cap_qty)

    assert g._block_sells is True


def test_a_flat_book_with_no_orders_is_unblocked():
    g = _grid()

    g.set_position_limit(0.0, 0.0, 1000.0)

    assert g._block_buys is False and g._block_sells is False


# ------------------------------------------------ the two jobs must stay separate
class _CancelSpy:
    def __init__(self):
        self.cancelled = []

    def __call__(self, side, reason):
        self.cancelled.append(side)


def test_a_commitment_block_does_not_cancel_the_orders_that_caused_it():
    """The oscillation guard, and the reason the cancel keeps a different trigger.

    Cancelling because of a commitment removes the very orders that created it: next
    poll the commitment is gone, the side unblocks, the ladder re-places, and it
    blocks again -- a loop that burns API calls and never converges.
    """
    g = _grid([_resting("sell", 400, oid="a"), _resting("sell", 400, oid="b"),
               _resting("sell", 400, oid="c")])
    spy = _CancelSpy()
    g._cancel_resting_orders = spy

    g.set_position_limit(0.0, 200.0, 1000.0)

    assert g._block_sells is True, "the placement gate should still fire"
    assert spy.cancelled == [], (
        "it cancelled the commitment that caused the block -- this oscillates"
    )


def test_a_position_already_over_the_cap_still_has_its_pending_adds_pulled():
    """The older remedy must survive: filled position past the cap pulls the resting
    orders behind it. Losing this would remove the backstop entirely."""
    g = _grid([_resting("sell", 100)])
    spy = _CancelSpy()
    g._cancel_resting_orders = spy

    g.set_position_limit(0.0, 1500.0, 1000.0)

    assert spy.cancelled == ["sell"]


def test_the_two_triggers_are_genuinely_different():
    """Guard on the guard: if both used the same input, one of the two tests above
    would be unreachable and neither would notice."""
    g = _grid([_resting("sell", 5000, oid="huge")])
    spy = _CancelSpy()
    g._cancel_resting_orders = spy

    g.set_position_limit(0.0, 10.0, 1000.0)   # filled tiny, committed enormous

    assert g._block_sells is True and spy.cancelled == []


# ------------------------------------------------- flat is where the ladder opens
def test_a_flat_ladder_is_gated_on_what_its_resting_orders_would_open():
    """M4: returning (0, 0) when flat left every other test in this file green.

    Flat is not a safe state to leave ungated -- it is the state the opening ladder
    starts from. AUDIT #138's breach began at flat: rungs went out, filled, and the
    short ran to ~1,103 against a 243 cap. From flat either side opens, so both are
    gated on the volume they would open.
    """
    g = _grid([_resting("buy", 600, oid="a"), _resting("sell", 900, oid="b")])

    committed_long, committed_short = g._committed_exposure(0.0, 0.0)

    assert committed_long == pytest.approx(600.0)
    assert committed_short == pytest.approx(900.0)


def test_a_flat_opening_ladder_over_the_cap_is_blocked_on_both_sides():
    """The end-to-end shape of the same mutant, through the public gate."""
    g = _grid([_resting("buy", 600, oid="a"), _resting("sell", 900, oid="b")])

    g.set_position_limit(0.0, 0.0, 500.0)

    assert g._block_buys is True, "a flat ladder committed 600 under a 500 cap"
    assert g._block_sells is True, "a flat ladder committed 900 under a 500 cap"
