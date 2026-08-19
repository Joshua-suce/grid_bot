"""A ladder that cannot trade may take a bounded loss to free itself. AUDIT #122.

AUDIT #32 refuses to place any exit that books a loss: "A grid is allowed to sit on
inventory and wait; that is what its levels are for." That is true only while those levels
can still TRADE. Waiting is free because the rest of the ladder keeps earning while you
wait -- once the position has eaten the position cap, that side is blocked, no rung on it
can be placed, and waiting earns nothing. It is a directional bet with the income off.

ADAUSDT, 2026-08-19, in order:

    14:56:27  POSITION LIMIT | short 6307.0 >= 5608.77 -- sell orders blocked
    15:06:28  KILL SWITCH: price 0.1776 above stop loss 0.17753
    16:30:22  SKIP BUY @ 0.1752 | below break-even 0.17475982
    16:30:23  SKIP BUY @ 0.1756 / 0.176 / 0.1765   (same reason, every rung)
    16:31:42  POSITION LIMIT | short 6307.0 >= 5433.6 -- sell orders blocked
    19:44:58  Cancelled 0 open orders for ADAUSDT

At 14:56 price was 0.176 -- 0.59% past break-even. Refusing to book that is what produced
a 3.3% loss by 19:44 and three hours in which no rung could trade at all. The hard stop is
not the answer either: it fired at 15:06 and the position was still open at shutdown.

So: while the OPPOSITE side is cap-blocked, an exit may price up to rung_loss_cap_pct
past break-even. Outside that state nothing changes at all, and rung_loss_cap_pct = 0
restores #32 exactly.
"""

from unittest.mock import MagicMock

import pytest

from grid import GridEngine, GridLevel

BE = 0.17475982        # the live break-even, from the SKIP lines
CAP = 0.01             # rung_loss_cap_pct default


def engine(loss_cap=CAP, spacing=0.00083636):
    ex = MagicMock()
    ex.exchange.amount_to_precision.side_effect = lambda s, a: str(int(float(a)))
    ex.exchange.price_to_precision.side_effect = lambda s, p: f"{float(p):.4f}"
    ex.can_place_order.return_value = True
    ex.get_open_orders.return_value = []
    ex.place_limit_order.return_value = {"id": "placed", "amount": 700}
    g = GridEngine(ex, "ADAUSDT", grid_lower=0.17556559, grid_upper=0.18563441,
                   grid_count=12, capital_per_grid_pct=0.018, stop_loss_pct=0.005,
                   capital_per_grid_usdt=5.0, leverage=25, rung_loss_cap_pct=loss_cap)
    g.grid_spacing = spacing
    g._position_break_even = lambda: ("short", BE)
    g._existing_open_order = lambda price, side: None
    return g


# --- the gate -------------------------------------------------------------------------

def test_a_capped_short_lets_the_buy_side_take_a_bounded_loss():
    g = engine()
    g._block_sells = True

    ceiling = g._stuck_exit_ceiling("buy")

    assert ceiling == pytest.approx(BE * 1.01)
    assert ceiling == pytest.approx(0.176507, abs=1e-5)


def test_nothing_changes_while_the_other_side_can_still_trade():
    """The whole safety of this: outside the stuck state it is inert."""
    g = engine()
    g._block_sells = False

    assert g._stuck_exit_ceiling("buy") is None


def test_a_capped_long_is_the_mirror():
    g = engine()
    g._position_break_even = lambda: ("long", BE)
    g._block_buys = True

    assert g._stuck_exit_ceiling("sell") == pytest.approx(BE * 0.99)


def test_it_never_licenses_an_order_that_adds_exposure():
    """A SELL while short grows the short; a BUY while long grows the long. Neither is an
    exit and neither may borrow this budget."""
    g = engine()
    g._block_sells = g._block_buys = True

    assert g._stuck_exit_ceiling("sell") is None            # short + sell

    g._position_break_even = lambda: ("long", BE)
    assert g._stuck_exit_ceiling("buy") is None             # long + buy


def test_flat_has_nothing_to_exit():
    g = engine()
    g._position_break_even = lambda: None
    g._block_sells = True

    assert g._stuck_exit_ceiling("buy") is None


def test_zero_cap_restores_audit_32_exactly():
    g = engine(loss_cap=0.0)
    g._block_sells = True

    assert g._stuck_exit_ceiling("buy") is None


def test_an_unreadable_break_even_licenses_nothing():
    g = engine()
    g._block_sells = True
    g._position_break_even = lambda: ("short", 0.0)

    assert g._stuck_exit_ceiling("buy") is None


# --- the bound actually bounds --------------------------------------------------------

def test_the_live_skipped_rung_is_now_inside_the_budget():
    """SKIP BUY @ 0.1752 -- 0.25% past break-even, well inside a 1% cap."""
    g = engine()
    g._block_sells = True

    assert 0.1752 <= g._stuck_exit_ceiling("buy")


def test_a_rung_far_past_break_even_is_still_refused():
    """0.1806 is 3.3% past break-even. The budget is a bound, not a licence to dump."""
    g = engine()
    g._block_sells = True

    assert 0.1806 > g._stuck_exit_ceiling("buy")


# --- wired into placement -------------------------------------------------------------

def place(g, price, side="buy"):
    lvl = GridLevel(price=price, side=side)
    lvl.status = "pending"
    return g._place_order_for_level(lvl, 4900.0), lvl


def test_the_stuck_rung_is_placed_instead_of_skipped():
    """16:30:22 SKIP BUY @ 0.1752 becomes an order."""
    g = engine()
    g._block_sells = True

    ok, lvl = place(g, 0.1752)

    assert ok is True
    assert g.exchange.place_limit_order.called
    assert lvl.price == pytest.approx(0.1752), "should not have been moved to break-even"


def test_a_rung_beyond_the_budget_still_falls_through_to_the_old_path():
    g = engine()
    g._block_sells = True
    g._nearest_legal_exit = lambda level: None

    ok, _ = place(g, 0.1806)

    assert ok is False
    assert not g.exchange.place_limit_order.called


def test_with_the_other_side_free_the_old_behaviour_is_untouched():
    """#32 must still hold in the ordinary case: a loss-making exit is refused."""
    g = engine()
    g._block_sells = False
    g._nearest_legal_exit = lambda level: None

    ok, _ = place(g, 0.1752)

    assert ok is False
    assert not g.exchange.place_limit_order.called


def test_zero_cap_reproduces_the_deadlock_exactly():
    """The opt-out is real: set the knob to 0 and 16:30:22 skips again."""
    g = engine(loss_cap=0.0)
    g._block_sells = True
    g._nearest_legal_exit = lambda level: None

    ok, _ = place(g, 0.1752)

    assert ok is False
