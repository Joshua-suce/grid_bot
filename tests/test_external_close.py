"""A stop leg that fires on the exchange must reach the ledger and the journal.

The hard stop-market order is placed on Binance and fires there. The bot never
initiates it, so it never passes through _handle_fill: the ladder keeps a position
that no longer exists, total_pnl never books the loss, and no journal row is written.

2026-08-20, from the Binance execution ledger (`py attribute_pnl.py 9`):

    stop_hard   6 taker execs   -76.98   worst single -74.84

That -74.84 is 105% of the account's entire -71.17 for the period, and it produced
no journal row at all. cycle_pnl for the same window showed 47 wins, 1 loss, +10.96.

reconcile_position_entry cannot cover this -- it returns early when the exchange
reads flat, because adopting "flat" was the AUDIT #80 hazard. So flat is handled
separately, and only when corroborated by a second read. AUDIT #143.
"""
from __future__ import annotations

import pytest

from grid import GridEngine


class _Ex:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def get_positions(self, symbol):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return reply


FLAT: list = []


def HOLDS(qty=100.0, side="long", entry=0.20):
    return [{"contracts": qty, "side": side, "entryPrice": entry,
             "info": {"positionAmt": str(qty)}}]


def _grid(exchange, pos_qty=0.0, pos_entry=0.0):
    g = GridEngine.__new__(GridEngine)
    g.exchange = exchange
    g.symbol = "ADAUSDT"
    g._pos_qty = pos_qty
    g._pos_entry = pos_entry
    g.total_fills = 0
    g.total_completed_cycles = 0
    g.total_pnl = 0.0
    g._event_journal = None
    return g


# ------------------------------------------------------------- nothing to report
def test_a_flat_ledger_reports_nothing():
    g = _grid(_Ex(FLAT), 0.0, 0.0)
    assert g.detect_external_close(0.20) is None
    assert g.exchange.calls == 0, "it read the account with nothing to check"


def test_a_position_still_open_reports_nothing():
    g = _grid(_Ex(HOLDS(100.0)), 100.0, 0.20)
    assert g.detect_external_close(0.20) is None
    assert g._pos_qty == 100.0, "it cleared a position that is still open"


def test_an_unreadable_account_reports_nothing():
    """UNKNOWN is not flat. Booking a close on a failed read invents a realised loss
    and throws away a live position's cost basis (AUDIT #128/#132/#134)."""
    g = _grid(_Ex(RuntimeError("endpoint down")), 100.0, 0.20)
    assert g.detect_external_close(0.20) is None
    assert g._pos_qty == 100.0


def test_one_transient_flat_reply_is_not_enough():
    """A single bad HTTP reply must not book a close that never happened."""
    g = _grid(_Ex(FLAT, HOLDS(100.0)), 100.0, 0.20)

    assert g.detect_external_close(0.20) is None
    assert g._pos_qty == 100.0
    assert g.total_pnl == 0.0


def test_a_flat_reply_followed_by_an_unreadable_one_is_not_enough():
    g = _grid(_Ex(FLAT, RuntimeError("boom")), 100.0, 0.20)
    assert g.detect_external_close(0.20) is None
    assert g._pos_qty == 100.0


# ------------------------------------------------------------------ the booking
def test_a_corroborated_close_is_booked_and_cleared():
    g = _grid(_Ex(FLAT, FLAT), 100.0, 0.20)

    fill = g.detect_external_close(0.18)

    assert fill is not None
    assert g._pos_qty == 0.0, "the phantom position survived"
    assert g.total_pnl == pytest.approx(-2.0), "the loss was not booked"
    assert g.total_fills == 1
    assert g.total_completed_cycles == 1


def test_a_long_closed_below_entry_books_a_loss():
    g = _grid(_Ex(FLAT, FLAT), 100.0, 0.20)
    fill = g.detect_external_close(0.18)
    assert fill["profit"] == pytest.approx(-2.0)
    assert fill["side"] == "sell", "a long is closed by selling"


def test_a_short_closed_above_entry_books_a_loss():
    """The 2026-08-20 shape: short stopped out into a rally."""
    g = _grid(_Ex(FLAT, FLAT), -100.0, 0.18)
    fill = g.detect_external_close(0.20)
    assert fill["profit"] == pytest.approx(-2.0)
    assert fill["side"] == "buy", "a short is closed by buying"
    # The quantity must be a magnitude. A short's ledger qty is negative, and only a
    # short exposes that -- checking it on a long is vacuous because abs(q) == q.
    assert fill["quantity"] == pytest.approx(100.0), (
        "a signed quantity would journal a negative-size row for every short"
    )


def test_a_profitable_external_close_is_booked_too():
    """Take-profit legs fire on the exchange the same way. Booking only losses would
    make total_pnl wrong in the other direction."""
    g = _grid(_Ex(FLAT, FLAT), 100.0, 0.18)
    assert g.detect_external_close(0.20)["profit"] == pytest.approx(+2.0)


def test_the_fill_carries_what_the_journal_needs():
    """It is folded into the same fill list main.py journals, so it must satisfy the
    same contract or the journal row raises instead of being written."""
    g = _grid(_Ex(FLAT, FLAT), 100.0, 0.20)
    fill = g.detect_external_close(0.18)
    for key in ("price", "side", "quantity", "profit", "fee", "completed_cycle"):
        assert key in fill, f"journal.record needs {key}"
    assert fill["quantity"] == pytest.approx(100.0)
    assert fill["quantity"] > 0, "a negative quantity would journal a nonsense row"


def test_the_estimate_is_labelled_as_one():
    """The exact exit is the exchange's; only the income reconciler knows it. Marking
    the row keeps the AUDIT #139 divergence alarm honest about what it is watching."""
    g = _grid(_Ex(FLAT, FLAT), 100.0, 0.20)
    fill = g.detect_external_close(0.18)
    assert fill.get("estimated") is True
    assert fill.get("external") is True


def test_no_entry_price_books_no_invented_profit():
    """Without a cost basis any P&L figure is fabricated. Clear the phantom, book
    zero, and let the reconciler carry the truth."""
    g = _grid(_Ex(FLAT, FLAT), 100.0, 0.0)
    fill = g.detect_external_close(0.18)
    assert fill["profit"] == 0.0
    assert g._pos_qty == 0.0


def test_it_only_fires_once_for_one_close():
    """Booking the same close every poll would manufacture a loss per iteration."""
    g = _grid(_Ex(FLAT), 100.0, 0.20)

    assert g.detect_external_close(0.18) is not None
    assert g.detect_external_close(0.18) is None
    assert g.total_pnl == pytest.approx(-2.0)


def test_a_zero_price_is_refused():
    """An unreadable price would book the whole position as a loss at 0."""
    g = _grid(_Ex(FLAT, FLAT), 100.0, 0.20)
    assert g.detect_external_close(0.0) is None
    assert g._pos_qty == 100.0


# ------------------------------------------------------- wired into the fill stream
def test_the_detected_close_is_folded_into_the_fills_main_journals():
    """M9: removing the wiring in main.py left every unit test above green. The
    detector can be perfect and still report to nobody -- the same "a helper nothing
    calls is not a check" failure that ladder_cap_room and the account recheck both
    needed pinning for.
    """
    import ast as _ast
    import inspect
    import textwrap

    import main as main_module

    tree = _ast.parse(textwrap.dedent(inspect.getsource(main_module.run_bot)))
    call = None
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute)
                and node.func.attr == "detect_external_close"):
            call = node
    assert call is not None, "main.py never asks whether the exchange closed us"

    # and its result must reach `fills`, which is what gets journalled
    src = inspect.getsource(main_module.run_bot)
    assert "fills = [_closed] + list(fills)" in src, (
        "the detected close is computed and then dropped -- it never reaches the "
        "journal or the fill handling"
    )


def test_the_protocol_and_both_strategies_agree():
    """main.py calls this on whichever strategy is live. A member the grid has and the
    trend follower does not is an AttributeError every iteration in router mode --
    which is exactly what tests/test_strategy.py caught when this was first written.
    """
    from grid import GridEngine
    from strategy import Strategy
    from trend_follower import TrendFollower

    for cls in (GridEngine, TrendFollower):
        assert hasattr(cls, "detect_external_close"), cls.__name__
    assert hasattr(Strategy, "detect_external_close")
