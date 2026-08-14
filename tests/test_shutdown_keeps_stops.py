"""A position that outlives the bot keeps its stops. AUDIT #65.

pause/shutdown deliberately does not flatten -- inventory has a ladder to unwind through
and booking a loss to tidy up is #32/#37's mistake. But `emergency_stop` called
`cancel_everything`, which takes the algo orders too. So the position stayed and its
protection did not.

Found live twice:

    8215 DOGE long  sat unhedged after the 11:35 shutdown
    2522 DOGE short sat unhedged after the 17:34 shutdown

Grid orders still go on shutdown -- they are this process's working state and would be
duplicated on restart. Stops are not working state, they are protection, and #54's
reconciler adopts live stop legs on restart rather than blindly re-placing them.
"""

import pytest

from grid import GridEngine


class _Ex:
    class exchange:
        @staticmethod
        def amount_to_precision(s, a): return f"{float(a):.0f}"
        @staticmethod
        def price_to_precision(s, p): return f"{float(p):.5f}"

    def __init__(self, contracts=0.0, positions_raise=False):
        self.contracts = contracts
        self.positions_raise = positions_raise
        self.calls = []

    def get_price(self, s): return 0.0698
    def get_balance(self, s="USDT"): return 4896.0
    def get_open_orders(self, s): return []
    def get_open_order_ids(self, s): return set()
    def can_place_order(self, s): return True
    def place_limit_order(self, *a, **k): return {"id": "x"}

    def get_positions(self, s):
        if self.positions_raise:
            raise ConnectionError("positions unreadable")
        return [{"side": "long", "contracts": self.contracts}]

    def cancel_everything(self, symbol, timeout_seconds=300.0):
        self.calls.append("cancel_everything")
        return 3

    def cancel_all_open_orders(self, symbol):
        self.calls.append("cancel_all_open_orders")
        return 3


def _engine(ex):
    g = GridEngine(exchange=ex, symbol="DOGEUSDT", grid_lower=0.0686,
                   grid_upper=0.0710, grid_count=8, capital_per_grid_pct=0.018,
                   capital_per_grid_usdt=5.0, stop_loss_pct=0.03,
                   max_exposure_pct=0.5, leverage=25)
    g.initialize(0.0698, balance=4896.0)
    return g


def test_stops_survive_shutdown_while_a_position_is_open():
    ex = _Ex(contracts=2522.0)
    g = _engine(ex)

    g.emergency_stop(reason="shutdown")

    assert ex.calls == ["cancel_all_open_orders"], (
        "cancel_everything takes the stop legs too — the position is left unprotected "
        "for as long as the bot stays down"
    )


def test_stops_survive_a_kill_switch_trip_too():
    """The kill switch stops trading; it does not make an open position safe to strip."""
    ex = _Ex(contracts=-2522.0)
    g = _engine(ex)

    g.emergency_stop(reason="daily_loss_limit")

    assert ex.calls == ["cancel_all_open_orders"]


def test_everything_is_cancelled_when_flat():
    """No position means no protection to preserve, and stray stops should not linger."""
    ex = _Ex(contracts=0.0)
    g = _engine(ex)

    g.emergency_stop(reason="shutdown")

    assert ex.calls == ["cancel_everything"]


def test_an_unreadable_position_book_is_treated_as_holding():
    """Assuming a position exists costs a few orphan stop orders. Assuming none exists
    costs an unhedged position. The asymmetry decides it."""
    ex = _Ex(positions_raise=True)
    g = _engine(ex)

    g.emergency_stop(reason="shutdown")

    assert ex.calls == ["cancel_all_open_orders"]


def test_grid_levels_are_still_released_either_way():
    """Grid orders are working state; they must not survive into the next process."""
    ex = _Ex(contracts=2522.0)
    g = _engine(ex)
    for level in g.levels:
        level.order_id = "live"
        level.status = "pending"

    g.emergency_stop(reason="shutdown")

    assert all(l.order_id is None for l in g.levels)
    assert g.active is False
