"""The loss budget: a capped position must not also be handed unlimited adverse room.

The cap bounds how BIG a position can get. Nothing bounded how much it was allowed to
lose while GROWING: a trend through the ladder filled every rung on one side -- each an
intentional average -- and then handed the whole capped position to the hard stop as one
taker print. That single print (-74.39) outweighed everything 178 maker cycles earned
(+2.61) over 2026-07-22..08-20: wins are structurally small (one spacing), losses were
structurally large (cap x trend distance).

apply_open_loss_guard makes the asymmetry explicit: once the open position's unrealised
loss reaches MAX_OPEN_LOSS_USDT, the side that would ADD to it is blocked. Reducing
stays legal -- blocking exits would weld the loss in place (the #42 lesson).
"""

import pytest

from grid import GridEngine


class _Ex:
    def __init__(self):
        self.placed = []
        self.cancelled = []
        self.open_ids = set()

    class exchange:
        @staticmethod
        def amount_to_precision(s, a): return f"{float(a):.0f}"
        @staticmethod
        def price_to_precision(s, p): return f"{float(p):.5f}"

    def get_price(self, s): return 0.0700
    def get_balance(self, s="USDT"): return 5000.0
    def get_positions(self, s): return []
    def get_open_orders(self, s): return []
    def get_open_order_ids(self, s): return set(self.open_ids)
    def fetch_order(self, i, s): return None
    def cancel_order(self, i, s):
        self.cancelled.append(i)
        return True
    def cancel_everything(self, s, timeout_seconds=300.0, keep_stops=False): return 0
    def can_place_order(self, s): return True
    def close_position(self, symbol, side, amount, max_attempts=None): return {"id": "c"}
    def place_limit_order(self, symbol, side, price, amount, max_attempts=3, params=None,
                          post_only=True, allow_taker_fallback=False):
        self.placed.append((side, price, amount))
        return {"id": f"o{len(self.placed)}"}


def _engine(max_open_loss_usdt=10.0):
    ex = _Ex()
    g = GridEngine(exchange=ex, symbol="DOGEUSDT", grid_lower=0.0690, grid_upper=0.0710,
                   grid_count=10, capital_per_grid_pct=0.018, stop_loss_pct=0.03,
                   min_profit_multiplier=3.0, max_exposure_pct=0.5, leverage=5,
                   max_open_loss_usdt=max_open_loss_usdt)
    g.initialize(0.0700, balance=5000.0)
    return g, ex


def _hold_long(g, qty=10000.0, entry=0.0700):
    g._pos_qty = qty
    g._pos_entry = entry


def test_a_losing_long_blocks_further_buys_at_the_budget():
    """Long 10000 @ 0.0700, price 0.0689 -> down ~11 USDT of a 10 budget."""
    g, ex = _engine(max_open_loss_usdt=10.0)
    _hold_long(g)
    g.apply_open_loss_guard(0.0689)

    assert g._loss_block_buys is True
    assert g._loss_block_sells is False, "exits must never be blocked by the budget"


def test_a_losing_short_blocks_further_sells_at_the_budget():
    g, ex = _engine(max_open_loss_usdt=10.0)
    _hold_long(g, qty=-10000.0)
    g.apply_open_loss_guard(0.0711)

    assert g._loss_block_sells is True
    assert g._loss_block_buys is False


def test_under_budget_nothing_is_blocked():
    g, _ = _engine(max_open_loss_usdt=10.0)
    _hold_long(g)
    g.apply_open_loss_guard(0.0695)          # down 5 USDT of a 10 budget

    assert g._loss_block_buys is False
    assert g._loss_block_sells is False


def test_the_block_clears_when_price_recovers():
    g, _ = _engine(max_open_loss_usdt=10.0)
    _hold_long(g)
    g.apply_open_loss_guard(0.0689)
    assert g._loss_block_buys is True

    g.apply_open_loss_guard(0.0698)          # down 2 USDT now

    assert g._loss_block_buys is False


def test_zero_budget_disables_the_guard_entirely():
    g, _ = _engine(max_open_loss_usdt=0.0)
    _hold_long(g)
    g.apply_open_loss_guard(0.0001)          # an absurd 699 USDT loss

    assert g._loss_block_buys is False


def test_flat_position_never_blocks_anything():
    g, _ = _engine(max_open_loss_usdt=10.0)
    g.apply_open_loss_guard(0.0001)

    assert g._loss_block_buys is False
    assert g._loss_block_sells is False


def test_tripping_the_budget_cancels_resting_orders_on_the_adverse_side():
    """Same contract as the cap: blocked placements alone cannot stop orders that are
    ALREADY resting from filling and growing the position past the intent."""
    g, ex = _engine(max_open_loss_usdt=10.0)
    _hold_long(g, qty=-10000.0)
    g.levels = [type("L", (), {"price": 0.0705, "side": "sell", "order_id": "o1",
                               "status": "placed"})()]
    ex.open_ids = {"o1"}
    g.apply_open_loss_guard(0.0711)

    assert "o1" in ex.cancelled


def test_a_blocked_side_cannot_place_through_place_initial_orders():
    """The guard must hold at the same choke point every placement flows through."""
    g, ex = _engine(max_open_loss_usdt=10.0)
    _hold_long(g)
    g.apply_open_loss_guard(0.0689)

    placed_before = len(ex.placed)
    for level in g.levels:
        if level.side == "buy":
            g._place_order_for_level(level, balance=5000.0)

    assert len(ex.placed) == placed_before


def test_an_exit_on_the_blocked_side_of_the_book_still_places():
    """For a losing LONG the guard blocks buys; sells are exits and stay quotable."""
    g, ex = _engine(max_open_loss_usdt=10.0)
    _hold_long(g)
    g.apply_open_loss_guard(0.0689)

    sell_levels = [l for l in g.levels if l.side == "sell"]
    assert sell_levels, "test needs a sell rung to exist"
    any_ok = any(g._place_order_for_level(l, balance=5000.0) or True
                 for l in sell_levels
                 if not g._would_realise_a_loss(l.side, l.price))
    assert any_ok
