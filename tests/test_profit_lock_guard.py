"""The profit lock: a good day's gains must not ride as continued exposure forever.

Every existing guard bounds LOSSES -- the hard stop-loss, MAX_OPEN_LOSS_USDT, the
daily-loss kill switch, the drawdown kill switch -- and none of them touch the profit
side: a day already won just kept trading with nothing to lock the gain in. That is the
mirror image of the shape MAX_OPEN_LOSS_USDT exists for (test_open_loss_guard.py):
hours of small grid profit erased in under a minute by one bad move.

apply_profit_lock_guard makes the protection explicit: once the day's REALISED P&L
reaches DAILY_PROFIT_LOCK_USDT, BOTH sides stop opening/adding new exposure -- symmetric,
unlike the loss guard, because this isn't defending one position, it's refusing to add
risk to a day already won. Reducing/exiting an open position stays legal -- blocking
exits would weld the day's inventory in place (the #42 lesson).
"""

import pytest

from grid import GridEngine


class _Ex:
    def __init__(self):
        self.placed = []
        self.cancelled = []
        self.open_ids = set()
        # _handle_fill re-reads the net position before it sizes the replacement/exit
        # order, so a test that wants the engine to believe a position is open has to
        # open it HERE too, the same discipline test_position_cap.py follows.
        self.positions = []

    class exchange:
        @staticmethod
        def amount_to_precision(s, a): return f"{float(a):.0f}"
        @staticmethod
        def price_to_precision(s, p): return f"{float(p):.5f}"

    def get_price(self, s): return 0.0700
    def get_balance(self, s="USDT"): return 5000.0
    def get_positions(self, s): return list(self.positions)
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
        self.placed.append((side, price, amount, (params or {}).get("reduceOnly", False)))
        return {"id": f"o{len(self.placed)}"}


def _engine(daily_profit_lock_usdt=3.0):
    ex = _Ex()
    g = GridEngine(exchange=ex, symbol="DOGEUSDT", grid_lower=0.0690, grid_upper=0.0710,
                   grid_count=10, capital_per_grid_pct=0.018, stop_loss_pct=0.03,
                   min_profit_multiplier=3.0, max_exposure_pct=0.5, leverage=5,
                   daily_profit_lock_usdt=daily_profit_lock_usdt)
    g.initialize(0.0700, balance=5000.0)
    return g, ex


def test_below_budget_neither_side_is_blocked():
    g, ex = _engine(daily_profit_lock_usdt=3.0)
    g.apply_profit_lock_guard(2.99)

    assert g._profit_lock_active is False
    buy = next(l for l in g.levels if l.side == "buy")
    sell = next(l for l in g.levels if l.side == "sell")
    assert g._place_order_for_level(buy, balance=5000.0) is True
    assert g._place_order_for_level(sell, balance=5000.0) is True


def test_at_budget_both_buy_and_sell_opening_are_blocked():
    """Symmetric, unlike the loss guard: profit lock blocks BOTH sides regardless of
    which direction the next trade would open, because it isn't defending a specific
    position -- it's refusing to add risk to a day already won."""
    g, ex = _engine(daily_profit_lock_usdt=3.0)
    g.apply_profit_lock_guard(3.0)

    assert g._profit_lock_active is True
    buy = next(l for l in g.levels if l.side == "buy")
    sell = next(l for l in g.levels if l.side == "sell")
    before = len(ex.placed)
    assert g._place_order_for_level(buy, balance=5000.0) is False
    assert g._place_order_for_level(sell, balance=5000.0) is False
    assert len(ex.placed) == before


def test_above_budget_also_blocks():
    g, _ = _engine(daily_profit_lock_usdt=3.0)
    g.apply_profit_lock_guard(6.08)   # this bot's own 16-day best-day figure

    assert g._profit_lock_active is True


def test_zero_budget_disables_the_guard_entirely():
    g, ex = _engine(daily_profit_lock_usdt=0.0)
    g.apply_profit_lock_guard(1_000_000.0)   # an absurd day's profit

    assert g._profit_lock_active is False
    buy = next(l for l in g.levels if l.side == "buy")
    assert g._place_order_for_level(buy, balance=5000.0) is True


def test_the_block_clears_when_pnl_dips_back_under_budget():
    """A losing fill later the same day can pull realised pnl back under the budget --
    the guard is stateless and recomputed fresh every call, so it must release."""
    g, _ = _engine(daily_profit_lock_usdt=3.0)
    g.apply_profit_lock_guard(4.0)
    assert g._profit_lock_active is True

    g.apply_profit_lock_guard(1.5)

    assert g._profit_lock_active is False


def test_it_can_re_engage_after_dipping_back_under():
    """The false-to-true-to-false-to-true round trip, mirroring the loss guard's own
    recovery behaviour: a fresh gain later the same day re-arms the lock."""
    g, _ = _engine(daily_profit_lock_usdt=3.0)
    g.apply_profit_lock_guard(4.0)
    assert g._profit_lock_active is True

    g.apply_profit_lock_guard(1.0)
    assert g._profit_lock_active is False

    g.apply_profit_lock_guard(3.5)
    assert g._profit_lock_active is True


def test_a_daily_reset_to_zero_releases_the_lock():
    """The guard's own self-clearing is what releases it at the next UTC day rollover:
    once daily_reset_check has rolled the day's realised P&L back to zero, the next
    call sees pnl=0 < budget and clears the flag automatically -- no separate reset
    path is needed."""
    g, _ = _engine(daily_profit_lock_usdt=3.0)
    g.apply_profit_lock_guard(6.08)
    assert g._profit_lock_active is True

    g.apply_profit_lock_guard(0.0)   # the new day's fresh (rolled-over) pnl

    assert g._profit_lock_active is False


def test_an_existing_positions_exit_still_places_through_a_fill():
    """#42's lesson, mirrored: refusing exit orders traps inventory regardless of which
    guard is doing the refusing. _handle_fill places the paired reduce-only replacement
    directly (it never goes through _place_order_for_level -- see test_position_cap.py's
    identically-shaped test_exits_are_never_blocked), so it must go out even while the
    profit lock has both sides blocked at the placement site."""
    g, ex = _engine(daily_profit_lock_usdt=3.0)
    ex.positions = [{"side": "short", "contracts": 50000.0, "entryPrice": 0.0700}]
    g._net_short_qty = 50000.0
    g.apply_profit_lock_guard(5.0)
    assert g._profit_lock_active is True

    sell = next(l for l in g.levels if l.side == "sell")
    before = len(ex.placed)
    g._handle_fill(sell, balance=5000.0)

    exits = [o for o in ex.placed[before:] if o[3]]
    assert exits, "the reduce-only exit was blocked -- inventory can never unwind"


def test_tripping_the_budget_cancels_resting_orders_on_both_sides():
    """Same contract as the loss guard's own cancellation test
    (test_open_loss_guard.py's test_tripping_the_budget_cancels_resting_orders_on_the_
    adverse_side): blocking placements alone cannot stop an order that is ALREADY
    resting on the book from filling and adding new exposure after the lock trips.
    Unlike the loss guard, the profit lock is symmetric, so both sides must clear."""
    g, ex = _engine(daily_profit_lock_usdt=3.0)
    g.levels = [
        type("L", (), {"price": 0.0695, "side": "buy", "order_id": "o1", "status": "placed"})(),
        type("L", (), {"price": 0.0705, "side": "sell", "order_id": "o2", "status": "placed"})(),
    ]
    ex.open_ids = {"o1", "o2"}
    g.apply_profit_lock_guard(3.0)

    assert "o1" in ex.cancelled
    assert "o2" in ex.cancelled


def test_a_blocked_side_cannot_place_through_place_initial_orders():
    """The guard must hold at the same choke point place_initial_orders flows through,
    the same contract test_open_loss_guard.py asserts for its own guard."""
    g, ex = _engine(daily_profit_lock_usdt=3.0)
    g.apply_profit_lock_guard(3.0)

    placed_before = len(ex.placed)
    for level in g.levels:
        g._place_order_for_level(level, balance=5000.0)

    assert len(ex.placed) == placed_before
