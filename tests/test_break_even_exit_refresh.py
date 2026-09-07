"""A resting exit quoted at break-even must not be left behind as the position's
real break-even keeps moving. AUDIT #175.

_place_order_for_level's break-even guard (AUDIT #32/#42) only ever runs at the
moment a level is (re)placed -- every one of its four call sites requires
order_id is None. _would_realise_a_loss/_nearest_legal_exit are otherwise called
from exactly one other place, accelerate_handoff_exit -- and that path only fires
during a strategy-router handoff (AUDIT #145), which never runs while
STRATEGY_MODE=grid trades alone.

So in ordinary grid operation, a level quoted at break-even sat there unwatched.
If a further fill landed on the SAME side afterward -- the common case, since a
level usually gets stuck at break-even exactly because price kept moving against
that side -- the position's real average entry moved past the frozen quote, and
the resting order ended up on the losing side of the NEW break-even. If it
filled, it booked a real loss: the precise outcome #32/#42 exist to prevent.

_refresh_break_even_exits fixes this the same way accelerate_handoff_exit already
fixes the analogous handoff-only case: it reuses that exact cancel/ambiguous-fill
/reprice machinery, just gated on "this resting price now loses" instead of
"this level is far from market during a handoff".
"""

from grid import GridEngine


class _Ex:
    """Exchange mock with a controllable position (side/qty/entryPrice) and
    controllable failure points for the cancel/replace path."""

    class exchange:
        @staticmethod
        def amount_to_precision(s, a):
            return f"{float(a):.0f}"

        @staticmethod
        def price_to_precision(s, p):
            return f"{float(p):.5f}"

    def __init__(self, position_qty=0.0, position_side="long", entry_price=0.0):
        self.orders = {}
        self.cancelled = []
        self._n = 0
        self.price = 0.2200
        self.position_qty = position_qty
        self.position_side = position_side
        self.entry_price = entry_price
        self.fail_cancel = False
        self.fail_place_after_cancel = False
        # When set, fetch_order/get_open_order_ids report this id as filled
        # (closed, non-zero filled) instead of ordinary cancel-and-replace.
        self.report_filled_instead = None

    def get_price(self, s):
        return self.price

    def get_balance(self, s="USDT"):
        return 5000.0

    def get_positions(self, s):
        if self.position_qty <= 0:
            return []
        return [{
            "contracts": self.position_qty, "side": self.position_side,
            "entryPrice": self.entry_price,
        }]

    def get_open_orders(self, s):
        return list(self.orders.values())

    def get_open_order_ids(self, s):
        if self.report_filled_instead is not None:
            return {oid for oid in self.orders if oid != self.report_filled_instead}
        return set(self.orders)

    def can_place_order(self, s):
        return True

    def cancel_order(self, oid, s):
        if self.fail_cancel:
            return False
        self.cancelled.append(oid)
        self.orders.pop(oid, None)
        return True

    def place_limit_order(self, symbol, side, price, amount, max_attempts=3,
                           params=None, post_only=True, allow_taker_fallback=False,
                           purpose="gg"):
        if self.fail_place_after_cancel:
            raise RuntimeError("simulated network failure")
        self._n += 1
        oid = f"o{self._n}"
        self.orders[oid] = {
            "id": oid, "side": side, "price": float(price), "amount": float(amount),
            "status": "open", "params": dict(params or {}),
        }
        return self.orders[oid]

    def fetch_order(self, oid, s):
        if oid == self.report_filled_instead:
            return {"id": oid, "status": "closed", "filled": self.orders.get(oid, {}).get("amount", 1.0)}
        return self.orders.get(oid) or {"id": oid, "status": "closed"}


def _engine(position_qty=0.0, position_side="long", entry_price=0.0):
    g = GridEngine(exchange=_Ex(position_qty, position_side, entry_price), symbol="ADAUSDT",
                    grid_lower=0.20, grid_upper=0.24, grid_count=10,
                    capital_per_grid_pct=0.018, stop_loss_pct=0.05,
                    min_profit_multiplier=3.0, max_exposure_pct=0.5, leverage=5)
    g.initialize(0.2200, balance=5000.0)
    g.activate(5000.0)
    return g


def _rig_exit_level(g, side, price, qty):
    """Trim to a single resting exit level, isolating _refresh_break_even_exits
    from the rest of the ladder (its own neighbour-crowding/grid-bounds checks
    are exercised separately, not the point of these tests)."""
    level = g.levels[0]
    level.side, level.price, level.quantity = side, price, qty
    level.order_id = "exit-live"
    level.status = "pending"
    g.levels = [level]
    g.exchange.orders["exit-live"] = {
        "id": "exit-live", "side": side, "price": price, "amount": qty, "status": "open",
    }
    return level


def test_reprices_a_resting_exit_that_now_loses_against_a_fresh_break_even():
    """The live scenario: a long's real average entry has climbed to 0.2200 (more
    buys landed on the same side after this exit was first quoted), so the exit
    still resting at 0.2160 would now book a real loss if it filled."""
    g = _engine(position_qty=200.0, position_side="long", entry_price=0.2200)
    level = _rig_exit_level(g, "sell", 0.2160, 200.0)
    g.exchange.price = 0.2205

    g._refresh_break_even_exits(balance=5000.0)

    assert level.price > 0.2160, "exit stayed at a price that loses against the real entry"
    assert level.price >= 0.2200, "repriced level must clear the real break-even, not just improve"
    assert level.order_id is not None and level.order_id != "exit-live"
    assert "exit-live" in g.exchange.cancelled
    new_order = g.exchange.orders[level.order_id]
    assert new_order["side"] == "sell"
    assert new_order["params"].get("reduceOnly") is True


def test_leaves_a_still_safe_exit_alone():
    """The real entry (0.2150) is BELOW the resting exit (0.2160) -- it already
    clears break-even, so nothing should be touched."""
    g = _engine(position_qty=200.0, position_side="long", entry_price=0.2150)
    level = _rig_exit_level(g, "sell", 0.2160, 200.0)

    g._refresh_break_even_exits(balance=5000.0)

    assert level.price == 0.2160
    assert level.order_id == "exit-live"
    assert g.exchange.cancelled == []


def test_skips_cleanly_when_cancel_fails():
    g = _engine(position_qty=200.0, position_side="long", entry_price=0.2200)
    level = _rig_exit_level(g, "sell", 0.2160, 200.0)
    g.exchange.price = 0.2205
    g.exchange.fail_cancel = True

    g._refresh_break_even_exits(balance=5000.0)  # must not raise

    assert level.order_id == "exit-live"
    assert level.price == 0.2160
    assert g.exchange.orders["exit-live"]["status"] == "open"


def test_leaves_the_level_pending_when_replacement_placement_fails():
    g = _engine(position_qty=200.0, position_side="long", entry_price=0.2200)
    level = _rig_exit_level(g, "sell", 0.2160, 200.0)
    g.exchange.price = 0.2205
    g.exchange.fail_place_after_cancel = True

    g._refresh_break_even_exits(balance=5000.0)  # must not raise

    assert "exit-live" in g.exchange.cancelled
    assert level.order_id is None
    assert level.status == "pending"
    assert level.price > 0.2160, "the safe target price is kept even though placing it failed"


def test_ambiguous_missing_order_that_was_actually_a_fill_is_processed_as_a_fill():
    """Missing from the open-orders snapshot is not automatically a stale cancel --
    it might have filled in the same breath. That must go through _handle_fill,
    not be treated as free to cancel-and-reprice (AUDIT #150's rule)."""
    g = _engine(position_qty=200.0, position_side="long", entry_price=0.2200)
    level = _rig_exit_level(g, "sell", 0.2160, 200.0)
    g.exchange.price = 0.2205
    g.exchange.report_filled_instead = "exit-live"
    fills_before = g.total_fills

    g._refresh_break_even_exits(balance=5000.0)

    assert g.total_fills == fills_before + 1, "the fill was never processed"
    assert "exit-live" not in g.exchange.cancelled, "a genuine fill must not also be cancelled"


def test_leaves_the_level_alone_when_no_legal_price_exists():
    """_nearest_legal_exit returning None (e.g. break-even now sits outside the
    grid's own bounds) must not be treated as 'reprice to nothing' -- leave the
    level exactly as it was."""
    g = _engine(position_qty=200.0, position_side="long", entry_price=0.30)
    level = _rig_exit_level(g, "sell", 0.2160, 200.0)
    g.grid_lower, g.grid_upper = 0.20, 0.24  # break-even (~0.30) falls outside this

    g._refresh_break_even_exits(balance=5000.0)

    assert level.price == 0.2160
    assert level.order_id == "exit-live"
    assert g.exchange.cancelled == []
