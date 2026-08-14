"""Held rungs must come back, but never at the price they just filled. AUDIT #62.

Two live failures, opposite directions, same subsystem:

  #58  A filled rung re-armed on its own side immediately. Price sat ON the rung and it
       refilled six times: 0 -> 8215 DOGE, 96% of the position cap, at one price.

  #61  So the rung was made to wait for its counter-slot instead. But in a ladder of
       buys below and sells above, EVERY fill's counter-target is another live rung, so
       held rungs waited on held rungs. Live 2026-08-14: `PLACED 10 initial grid orders`
       at 12:06, `BATCH CANCELLED 5 orders` at 17:34. The book drained by half.

The gate that separates them: a rung may come back, but only once price has moved a
full spacing clear of it.

Plus the collision that run also exposed:

    16:20:51  COUNTER SLOT FREED | BUY 0.06922 -> SELL 0.06955
    16:20:52  COUNTER SLOT FREED | BUY 0.06939 -> SELL 0.06955

Two rungs onto one price, because a just-released level has order_id None and the
occupancy check only looked at live orders.
"""

import pytest

from grid import GridEngine


class _Ex:
    class exchange:
        @staticmethod
        def amount_to_precision(s, a): return f"{float(a):.0f}"
        @staticmethod
        def price_to_precision(s, p): return f"{float(p):.5f}"

    def __init__(self):
        self.orders = {}
        self._n = 0
        self.price = 0.07026

    def get_price(self, s): return self.price
    def get_balance(self, s="USDT"): return 5000.0
    def get_positions(self, s): return []
    def get_open_orders(self, s): return list(self.orders.values())
    def get_open_order_ids(self, s): return set(self.orders)
    def can_place_order(self, s): return True
    def cancel_order(self, oid, s):
        self.orders.pop(oid, None)
        return True

    def place_limit_order(self, symbol, side, price, amount, max_attempts=3,
                          params=None, post_only=True, allow_taker_fallback=False):
        self._n += 1
        oid = f"o{self._n}"
        self.orders[oid] = {"id": oid, "side": side, "price": float(price),
                            "amount": float(amount), "status": "open"}
        return self.orders[oid]

    def fetch_order(self, oid, s):
        return self.orders.get(oid) or {"id": oid, "status": "closed"}


def _engine():
    g = GridEngine(exchange=_Ex(), symbol="DOGEUSDT", grid_lower=0.06936,
                   grid_upper=0.07116, grid_count=10, capital_per_grid_pct=0.018,
                   stop_loss_pct=0.03, min_profit_multiplier=3.0,
                   max_exposure_pct=0.5, leverage=5)
    g.initialize(0.07026, balance=5000.0)
    g.activate(5000.0)
    g.place_initial_orders(5000.0)
    return g


def _hold(level, side, price):
    level.order_id = None
    level.status = "awaiting_counter"
    level.awaiting_side = side
    level.awaiting_price = price


# --- the collision -----------------------------------------------------------

def test_two_held_rungs_never_release_onto_the_same_price():
    """The live sequence: BUY 0.06922 and BUY 0.06939 both freed to SELL 0.06955."""
    g = _engine()
    for level in g.levels:
        level.order_id = None
        level.status = "pending"

    a, b = g.levels[0], g.levels[1]
    a.side = b.side = "buy"
    a.price, b.price = 0.06922, 0.06939
    _hold(a, "sell", 0.06955)
    _hold(b, "sell", 0.06955)

    g._release_awaiting_levels(current_price=0.06950)

    released = [l for l in (a, b) if l.status == "pending"]
    assert len(released) == 1, (
        f"{len(released)} rungs released onto 0.06955 — they collide and one will "
        f"strand as a duplicate"
    )
    assert sum(1 for l in (a, b) if l.price == 0.06955 and l.status == "pending") == 1


# --- the starvation ----------------------------------------------------------

def test_a_held_rung_re_arms_in_place_once_price_clears_it():
    """#61's fix left rungs waiting forever on counters that were themselves held.
    Price moving clear of the rung has to be enough to bring it back."""
    g = _engine()
    level = g.levels[0]
    level.side = "buy"
    level.price = 0.06939
    _hold(level, "sell", 0.06955)

    # counter-slot deliberately kept busy by a live sibling
    other = g.levels[1]
    other.side, other.price, other.order_id, other.status = "sell", 0.06955, "live", "pending"

    g._release_awaiting_levels(current_price=0.06939 + g.grid_spacing)

    assert level.status == "pending", "rung stayed held even though price had cleared it"
    assert level.side == "buy" and level.price == 0.06939, "it should re-arm in PLACE"
    assert level.awaiting_price is None


def test_a_held_rung_does_not_re_arm_at_the_price_it_just_filled():
    """#58's defect: price sitting on the rung, refilling forever."""
    g = _engine()
    level = g.levels[0]
    level.side = "buy"
    level.price = 0.06945
    _hold(level, "sell", 0.06961)

    other = g.levels[1]
    other.side, other.price, other.order_id, other.status = "sell", 0.06961, "live", "pending"

    for price in (0.06945, 0.06944, 0.06946):        # never a full spacing clear
        g._release_awaiting_levels(current_price=price)
        assert level.status == "awaiting_counter", (
            f"rung re-armed at {price} with its fill price 0.06945 — that is the "
            f"8215 DOGE accumulation"
        )


def test_a_sell_rung_needs_price_to_fall_clear():
    """Mirror of the buy case: a sell rung rests above market."""
    g = _engine()
    level = g.levels[0]
    level.side = "sell"
    level.price = 0.07005
    _hold(level, "buy", 0.06989)

    other = g.levels[1]
    other.side, other.price, other.order_id, other.status = "buy", 0.06989, "live", "pending"

    g._release_awaiting_levels(current_price=0.07005)
    assert level.status == "awaiting_counter"

    g._release_awaiting_levels(current_price=0.07005 - g.grid_spacing)
    assert level.status == "pending" and level.side == "sell"


def test_the_counter_flip_still_wins_when_the_slot_is_free():
    """Flipping to the counter remains the preferred outcome -- that is the cycle."""
    g = _engine()
    for level in g.levels:
        level.order_id = None
        level.status = "pending"

    level = g.levels[0]
    level.side = "buy"
    level.price = 0.06939
    _hold(level, "sell", 0.06955)

    g._release_awaiting_levels(current_price=0.06939 + g.grid_spacing * 5)

    assert level.side == "sell" and level.price == 0.06955, (
        "a free counter-slot should produce the flip, not an in-place re-arm"
    )


def test_no_price_means_counter_only_never_a_guess():
    g = _engine()
    level = g.levels[0]
    level.side = "buy"
    level.price = 0.06939
    _hold(level, "sell", 0.06955)
    other = g.levels[1]
    other.side, other.price, other.order_id, other.status = "sell", 0.06955, "live", "pending"

    g._release_awaiting_levels(current_price=None)

    assert level.status == "awaiting_counter"


# --- the end-to-end symptom --------------------------------------------------

def test_the_ladder_does_not_drain_as_rungs_fill():
    """10 orders in. Fill rungs and walk price around; the book must refill, not halve."""
    g = _engine()
    assert len(g.exchange.orders) == 10

    for step, price in enumerate([0.06990, 0.07060, 0.06990, 0.07060, 0.06990]):
        live = list(g.exchange.orders.values())
        if live:
            victim = live[0]
            g.exchange.orders.pop(victim["id"], None)
        g.exchange.price = price
        g.check_fills(5000.0)
        g.place_initial_orders(5000.0)

    live = len(g.exchange.orders)
    assert live >= 7, (
        f"ladder drained to {live} of 10 orders — that is the 10->5 the live run showed"
    )
