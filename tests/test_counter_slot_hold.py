"""A rung must not double down when its exit is already resting. AUDIT #61.

#58 stopped a filled rung migrating onto an occupied counter-slot, where it could never
place again. But it re-armed the rung on its OWN side instead, which breaks the
alternation a grid depends on. With price sitting at the rung, it simply buys again.

Observed live on 2026-08-14, six consecutive fills at one price:

    09:42  FILL #8  BUY @ 0.06945  qty=2397
    09:43  FILL #9  BUY @ 0.06945  qty=2478   -> LONG 2181
    10:00  FILL #10 BUY @ 0.06945  qty=2463   -> LONG 4644
    10:37  FILL #11 BUY @ 0.06945  qty=2246   -> LONG 6890
    11:05  FILL #12 BUY @ 0.06945  qty=927    -> LONG 7817
    11:22  FILL #13 BUY @ 0.06945  qty=398    -> LONG 8215
                                    BUY SCALE | long=8215.0/8515.3 | scale=0.07

96% of the position cap, every fill booking profit=-0.000000 and paying a fee. The
occupying order IS the exit for the first fill; a second buy adds exposure with no
matching exit. The rung has to wait, then flip.
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


def _fill(engine, level):
    """Simulate `level`'s resting order filling."""
    engine.exchange.orders.pop(level.order_id, None)
    engine.check_fills(5000.0)


def test_a_rung_whose_exit_is_resting_does_not_buy_again():
    """The exact live sequence: repeatedly fill the same buy rung and confirm the
    engine stops re-buying once its counter-slot is occupied."""
    g = _engine()
    rung = max((l for l in g.levels if l.side == "buy" and l.order_id), key=lambda l: l.price)
    rung_price = rung.price

    buys_at_rung = 0
    for _ in range(6):
        live = [o for o in g.exchange.orders.values()
                if o["side"] == "buy" and abs(o["price"] - rung_price) < 1e-12]
        if not live:
            break
        buys_at_rung += 1
        g.exchange.orders.pop(live[0]["id"], None)
        g.exchange.price = rung_price
        g.check_fills(5000.0)
        g.place_initial_orders(5000.0)

    assert buys_at_rung <= 2, (
        f"the rung filled as a BUY {buys_at_rung} times in a row without ever flipping "
        f"to a sell -- that is the 8215 DOGE accumulation"
    )


def test_a_held_rung_places_nothing():
    g = _engine()
    rung = next(l for l in g.levels if l.order_id)
    rung.status = "awaiting_counter"
    rung.awaiting_side = "sell"
    rung.awaiting_price = 0.07035
    rung.order_id = None

    assert g._place_order_for_level(rung, 5000.0) is False


def test_a_held_rung_flips_to_the_counter_side_once_the_slot_frees():
    """The waiting is temporary. When the occupying exit fills, the rung becomes the
    replacement it was always meant to be."""
    g = _engine()
    # Clear the ladder so exactly one level occupies the awaited price -- otherwise a
    # sibling rung legitimately holding 0.07035 keeps the slot busy.
    for level in g.levels:
        level.order_id = None
        level.status = "pending"

    held = g.levels[0]
    occupier = g.levels[1]

    occupier.price = 0.07035
    occupier.order_id = "live"
    occupier.status = "pending"

    held.side = "buy"
    held.price = 0.07017
    held.order_id = None
    held.status = "awaiting_counter"
    held.awaiting_side = "sell"
    held.awaiting_price = 0.07035

    g._release_awaiting_levels()
    assert held.status == "awaiting_counter", "released while the slot was still taken"

    occupier.order_id = None            # the exit filled
    g._release_awaiting_levels()

    assert held.status == "pending"
    assert held.side == "sell" and held.price == 0.07035
    assert held.awaiting_price is None and held.awaiting_side is None


def test_holding_is_not_reported_as_a_failure():
    """A held rung is doing the right thing. Counting it as failed would read as the
    ladder being broken."""
    g = _engine()
    for level in g.levels:
        level.order_id = None
        level.status = "awaiting_counter"
        level.awaiting_side = "sell"
        level.awaiting_price = 0.09
        # keep the slot occupied so nothing releases
    g.levels[0].order_id = "live"
    g.levels[0].status = "pending"
    g.levels[0].price = 0.09

    assert g.place_initial_orders(5000.0) == 0


def test_the_hold_survives_a_restart():
    """Persisted, or a restart re-arms the rung and resumes doubling down."""
    import json

    g = _engine()
    held = g.levels[0]
    held.status = "awaiting_counter"
    held.awaiting_side = "sell"
    held.awaiting_price = 0.07035
    held.order_id = None

    revived = _engine()
    revived.load_from_dict(json.loads(json.dumps(g.to_dict())), current_price=0.07026)

    match = [l for l in revived.levels if l.status == "awaiting_counter"]
    assert match, "the held state was lost across a restart"
    assert match[0].awaiting_price == 0.07035
    assert match[0].awaiting_side == "sell"
