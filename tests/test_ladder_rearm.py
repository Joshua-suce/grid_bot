"""A filled rung has to come back. AUDIT #58.

Observed live on 2026-08-14, in a 68-minute DEMO run:

    01:01:03  PLACED 10 initial grid orders
    01:11:14  FILL #1 | BUY @ 0.07017
    01:11:14  SKIP REPLACEMENT | level at 0.07035 already occupied by active order
    ...       (no ORDER PLACED for the next 57 minutes)
    02:08:02  BATCH CANCELLED 9 orders

Ten orders placed, one filled, nine cancelled at shutdown -- the replacement was never
made. The buy's counter-sell targeted 0.07035, where the initial ladder's own sell was
already resting, so the level was MOVED onto that price. From there
`_place_order_for_level` finds an order already tracked by another level, returns False
and logs at DEBUG. The rung retries forever and never places.

The ladder therefore ran on 9 of 10 rungs, with the hole at 0.07017 -- the rung NEAREST
the price, the one most likely to fill next. `_refill_missing_grid_lines` would repair
it, but it only runs on state-load and reset, never in the live loop.
"""

import pytest

from grid import GridEngine


class _Ex:
    """Tracks the live book so an occupied slot behaves like the real thing."""

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


def test_a_fill_whose_slot_is_taken_re_arms_its_own_rung():
    """The level must keep its own price and side, not migrate onto the occupant."""
    g = _engine()
    buy = max((l for l in g.levels if l.side == "buy" and l.order_id), key=lambda l: l.price)
    price_before, side_before = buy.price, buy.side

    # Make the counter-slot occupied, then fill this level.
    g.exchange.orders.pop(buy.order_id, None)
    g.check_fills(5000.0)

    if buy.order_id is None:                      # replacement was skipped
        assert buy.price == price_before, (
            f"level migrated from {price_before} to {buy.price} -- onto the occupied "
            f"slot, where it can never place again"
        )
        assert buy.side == side_before


def test_the_ladder_keeps_all_its_rungs_working_after_a_fill():
    """The end-to-end symptom: 10 rungs in, one fills, and the book must not be left
    permanently one order short."""
    g = _engine()
    assert len(g.exchange.orders) == 10

    target = max((l for l in g.levels if l.side == "buy" and l.order_id), key=lambda l: l.price)
    g.exchange.orders.pop(target.order_id, None)
    g.exchange.price = target.price

    for _ in range(3):                            # a few loop iterations
        g.check_fills(5000.0)
        g.place_initial_orders(5000.0)

    live = len(g.exchange.orders)
    assert live >= 10, (
        f"ladder is running on {live} orders after one fill -- a rung was stranded. "
        f"That is exactly the 9-of-10 the live run showed for 57 minutes."
    )


def test_no_two_levels_claim_the_same_order():
    """The stranding is only possible because two levels ended up on one price. If that
    cannot happen, the dead-end cannot either."""
    g = _engine()
    target = max((l for l in g.levels if l.side == "buy" and l.order_id), key=lambda l: l.price)
    g.exchange.orders.pop(target.order_id, None)

    for _ in range(3):
        g.check_fills(5000.0)
        g.place_initial_orders(5000.0)

    claimed = [l.order_id for l in g.levels if l.order_id is not None]
    assert len(claimed) == len(set(claimed)), f"two levels claim one order: {claimed}"


def test_a_stranded_rung_is_reported_not_swallowed():
    """Defense in depth: if a level ever does end up unable to place, it must say so at
    a level the operator actually sees, and say it once."""
    from loguru import logger

    g = _engine()
    a, b = [l for l in g.levels if l.order_id][:2]
    b.price = a.price
    b.side = a.side
    b.order_id = None
    b.status = "pending"

    sink = []
    logger.remove()
    handle = logger.add(lambda m: sink.append(str(m)), level="INFO")
    try:
        for _ in range(5):
            g._place_order_for_level(b, 5000.0)
    finally:
        logger.remove(handle)
        import sys
        logger.add(sys.stderr, level="INFO")

    out = "".join(sink)
    assert "LEVEL STRANDED" in out, f"a rung that can never place said nothing:\n{out}"
    assert out.count("LEVEL STRANDED") == 1, (
        f"logged {out.count('LEVEL STRANDED')} times over 5 attempts -- that is the #42 "
        f"mistake of 1,300 identical lines"
    )
