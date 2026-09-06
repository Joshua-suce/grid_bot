"""An exit order must grow with the position it is covering. AUDIT #173.

REPLACEMENT SLOT TAKEN (AUDIT #61) correctly stops a rung from doubling down when its
counter-slot is already live -- but it only ever recorded that the fill was WAITING on
that order. It never touched the order itself. Live on 2026-09-05, ADAUSDT, one session:

    06:34:32  RUNG FLIPPED | SELL 0.2114 -> BUY -- places BUY 229 @ 0.2114
    07:34:52  FILL #141 SELL @ 0.2138 | REPLACEMENT SLOT TAKEN | BUY @ 0.2114 is this
              fill's exit -- holding                                  (SHORT 458 -> 685)
    09:03:45  FILL #142 SELL @ 0.2138 | REPLACEMENT SLOT TAKEN | BUY @ 0.2114 ... again
                                                                        (SHORT 685 -> 841)

Both fills named the SAME BUY @ 0.2114 as "this fill's exit". That order was still
sized at 229 -- what it was given hours earlier -- while the short it was nominally
covering had tripled to 841. Even in the best case (price falling all the way back to
0.2114), that order would have closed barely a quarter of the position.

The fix is narrow: when a fill finds its counter-slot occupied, grow the occupying
order by this fill's own quantity (capped at what is actually still closable), so the
one thing a fill's exit needs -- to be sized for what it just added -- stays true. It
does not touch _release_awaiting_levels's own re-arm choice (AUDIT #61/#62), which has
its own tested tradeoffs.
"""

from grid import GridEngine


class _Ex:
    """Like the other REPLACEMENT SLOT / rung tests' mock, but get_positions actually
    reports a position, so _refresh_net_counters (and therefore _grow_occupied_exit) has
    something real to compute against instead of silently no-op'ing on an empty book."""

    class exchange:
        @staticmethod
        def amount_to_precision(s, a):
            return f"{float(a):.0f}"

        @staticmethod
        def price_to_precision(s, p):
            return f"{float(p):.5f}"

    def __init__(self, position_qty=0.0, position_side="short"):
        self.orders = {}
        self.cancelled = []
        self._n = 0
        self.price = 0.07026
        self.position_qty = position_qty
        self.position_side = position_side
        self.fail_cancel = False
        self.fail_place_after_cancel = False

    def get_price(self, s):
        return self.price

    def get_balance(self, s="USDT"):
        return 5000.0

    def get_positions(self, s):
        if self.position_qty <= 0:
            return []
        return [{"contracts": self.position_qty, "side": self.position_side}]

    def get_open_orders(self, s):
        return list(self.orders.values())

    def get_open_order_ids(self, s):
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
        return self.orders.get(oid) or {"id": oid, "status": "closed"}


def _engine(position_qty=0.0, position_side="short"):
    g = GridEngine(exchange=_Ex(position_qty, position_side), symbol="DOGEUSDT",
                    grid_lower=0.06936, grid_upper=0.07116, grid_count=10,
                    capital_per_grid_pct=0.018, stop_loss_pct=0.03,
                    min_profit_multiplier=3.0, max_exposure_pct=0.5, leverage=5)
    g.initialize(0.07026, balance=5000.0)
    g.activate(5000.0)
    for level in g.levels:
        level.order_id = None
        level.status = "pending"
    return g


def _rig_occupied_fill(g, occ_qty, fill_qty, spacing=None):
    """Build the exact live shape: a live BUY exit (`occ`) and a SELL rung one spacing
    above it that is about to fill for the first time, whose natural counter snaps
    straight onto occ's price.

    Trimmed to exactly these two levels: the dynamic grid's other rungs can otherwise
    sit closer to the snap target than occ does, so _handle_fill's snap-to-nearest-line
    search picks one of THEM instead of occ and the "occupied" branch this test exists
    to exercise never fires.
    """
    spacing = g.grid_spacing if spacing is None else spacing
    occ, filling = g.levels[0], g.levels[1]
    g.levels = [occ, filling]

    occ.side, occ.price, occ.quantity = "buy", 0.07000, occ_qty
    occ.order_id = "occ-live"
    occ.status = "pending"
    g.exchange.orders["occ-live"] = {
        "id": "occ-live", "side": "buy", "price": occ.price, "amount": occ_qty,
        "status": "open",
    }

    filling.side, filling.price, filling.quantity = "sell", occ.price + spacing, fill_qty
    filling.order_id = "filling-live"
    filling.status = "pending"
    filling.fill_count = 0

    return occ, filling


def test_a_growing_short_grows_its_resting_exit():
    """The exact live shape: a sell fills, its counter (a buy) is already resting, and
    the exchange now reports the position that fill just grew. The occupying buy must
    come back sized for the WHOLE position, not just its own original share."""
    g = _engine(position_qty=451.0, position_side="short")
    occ, filling = _rig_occupied_fill(g, occ_qty=229.0, fill_qty=222.0)

    g._handle_fill(filling, balance=5000.0)

    assert occ.quantity == 451.0, (
        f"exit stayed at {occ.quantity} while the short it covers is 451 -- it would "
        f"close only {occ.quantity / 451.0:.0%} of the position if it filled"
    )
    assert occ.order_id is not None and occ.order_id != "occ-live", (
        "the resized order was never tracked back onto the level"
    )
    assert "occ-live" in g.exchange.cancelled, "the old, undersized order was never cancelled"
    new_order = g.exchange.orders[occ.order_id]
    assert new_order["side"] == "buy" and new_order["price"] == 0.07000
    assert new_order["amount"] == 451.0
    assert new_order["params"].get("reduceOnly") is True, "grown exit must stay reduce-only"


def test_the_filled_level_still_goes_to_awaiting_counter_exactly_as_before():
    """AUDIT #173 must not change what happens to the level that just filled -- only
    what happens to the order it named as its exit. #61/#62's contract for THIS level
    is unaffected."""
    g = _engine(position_qty=451.0, position_side="short")
    occ, filling = _rig_occupied_fill(g, occ_qty=229.0, fill_qty=222.0)

    g._handle_fill(filling, balance=5000.0)

    assert filling.status == "awaiting_counter"
    assert filling.order_id is None
    assert filling.awaiting_side == "buy"
    assert filling.awaiting_price == 0.07000


def test_no_resize_when_nothing_more_is_closable():
    """If other resting exits (or a stale position read) already claim the whole
    position, growing occ further would over-commit reduce-only capacity. Leave it."""
    g = _engine(position_qty=229.0, position_side="short")  # no room beyond occ's own 229
    occ, filling = _rig_occupied_fill(g, occ_qty=229.0, fill_qty=222.0)

    g._handle_fill(filling, balance=5000.0)

    assert occ.quantity == 229.0, "resized an exit with no more position left to cover"
    assert occ.order_id == "occ-live", "cancelled a correctly-sized order for no reason"
    assert g.exchange.cancelled == []


def test_resize_skips_cleanly_when_cancel_fails():
    """A failed cancel must not leave two orders resting at the same price+side, and
    must not raise out of the fill handler."""
    g = _engine(position_qty=451.0, position_side="short")
    occ, filling = _rig_occupied_fill(g, occ_qty=229.0, fill_qty=222.0)
    g.exchange.fail_cancel = True

    g._handle_fill(filling, balance=5000.0)  # must not raise

    assert occ.order_id == "occ-live", "order_id lost even though the cancel never happened"
    assert occ.quantity == 229.0
    assert g.exchange.orders["occ-live"]["status"] == "open"


def test_resize_failure_after_cancel_leaves_the_level_pending_for_reconcile():
    """The one case where occ ends up temporarily unprotected: cancel succeeds, the
    replacement placement then fails. It must not raise, and it must mark occ recoverable
    (pending, no order_id) rather than silently pretending the old order still exists."""
    g = _engine(position_qty=451.0, position_side="short")
    occ, filling = _rig_occupied_fill(g, occ_qty=229.0, fill_qty=222.0)
    g.exchange.fail_place_after_cancel = True

    g._handle_fill(filling, balance=5000.0)  # must not raise

    assert "occ-live" in g.exchange.cancelled
    assert occ.order_id is None
    assert occ.status == "pending"


def test_repeated_fills_against_the_same_exit_keep_it_sized_to_the_whole_position():
    """The live incident, end to end: several fills in a row all naming the same
    occupied slot as their exit. Its size must track the running total, not just the
    most recent fill."""
    g = _engine(position_qty=229.0, position_side="short")
    occ, filling = _rig_occupied_fill(g, occ_qty=229.0, fill_qty=222.0)

    running = 229.0
    for fill_qty in (222.0, 156.0, 98.0):
        previous_order_id = occ.order_id
        g.exchange.position_qty = running + fill_qty
        filling.side, filling.price, filling.quantity = "sell", occ.price + g.grid_spacing, fill_qty
        filling.order_id, filling.status, filling.fill_count = "filling-live", "pending", 0
        g.exchange.orders["filling-live"] = {
            "id": "filling-live", "side": "sell", "price": filling.price,
            "amount": fill_qty, "status": "open",
        }

        g._handle_fill(filling, balance=5000.0)

        running += fill_qty
        assert occ.quantity == running, (
            f"after fill of {fill_qty}, exit is {occ.quantity} but the position is "
            f"{running} -- it fell behind again"
        )
        assert occ.order_id is not None and occ.order_id != previous_order_id, (
            "the exit's order_id did not change even though it was resized"
        )
