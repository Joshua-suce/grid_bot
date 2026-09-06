"""When the position's real cost basis drifts past a shared exit's frozen price, that
exit must move to protect its margin, not just its size. AUDIT #174.

_grow_occupied_exit (AUDIT #173) grows the SIZE of an occupied exit as more fills
stack behind it, but leaves its PRICE exactly where it was for the first fill that
claimed it. Binance nets every fill into one blended average entry, and each fill
that stacks onto the same occupied slot afterward can move that average to either
side of the occupied exit's frozen price. Live on 2026-09-06, ADAUSDT: a run of sells
kept naming the same BUY @ 0.2188 as their exit while the short's real average entry
drifted through it, and the resulting closes came back from Binance's own income
ledger at a few cents each -- some negative -- not because the strategy called price
wrong, but because the frozen exit price was, by the time it filled, on the wrong
side of the position it was meant to close at a profit. A run of those is
indistinguishable from a run of genuine losers to the streak-based kill switch in
risk.py, which shut the grid down for 4 hours twice in one session over what was
mostly this.

The fix only ever moves the occupying order to a MORE conservative price -- further
from market, requiring a better fill before it executes. It never loosens toward a
worse price, so it needs no exposure/risk-threshold changes: it can only delay a fill
in exchange for a real margin.
"""

from grid import GridEngine


class _Ex:
    """Like AUDIT #173's mock, but get_positions also reports an entryPrice, so
    _grow_occupied_exit's reprice math has a real cost basis to work against."""

    class exchange:
        @staticmethod
        def amount_to_precision(s, a):
            return f"{float(a):.0f}"

        @staticmethod
        def price_to_precision(s, p):
            return f"{float(p):.5f}"

    def __init__(self, position_qty=0.0, position_side="short", entry_price=0.0):
        self.orders = {}
        self.cancelled = []
        self._n = 0
        self.price = 0.07026
        self.position_qty = position_qty
        self.position_side = position_side
        self.entry_price = entry_price
        self.fail_cancel = False
        self.fail_place_after_cancel = False

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


def _engine(position_qty=0.0, position_side="short", entry_price=0.0):
    g = GridEngine(exchange=_Ex(position_qty, position_side, entry_price), symbol="DOGEUSDT",
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
    """Same trimmed two-level shape as AUDIT #173's tests -- the dynamic grid's other
    rungs can otherwise sit closer to the snap target than occ does, so _handle_fill's
    snap-to-nearest-line search picks one of THEM instead of occ and the "occupied"
    branch this test exists to exercise never fires."""
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


def test_reprices_down_when_the_real_entry_has_drifted_below_the_frozen_price():
    """The live incident: the short's real (exchange-reported) average entry has
    drifted to within a spacing of occ's frozen price -- buying back there would
    barely profit, or lose money outright. The exit must move to guarantee its
    margin, in the same cancel-and-replace that grows its size."""
    entry_price = 0.06985
    g = _engine(position_qty=451.0, position_side="short", entry_price=entry_price)
    occ, filling = _rig_occupied_fill(g, occ_qty=229.0, fill_qty=222.0)

    expected_price = g._round_price(entry_price - g.grid_spacing)
    assert expected_price < 0.07000, "test setup must target a price below occ's original"

    g._handle_fill(filling, balance=5000.0)

    assert occ.price == expected_price, (
        f"exit stayed at 0.07000 even though the real entry {entry_price} means buying "
        f"back there would barely profit -- or lose -- instead of a full spacing"
    )
    assert occ.quantity == 451.0
    new_order = g.exchange.orders[occ.order_id]
    assert new_order["price"] == expected_price
    assert new_order["amount"] == 451.0
    assert new_order["params"].get("reduceOnly") is True
    assert "occ-live" in g.exchange.cancelled


def test_does_not_loosen_when_the_current_price_already_beats_the_real_entry():
    """occ already sits at a price that guarantees MORE than a spacing of margin over
    the real entry -- e.g. later fills raised the short's average. Moving it would
    only give back margin that is already locked in, so price must be left alone
    (quantity still grows -- only the price decision is at stake here)."""
    entry_price = 0.07100
    g = _engine(position_qty=451.0, position_side="short", entry_price=entry_price)
    occ, filling = _rig_occupied_fill(g, occ_qty=229.0, fill_qty=222.0)

    ideal = g._round_price(entry_price - g.grid_spacing)
    assert ideal > 0.07000, "test setup must target a price above occ's original"

    g._handle_fill(filling, balance=5000.0)

    assert occ.price == 0.07000, "a correctly-conservative exit was loosened"
    assert occ.quantity == 451.0


def test_reprices_even_when_nothing_more_is_closable_to_grow():
    """No room to grow the size (occ already covers the whole closable position), but
    the real entry has drifted unfavourably -- the price must still move on its own."""
    entry_price = 0.06985
    g = _engine(position_qty=229.0, position_side="short", entry_price=entry_price)
    occ, filling = _rig_occupied_fill(g, occ_qty=229.0, fill_qty=222.0)

    expected_price = g._round_price(entry_price - g.grid_spacing)

    g._handle_fill(filling, balance=5000.0)

    assert occ.price == expected_price
    assert occ.quantity == 229.0, "quantity must not change -- nothing more was closable"
    assert "occ-live" in g.exchange.cancelled


def test_skips_reprice_when_the_exchange_reports_no_entry_price():
    """0.0 means unknown, never 'entered at zero' -- growth (AUDIT #173) still works
    with no entry data, but there is nothing to reprice against."""
    g = _engine(position_qty=451.0, position_side="short", entry_price=0.0)
    occ, filling = _rig_occupied_fill(g, occ_qty=229.0, fill_qty=222.0)

    g._handle_fill(filling, balance=5000.0)

    assert occ.price == 0.07000, "repriced against an entry the exchange never reported"
    assert occ.quantity == 451.0


def test_does_not_reprice_outside_the_grid_bounds():
    """An entry so far away that a full-spacing-conservative target would fall
    outside the grid's own range -- leave occ where it was rather than quote
    somewhere the grid was never configured to trade. Growth is unaffected."""
    g = _engine(position_qty=451.0, position_side="short", entry_price=0.02)
    occ, filling = _rig_occupied_fill(g, occ_qty=229.0, fill_qty=222.0)

    ideal = g._round_price(0.02 - g.grid_spacing)
    assert ideal < g.grid_lower, "test setup must target a price below grid_lower"

    g._handle_fill(filling, balance=5000.0)

    assert occ.price == 0.07000
    assert occ.quantity == 451.0
