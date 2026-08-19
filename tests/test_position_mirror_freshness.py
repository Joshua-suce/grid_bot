"""An exit must be sized against the position that exists. AUDIT #98.

_net_long_qty/_net_short_qty decide how much a reduce-only order may close
(_exit_order_params clamps to them). They are written ONLY by set_position_limit, which
main.py calls at :1773 -- more than a hundred lines AFTER check_fills, in the same
iteration. So every order placed during the fill sweep is sized against the position as
it stood BEFORE the fill that triggered it, and the worst case is the common one: a
level's exit goes out in the same breath as its own entry fill.

Twice in the 2026-08-17 12:55 session:

    14:06:46  FILL #47 BUY 1782 @ 0.07012      position 302 -> 1789
    14:06:47  ORDER PLACED SELL 302 @ 0.07027  <- the pre-fill size
    14:50:30  FILL #49 SELL 302, profit +0.045377   (1778 was worth about +0.27)

    15:19:57  FILL #51 BUY 1778 @ 0.07027      position 1494 -> 3272
    15:19:58  ORDER PLACED SELL 1494 @ 0.07041 <- the pre-fill size again

The fix asks the exchange rather than incrementing a local count, and that is not
fastidiousness: the 1782 order had already put 295 into the position as a partial before
it completed (302 + 1487 = 1789, not 302 + 1782). An increment would have claimed 2084
against a real 1789, and a reduce-only order larger than the position is rejected -2022 --
turning an undersized exit into no exit at all.
"""

import pytest

from grid import GridEngine, GridLevel


class FakeInner:
    @staticmethod
    def price_to_precision(symbol, price):
        return f"{float(price):.5f}"

    @staticmethod
    def amount_to_precision(symbol, qty):
        return f"{float(qty):.0f}"


class FakeExchange:
    """Reports a position and records the params every order was placed with."""

    def __init__(self, positions=None, raises=False):
        self.exchange = FakeInner()
        self._positions = positions if positions is not None else []
        self.raises = raises
        self.placed = []
        self.position_reads = 0

    def get_positions(self, symbol):
        self.position_reads += 1
        if self.raises:
            raise RuntimeError("positions unreadable")
        return list(self._positions)

    def get_price(self, symbol):
        return 0.07020

    def place_limit_order(self, symbol, side, price, qty, params=None, max_attempts=1, post_only=True, allow_taker_fallback=False):
        self.placed.append({"side": side, "price": price, "qty": qty,
                            "params": dict(params or {})})
        return {"id": f"o{len(self.placed)}"}

    def get_open_order_ids(self, symbol):
        return set()

    def cancel_order(self, order_id, symbol):
        return True


def long_pos(qty, entry=0.07012):
    return [{"side": "long", "contracts": qty, "entryPrice": entry}]


def short_pos(qty, entry=0.07027):
    return [{"side": "short", "contracts": qty, "entryPrice": entry}]


def engine(ex, net_long=0.0, net_short=0.0):
    g = GridEngine.__new__(GridEngine)
    g.symbol = "DOGEUSDT"
    g.exchange = ex
    g.levels = []
    g._net_long_qty = net_long
    g._net_short_qty = net_short
    g._pos_qty = 0.0
    g._pos_entry = 0.0
    return g


# --- the mirror itself ----------------------------------------------------------------

def test_the_mirror_takes_the_exchanges_long():
    """The 14:06 state: the mirror still holds the pre-fill 302, the exchange holds 1789."""
    ex = FakeExchange(long_pos(1789.0))
    g = engine(ex, net_long=302.0)

    assert g._refresh_net_counters() is True
    assert g._net_long_qty == 1789.0
    assert g._net_short_qty == 0.0


def test_the_exit_is_then_sized_against_the_real_position():
    """The consequence, stated as the order that actually goes out. 302 was booked at
    +0.045377 when it filled; the same crossing at 1778 was worth about +0.27."""
    ex = FakeExchange(long_pos(1789.0))
    g = engine(ex, net_long=302.0)

    stale_params, stale_qty = g._exit_order_params("sell", 1778.0)
    g._refresh_net_counters()
    fresh_params, fresh_qty = g._exit_order_params("sell", 1778.0)

    assert stale_qty == 302.0, "fixture no longer reproduces the live undersizing"
    assert fresh_qty == 1778.0
    assert stale_params["reduceOnly"] is True and fresh_params["reduceOnly"] is True


def test_a_short_is_not_read_as_a_long():
    """One-way mode spells a short either way. Reading side='short' as a long would send
    reduceOnly on the wrong side, which the exchange refuses outright."""
    for positions in (short_pos(5342.0), [{"side": "long", "contracts": -5342.0,
                                           "entryPrice": 0.07027}]):
        g = engine(FakeExchange(positions))
        g._refresh_net_counters()

        assert (g._net_long_qty, g._net_short_qty) == (0.0, 5342.0), positions
        assert g._reduce_only_qty("buy") == 5342.0
        assert g._reduce_only_qty("sell") == 0.0


def test_a_flat_account_clears_the_mirror():
    """Otherwise the next order goes out reduceOnly against nothing (-2022) instead of
    opening the position the ladder wants."""
    g = engine(FakeExchange([]), net_long=1789.0)

    g._refresh_net_counters()

    assert (g._net_long_qty, g._net_short_qty) == (0.0, 0.0)
    assert g._exit_order_params("sell", 500.0) == (None, 500.0), "still marked reduceOnly"


def test_a_shrunken_position_is_also_corrected():
    """The mirror can be stale in the dangerous direction too: an exit filled earlier in
    the same sweep leaves it OVER-stating, and a reduce-only order bigger than the
    position is rejected -2022. The same refresh fixes both directions."""
    ex = FakeExchange(long_pos(11.0))
    g = engine(ex, net_long=1789.0)

    g._refresh_net_counters()

    assert g._exit_order_params("sell", 1778.0) == ({"reduceOnly": True, "postOnly": False}, 11.0)


# --- it must never be the thing that stops a fill --------------------------------------

def test_an_unreadable_position_keeps_the_previous_mirror():
    """A failed read degrades to exactly what this code used before the refresh existed.
    Zeroing the mirror instead would send the exit out WITHOUT reduceOnly and open
    exposure the ladder has no exit for."""
    g = engine(FakeExchange(raises=True), net_long=302.0)

    assert g._refresh_net_counters() is False
    assert g._net_long_qty == 302.0


def test_a_position_payload_without_contracts_is_ignored():
    g = engine(FakeExchange([{"side": "long"}, {"side": "long", "contracts": 0}]),
               net_long=5.0)

    g._refresh_net_counters()

    assert (g._net_long_qty, g._net_short_qty) == (0.0, 0.0)


# --- wired into the fill path ----------------------------------------------------------

def test_handle_fill_refreshes_before_it_places_the_replacement():
    """The refresh is only worth anything if it happens BEFORE _handle_fill sizes and
    places the exit -- which it does itself, a hundred lines further down the method."""
    import inspect

    src = inspect.getsource(GridEngine._handle_fill)

    assert "_refresh_net_counters()" in src, "_handle_fill no longer refreshes the mirror"
    assert src.index("_refresh_net_counters()") < src.index("_place_order_for_level"), (
        "the mirror is refreshed after the replacement order has already been sized"
    )
    assert src.index("_apply_to_position") < src.index("_refresh_net_counters()"), (
        "refreshed before the fill was applied — the P&L ledger would disagree with it"
    )


def test_set_position_limit_still_owns_the_mirror_between_fills():
    """The refresh supplements the once-per-poll write, it does not replace it: a poll
    with no fills must still pick up a position the stop legs moved."""
    g = engine(FakeExchange(long_pos(500.0)), net_long=0.0)

    g._block_buys = g._block_sells = False
    g._buy_scale = g._sell_scale = 1.0
    g.set_position_limit(500.0, 0.0, 10_000.0)

    assert g._net_long_qty == 500.0


def test_the_refresh_costs_one_read_per_fill():
    """It runs on the fill path, so its cost has to stay proportional to fills (11 in
    ~640 iterations that session), not to polls."""
    ex = FakeExchange(long_pos(1789.0))
    g = engine(ex, net_long=302.0)

    for _ in range(3):
        g._refresh_net_counters()

    assert ex.position_reads == 3
