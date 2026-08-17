"""The position cap must actually cap. AUDIT #49.

`_handle_fill` places the paired replacement order by calling `place_limit_order`
directly. Every guard -- `_block_buys`/`_block_sells`, the `_buy_scale`/`_sell_scale`
taper, the minimum-notional check -- lives in `_place_order_for_level`, which that path
skips. The cap was therefore advisory: each fill armed another order regardless of it,
that order filled, and its replacement did the same.

On 2026-08-08 the position reached 31,761 DOGE against a 17,467 cap, while the log was
concurrently printing `BUY SCALE | long=11812.0/17476.7 | scale=0.65` as though the
limit were being enforced. That day realised -48.92 on six closes: 60% of the fortnight's
entire loss.

Exits must stay unconditional -- refusing those traps inventory, which is #42.
"""

import pytest

from grid import GridEngine, MIN_NOTIONAL_USDT


class _Ex:
    def __init__(self):
        self.placed = []
        # _handle_fill re-reads the net position before it sizes the replacement order
        # (AUDIT #98), so a test that wants the engine to believe a position is open has
        # to open it HERE as well. A mirror the exchange contradicts is not a state the
        # bot can be in for longer than one fill.
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
    def get_open_order_ids(self, s): return set()
    def fetch_order(self, i, s): return None
    def cancel_order(self, i, s): return True
    def cancel_everything(self, s, timeout_seconds=300.0): return 0
    def can_place_order(self, s): return True
    def close_position(self, symbol, side, amount, max_attempts=None): return {"id": "c"}
    def place_limit_order(self, symbol, side, price, amount, max_attempts=3, params=None,
                          post_only=True, allow_taker_fallback=False):
        self.placed.append((side, price, amount, (params or {}).get("reduceOnly", False)))
        return {"id": f"o{len(self.placed)}"}


def _engine():
    ex = _Ex()
    g = GridEngine(exchange=ex, symbol="DOGEUSDT", grid_lower=0.0690, grid_upper=0.0710,
                   grid_count=10, capital_per_grid_pct=0.018, stop_loss_pct=0.03,
                   min_profit_multiplier=3.0, max_exposure_pct=0.5, leverage=5)
    g.initialize(0.0700, balance=5000.0)
    return g, ex


def _fill(g, level, balance=5000.0):
    return g._handle_fill(level, balance)


def test_a_capped_grid_stops_adding_exposure():
    """The regression. A SELL fill arms a BUY replacement, so blocking buys is what
    must stop it -- the sides flip across a fill, which is why this bypass was easy
    to miss."""
    g, ex = _engine()
    g.set_position_limit(long_position=20000.0, short_position=0.0, max_position_qty=17467.0)
    assert g._block_buys, "precondition: the cap should have blocked buys"

    sell = next(l for l in g.levels if l.side == "sell")
    before = len(ex.placed)
    _fill(g, sell)

    added = [o for o in ex.placed[before:] if o[0] == "buy" and not o[3]]
    assert added == [], (
        f"placed {added} while buys were blocked -- the cap is advisory and the "
        f"position can ratchet past it, exactly as it did on 2026-08-08"
    )


def test_the_blocked_level_is_left_pending_for_retry():
    """Blocked is not broken: the level must come back when the cap clears."""
    g, ex = _engine()
    g.set_position_limit(20000.0, 0.0, 17467.0)
    sell = next(l for l in g.levels if l.side == "sell")

    _fill(g, sell)

    assert sell.order_id is None and sell.status == "pending"


def test_exits_are_never_blocked():
    """#42's lesson: refusing exit orders traps inventory. A reduceOnly replacement
    must go out even with that side blocked."""
    g, ex = _engine()
    # A short is open, so a BUY reduces it. The exchange has to say so too: the engine
    # re-reads the position before sizing the exit, and a BUY against an account that is
    # genuinely flat is not an exit at all -- blocking that one is correct (AUDIT #98).
    ex.positions = [{"side": "short", "contracts": 50000.0, "entryPrice": 0.0700}]
    g._net_short_qty = 50000.0
    g.set_position_limit(0.0, 50000.0, 17467.0)
    assert g._block_buys or g._block_sells

    sell = next(l for l in g.levels if l.side == "sell")
    before = len(ex.placed)
    _fill(g, sell)

    exits = [o for o in ex.placed[before:] if o[3]]
    assert exits, "the reduce-only exit was blocked -- inventory can never unwind"


def test_the_size_taper_reaches_the_replacement_path():
    """_buy_scale/_sell_scale shrink orders as the cap approaches. The replacement
    path ignored them, so sizing near the cap was full-size."""
    g, ex = _engine()
    g.set_position_limit(long_position=14000.0, short_position=0.0, max_position_qty=17467.0)
    assert 0 < g._buy_scale < 1.0, f"precondition: expected a taper, got {g._buy_scale}"

    sell = next(l for l in g.levels if l.side == "sell")   # -> BUY replacement
    sell.quantity = baseline = 1000.0     # levels start at 0, which falls back to a
                                          # computed size and makes the check vacuous
    before = len(ex.placed)
    _fill(g, sell)

    placed = [o for o in ex.placed[before:] if o[0] == "buy"]
    assert placed, "no buy replacement was placed at all -- test is not exercising the path"
    assert placed[0][2] < baseline, (
        f"replacement placed {placed[0][2]} against an untapered {baseline} -- "
        f"the scale factor is not reaching this path"
    )


def test_replacements_below_min_notional_are_not_attempted():
    """A sub-minimum order is a guaranteed -4164; attempting it burns rate limit."""
    g, ex = _engine()
    buy = next(l for l in g.levels if l.side == "buy")
    buy.quantity = 1.0                  # ~0.07 USDT notional
    before = len(ex.placed)

    _fill(g, buy)

    tiny = [o for o in ex.placed[before:] if o[2] * 0.07 < MIN_NOTIONAL_USDT]
    assert tiny == [], f"attempted guaranteed-reject orders: {tiny}"
