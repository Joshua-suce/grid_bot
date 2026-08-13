"""A blocked level must keep quoting somewhere legal, not go dead. AUDIT #42.

The #32 break-even guard was right about the economics and wrong about the consequence.
It returned False and left the level dead -- nothing re-sited it, nothing replaced it --
so the ladder kept a permanent hole exactly where trading happens: next to the price.

On 2026-08-13 that hole was the entire strategy. A short at 0.07024719 made the buy
level at 0.07034 permanently illegal, and it was the only level within 1.2% of the
price. From 10:46 to 15:29 the bot logged

    SKIP BUY @ 0.07034 | below break-even 0.07021909 on the open short ...

every fifteen seconds and did not trade once: seven fills in seven hours, six of them
inside a single 90-second burst.

Waiting out inventory does not require refusing to quote. It requires quoting where it
does not lose.
"""

import sys

import pytest

from grid import GridEngine

ENTRY = 0.07024719          # the real short from that session
QTY = 2479.0


class _Ex:
    """Exchange stub carrying the production position."""

    def __init__(self, price=0.07020, side="short", qty=QTY, entry=ENTRY):
        self.price, self.side, self.qty, self.entry = price, side, qty, entry
        self.placed = []

    class exchange:
        @staticmethod
        def amount_to_precision(s, a): return f"{float(a):.0f}"
        @staticmethod
        def price_to_precision(s, p): return f"{float(p):.5f}"

    def get_price(self, s): return self.price
    def get_balance(self, s="USDT"): return 4751.0
    def get_positions(self, s):
        if self.qty == 0:
            return []
        return [{"side": self.side, "contracts": self.qty, "entryPrice": self.entry}]
    def get_open_orders(self, s): return []
    def get_open_order_ids(self, s): return set()
    def fetch_order(self, i, s): return None
    def cancel_order(self, i, s): return True
    def cancel_everything(self, s, timeout_seconds=300.0): return 0
    def can_place_order(self, s): return True
    def close_position(self, symbol, side, amount, max_attempts=None): return {"id": "c"}
    def place_limit_order(self, symbol, side, price, amount, max_attempts=3, params=None,
                          post_only=True, allow_taker_fallback=False):
        self.placed.append((side, price, amount))
        return {"id": f"o{len(self.placed)}"}


def _engine(ex):
    g = GridEngine(exchange=ex, symbol="DOGEUSDT",
                   grid_lower=0.06929107, grid_upper=0.07190893, grid_count=10,
                   capital_per_grid_pct=0.018, stop_loss_pct=0.03,
                   min_profit_multiplier=3.0, max_exposure_pct=0.5, leverage=5)
    g.initialize(ex.price, balance=4751.0)
    return g


def _blocked_buy_level(g):
    lvl = next(l for l in g.levels if l.side == "buy")
    lvl.price, lvl.order_id, lvl.status = 0.07034, None, "pending"
    return lvl


def test_a_blocked_level_starts_quoting_again():
    """The regression itself: 4h43m of silence because one level had nowhere to go."""
    ex = _Ex()
    g = _engine(ex)
    lvl = _blocked_buy_level(g)

    for _ in range(20):
        g._place_order_for_level(lvl, 4751.0)
        g._break_even_time = 0.0

    assert ex.placed, (
        "the level is still dead -- with no other level near the price the grid quotes "
        "nothing and trades nothing, exactly as it did for 4h43m on 2026-08-13"
    )


def test_it_never_quotes_at_a_losing_price():
    """#42 must not undo #32: the level moves, but only to somewhere that does not lose."""
    ex = _Ex()
    g = _engine(ex)
    lvl = _blocked_buy_level(g)
    break_even = ENTRY * (1 - 2 * g.maker_fee_pct)

    for _ in range(20):
        g._place_order_for_level(lvl, 4751.0)
        g._break_even_time = 0.0

    assert lvl.price <= break_even, f"level at {lvl.price} is above break-even {break_even}"
    for side, price, _ in ex.placed:
        assert side == "buy" and price <= break_even, (
            f"covering the short at {price} is above break-even {break_even} -- a loss"
        )


def test_the_moved_level_rests_instead_of_crossing():
    """A buy above the market is post-only rejected and retried forever at debug level:
    the same dormancy, just silent. The move must land on the maker side too."""
    ex = _Ex(price=0.07020)
    g = _engine(ex)
    lvl = _blocked_buy_level(g)

    g._place_order_for_level(lvl, 4751.0)

    assert lvl.price <= ex.price, (
        f"moved buy to {lvl.price} with the market at {ex.price} -- that crosses, "
        f"post-only rejects it, and the level goes quiet again"
    )


def test_the_same_block_is_not_logged_every_single_poll(capsys):
    """~1,300 identical lines in one session. Once is information; 1,300 is noise that
    buries the fills between them."""
    from loguru import logger

    ex = _Ex()
    g = _engine(ex)
    lvl = _blocked_buy_level(g)
    # wall break-even in with neighbours so no legal re-site exists
    for other in g.levels:
        if other is not lvl:
            other.price = round(ENTRY * (1 - 2 * g.maker_fee_pct), 5)

    sink = []
    logger.remove()
    handler = logger.add(lambda m: sink.append(m), level="INFO")
    try:
        for _ in range(40):
            g._place_order_for_level(lvl, 4751.0)
            g._break_even_time = 0.0
    finally:
        logger.remove(handler)
        logger.add(sys.stderr, level="INFO")

    skips = [m for m in sink if "SKIP BUY" in m]
    assert len(skips) <= 1, f"logged the same unresolvable skip {len(skips)} times"


def test_a_long_gets_the_same_treatment_upward():
    """Symmetry: a sell level below a long's break-even moves UP, not into a loss."""
    ex = _Ex(price=0.07100, side="long", entry=0.07080)
    g = _engine(ex)
    lvl = next(l for l in g.levels if l.side == "sell")
    lvl.price, lvl.order_id, lvl.status = 0.06950, None, "pending"
    break_even = 0.07080 * (1 + 2 * g.maker_fee_pct)

    g._place_order_for_level(lvl, 4751.0)

    assert lvl.price >= break_even, f"selling the long at {lvl.price} < {break_even} loses"
    for side, price, _ in ex.placed:
        assert price >= break_even


def test_no_move_when_it_would_deform_the_ladder():
    """#42 must not undo #34 either: if break-even sits on top of a neighbour, the level
    stays put rather than creating a pair inside the fee floor."""
    ex = _Ex()
    g = _engine(ex)
    lvl = _blocked_buy_level(g)
    break_even = ENTRY * (1 - 2 * g.maker_fee_pct)
    for other in g.levels:
        if other is not lvl:
            other.price = round(break_even, 5)

    moved = g._nearest_legal_exit(lvl)

    assert moved is None, f"moved onto a neighbour at {moved} and deformed the ladder"
