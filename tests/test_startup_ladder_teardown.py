"""A restart must not tear down the ladder it just restored. AUDIT #90.

2026-08-17 05:00. Fourteen orders went out restoring a saved grid, reconcile_positions
added a fifteenth, and four seconds after GRID ACTIVATED the deformation check cancelled
all sixteen and rebuilt from scratch -- with 5342 DOGE of short open the whole time.
About thirty orders placed and killed inside ninety seconds, and it had happened on the
previous restart too.

Two independent defects had to line up, so this module tests both:

  1. reconcile_positions re-sited a level onto 0.07008, a line a restored grid buy
     already held: two levels, one price, "tightest 0.00%". Every other re-siting path
     in grid.py checks for a crowded neighbour; this one did not.

  2. recenter's `flat` guard reads _net_short_qty, which set_position_limit writes at
     main.py:1593 -- 139 lines AFTER recenter is called at main.py:1454, in the same
     iteration. On the first poll of a restart it is therefore still 0.0, and the guard
     read an account holding 5342 DOGE as flat on precisely the iteration where a
     restored position is most likely to exist.
"""

import pytest

from grid import GridEngine, GridLevel


class FakeInner:
    """The ccxt handle: only the precision helpers are reached."""

    @staticmethod
    def price_to_precision(symbol, price):
        return f"{float(price):.5f}"

    @staticmethod
    def amount_to_precision(symbol, qty):
        return f"{float(qty):.0f}"


class FakeExchange:
    def __init__(self, positions=None):
        self.exchange = FakeInner()
        self._positions = positions or []
        self.placed = []
        self.cancelled = []
        self.cancel_everything_calls = 0

    def get_positions(self, symbol):
        return self._positions

    def get_price(self, symbol):
        return 0.0702

    def cancel_order(self, order_id, symbol):
        self.cancelled.append(order_id)
        return True

    def place_limit_order(self, symbol, side, price, qty, params=None, max_attempts=1, post_only=True, allow_taker_fallback=False):
        self.placed.append({"side": side, "price": price, "qty": qty})
        return {"id": f"o{len(self.placed)}"}

    # recenter -> pause() reaches this; counting it is how we tell whether the guard
    # let a rebuild through, without needing the rest of the rebuild to work.
    def cancel_everything(self, symbol, timeout_seconds=300.0, keep_stops=False):
        self.cancel_everything_calls += 1
        raise AssertionError("rebuild reached the exchange")

    def get_open_order_ids(self, symbol):
        return []


def engine(levels, positions=None):
    e = GridEngine.__new__(GridEngine)
    e.symbol = "DOGEUSDT"
    e.exchange = FakeExchange(positions)
    e.levels = levels
    e.grid_spacing = 0.00014154
    e.grid_lower = 0.06921204
    e.grid_upper = 0.07118796
    # round_trip_fee_pct is derived, not settable: 2 * blended per-side at the measured
    # 11.8% taker share = 0.04472%, and the 3.0x profit multiplier makes the fee floor
    # 0.13416% -- the "0.13% fee floor" the live DEFORMED LADDER line reports.
    e.maker_fee_pct = 0.0002
    e.taker_fee_pct = 0.0004
    e.taker_fill_share = 0.118
    e._min_profit_multiplier = 3.0
    e.use_market_close_on_replace = False
    e._open_orders_fetch_time = 0.0
    return e


SHORT = [{"side": "short", "contracts": 5342.0, "entryPrice": 0.07015001}]


# --- 1. the hedge must not stack on a line the ladder already holds -----------------

def test_the_hedge_does_not_stack_on_an_occupied_line():
    """The live collision. Entry 0.07015001 clamps the hedge to 0.07011, and a restored
    grid buy is already resting there. Placing a second order on that line is what made
    the tightest spacing 0.00% and triggered DEFORMED LADDER."""
    occupied = GridLevel(price=0.07011, side="buy", order_id="grid-buy")
    nearest_sell = GridLevel(price=0.07036, side="sell", order_id="grid-sell")
    e = engine([occupied, nearest_sell], SHORT)

    e.reconcile_positions()

    assert e.exchange.placed == [], "stacked a second order on an occupied line"
    assert nearest_sell.side == "sell", "repurposed the level anyway"
    assert nearest_sell.price == 0.07036
    assert occupied.order_id == "grid-buy", "disturbed the level already doing the job"


def test_the_ladder_is_not_deformed_by_the_reconcile():
    """The consequence, stated as the ladder property that actually matters: after
    reconcile every pair of levels still clears the fee floor. Before the fix this
    produced a 0.00% pair, which ladder_defects reports and recenter acts on."""
    e = engine(
        [GridLevel(price=0.07011, side="buy", order_id="grid-buy"),
         GridLevel(price=0.07036, side="sell", order_id="grid-sell")],
        SHORT,
    )

    e.reconcile_positions()

    prices = sorted(l.price for l in e.levels)
    floor = e.round_trip_fee_pct * e._min_profit_multiplier
    assert all((b - a) / a >= floor for a, b in zip(prices, prices[1:]))
    assert e.ladder_defects(0.0702) == []


def test_a_hedge_with_room_still_repurposes_a_level():
    """The guard must not disable the feature. With no level near the hedge price the
    reconcile does exactly what it always did."""
    nearest_sell = GridLevel(price=0.07036, side="sell", order_id="grid-sell")
    far_buy = GridLevel(price=0.06950, side="buy", order_id="grid-buy")
    e = engine([far_buy, nearest_sell], SHORT)

    e.reconcile_positions()

    assert len(e.exchange.placed) == 1
    assert e.exchange.placed[0]["side"] == "buy"
    assert nearest_sell.side == "buy", "the level was not repurposed"


def test_a_near_neighbour_does_not_suppress_the_cover():
    """The guard is deliberately narrow. A level one tick away is a spacing question the
    ladder already handles; refusing there would suppress a legitimate full-size cover.
    That is exactly the AUDIT #41 state -- short 9916 @ 0.07024719 covers at 0.07021 with
    a grid buy 0.00008 below it -- and that cover must still reach the exchange."""
    e = engine(
        [GridLevel(price=0.07012, side="buy", order_id="grid-buy"),   # 1 tick off 0.07011
         GridLevel(price=0.07036, side="sell", order_id="grid-sell")],
        SHORT,
    )

    e.reconcile_positions()

    assert len(e.exchange.placed) == 1, "a near neighbour suppressed the position cover"
    assert e.exchange.placed[0]["side"] == "buy"


def test_a_near_neighbour_is_nudged_clear_when_there_is_room():
    """2026-08-29. The "near neighbour" guard above stops the cover from being
    suppressed, but it left the cover sitting wherever the raw calculation put it --
    0.00001 from a live grid buy, both under the fee floor. That pair, both real (a
    -1984 ADA short, not dust), sat deformed for the rest of the session: the one
    repair that could have fixed it -- a full rebuild -- refuses to run while any real
    position is open. When there is room to move without crossing break-even, the
    cover should land somewhere that actually clears its own fees instead."""
    occupied = GridLevel(price=0.07012, side="buy", order_id="grid-buy")   # 1 tick off
    nearest_sell = GridLevel(price=0.07036, side="sell", order_id="grid-sell")
    e = engine([occupied, nearest_sell], SHORT)

    e.reconcile_positions()

    assert len(e.exchange.placed) == 1, "a near neighbour suppressed the position cover"
    placed_price = e.exchange.placed[0]["price"]
    floor = placed_price * e.round_trip_fee_pct * e._min_profit_multiplier
    assert abs(placed_price - occupied.price) >= floor, "still crowds the neighbour"
    assert placed_price <= 0.07011, "moved past the position's own break-even"
    assert occupied.price == 0.07012, "disturbed the level it nudged away from"


def test_the_nudge_never_crosses_break_even():
    """A neighbour can sit close enough that the only way to clear it is through
    break-even. AUDIT #41 still applies here: the cover must go out regardless, so
    when no safe room exists this is a no-op and the cover lands exactly where the
    unmodified calculation put it -- crowded, but not a loss, and not suppressed."""
    buy_near = GridLevel(price=0.07018, side="buy", order_id="grid-buy2")
    sell_far = GridLevel(price=0.07050, side="sell", order_id="grid-sell2")
    e = engine([buy_near, sell_far], [{"side": "short", "contracts": 5342.0, "entryPrice": 0.07030}])

    e.reconcile_positions()

    assert len(e.exchange.placed) == 1, "no safe nudge existed, but the cover was suppressed anyway"
    assert e.exchange.placed[0]["price"] == 0.07026, "moved the price despite no safe room to do so"
    assert buy_near.price == 0.07018, "disturbed the crowding level"


def test_a_neighbour_below_the_hedge_nudges_up_capped_at_break_even():
    """The mirror of the first nudge case: the neighbour sits below the hedge, so
    clearing it means moving toward break-even, not away from it. The nudge must
    still stop at break-even rather than sail past it chasing clearance."""
    buy_below = GridLevel(price=0.07033, side="buy", order_id="grid-buy3")
    sell_near = GridLevel(price=0.07051, side="sell", order_id="grid-sell3")
    e = engine([buy_below, sell_near], [{"side": "short", "contracts": 5342.0, "entryPrice": 0.07050}])

    e.reconcile_positions()

    assert len(e.exchange.placed) == 1
    placed_price = e.exchange.placed[0]["price"]
    breakeven_bound = e._round_price_toward(0.07050 * (1 - e.round_trip_fee_pct), -1)
    assert placed_price <= breakeven_bound, "nudged past the position's own break-even"
    assert buy_below.price == 0.07033, "disturbed the crowding level"


def test_an_exact_duplicate_is_still_skipped_not_nudged():
    """The exact-match guard above and the nudge added here must not overlap: an exact
    duplicate is a different kind of problem (AUDIT #90's own words: "two orders where
    the ladder believes there is one") and stays on the skip path, never the nudge
    path."""
    occupied = GridLevel(price=0.07011, side="buy", order_id="grid-buy")   # exact match
    nearest_sell = GridLevel(price=0.07036, side="sell", order_id="grid-sell")
    e = engine([occupied, nearest_sell], SHORT)

    e.reconcile_positions()

    assert e.exchange.placed == [], "an exact duplicate took the nudge path instead of skipping"
    assert occupied.order_id == "grid-buy"


def test_an_occupant_on_the_wrong_side_does_not_suppress_the_cover():
    """Only a level that can actually close the position may stand in for the hedge. A
    SELL resting on that line does not cover a short -- deferring to it would leave the
    position genuinely unhedged, which is worse than a crowded ladder."""
    e = engine(
        [GridLevel(price=0.07011, side="sell", order_id="grid-sell-low"),
         GridLevel(price=0.07036, side="sell", order_id="grid-sell")],
        SHORT,
    )

    e.reconcile_positions()

    assert len(e.exchange.placed) == 1, "deferred to a level on the wrong side"
    assert e.exchange.placed[0]["side"] == "buy"


# --- 2. a restored position is not flat, whatever the ladder counters say -----------

def deformed_engine(pos_qty):
    """In-band, two-sided, and carrying a 0.00% level pair -- the 05:00:56 shape."""
    e = engine([
        GridLevel(price=0.07008, side="buy", order_id="a"),
        GridLevel(price=0.07008, side="buy", order_id="b"),      # the duplicate line
        GridLevel(price=0.07036, side="sell", order_id="c"),
    ])
    e._last_recenter_time = 0.0
    e.recenter_cooldown = 0.0
    e._block_buys = False
    e._block_sells = False
    e._net_long_qty = 0.0        # set_position_limit has not run yet this iteration
    e._net_short_qty = 0.0
    e._pos_qty = pos_qty
    e.active = True
    e._event_journal = None
    e._notifier = None
    return e


def test_a_restored_position_is_not_read_as_flat_on_the_first_poll():
    """The live failure. Ladder counters 0/0 because set_position_limit runs 139 lines
    later; the mirror says SHORT 5342. Recentring here cancels the exit orders the
    position has to unwind through."""
    e = deformed_engine(pos_qty=-5342.0)

    assert e.recenter(0.0702, balance=4920.0, margin_pct=0.02) is False
    assert e.exchange.cancel_everything_calls == 0, "tore down the ladder with a position open"


def test_a_long_is_caught_by_the_same_guard():
    e = deformed_engine(pos_qty=8516.0)

    assert e.recenter(0.0702, balance=4920.0, margin_pct=0.02) is False
    assert e.exchange.cancel_everything_calls == 0


def test_a_genuinely_flat_ladder_still_rebuilds_when_deformed():
    """The guard must not swallow AUDIT #34. Flat by both measures, deformed, in band:
    the rebuild has to go ahead, or a restored grid keeps a hole where price trades."""
    e = deformed_engine(pos_qty=0.0)

    with pytest.raises(AssertionError, match="rebuild reached the exchange"):
        e.recenter(0.0702, balance=4920.0, margin_pct=0.02)
    assert e.exchange.cancel_everything_calls == 1


def test_the_ladder_counters_alone_still_block_a_rebuild():
    """The original guard is kept, not replaced: inventory the counters know about but
    the mirror has not caught up on must still block."""
    e = deformed_engine(pos_qty=0.0)
    e._net_short_qty = 5342.0

    assert e.recenter(0.0702, balance=4920.0, margin_pct=0.02) is False
    assert e.exchange.cancel_everything_calls == 0
