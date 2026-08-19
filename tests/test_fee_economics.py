"""Tests for the fee-economics fixes (AUDIT.md issues #17-#20).

The 2026-08-11 session booked ~1442 fills and still ended down. Binance's own income
ledger explained why: realized +13.21 against commission -45.26. Fees were 2.2x gross
profit. Two mechanisms let that happen -- crossing orders being silently downgraded to
taker fills, and a profitability gate pinned at break-even -- and two config settings
guaranteed it stayed that way.
"""

import pytest

import ccxt

from exchange import PostOnlyWouldCross
from grid import GridEngine


class RecordingExchange:
    """Records placements. Optionally raises PostOnlyWouldCross for a given side."""

    def __init__(self, cross_side=None):
        self.placed: list[dict] = []
        self._cross_side = cross_side
        self._next_id = 0

        class _inner:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{float(amount):.0f}"

            @staticmethod
            def price_to_precision(symbol, price):
                return f"{float(price):.5f}"

        self.exchange = _inner()

    def get_positions(self, symbol):
        return []

    def get_open_orders(self, symbol):
        return []

    def get_open_order_ids(self, symbol):
        return set()

    def can_place_order(self, symbol):
        return True

    def get_balance(self):
        return 5000.0

    def cancel_order(self, order_id, symbol):
        return True

    def get_orderbook_depth(self, symbol, limit=10):
        return {"bid_vol": 0, "ask_vol": 0, "imbalance": 0, "spread_pct": 0}

    def place_limit_order(self, symbol, side, price, amount, max_attempts=3, params=None, post_only=True, allow_taker_fallback=False):
        if self._cross_side is not None and side == self._cross_side:
            raise PostOnlyWouldCross(f"{side} @ {price} would cross the spread")
        self._next_id += 1
        order = {"id": f"o{self._next_id}", "side": side, "price": price, "amount": amount}
        self.placed.append(order)
        return order


class SpyNotifier:
    def __init__(self):
        self.failures = []
        self.placements = []

    def on_order_placed(self, *a, **kw):
        self.placements.append(a)

    def on_order_failed(self, *a, **kw):
        self.failures.append(a)


class SpyJournal:
    def __init__(self):
        self.failures = []

    def order_placed(self, *a, **kw):
        pass

    def order_failed(self, *a, **kw):
        self.failures.append(a)


def make_engine(exchange, grid_count=10, min_profit_multiplier=3.0, **kw):
    defaults = dict(
        symbol="DOGEUSDT",
        grid_lower=0.0710,
        grid_upper=0.0730,
        grid_count=grid_count,
        capital_per_grid_pct=0.05,
        stop_loss_pct=0.05,
        maker_fee_pct=0.0002,
        taker_fee_pct=0.0004,
        max_exposure_pct=1.0,
        min_profit_multiplier=min_profit_multiplier,
        order_pacing_seconds=0.0,
    )
    defaults.update(kw)
    return GridEngine(exchange=exchange, **defaults)


# --- #17: crossing orders must never become taker fills --------------------

def test_crossing_level_is_skipped_not_filled_as_taker():
    """A level on the wrong side of the book is left unplaced, not crossed.

    Two sells went out below market on 2026-08-12 at 01:22:40 and 01:22:43 with
    postOnly=False and filled instantly as taker 22 seconds later, both at a loss.
    """
    ex = RecordingExchange(cross_side="sell")
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)

    grid.place_initial_orders(balance=5000)

    assert all(o["side"] == "buy" for o in ex.placed), "a crossing sell was placed anyway"
    unplaced = [lv for lv in grid.levels if lv.side == "sell" and lv.order_id is None]
    assert unplaced, "crossing sells should be left pending for a later retry"


def test_crossing_level_is_not_reported_as_a_failure():
    """It is a normal transient condition, not an error worth alerting on."""
    ex = RecordingExchange(cross_side="sell")
    notifier, journal = SpyNotifier(), SpyJournal()
    grid = make_engine(ex, event_journal=journal, notifier=notifier)
    grid.initialize(0.0720, balance=5000)

    grid.place_initial_orders(balance=5000)

    assert notifier.failures == [], "crossing level raised a Telegram alert"
    assert journal.failures == [], "crossing level was journalled as a failure"


def test_crossing_level_stays_retryable():
    """The level must keep its identity so the next pass can place it."""
    ex = RecordingExchange(cross_side="sell")
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.place_initial_orders(balance=5000)

    sells = [lv for lv in grid.levels if lv.side == "sell"]
    assert sells and all(lv.order_id is None for lv in sells)
    assert all(lv.status == "pending" for lv in sells)

    # Book moves back; the same levels now rest normally.
    ex._cross_side = None
    grid.place_initial_orders(balance=5000)
    assert any(o["side"] == "sell" for o in ex.placed)


def test_real_placement_errors_are_still_reported():
    """Only PostOnlyWouldCross is quiet -- genuine failures must still surface."""
    class BrokenExchange(RecordingExchange):
        def place_limit_order(self, *a, **kw):
            raise ccxt.ExchangeError("-1013 something genuinely wrong")

    ex = BrokenExchange()
    notifier, journal = SpyNotifier(), SpyJournal()
    grid = make_engine(ex, event_journal=journal, notifier=notifier)
    grid.initialize(0.0720, balance=5000)

    grid.place_initial_orders(balance=5000)

    assert notifier.failures, "a real placement error was swallowed"
    assert journal.failures


# --- #18/#20: the profitability gate must demand a real margin -------------

def test_level_whose_spacing_barely_covers_fees_is_skipped():
    """30 levels across this range gives 0.0069% spacing against a 0.04% round trip.

    Under the old hardcoded multiplier of 1.0 this passed the gate, because spacing
    exceeded fees by a hair. It is a guaranteed net loss once anything goes against it.
    """
    ex = RecordingExchange()
    grid = make_engine(ex, grid_count=30, min_profit_multiplier=3.0)
    grid.initialize(0.0720, balance=5000)

    grid.place_initial_orders(balance=5000)
    assert ex.placed == [], "placed levels that cannot cover their own fees"


def test_level_with_healthy_spacing_is_placed():
    ex = RecordingExchange()
    grid = make_engine(ex, grid_count=10, min_profit_multiplier=3.0)
    grid.initialize(0.0720, balance=5000)

    grid.place_initial_orders(balance=5000)
    assert ex.placed, "refused levels that clear fees 5x over"


def test_gate_tightens_as_the_multiplier_rises():
    """The same grid passes at 1.0 and fails at 5.0 -- the knob actually binds."""
    spacings = {}
    for mult in (1.0, 5.0):
        ex = RecordingExchange()
        grid = make_engine(ex, grid_count=24, min_profit_multiplier=mult)
        grid.initialize(0.0720, balance=5000)
        grid.place_initial_orders(balance=5000)
        spacings[mult] = len(ex.placed)

    assert spacings[1.0] > 0, "break-even multiplier should still place this grid"
    assert spacings[5.0] == 0, "a 5x margin requirement should reject it"


def test_profitability_gate_uses_round_trip_not_single_leg_fees():
    """A cycle pays two fees, not one. Sizing the gate on a single leg would let
    through levels that lose exactly the second fee on every completed cycle."""
    ex = RecordingExchange()
    grid = make_engine(ex, grid_count=10, min_profit_multiplier=1.0)
    price = 0.0720
    grid.initialize(price, balance=5000)

    single_leg = grid.maker_fee_pct * price
    assert grid._is_level_profitable(price) is (grid.grid_spacing > 2 * single_leg)


# --- #32: never book a loss to keep the ladder tidy ------------------------

class _PositionExchange:
    """Stub reporting one open long, as Binance does: netted, one blended entry."""

    def __init__(self, qty=10456.0, entry=0.06962068):
        self.qty = qty
        self.entry = entry
        self.placed: list[dict] = []
        self._next = 0

    class exchange:
        @staticmethod
        def amount_to_precision(symbol, amount):
            return f"{float(amount):.0f}"

        @staticmethod
        def price_to_precision(symbol, price):
            return f"{float(price):.5f}"

    def get_positions(self, symbol):
        if self.qty == 0:
            return []
        return [{"side": "long", "contracts": self.qty, "entryPrice": self.entry}]

    def get_open_orders(self, symbol):
        return []

    def get_open_order_ids(self, symbol):
        return set()

    def can_place_order(self, symbol):
        return True

    def get_balance(self):
        return 4900.0

    def place_limit_order(self, symbol, side, price, amount, max_attempts=3,
                          params=None, post_only=True, allow_taker_fallback=False):
        self._next += 1
        order = {"id": f"o{self._next}", "side": side, "price": price, "amount": amount}
        self.placed.append(order)
        return order


def _engine_with_position(ex, lower=0.06771107, upper=0.07032893):
    from grid import GridEngine

    engine = GridEngine(
        exchange=ex, symbol="DOGEUSDT",
        grid_lower=lower, grid_upper=upper, grid_count=10,
        capital_per_grid_pct=0.018, stop_loss_pct=0.03,
    )
    engine.initialize((lower + upper) / 2, balance=4900.0)
    return engine


def test_unwind_does_not_sell_below_the_blended_entry():
    """AUDIT #32, replaying the 2026-08-12 22:04 recenter exactly.

    Price fell out of the grid, the ladder was rebuilt lower, and the unwind placed
    reduce-only sells at 0.06928 and 0.06954 against a long whose average entry was
    0.06962068. Both filled: -0.712368 and -0.168708, over half the whole session's
    loss. The position was not in trouble -- the hard stop sat at 0.06697 and price was
    back above 0.0694 within minutes.

    Binance nets everything into one position at one average entry, so a sell below it
    realises a loss no matter which level the engine has paired it with.
    """
    ex = _PositionExchange(qty=10456.0, entry=0.06962068)
    engine = _engine_with_position(ex)

    engine._unwind_position_through_grid(4900.0)

    sells = [o for o in ex.placed if o["side"] == "sell"]
    assert sells, "placed no exit orders at all"
    break_even = 0.06962068 * (1 + 2 * engine.maker_fee_pct)
    below = [o["price"] for o in sells if o["price"] < break_even]
    assert below == [], f"unwind sold below break-even {break_even:.8f} at {below}"


def test_unwind_places_nothing_rather_than_dumping_when_every_level_is_underwater():
    """A whole ladder below cost means wait, not sell. The hard stop is the backstop."""
    ex = _PositionExchange(qty=10456.0, entry=0.08000)
    engine = _engine_with_position(ex)

    engine._unwind_position_through_grid(4900.0)

    assert [o for o in ex.placed if o["side"] == "sell"] == []


def test_the_grid_will_not_place_a_sell_below_break_even_while_long():
    """The unwind is not the only path to the same mistake: after a downward recenter
    the ordinary ladder has sell levels below the open position's cost too."""
    ex = _PositionExchange(qty=10456.0, entry=0.06962068)
    engine = _engine_with_position(ex)

    engine.place_initial_orders(4900.0)

    break_even = 0.06962068 * (1 + 2 * engine.maker_fee_pct)
    bad = [o["price"] for o in ex.placed if o["side"] == "sell" and o["price"] < break_even]
    assert bad == [], f"grid placed losing sells at {bad}"


def test_a_flat_engine_is_completely_unconstrained():
    """No position, no break-even, no new restriction -- the ordinary grid is unchanged."""
    ex = _PositionExchange(qty=0.0)
    engine = _engine_with_position(ex)

    placed = engine.place_initial_orders(4900.0)

    assert placed > 0
    assert any(o["side"] == "sell" for o in ex.placed)
    assert any(o["side"] == "buy" for o in ex.placed)


def test_an_unreadable_position_does_not_silently_block_the_grid():
    """If the position cannot be read we must not invent a break-even and freeze."""
    class Unreadable(_PositionExchange):
        def get_positions(self, symbol):
            raise RuntimeError("API down")

    ex = Unreadable(qty=10456.0)
    engine = _engine_with_position(ex)

    assert engine._would_realise_a_loss("sell", 0.0680) is False


def test_buying_is_never_blocked_by_the_guard_so_it_unsticks_itself():
    """The guard restricts the *closing* side only, and that is what stops it becoming
    a dormancy trap.

    Holding a long at 0.06962, sells below cost are refused -- but the grid keeps
    buying underneath. Each lower buy drags the blended average entry down, which drags
    break-even down with it, which makes previously-refused sell levels eligible. The
    position works its own way out instead of waiting on one price level.
    """
    ex = _PositionExchange(qty=10456.0, entry=0.06962068)
    engine = _engine_with_position(ex)

    engine.place_initial_orders(4900.0)
    assert [o for o in ex.placed if o["side"] == "buy"], "guard blocked the buy side"

    high = engine._position_break_even()[1]
    ex.entry = 0.06900          # the lower buys filled; average entry drops
    engine._break_even_time = 0.0
    low = engine._position_break_even()[1]

    assert low < high, "break-even did not follow the average entry down"


def test_the_stop_loss_path_is_untouched_by_the_guard():
    """Refusing to sell below cost must not also refuse to protect the position. Stops
    are placed by main.py straight through the exchange, never as grid levels, so the
    backstop survives however long the grid waits."""
    import inspect

    import grid as grid_module

    callers = [
        name for name, fn in inspect.getmembers(grid_module.GridEngine, inspect.isfunction)
        if "_would_realise_a_loss(" in inspect.getsource(fn) and name != "_would_realise_a_loss"
    ]
    assert sorted(callers) == ["_place_order_for_level"], (
        f"the break-even guard reached beyond order placement into {callers}"
    )

    # And the stop prices themselves never consult it.
    for name in ("get_stop_loss_price", "get_hard_stop_loss_price",
                 "get_short_stop_loss_price", "get_short_hard_stop_loss_price"):
        src = inspect.getsource(getattr(grid_module.GridEngine, name))
        assert "_would_realise_a_loss" not in src and "_position_break_even" not in src, (
            f"{name} consults the break-even guard -- protection must not depend on it"
        )


# --- #41: reconcile_positions bypassed the break-even rule -----------------

class _ShortExchange(_PositionExchange):
    """The 2026-08-13 account: short 9916 DOGE @ 0.07024719."""

    def get_positions(self, symbol):
        if self.qty == 0:
            return []
        return [{"side": "short", "contracts": self.qty, "entryPrice": self.entry}]

    def place_limit_order(self, symbol, side, price, amount, max_attempts=3,
                          params=None, post_only=True, allow_taker_fallback=False):
        self._next += 1
        order = {"id": f"o{self._next}", "side": side, "price": price, "amount": amount,
                 "reduceOnly": (params or {}).get("reduceOnly", False)}
        self.placed.append(order)
        return order


def _short_engine():
    ex = _ShortExchange(qty=9916.0, entry=0.07024719)
    from grid import GridEngine

    engine = GridEngine(
        exchange=ex, symbol="DOGEUSDT",
        grid_lower=0.06902107, grid_upper=0.07163893, grid_count=10,
        capital_per_grid_pct=0.018, stop_loss_pct=0.03,
    )
    engine.initialize(0.07060, balance=4783.0)
    return engine, ex


def test_covering_a_short_never_buys_above_break_even():
    """AUDIT #41. reconcile_positions picked the level nearest the entry and stepped one
    spacing toward profit -- but from the LEVEL, not from the entry. With the real saved
    state (short @ 0.07024719, nearest sell level 0.07059) that produced a cover at
    0.07030, above break-even, booking -0.52 on 9,916 DOGE. The #32 guard lives in
    _place_order_for_level and the unwind; this path reached the exchange directly.
    """
    engine, ex = _short_engine()

    engine.reconcile_positions()

    covers = [o for o in ex.placed if o["side"] == "buy" and o["amount"] >= 5000]
    assert covers, "no cover order placed for the open short"
    break_even = ex.entry * (1 - 2 * engine.maker_fee_pct)
    for o in covers:
        assert o["price"] <= break_even, (
            f"cover at {o['price']:.8f} is above break-even {break_even:.8f} -- a loss"
        )


def test_the_cover_is_reduce_only_on_the_short_side_too():
    """reduceOnly was set only when hedging a long. Covering a short went out as a plain
    buy, so if the position had already closed it would open a fresh long instead."""
    engine, ex = _short_engine()

    engine.reconcile_positions()

    covers = [o for o in ex.placed if o["side"] == "buy" and o["amount"] >= 5000]
    assert covers and all(o["reduceOnly"] for o in covers), (
        "cover order is not reduceOnly -- it can open a position rather than close one"
    )


def test_no_reduce_only_sells_are_placed_while_short():
    """A reduceOnly SELL can only reduce a LONG; with a short open every one is a
    guaranteed -2022 rejection."""
    engine, ex = _short_engine()
    for level in engine.levels:
        if level.side == "sell":
            level.quantity = 300.0

    engine.reconcile_positions()

    bad = [o for o in ex.placed if o["side"] == "sell" and o["reduceOnly"]]
    assert bad == [], f"placed reduce-only sells against a short: {bad}"


def test_rounding_never_crosses_the_break_even_boundary():
    """0.07021909 rounds to 0.07022 at five decimals -- above break-even, so the 'safe'
    price was a loss by 0.0000009. Small per unit; on 9,916 DOGE it flips the sign."""
    engine, _ = _short_engine()

    for value in (0.07021909, 0.0702, 0.070215, 0.07019999):
        down = engine._round_price_toward(value, -1)
        up = engine._round_price_toward(value, +1)
        assert down <= value, f"{down} > {value}"
        assert up >= value, f"{up} < {value}"
