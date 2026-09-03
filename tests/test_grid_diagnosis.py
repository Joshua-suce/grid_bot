"""Regression tests for the 2026-08-12 technical diagnosis.

Each test pins one defect found by tracing the 2026-08-11 production log, where the
bot recentered 89 times (median 196s apart -- exactly the recenter cooldown), logged
740 ReduceOnly rejections, and held a 9148 DOGE long for over an hour whose stop-loss
drifted DOWN with the market. See AUDIT.md issues #11-#16.
"""

import pytest

from grid import GridEngine, GridLevel


class FakeExchange:
    """Records placed orders instead of hitting an API. Mirrors the ccxt surface the
    engine actually touches (amount_to_precision / price_to_precision)."""

    def __init__(self, positions=None, reject_reduce_only_over=None):
        self.placed: list[dict] = []
        self.cancelled: list[str] = []
        self._positions = positions or []
        # When set, emulates Binance -2022: reject a reduceOnly order once the
        # cumulative reduceOnly quantity exceeds the open position.
        self._reject_over = reject_reduce_only_over
        self._reduce_only_total = 0.0
        self._next_id = 0

        outer = self

        class _inner:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{float(amount):.0f}"

            @staticmethod
            def price_to_precision(symbol, price):
                return f"{float(price):.5f}"

        self.exchange = _inner()
        self._outer = outer

    def get_positions(self, symbol):
        return self._positions

    def get_open_orders(self, symbol):
        return []

    def get_open_order_ids(self, symbol):
        return set()

    def can_place_order(self, symbol):
        return True

    def cancel_order(self, order_id, symbol):
        self.cancelled.append(order_id)
        return True

    def cancel_everything(self, symbol, timeout_seconds=300.0, keep_stops=False):
        return 0

    def get_orderbook_depth(self, symbol, limit=10):
        return {"bid_vol": 0, "ask_vol": 0, "imbalance": 0, "spread_pct": 0}

    def place_limit_order(self, symbol, side, price, amount, max_attempts=3, params=None, post_only=True, allow_taker_fallback=False):
        params = params or {}
        if params.get("reduceOnly") and self._reject_over is not None:
            if self._reduce_only_total + amount > self._reject_over + 1e-9:
                raise ValueError('binanceusdm {"code":-2022,"msg":"ReduceOnly Order is rejected."}')
            self._reduce_only_total += amount
        self._next_id += 1
        order = {
            "id": f"o{self._next_id}", "side": side, "price": price,
            "amount": amount, "params": params,
        }
        self.placed.append(order)
        return order


def make_engine(exchange, **kw):
    defaults = dict(
        symbol="DOGEUSDT",
        grid_lower=0.0710,
        grid_upper=0.0730,
        grid_count=10,
        capital_per_grid_pct=0.05,
        stop_loss_pct=0.05,
        max_exposure_pct=1.0,
    )
    defaults.update(kw)
    return GridEngine(exchange=exchange, **defaults)


# --- #11: reduceOnly hard-coded on every replacement sell ------------------

def test_replacement_sell_is_not_reduce_only_while_net_short():
    """A SELL while net SHORT opens exposure, so reduceOnly is illegal (-2022).

    The engine used to set reduceOnly on every replacement sell unconditionally, so
    the whole sell side was un-armable while short: 265 rejections in one session.
    """
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=0.0, short_position=5000.0, max_position_qty=8000.0)

    params, qty = grid._exit_order_params("sell", 1500.0)
    assert params is None, "sell while short opens exposure — must not be reduceOnly"
    assert qty == 1500.0


def test_replacement_sell_is_reduce_only_while_net_long():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.set_position_limit(long_position=5000.0, short_position=0.0, max_position_qty=8000.0)

    params, qty = grid._exit_order_params("sell", 1500.0)
    assert params == {"reduceOnly": True, "postOnly": False}
    assert qty == 1500.0


def test_reduce_only_quantity_is_clamped_to_remaining_position():
    """Binance also rejects a reduceOnly order larger than the position."""
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.set_position_limit(long_position=800.0, short_position=0.0, max_position_qty=8000.0)

    params, qty = grid._exit_order_params("sell", 1500.0)
    assert params == {"reduceOnly": True, "postOnly": False}
    assert qty == 800.0, "must not ask to close more than is open"


def test_reduce_only_qty_accounts_for_other_resting_orders_on_the_same_side():
    """AUDIT (live incident, 2026-09-02). Restart with SHORT 984 open: reconcile_state
    rebuilt six buy rungs in one pass, each asking _reduce_only_qty the same question
    and each getting back the SAME un-shared 984 -- so 247+257+255+254+253+250 = 1516
    of reduceOnly BUY went resting against an actual 984 short (and
    reconcile_positions's own separate 984-sized hedge stacked another 984 on top of
    that). Binance accepted them individually, then started rejecting the excess with
    -2022 once its own aggregate bookkeeping caught up -- and because every rung
    recomputed against the same unshared figure on every poll, the rejections never
    resolved: a permanent retry deadlock, one ERROR + Telegram alert per rung per
    ~10s poll, until the process was killed by hand.

    The position is one finite pool; a level already resting on the closing side has
    already drawn from it.
    """
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.set_position_limit(long_position=0.0, short_position=984.0, max_position_qty=8000.0)

    # 400 + 400 + 100 = 900 already resting; only 84 of the 984 short is unclaimed.
    grid.levels = [
        GridLevel(price=0.0690, side="buy", order_id="B1", quantity=400.0),
        GridLevel(price=0.0691, side="buy", order_id="B2", quantity=400.0),
        GridLevel(price=0.0692, side="buy", order_id="B3", quantity=100.0),
    ]

    params, qty = grid._exit_order_params("buy", 300.0)

    assert params == {"reduceOnly": True, "postOnly": False}
    assert qty == pytest.approx(84.0), (
        "must clamp to what's actually still unclaimed (84), not the raw 984"
    )


def test_sequential_exit_order_params_calls_never_collectively_overcommit_the_short():
    """Same fix, exercised the way reconcile_state actually drives it: several buy
    rungs asking _exit_order_params one after another in the same pass, each getting
    its own resting order attached before the next one asks. The SUM of everything
    reduceOnly-tagged must never exceed the real 984 short, and once it's fully
    claimed, later rungs must fall back to plain entries instead of each
    independently reclaiming the whole 984 for itself.
    """
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.set_position_limit(long_position=0.0, short_position=984.0, max_position_qty=8000.0)
    grid.levels = []

    requested = [247.0, 257.0, 255.0, 254.0, 253.0, 250.0]  # sums to 1516, well over 984
    reduce_only_claimed = 0.0
    for i, qty in enumerate(requested):
        params, clamped = grid._exit_order_params("buy", qty)
        # Simulate this rung now resting, the way _place_order_for_level would leave
        # it after a successful placement -- before the NEXT rung asks its question.
        grid.levels.append(GridLevel(
            price=0.0680 + i * 0.0005, side="buy", order_id=f"o{i}", quantity=clamped,
        ))
        if params:
            reduce_only_claimed += clamped

    assert reduce_only_claimed <= 984.0 + 1e-6, (
        f"reduceOnly buys collectively claimed {reduce_only_claimed}, over the real 984 short"
    )


def test_reconcile_positions_hedge_respects_already_resting_reduce_only_coverage():
    """The other contributor to the same 2026-09-02 incident: reconcile_positions's
    own hedge order was hardcoded to the FULL detected position size, unconditionally,
    on top of whatever the ladder's own rungs already had resting as reduceOnly
    coverage. With SHORT 9840 open and 9000 of that already covered by three resting
    buy rungs, the hedge still asked for the full 9840 -- 18840 total reduceOnly BUY
    resting against a real 9840 short.
    """
    ex = FakeExchange(positions=[{"side": "short", "contracts": 9840.0, "entryPrice": 0.0700}])
    # Wide enough that the computed hedge price (below entry, for a short) stays
    # inside the grid regardless of spacing -- make_engine's own default bounds
    # (0.0710-0.0730) sit entirely ABOVE this test's 0.0700 entry.
    grid = make_engine(ex, grid_lower=0.0600, grid_upper=0.0800, grid_count=10)
    grid.levels = [
        GridLevel(price=0.0700, side="sell", quantity=1.0, entry_price=0.0700),
        GridLevel(price=0.0670, side="buy", order_id="B1", quantity=4000.0),
        GridLevel(price=0.0675, side="buy", order_id="B2", quantity=4000.0),
        GridLevel(price=0.0680, side="buy", order_id="B3", quantity=1000.0),
    ]

    grid.reconcile_positions()

    new_reduce_only_buy = sum(
        o["amount"] for o in ex.placed
        if o.get("side") == "buy" and o.get("params", {}).get("reduceOnly")
    )
    already_resting = 4000.0 + 4000.0 + 1000.0
    assert new_reduce_only_buy + already_resting <= 9840.0 + 1e-6, (
        f"hedge ({new_reduce_only_buy}) + already-resting ({already_resting}) reduceOnly "
        f"buys totalled {new_reduce_only_buy + already_resting}, over the real 9840 short"
    )
    assert new_reduce_only_buy == pytest.approx(840.0), "must only ask for the remaining uncovered 840"


def test_buy_while_net_long_is_not_reduce_only():
    """Mirror case: a BUY adds to a long, so it can never be reduceOnly."""
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.set_position_limit(long_position=5000.0, short_position=0.0, max_position_qty=8000.0)

    params, _ = grid._exit_order_params("buy", 1000.0)
    assert params is None


def test_flat_position_never_sends_reduce_only():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.set_position_limit(long_position=0.0, short_position=0.0, max_position_qty=8000.0)

    assert grid._exit_order_params("sell", 100.0)[0] is None
    assert grid._exit_order_params("buy", 100.0)[0] is None


# --- #12: unwind over-allocated every exit level ---------------------------

def test_unwind_slices_the_position_not_the_grid_notional():
    """The unwind must distribute the ACTUAL position across exit levels.

    It used to size every level at the full grid notional, so it asked to close
    len(levels) x grid_qty against a smaller position and Binance rejected the
    overflow (105 rejections in one session).
    """
    position = 3000.0
    ex = FakeExchange(
        positions=[{"side": "long", "contracts": position, "entryPrice": 0.0723}],
        reject_reduce_only_over=position,
    )
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=position, short_position=0.0, max_position_qty=8000.0)

    grid._unwind_position_through_grid(balance=5000)

    reduce_orders = [o for o in ex.placed if o["params"].get("reduceOnly")]
    assert reduce_orders, "unwind should place exit orders"
    total = sum(o["amount"] for o in reduce_orders)
    assert total <= position + 1e-6, (
        f"unwind tried to close {total} against a {position} position"
    )


def test_unwind_places_no_order_exceeding_the_position():
    position = 500.0
    ex = FakeExchange(
        positions=[{"side": "long", "contracts": position, "entryPrice": 0.0723}],
        reject_reduce_only_over=position,
    )
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=position, short_position=0.0, max_position_qty=8000.0)

    grid._unwind_position_through_grid(balance=5000)
    for o in ex.placed:
        assert o["amount"] <= position + 1e-6


# --- #13: dead-grid false positive drove the recenter loop -----------------

def test_capped_long_with_exit_sells_is_not_a_dead_grid():
    """The exact production state that looped: long at the cap, buys blocked by the
    position limit, exit sells resting above price. That is a healthy unwind, not a
    dead grid — recentering it cancels the exits that were about to fill."""
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=9148.0, short_position=0.0, max_position_qty=8197.0)
    assert grid._block_buys, "precondition: the position cap blocked the buy side"

    for lv in grid.levels:
        lv.order_id = "x" if lv.side == "sell" else None

    grid._last_recenter_time = 0.0
    assert grid.recenter(0.0719, balance=5000, margin_pct=0.02) is False, (
        "recenter fired on a healthy one-sided unwind — this is the 196s thrash loop"
    )


def test_genuinely_dead_grid_still_recenters():
    """The real failure the check exists for: no position, no reason for the missing
    side, and price stranded below every resting sell."""
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=0.0, short_position=0.0, max_position_qty=8000.0)
    assert not grid._block_buys

    for lv in grid.levels:
        lv.order_id = "x" if lv.side == "sell" else None

    grid._last_recenter_time = 0.0
    assert grid.recenter(0.0711, balance=5000, margin_pct=0.02) is True


def test_capped_short_with_exit_buys_is_not_a_dead_grid():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=0.0, short_position=9000.0, max_position_qty=8000.0)
    assert grid._block_sells

    for lv in grid.levels:
        lv.order_id = "x" if lv.side == "buy" else None

    grid._last_recenter_time = 0.0
    assert grid.recenter(0.0729, balance=5000, margin_pct=0.02) is False


# --- #14: trailing stop-loss was not a ratchet -----------------------------

def test_long_trailing_stop_never_moves_down():
    ex = FakeExchange()
    grid = make_engine(ex)

    grid.update_trailing_sl(0.0730)
    high_water = grid.get_stop_loss_price()

    for price in (0.0725, 0.0718, 0.0705, 0.0690):
        grid.update_trailing_sl(price)
        assert grid.get_stop_loss_price() >= high_water - 1e-12, (
            f"long stop dropped from {high_water} to {grid.get_stop_loss_price()} at price {price}"
        )


def test_long_trailing_stop_still_rises_with_price():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.update_trailing_sl(0.0730)
    first = grid.get_stop_loss_price()
    grid.update_trailing_sl(0.0800)
    assert grid.get_stop_loss_price() > first


def test_short_trailing_stop_never_moves_up():
    ex = FakeExchange()
    grid = make_engine(ex)

    grid.update_trailing_sl_short(0.0710)
    low_water = grid.get_short_stop_loss_price()

    for price in (0.0715, 0.0725, 0.0740):
        grid.update_trailing_sl_short(price)
        assert grid.get_short_stop_loss_price() <= low_water + 1e-12


def test_side_flip_releases_the_ratchet():
    """reset_trailing() is the one sanctioned way to release the ratchet."""
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.update_trailing_sl(0.0730)
    assert grid.get_stop_loss_price() > 0

    grid.reset_trailing()
    assert grid._trailing_sl_price is None
    assert grid._peak_price == 0.0


# --- #15: recenter reset the trailing anchor mid-position ------------------

def test_recenter_preserves_trailing_stop_while_long_is_open():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=5000.0, short_position=0.0, max_position_qty=8000.0)

    grid.update_trailing_sl(0.0760)
    protected = grid.get_stop_loss_price()

    grid._last_recenter_time = 0.0
    grid.recenter(0.0700, balance=5000, margin_pct=0.001)

    assert grid._peak_price == pytest.approx(0.0760), "recenter wiped the high-water mark"
    assert grid.get_stop_loss_price() >= protected - 1e-12


def test_recenter_preserves_trailing_stop_when_pos_qty_shows_a_position_but_net_mirror_is_stale():
    """AUDIT (rejected-finding re-examination, order-placement round). recenter()'s
    trailing-stop reanchor block checked ONLY _net_long_qty/_net_short_qty -- the
    exchange-position mirror, refreshed by set_position_limit/_refresh_net_counters
    and left at its LAST value on a failed read rather than raising -- while the
    'flat' gate a few dozen lines above it in the same function was already
    hardened to also check _pos_qty, this ladder's own locally-updated tracker,
    current the instant a fill is applied. The trailing-stop block never got the
    same fix.

    test_recenter_preserves_trailing_stop_while_long_is_open (above) only ever
    exercises the case where set_position_limit has already run and the mirror
    agrees with _pos_qty -- it cannot catch this gap. This pins the case the
    mirror hasn't caught up yet: _pos_qty shows a real long (set the way a fill
    actually sets it) while _net_long_qty/_net_short_qty are still their default
    0.0, exactly the state a failed exchange-position read leaves them in.
    """
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid._pos_qty = 5000.0
    grid._pos_entry = 0.0720

    grid.update_trailing_sl(0.0760)
    protected = grid.get_stop_loss_price()

    grid._last_recenter_time = 0.0
    grid.recenter(0.0700, balance=5000, margin_pct=0.001)

    assert grid._peak_price == pytest.approx(0.0760), "recenter wiped the high-water mark"
    assert grid.get_stop_loss_price() >= protected - 1e-12


def test_recenter_reanchors_trailing_stop_when_flat():
    ex = FakeExchange()
    grid = make_engine(ex)
    grid.initialize(0.0720, balance=5000)
    grid.set_position_limit(long_position=0.0, short_position=0.0, max_position_qty=8000.0)
    grid.update_trailing_sl(0.0760)

    grid._last_recenter_time = 0.0
    grid.recenter(0.0700, balance=5000, margin_pct=0.001)

    assert grid._trailing_sl_price is None, "with no position open the anchor should reset"
