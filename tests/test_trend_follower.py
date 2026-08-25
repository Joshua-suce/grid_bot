"""Behavioural tests for the trend-following strategy (step 3).

The ratchet tests matter most. AUDIT #14 was a stop that followed the market down and
stopped protecting anything; the same mistake is easy to reproduce in a fresh strategy,
so the monotonicity is pinned here rather than assumed.
"""

import pytest

from trend_follower import TrendFollower


class FakeExchange:
    def __init__(self, price=0.0720):
        self.price = price
        self.placed: list[dict] = []
        self.cancelled: list[str] = []
        self.closed = 0
        self.close_args = []
        self._orders: dict[str, dict] = {}
        self._next = 0
        self._positions: list[dict] = []
        self.fill_immediately = True

        class _inner:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{float(amount):.0f}"

            @staticmethod
            def price_to_precision(symbol, price):
                return f"{float(price):.5f}"

        self.exchange = _inner()

    def get_price(self, symbol):
        return self.price

    def get_positions(self, symbol):
        return self._positions

    def get_open_order_ids(self, symbol):
        return {i for i, o in self._orders.items() if o["status"] == "open"}

    def fetch_order(self, order_id, symbol):
        return self._orders.get(order_id)

    def cancel_order(self, order_id, symbol):
        self.cancelled.append(order_id)
        if order_id in self._orders:
            self._orders[order_id]["status"] = "canceled"
        return True

    def cancel_everything(self, symbol, timeout_seconds=300.0, keep_stops=False):
        return 0

    def close_position(self, symbol, side, amount, max_attempts=None):
        """Signature mirrors the real Exchange exactly -- see AUDIT #38."""
        self.closed += 1
        self.close_args.append((symbol, side, amount))
        self._positions = []
        return True

    def place_limit_order(self, symbol, side, price, amount, max_attempts=3,
                          params=None, post_only=True, allow_taker_fallback=False):
        self._next += 1
        oid = f"o{self._next}"
        status = "closed" if self.fill_immediately else "open"
        order = {"id": oid, "side": side, "price": price, "amount": amount,
                 "filled": amount, "average": price, "status": status}
        self._orders[oid] = order
        self.placed.append(order)
        if status == "closed":
            self._positions = [{
                "side": "long" if side == "buy" else "short",
                "contracts": amount, "entryPrice": price,
            }]
        return order


def make(ex=None, **kw):
    ex = ex or FakeExchange()
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT", min_hold_seconds=0, **kw)
    tf.initialize(ex.price, balance=5000)
    tf.activate(5000)
    return tf, ex


# --- direction -------------------------------------------------------------

def test_enters_long_on_an_uptrend():
    tf, ex = make()
    tf.update_regime("uptrend")
    tf.place_initial_orders(5000)
    assert [o["side"] for o in ex.placed] == ["buy"]


def test_enters_short_on_a_downtrend():
    tf, ex = make()
    tf.update_regime("downtrend")
    tf.place_initial_orders(5000)
    assert [o["side"] for o in ex.placed] == ["sell"]


@pytest.mark.parametrize("regime", ["ranging", "uncertain"])
def test_stays_flat_when_the_regime_gives_no_direction(regime):
    tf, ex = make()
    tf.update_regime(regime)
    tf.place_initial_orders(5000)
    assert ex.placed == []


def test_does_not_stack_a_second_position():
    """place_initial_orders is called every loop iteration; it must be idempotent."""
    tf, ex = make()
    tf.update_regime("uptrend")
    tf.place_initial_orders(5000)
    tf.check_fills(5000)
    for _ in range(5):
        tf.place_initial_orders(5000)
    assert len(ex.placed) == 1


def test_exits_when_the_regime_stops_supporting_the_side():
    tf, ex = make()
    tf.update_regime("uptrend")
    tf.place_initial_orders(5000)
    tf.check_fills(5000)
    assert tf._side == "long"

    tf.update_regime("downtrend")
    tf.place_initial_orders(5000)
    assert ex.closed == 1
    assert tf._side is None


# --- the ratchet -----------------------------------------------------------

def test_long_stop_never_moves_down():
    tf, ex = make()
    tf.update_trailing_sl(0.0730)
    high_water = tf.get_stop_loss_price()
    assert high_water is not None
    for price in (0.0725, 0.0710, 0.0690, 0.0650):
        tf.update_trailing_sl(price)
        assert tf.get_stop_loss_price() >= high_water - 1e-12, (
            f"stop fell to {tf.get_stop_loss_price()} at price {price}"
        )


def test_long_stop_rises_with_a_new_high():
    tf, ex = make()
    tf.update_trailing_sl(0.0730)
    first = tf.get_stop_loss_price()
    tf.update_trailing_sl(0.0800)
    assert tf.get_stop_loss_price() > first


def test_short_stop_never_moves_up():
    tf, ex = make()
    tf.update_trailing_sl_short(0.0700)
    low_water = tf.get_short_stop_loss_price()
    assert low_water is not None
    for price in (0.0710, 0.0730, 0.0780):
        tf.update_trailing_sl_short(price)
        assert tf.get_short_stop_loss_price() <= low_water + 1e-12


def test_reset_trailing_releases_the_ratchet():
    tf, ex = make()
    tf.update_trailing_sl(0.0730)
    tf.reset_trailing()
    assert tf.get_stop_loss_price() is None
    assert tf._peak_price == 0.0


def test_stop_is_set_at_entry_not_left_none():
    """A position with no stop is unprotected until the next tick."""
    tf, ex = make()
    tf.update_regime("uptrend")
    tf.place_initial_orders(5000)
    tf.check_fills(5000)
    assert tf.get_stop_loss_price() is not None


def test_stop_distance_floors_at_stop_loss_pct_in_a_quiet_market():
    tf, ex = make(stop_loss_pct=0.03)
    tf.update_volatility(0.0001)          # near-zero ATR
    assert tf._stop_distance(0.0720) == pytest.approx(0.0720 * 0.03)


# --- exit guards -----------------------------------------------------------

def test_trailing_stop_closes_the_position():
    ex = FakeExchange(price=0.0720)
    tf, _ = make(ex)
    tf.update_regime("uptrend")
    tf.place_initial_orders(5000)
    tf.check_fills(5000)

    ex.price = 0.0600                      # well through the stop
    tf.check_fills(5000)
    assert ex.closed == 1
    assert tf._side is None


def test_min_hold_prevents_an_instant_stop_out():
    """Without this the position can close on the same tick it opened, paying two
    taker fees for zero exposure."""
    ex = FakeExchange(price=0.0720)
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT", min_hold_seconds=3600)
    tf.initialize(ex.price, 5000)
    tf.activate(5000)
    tf.update_regime("uptrend")
    tf.place_initial_orders(5000)
    tf.check_fills(5000)

    ex.price = 0.0500
    tf.check_fills(5000)
    assert ex.closed == 0, "closed inside the minimum hold window"
    assert tf._side == "long"


# --- sizing ----------------------------------------------------------------

def test_entry_is_clamped_by_the_position_cap():
    tf, ex = make(capital_pct=0.50)
    tf.set_position_limit(0.0, 0.0, max_position_qty=1000.0)
    tf.update_regime("uptrend")
    tf.place_initial_orders(5000)
    assert ex.placed[0]["amount"] <= 1000.0


def test_entry_below_minimum_notional_is_skipped():
    tf, ex = make(capital_pct=0.00001)
    tf.update_regime("uptrend")
    assert tf.place_initial_orders(5000) == 0
    assert ex.placed == []


# --- lifecycle -------------------------------------------------------------

def test_pause_cancels_the_entry_but_keeps_the_position():
    ex = FakeExchange()
    ex.fill_immediately = False
    tf, _ = make(ex)
    tf.update_regime("uptrend")
    tf.place_initial_orders(5000)
    assert tf._order_id is not None

    tf.pause()
    assert ex.cancelled, "resting entry order was not cancelled"
    assert ex.closed == 0, "pause must not flatten -- that is the router's decision"
    assert tf.active is False


def test_paused_strategy_places_nothing():
    tf, ex = make()
    tf.pause()
    tf.update_regime("uptrend")
    assert tf.place_initial_orders(5000) == 0


# --- persistence -----------------------------------------------------------

def test_state_round_trips():
    tf, ex = make()
    tf.update_regime("uptrend")
    tf.place_initial_orders(5000)
    tf.check_fills(5000)
    snapshot = tf.to_dict()

    restored = TrendFollower(exchange=FakeExchange(), symbol="DOGEUSDT")
    restored.load_from_dict(snapshot, current_price=0.0720)
    assert restored._side == tf._side
    assert restored._qty == tf._qty
    assert restored._entry_price == tf._entry_price
    assert restored.get_stop_loss_price() == tf.get_stop_loss_price()
    assert restored.state_corrupted is False


def test_corrupt_state_is_flagged_not_traded_on():
    tf = TrendFollower(exchange=FakeExchange(), symbol="DOGEUSDT")
    tf.load_from_dict({"qty": "not-a-number"}, current_price=0.0720)
    assert tf.state_corrupted is True


def test_state_claiming_a_position_with_no_quantity_is_cleared():
    tf = TrendFollower(exchange=FakeExchange(), symbol="DOGEUSDT")
    tf.load_from_dict({"side": "long", "qty": 0.0}, current_price=0.0720)
    assert tf._side is None


# --- reconciliation --------------------------------------------------------

def test_reconcile_adopts_the_exchange_position():
    ex = FakeExchange()
    ex._positions = [{"side": "long", "contracts": 4000.0, "entryPrice": 0.0715}]
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT")
    tf.reconcile_positions()
    assert tf._side == "long"
    assert tf._qty == 4000.0
    assert tf._entry_price == pytest.approx(0.0715)


def test_reconcile_clears_a_position_the_exchange_does_not_have():
    ex = FakeExchange()
    tf, _ = make(ex)
    tf._side, tf._qty, tf._entry_price = "long", 4000.0, 0.0715
    ex._positions = []
    tf.reconcile_positions()
    assert tf._side is None


def test_clearing_a_phantom_position_also_clears_its_regime():
    """AUDIT #146. A restored state file's claimed position and its regime label are
    exactly as stale as each other -- proving one false and trusting the other is how
    a bot with a flat exchange still opens a fresh trade on the next activate().

    ADAUSDT 2026-08-25 00:01: cleanup.py flattened the book the night before while the
    bot was stopped. On restart, load_from_dict restored _side="long" and
    _regime="uptrend" from before the flatten. reconcile_positions correctly found the
    exchange flat and cleared _side -- but left _regime alone, and activate() (called
    moments later, before update_regime ever ran this session) opened a brand new long
    off that stale label alone.
    """
    ex = FakeExchange()
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT")
    tf.load_from_dict(
        {"side": "long", "qty": 4000.0, "entry_price": 0.0715, "regime": "uptrend"},
        current_price=0.0720,
    )
    assert tf._regime == "uptrend"      # sanity: the stale claim really did restore

    ex._positions = []                  # the exchange disagrees: flat
    tf.reconcile_positions()

    assert tf._side is None
    assert tf._regime == "uncertain", (
        "the phantom position's regime survived the position that carried it"
    )

    tf.active = True
    tf.place_initial_orders(5000)
    assert ex.placed == [], "opened a fresh position on a regime reconciliation just disproved"


def test_loading_state_revokes_a_prior_live_confirmation():
    """load_from_dict can run on an instance that already had a genuine, live-confirmed
    regime -- restoring on top of it must not leave that confirmation standing, or a
    stale restored regime would be trusted as if it were the live one that preceded it."""
    tf = TrendFollower(exchange=FakeExchange(), symbol="DOGEUSDT")
    tf.update_regime("uptrend")
    assert tf._regime_confirmed is True

    tf.load_from_dict({"regime": "downtrend"}, current_price=0.0720)

    assert tf._regime_confirmed is False


def test_a_genuinely_resumed_position_keeps_its_regime():
    """The mirror case: when the exchange DOES still hold what state claimed, the
    regime that explains why must survive -- it is what lets a later regime flip
    still trigger the regime_change exit."""
    ex = FakeExchange()
    ex._positions = [{"side": "long", "contracts": 4000.0, "entryPrice": 0.0715}]
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT")
    tf.load_from_dict(
        {"side": "long", "qty": 4000.0, "entry_price": 0.0715, "regime": "uptrend"},
        current_price=0.0720,
    )

    tf.reconcile_positions()

    assert tf._side == "long"
    assert tf._regime == "uptrend"


def test_reconcile_handles_negative_contracts_encoding():
    """Binance reports a short as side=long with negative contracts in some payloads."""
    ex = FakeExchange()
    ex._positions = [{"side": "long", "contracts": -4000.0, "entryPrice": 0.0715}]
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT")
    tf.reconcile_positions()
    assert tf._side == "short"
    assert tf._qty == 4000.0


# --- #30: the strategy must arm itself, not wait to be driven --------------

def test_activating_into_a_known_regime_opens_immediately():
    """GridEngine.activate places its ladder there and then. This must too: the router
    calls activate() at the end of a handoff and nothing else in main.py's loop opens a
    trend position."""
    ex = FakeExchange()
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT", min_hold_seconds=0)
    tf.initialize(ex.price, balance=5000)
    tf.update_regime("uptrend")

    tf.activate(5000)

    assert [o["side"] for o in ex.placed] == ["buy"]


def test_entry_is_armed_from_check_fills_like_the_live_loop():
    """AUDIT #30. main.py calls place_initial_orders exactly once, at startup, before
    the loop; inside the loop the grid re-places filled levels from within check_fills.
    A trend follower whose only entry path was place_initial_orders would take over on
    a confirmed trend and then stand flat for the entire move -- the bot would look
    dormant in exactly the conditions the router exists to trade. So check_fills, which
    does run every iteration, arms the entry too."""
    ex = FakeExchange()
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT", min_hold_seconds=0)
    tf.initialize(ex.price, balance=5000)
    tf.active = True                 # activated without a regime yet
    tf.update_regime("uptrend")      # regime arrives afterwards, as it does live

    tf.check_fills(5000)             # the only call main.py's loop makes

    assert [o["side"] for o in ex.placed] == ["buy"], "no position opened during a trend"


def test_being_driven_twice_in_one_iteration_still_opens_one_position():
    """The backtester calls check_fills and place_initial_orders in the same candle."""
    ex = FakeExchange()
    ex.fill_immediately = False
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT", min_hold_seconds=0)
    tf.initialize(ex.price, balance=5000)
    tf.activate(5000)
    tf.update_regime("uptrend")

    tf.check_fills(5000)
    tf.place_initial_orders(5000)
    tf.check_fills(5000)

    assert len(ex.placed) == 1, f"placed {len(ex.placed)} entries for one signal"


def test_no_entry_is_armed_while_the_regime_is_undecided():
    ex = FakeExchange()
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT", min_hold_seconds=0)
    tf.initialize(ex.price, balance=5000)
    tf.activate(5000)
    tf.update_regime("ranging")

    tf.check_fills(5000)

    assert ex.placed == []


def test_a_paused_strategy_does_not_arm_itself():
    ex = FakeExchange()
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT", min_hold_seconds=0)
    tf.initialize(ex.price, balance=5000)
    tf.update_regime("uptrend")
    tf.active = False

    tf.check_fills(5000)

    assert ex.placed == []


def test_price_is_never_outside_a_strategy_that_has_no_range():
    """main.py's escape checks (`price < grid_lower`, `price > grid_upper`) run against
    whatever the router has live. A zero-width range at the origin makes the upper test
    true at every price, which logged GRID EXIT and journalled an event every iteration
    while the follower was flat (AUDIT #31)."""
    tf, ex = make()
    for price in (0.00001, 0.0705, 1_000_000.0):
        assert not (price < tf.grid_lower)
        assert not (price > tf.grid_upper)


# --- handoff acceleration (AUDIT #148) --------------------------------------
#
# GridEngine has had accelerate_handoff_exit since AUDIT #145 (tests/test_handoff_accel.py);
# this strategy never did, so a trend-to-grid handoff had nothing to accelerate and just
# waited out the full grace period. 2026-08-24/25: 31 minutes of "still waiting for flat"
# with zero repricing, ending in a forced market close at grace expiry.

def positioned(ex=None, side="long", entry=0.2179, qty=114.0, price_now=0.2182,
               tp=None, tp_order_id=None, take_profit_r=3.0):
    """A TrendFollower already holding a position, without driving the full
    entry-fill pipeline -- mirrors tests/test_handoff_accel.py's engine() helper for
    the same reason: this is a unit test of accelerate_handoff_exit's own decision,
    not of order placement mechanics."""
    ex = ex or FakeExchange(price=price_now)
    ex.fill_immediately = False
    tf = TrendFollower(exchange=ex, symbol="DOGEUSDT", min_hold_seconds=0, take_profit_r=take_profit_r)
    tf.active = True
    tf._side = side
    tf._entry_price = entry
    tf._qty = qty
    tf._take_profit_price = tp
    tf._tp_order_id = tp_order_id
    return tf, ex


# --------------------------------------------------------------- the live incident
def test_a_target_far_from_market_is_repriced_toward_break_even():
    tf, ex = positioned(tp=0.2450, tp_order_id="old-tp")

    result = tf.accelerate_handoff_exit(0.2182, balance=4866.53)

    assert result is True
    assert tf._take_profit_price < 0.2450, "the target was left exactly where it started"
    assert tf._take_profit_price >= tf._entry_price - 1e-9, (
        "repricing must never quote below the position's own entry price"
    )
    assert "old-tp" in ex.cancelled, "the stale target was never cancelled"
    assert ex.placed, "nothing was placed at the new target"
    assert ex.placed[-1]["side"] == "sell"


def test_price_below_entry_still_clamps_to_break_even_not_a_loss():
    """The clamp only does real work when price has moved PAST entry -- the case
    above (price still above entry for a long) would pass even with no clamp at all,
    since max(entry, price) == price there regardless. This is the case that proves
    it: price has moved against the position, and the target must still floor at
    entry rather than follow price down into a loss."""
    tf, ex = positioned(entry=0.2179, price_now=0.2150, tp=0.2450, tp_order_id="old-tp")

    result = tf.accelerate_handoff_exit(0.2150, balance=4866.53)

    assert result is True
    assert tf._take_profit_price == tf._entry_price, (
        f"target {tf._take_profit_price} followed price below entry {tf._entry_price}"
    )


def test_a_short_is_the_mirror():
    """The exit for a short is a BUY, and 'stale' means too far BELOW market -- moving
    it UP toward break-even is the improvement, not down."""
    tf, ex = positioned(side="short", entry=0.2179, price_now=0.2176,
                         tp=0.1900, tp_order_id="old-tp")

    result = tf.accelerate_handoff_exit(0.2176, balance=4866.53)

    assert result is True
    assert tf._take_profit_price > 0.1900, "a short's stale target must move UP toward the market"
    assert tf._take_profit_price <= tf._entry_price + 1e-9
    assert ex.placed[-1]["side"] == "buy"


def test_price_above_entry_still_clamps_a_short_to_break_even_not_a_loss():
    """The mirror of the long clamp test: price has moved AGAINST the short (up, past
    entry), and the buy-to-cover target must floor at entry rather than chase price
    up into a loss."""
    tf, ex = positioned(side="short", entry=0.2179, price_now=0.2210,
                         tp=0.1900, tp_order_id="old-tp")

    result = tf.accelerate_handoff_exit(0.2210, balance=4866.53)

    assert result is True
    assert tf._take_profit_price == tf._entry_price, (
        f"target {tf._take_profit_price} followed price above entry {tf._entry_price}"
    )


def test_a_position_with_no_configured_target_gets_one():
    """trend_take_profit_r defaults to 0.0 -- ride the trailing stop only, no resting
    exit at all. This is very likely what the live incident actually held: nothing to
    reprice because nothing was ever armed in the first place. The handoff still needs
    a real order to close through, so acceleration must create one, not require one."""
    tf, ex = positioned(tp=None, tp_order_id=None, take_profit_r=0.0)
    assert tf._take_profit_price is None

    result = tf.accelerate_handoff_exit(0.2182, balance=4866.53)

    assert result is True
    assert tf._take_profit_price is not None
    assert tf._take_profit_price >= tf._entry_price - 1e-9
    assert ex.placed and ex.placed[-1]["side"] == "sell"


# ------------------------------------------------------------------------ inertness
def test_flat_position_does_nothing():
    tf, ex = positioned()
    tf._side = None
    tf._qty = 0.0

    assert tf.accelerate_handoff_exit(0.2182, balance=1000) is False
    assert ex.placed == []


def test_the_stop_is_never_touched():
    """Only the take-profit ever moves here -- accelerating the stop would mean
    voluntarily taking a WORSE exit, the opposite of the point."""
    tf, ex = positioned(tp=0.2450, tp_order_id="old-tp")
    tf.update_trailing_sl(0.2182)
    stop_before = tf.get_stop_loss_price()

    tf.accelerate_handoff_exit(0.2182, balance=4866.53)

    assert tf.get_stop_loss_price() == stop_before


def test_a_target_already_at_break_even_or_better_is_left_alone():
    tf, ex = positioned(tp=0.2180, tp_order_id="old-tp")   # already inside break-even

    result = tf.accelerate_handoff_exit(0.2182, balance=1000)

    assert result is False
    assert ex.cancelled == []
    assert tf._take_profit_price == 0.2180


# ------------------------------------------------------------------------- cooldown
def test_a_second_call_inside_the_cooldown_is_a_no_op():
    tf, ex = positioned(tp=0.2450, tp_order_id="old-tp")
    assert tf.accelerate_handoff_exit(0.2182, balance=1000) is True

    # Put the target back exactly where it would be genuinely improvable again -- the
    # cooldown, not "nothing left to improve", must be what blocks the second call.
    reprised_id = tf._tp_order_id
    tf._take_profit_price = 0.2450
    tf._tp_order_id = "old-tp-2"
    ex.cancelled = []

    assert tf.accelerate_handoff_exit(0.2182, balance=1000) is False, (
        "repriced again inside HANDOFF_ACCEL_COOLDOWN_SECONDS"
    )
    assert ex.cancelled == []
    assert tf._tp_order_id == "old-tp-2"


# ------------------------------------------------------------- unconfirmed cancels
def test_an_unconfirmed_cancel_keeps_the_target_claimed():
    """Same discipline as _disarm_take_profit's own contract: if the exchange cannot
    confirm the cancel, the target must stay exactly as it was."""
    tf, ex = positioned(tp=0.2450, tp_order_id="live-order")

    def refuse_cancel(order_id, symbol):
        return False
    ex.cancel_order = refuse_cancel

    result = tf.accelerate_handoff_exit(0.2182, balance=1000)

    assert result is False
    assert tf._tp_order_id == "live-order"
    assert tf._take_profit_price == 0.2450
    assert ex.placed == []
