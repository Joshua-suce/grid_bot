"""Regression tests for the 2026-08-20 dormancy incident.

Two wedges killed liveness that day:

  A. A held trend position whose only resting order was an UNTRACKED stop leg --
     the watchdog counted zero working orders forever and restarted the bot every
     45 minutes ("DORMANT WITH EXPOSURE"). The fix rests a tracked reduce-only
     take-profit against every filled entry, so a held position always works
     something the watchdog can see.

  B. The exchange-side stop fired behind the bot's back and main.py's side-flip
     handler wiped the ratcheted stop anchor before the software stop-check could
     see the breach -- the strategy carried a ghost "long" forever and never
     re-entered. The fix tells the strategy the truth (reconcile_positions)
     BEFORE resetting the trail, at startup, on side flips, and after any
     router-forced flatten.

These tests use a fake where limit orders REST until explicitly filled -- unlike
test_trend_follower.py's fill_immediately fake, which cannot model a resting
target.
"""

from pathlib import Path

import pytest

import exchange as exchange_mod
from trend_follower import MIN_NOTIONAL_USDT, TrendFollower

SYMBOL = "ADAUSDT"


class RestingFakeExchange:
    """Limit orders rest until fill() is called; positions are explicit."""

    def __init__(self, price=0.20):
        self.price = price
        self.placed: list[dict] = []
        self.cancelled: list[str] = []
        self.closed = 0
        self.close_args: list[tuple] = []
        self.last_limit_params: dict | None = None
        self.place_error: Exception | None = None
        self.cancel_fail = False
        self._orders: dict[str, dict] = {}
        self._next = 0
        self._positions: list[dict] = []

        class _inner:
            @staticmethod
            def amount_to_precision(symbol, amount):
                return f"{float(amount):.0f}"

            @staticmethod
            def price_to_precision(symbol, price):
                return f"{float(price):.5f}"

        self.exchange = _inner()

    # ---- market data -------------------------------------------------
    def get_price(self, symbol):
        return self.price

    def get_positions(self, symbol):
        return self._positions

    # ---- order placement --------------------------------------------
    def place_limit_order(self, symbol, side, price, amount, max_attempts=3,
                          params=None, post_only=True, allow_taker_fallback=False):
        if self.place_error is not None:
            raise self.place_error
        self.last_limit_params = dict(params or {})
        self._next += 1
        oid = f"o{self._next}"
        self._orders[oid] = {
            "id": oid,
            "side": side,
            "price": float(price),
            "amount": float(amount),
            "filled": 0.0,
            "average": None,
            "status": "open",
            "reduce_only": bool((params or {}).get("reduceOnly")),
        }
        self.placed.append({"id": oid, "side": side, "price": float(price),
                            "amount": float(amount)})
        return self._orders[oid]

    # ---- order management --------------------------------------------
    def get_open_order_ids(self, symbol):
        return {i for i, o in self._orders.items() if o["status"] == "open"}

    def get_open_orders(self, symbol):
        return [dict(o) for o in self._orders.values() if o["status"] == "open"]

    def fetch_order(self, order_id, symbol):
        return self._orders.get(order_id)

    def cancel_order(self, order_id, symbol):
        if self.cancel_fail:
            return False
        self.cancelled.append(order_id)
        if order_id in self._orders:
            self._orders[order_id]["status"] = "canceled"
        return True

    def cancel_everything(self, symbol, timeout_seconds=300.0, keep_stops=False):
        n = len(self.get_open_order_ids(symbol))
        for o in self._orders.values():
            if o["status"] == "open":
                o["status"] = "canceled"
        return n

    def close_position(self, symbol, side, amount, max_attempts=None):
        self.closed += 1
        self.close_args.append((symbol, side, amount))
        self._positions = []
        return {"id": "mkt-close", "status": "closed"}

    # ---- test helpers --------------------------------------------------
    def fill(self, order_id, price=None):
        o = self._orders[order_id]
        o["status"] = "closed"
        o["filled"] = o["amount"]
        o["average"] = float(price) if price is not None else o["price"]
        if o.get("reduce_only"):
            self._positions = []      # a reduce-only fill closes, never flips
        else:
            side = "long" if o["side"] == "buy" else "short"
            self._positions = [{"side": side, "contracts": o["amount"],
                                "entryPrice": o["average"], "symbol": SYMBOL}]

    def hold_long(self, qty=600.0, entry=0.20):
        self._positions = [{"side": "long", "contracts": qty,
                            "entryPrice": entry, "symbol": SYMBOL}]


def make(price=0.20, **overrides):
    ex = RestingFakeExchange(price)
    kwargs = dict(
        exchange=ex,
        symbol=SYMBOL,
        capital_pct=0.10,
        stop_loss_pct=0.02,
        atr_stop_multiplier=2.0,
        take_profit_r=3.0,
        leverage=1,
        min_hold_seconds=0,
    )
    kwargs.update(overrides)
    tf = TrendFollower(**kwargs)
    tf._atr_pct = 0.01
    tf.update_regime("uptrend")
    tf.activate(5000.0)
    return tf, ex


def enter_long(tf, ex, balance=5000.0):
    """Drive one full entry cycle: rest the entry, fill it, detect the fill.

    activate() already places the entry when the regime supports a side, so only
    place when nothing rests yet.
    """
    if tf._order_id is None:
        tf.place_initial_orders(balance)
    assert tf._order_id is not None, "entry never rested"
    ex.fill(tf._order_id)
    fills = tf.check_fills(1000)
    assert tf._side == "long"
    assert tf._qty > 0
    return fills


# ---------------------------------------------------------------------------
# Wedge A: a held position must always work a TRACKED order
# ---------------------------------------------------------------------------

class TestHeldPositionAlwaysWorksSomething:
    def test_a_filled_entry_arms_a_resting_tracked_target(self):
        tf, ex = make()
        enter_long(tf, ex)

        assert tf._tp_order_id is not None
        assert tf._take_profit_price is not None
        assert tf._take_profit_price > tf._entry_price

        book = ex.get_open_order_ids(SYMBOL)
        assert tf._tp_order_id in book, "target must actually rest on the book"

    def test_the_target_is_reduce_only_and_tagged(self):
        tf, ex = make()
        enter_long(tf, ex)

        params = ex.last_limit_params
        assert params.get("reduceOnly") is True, \
            "an un-reduced target fill would flip us into a phantom short"
        assert exchange_mod.PURPOSE_TAGS.get(params.get("purpose")) == "tp"

    def test_watchdog_counting_sees_the_target_as_working(self):
        """The exact arithmetic main.py's dormancy clock performs (AUDIT #130)."""
        tf, ex = make()
        enter_long(tf, ex)

        open_orders = ex.get_open_orders(SYMBOL)
        tracked = tf.get_tracked_order_ids()
        working = sum(1 for o in open_orders if str(o.get("id")) in tracked)
        assert working >= 1, \
            "a held position with zero working orders is what restarted the bot hourly"

    def test_a_resting_target_that_fills_books_the_exit_and_rearms(self):
        tf, ex = make()
        enter_long(tf, ex)
        tp_id = tf._tp_order_id

        ex.fill(tp_id)
        fills = tf.check_fills(2000)

        assert len(fills) == 1
        assert fills[0]["reason"] == "take_profit"
        assert fills[0]["profit"] > 0
        assert tf._side is None and tf._qty == 0.0
        assert tf._tp_order_id is None

        # Liveness: the very next poll re-arms a fresh entry.
        tf.check_fills(2100)
        assert tf._order_id is not None
        assert tf._order_id in ex.get_open_order_ids(SYMBOL)


# ---------------------------------------------------------------------------
# Wedge B: the exchange closing us out behind our back must not wedge us
# ---------------------------------------------------------------------------

class TestExternalCloseIsNoticed:
    def test_reconcile_clears_a_position_the_exchange_already_closed(self):
        tf, ex = make()
        enter_long(tf, ex)
        stray_tp = tf._tp_order_id

        ex._positions = []          # the stop leg fired; we never saw it
        tf.reconcile_positions()

        assert tf._side is None
        assert tf._qty == 0.0
        assert tf._tp_order_id is None, "stray target must be disarmed"
        assert stray_tp not in ex.get_open_order_ids(SYMBOL)

        # ...and the strategy can trade again immediately.
        tf.check_fills(3000)
        assert tf._order_id is not None

    def test_software_target_still_fires_when_placement_fails(self):
        tf, ex = make()
        ex.place_error = RuntimeError("rate limited")
        enter_long(tf, ex)

        assert tf._tp_order_id is None, "failed placement must fall back to software"
        target = tf._take_profit_price
        assert target is not None

        ex.price = target * 1.001   # drive through the target
        fills = tf.check_fills(4000)

        assert len(fills) == 1
        assert fills[0]["reason"] == "take_profit"
        assert ex.closed == 1
        assert tf._side is None

    def test_an_unconfirmed_cancel_keeps_the_claim_until_resolved(self):
        tf, ex = make()
        enter_long(tf, ex)
        tp_id = tf._tp_order_id

        ex.cancel_fail = True
        tf.update_regime("ranging")
        tf.place_initial_orders(5000)     # regime change -> try to flatten

        assert tf._tp_order_id == tp_id, \
            "a cancel we could not confirm must keep its claim"

        # The exchange processes the cancel later; the next poll resolves it.
        ex.cancel_fail = False
        ex._orders[tp_id]["status"] = "canceled"
        tf.check_fills(5000)
        assert tf._tp_order_id is None


# ---------------------------------------------------------------------------
# Restarts: state files, lost targets, legacy formats
# ---------------------------------------------------------------------------

class TestRestartPaths:
    def test_persistence_roundtrip_keeps_the_target(self):
        tf, ex = make()
        enter_long(tf, ex)

        tf2, ex2 = make()
        tf2.load_from_dict(tf.to_dict(), ex2.price)

        assert tf2._tp_order_id == tf._tp_order_id
        assert tf2._take_profit_price == tf._take_profit_price

    def test_a_restart_into_a_held_position_replaces_a_lost_target(self):
        tf, ex = make()
        enter_long(tf, ex)

        tf2 = TrendFollower(exchange=ex, symbol=SYMBOL, capital_pct=0.10,
                            stop_loss_pct=0.02, atr_stop_multiplier=2.0,
                            take_profit_r=3.0, leverage=1, min_hold_seconds=0)
        tf2.load_from_dict(tf.to_dict(), ex.price)
        tf2.active = True
        tf2._tp_order_id = None            # the state file predates the fix

        tf2.reconcile_positions()

        assert tf2._tp_order_id is not None
        assert tf2._tp_order_id in ex.get_open_order_ids(SYMBOL)
        assert tf2._take_profit_price is not None

    def test_legacy_state_gets_a_reconstructed_target(self):
        ex = RestingFakeExchange(0.20)
        ex.hold_long(qty=600.0, entry=0.20)
        tf = TrendFollower(exchange=ex, symbol=SYMBOL, capital_pct=0.10,
                           stop_loss_pct=0.02, atr_stop_multiplier=2.0,
                           take_profit_r=3.0, leverage=1, min_hold_seconds=0)
        tf.active = True
        tf.load_from_dict({
            "side": "long",
            "entry_price": 0.20,
            "qty": 600.0,
            "trailing_sl_price": 0.196,    # pre-fix files carry no tp fields
        }, current_price=0.20)

        tf.reconcile_positions()

        assert tf._tp_order_id is not None
        assert tf._take_profit_price is not None
        assert tf._take_profit_price > 0.20


# ---------------------------------------------------------------------------
# The wiring in main.py / router.py / exchange.py that makes it all stick
# ---------------------------------------------------------------------------

class TestWiring:
    def _src(self, name):
        return Path(__file__).resolve().parent.parent.joinpath(name).read_text()

    def test_main_tells_the_strategy_the_truth_before_resetting_the_trail(self):
        src = self._src("main.py")
        i = src.index("if position_side != _last_side:")
        block = src[i:i + 3000]
        r = block.index("grid.reconcile_positions()")
        t = block.index("grid.reset_trailing()")
        assert r < t, \
            "reset_trailing wipes the stop anchor; the strategy must reconcile FIRST"

    def test_a_failed_reconcile_does_not_commit_the_flip(self):
        """AUDIT #160. reconcile_positions() failing after an external close used to
        still advance _last_side/reset_trailing unconditionally -- committing to a flat
        belief the code had not actually managed to reconcile, and permanently losing
        this branch's own retry (the guard is `position_side != _last_side`; once
        _last_side matches, it never fires again for this flip)."""
        src = self._src("main.py")
        i = src.index("if position_side != _last_side:")
        block = src[i:i + 3000]
        commit_at = block.index("commit_flip = True")
        reconcile_try_at = block.index("grid.reconcile_positions()")
        reconcile_except_at = block.index("except Exception as e:", reconcile_try_at)
        commit_false_at = block.index("commit_flip = False", reconcile_except_at)
        gate_at = block.index("if commit_flip:")
        last_side_at = block.index("_last_side = position_side", gate_at)
        assert commit_at < reconcile_try_at < reconcile_except_at < commit_false_at < gate_at < last_side_at, (
            "a failed reconcile no longer withholds the flip from being committed"
        )

    def test_an_unverifiable_or_still_held_flat_reading_is_not_committed_either(self):
        """The still_held re-verify already fails safe (defaults still_held=True), but
        that alone did nothing if _last_side/reset_trailing then advanced anyway --
        AUDIT #160 closes that gap too."""
        src = self._src("main.py")
        i = src.index("if position_side != _last_side:")
        block = src[i:i + 3000]
        still_held_at = block.index("still_held = True")
        if_still_held_at = block.index("if still_held:", still_held_at)
        commit_false_at = block.index("commit_flip = False", if_still_held_at)
        gate_at = block.index("if commit_flip:", commit_false_at)
        assert if_still_held_at < commit_false_at < gate_at, (
            "a still-held (or unverifiable) flat reading is not withheld from being "
            "committed as though the flip were handled"
        )

    def test_main_reconciles_at_startup_unconditionally(self):
        src = self._src("main.py")
        assert src.count("grid.reconcile_positions()") >= 2, \
            "startup reconcile must not be gated on has_exchange_positions"

    def test_dust_positions_do_not_arm_the_dormancy_clock(self):
        src = self._src("main.py")
        assert "_pos_notional >= MIN_NOTIONAL_USDT" in src

    def test_close_position_is_reduce_only(self):
        src = self._src("exchange.py")
        i = src.index("def close_position")
        body = src[i:src.index("\n    def ", i)]
        assert '"reduceOnly": True' in body, \
            "closing an already-flat account must not open a phantom opposite leg"

    def test_router_tells_the_outgoing_strategy_after_a_forced_flatten(self):
        src = self._src("router.py")
        assert "post-flatten reconcile failed" in src, \
            "a forced close behind the strategy's back leaves a ghost otherwise"

    def test_purpose_tags_know_the_target(self):
        assert exchange_mod.PURPOSE_TAGS.get("trend_tp") == "tp"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
