import pytest
from loguru import logger

import main as main_module
from main import (
    build_scale_out_orders,
    get_net_position,
    get_position_details,
    get_short_position,
    get_total_position,
    _position_unrealized_pnl,
)


class FakeExchange:
    def __init__(self, positions):
        self._positions = positions

    def get_positions(self, symbol):
        return self._positions


def test_get_total_position_counts_only_longs():
    exchange = FakeExchange([
        {"side": "long", "contracts": 10},
        {"side": "short", "contracts": 5},
    ])
    assert get_total_position(exchange, "DOGEUSDT") == 10.0


def test_get_net_position_long():
    exchange = FakeExchange([{"side": "long", "contracts": 10}])
    assert get_net_position(exchange, "DOGEUSDT") == ("long", 10.0)


def test_get_net_position_short():
    exchange = FakeExchange([{"side": "short", "contracts": 5}])
    assert get_net_position(exchange, "DOGEUSDT") == ("short", 5.0)


def test_get_net_position_flat():
    exchange = FakeExchange([])
    assert get_net_position(exchange, "DOGEUSDT") == ("", 0.0)


def test_get_net_position_negative_contracts_short():
    exchange = FakeExchange([{"side": "long", "contracts": -5}])
    assert get_net_position(exchange, "DOGEUSDT") == ("short", 5.0)


def test_get_net_position_nets_long_and_short():
    exchange = FakeExchange([
        {"side": "long", "contracts": 10},
        {"side": "short", "contracts": 4},
    ])
    assert get_net_position(exchange, "DOGEUSDT") == ("long", 6.0)


def test_get_short_position_returns_qty_and_entry():
    exchange = FakeExchange([{"side": "short", "contracts": 5, "entryPrice": 0.07}])
    assert get_short_position(exchange, "DOGEUSDT") == (5.0, 0.07)


def test_get_short_position_negative_contracts_encoding():
    exchange = FakeExchange([{"side": "long", "contracts": -5, "entryPrice": 0.07}])
    assert get_short_position(exchange, "DOGEUSDT") == (5.0, 0.07)


def test_get_short_position_ignores_longs_and_flat():
    exchange = FakeExchange([
        {"side": "long", "contracts": 10, "entryPrice": 0.06},
        {"side": "long", "contracts": -5, "entryPrice": 0.07},
    ])
    assert get_short_position(exchange, "DOGEUSDT") == (5.0, 0.07)
    assert get_short_position(FakeExchange([]), "DOGEUSDT") == (0.0, 0.0)


def test_get_short_position_weighted_entry_across_legs():
    exchange = FakeExchange([
        {"side": "short", "contracts": 100, "entryPrice": 0.08},
        {"side": "short", "contracts": 300, "entryPrice": 0.10},
    ])
    qty, entry = get_short_position(exchange, "DOGEUSDT")
    assert qty == 400.0
    assert entry == pytest.approx((100 * 0.08 + 300 * 0.10) / 400)


def test_get_position_details_includes_short_positions():
    exchange = FakeExchange([
        {"side": "short", "contracts": 5, "entryPrice": 0.07},
        {"side": "long", "contracts": 3, "entryPrice": 0.069},
    ])
    details = get_position_details(exchange, "DOGEUSDT")
    assert len(details) == 2
    assert any(p["side"] == "short" and p["qty"] == 5.0 for p in details)
    assert any(p["side"] == "long" and p["qty"] == 3.0 for p in details)


def test_get_position_details_passes_through_exchange_unrealized_pnl():
    """The exchange's own mark-price-based unrealizedPnl should pass through
    verbatim rather than being silently dropped/recomputed -- see AUDIT.md
    follow-up on not re-deriving numbers the exchange already provides."""
    exchange = FakeExchange([
        {"side": "long", "contracts": 3, "entryPrice": 0.069, "unrealizedPnl": 1.23},
    ])
    details = get_position_details(exchange, "DOGEUSDT")
    assert details[0]["unrealized_pnl"] == 1.23


def test_get_position_details_unrealized_pnl_none_when_exchange_omits_it():
    exchange = FakeExchange([{"side": "long", "contracts": 3, "entryPrice": 0.069}])
    details = get_position_details(exchange, "DOGEUSDT")
    assert details[0]["unrealized_pnl"] is None


def test_position_unrealized_pnl_prefers_exchange_field():
    pos = {"side": "short", "entry_price": 0.07, "qty": 5.0, "unrealized_pnl": -9.99}
    # Even though the manual short-side formula would give a different (positive)
    # number here, the exchange's own figure must win.
    assert _position_unrealized_pnl(pos, current_price=0.06) == -9.99


def test_position_unrealized_pnl_falls_back_for_long_when_exchange_omits_it():
    pos = {"side": "long", "entry_price": 0.069, "qty": 3.0, "unrealized_pnl": None}
    assert _position_unrealized_pnl(pos, current_price=0.079) == pytest.approx(0.03)


def test_position_unrealized_pnl_fallback_respects_short_sign():
    """Regression: a prior inline fallback in main.py applied the long-side
    formula unconditionally, which silently inverted the sign for shorts
    (profitable-when-price-drops shorts would show as a loss, and vice versa)."""
    pos = {"side": "short", "entry_price": 0.07, "qty": 5.0, "unrealized_pnl": None}
    # Price dropped below entry -- a short should show a profit.
    assert _position_unrealized_pnl(pos, current_price=0.06) == pytest.approx(0.05)


def test_build_scale_out_orders_splits_qty_at_half():
    orders = build_scale_out_orders("long", 16343.0, 0.5, trail_price=0.0700, hard_price=0.0680)
    assert orders == [("trail", 8171.5, 0.0700), ("hard", 8171.5, 0.0680)]


def test_build_scale_out_orders_uses_rounder_for_qty():
    orders = build_scale_out_orders(
        "long", 16343.0, 0.5, trail_price=0.0700, hard_price=0.0680,
        rounder=lambda q: int(q),
    )
    assert orders == [("trail", 8171, 0.0700), ("hard", 8172, 0.0680)]


def test_build_scale_out_orders_single_stop_when_levels_equal():
    orders = build_scale_out_orders("long", 1000.0, 0.5, trail_price=0.0680, hard_price=0.0680)
    assert orders == [("hard", 1000.0, 0.0680)]


def test_build_scale_out_orders_split_arms_with_startup_trail_price():
    orders = build_scale_out_orders(
        "long", 1000.0, 0.5, trail_price=0.0680, hard_price=0.0680,
        startup_trail_price=0.0687,
    )
    assert orders == [("trail", 500.0, 0.0687), ("hard", 500.0, 0.0680)]


def test_build_scale_out_orders_startup_trail_equal_to_hard_stays_single():
    orders = build_scale_out_orders(
        "long", 1000.0, 0.5, trail_price=0.0680, hard_price=0.0680,
        startup_trail_price=0.0680,
    )
    assert orders == [("hard", 1000.0, 0.0680)]


def test_build_scale_out_orders_ignores_startup_trail_when_trail_armed():
    orders = build_scale_out_orders(
        "long", 1000.0, 0.5, trail_price=0.0700, hard_price=0.0680,
        startup_trail_price=0.0687,
    )
    assert orders == [("trail", 500.0, 0.0700), ("hard", 500.0, 0.0680)]


def test_build_scale_out_orders_startup_trail_ignored_after_scale_out_done():
    orders = build_scale_out_orders(
        "long", 1000.0, 0.5, trail_price=0.0680, hard_price=0.0680,
        scale_out_done=True, startup_trail_price=0.0687,
    )
    assert orders == [("hard", 1000.0, 0.0680)]


def test_build_scale_out_orders_startup_trail_short_side():
    orders = build_scale_out_orders(
        "short", 8000.0, 0.5, trail_price=0.0740, hard_price=0.0740,
        startup_trail_price=0.0730,
    )
    assert orders == [("trail", 4000.0, 0.0730), ("hard", 4000.0, 0.0740)]


def test_build_scale_out_orders_single_stop_after_scale_out_done():
    orders = build_scale_out_orders("long", 8171.5, 0.5, trail_price=0.0700, hard_price=0.0680, scale_out_done=True)
    assert orders == [("hard", 8171.5, 0.0680)]


def test_build_scale_out_orders_short_side():
    orders = build_scale_out_orders("short", 8000.0, 0.5, trail_price=0.0720, hard_price=0.0740)
    assert orders == [("trail", 4000.0, 0.0720), ("hard", 4000.0, 0.0740)]


def test_build_scale_out_orders_zero_qty():
    assert build_scale_out_orders("long", 0.0, 0.5, trail_price=0.07, hard_price=0.068) == []


def test_build_scale_out_orders_clamps_scale_pct():
    orders = build_scale_out_orders("long", 100.0, 2.0, trail_price=0.07, hard_price=0.068)
    assert orders[0][1] == 95.0


class DirtyBookExchange:
    """Exchange stub that leaves stale orders open after cleanup, forcing the
    startup dirty-book abort path to trigger.

    It has to satisfy every check that now runs BEFORE the cleanup verification --
    account config, balance, positions (AUDIT #69/#72). Omitting them does not make the
    test fail; it makes it abort earlier and pass without ever reaching the dirty book,
    which is the same false pass the account checks exist to prevent.
    """

    def __init__(self, config, demo=False):
        self.config = config
        self.demo = demo

    def set_leverage(self, symbol, leverage):
        return True

    def get_balance(self, asset="USDT"):
        return 4931.09

    def get_account_config(self, symbol):
        return {"leverage": main_module.settings.leverage, "margin_mode": "cross",
                "isolated": False, "dual_side": False, "max_notional": 600000.0}

    def get_maint_margin_ratio(self, symbol, notional):
        return 0.006

    def get_commission_rates(self, symbol):
        return {"maker_pct": main_module.settings.maker_fee_pct,
                "taker_pct": main_module.settings.taker_fee_pct}

    def get_min_notional(self, symbol):
        # verify_account_config checks the order floor against the
        # exchange now, same as it checks leverage and fees (AUDIT #107).
        from grid import MIN_NOTIONAL_USDT

        return MIN_NOTIONAL_USDT
    def get_positions(self, symbol):
        return []

    def cancel_everything(self, symbol, timeout_seconds=300.0, keep_stops=False):
        return 0

    def close_all_positions(self, symbol):
        return 0

    def get_open_orders(self, symbol):
        return [{"id": "stale-1"}, {"id": "stale-2"}]


def _run_and_capture(monkeypatch, exchange_cls, state_dir):
    """Drive run_bot() against a stub and return everything it logged.

    state_dir is pinned to a tmp path so the first-run/orphan branch depends on the
    test, not on whatever happens to be sitting in the real state/ directory."""
    monkeypatch.setattr(main_module, "Exchange", exchange_cls)
    monkeypatch.setattr(main_module.settings, "telegram_enabled", False)
    monkeypatch.setattr(main_module.settings, "state_dir", str(state_dir))
    monkeypatch.setattr(main_module, "setup_logging", lambda *a, **k: None)

    sink = []
    handle = logger.add(lambda m: sink.append(str(m)), level="INFO")
    # Startup guards end in abort_startup, which RAISES SystemExit -- the exit code is
    # the contract with supervise.py, so the code is part of what these tests assert
    # (AUDIT #126). A guard that merely returned would exit 0 by accident rather than
    # by decision, which is the bug.
    code = 0
    try:
        main_module.run_bot()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    finally:
        logger.remove(handle)
    return "".join(sink), code


class WrongLeverageExchange(DirtyBookExchange):
    """The measured case: .env says one thing, the exchange is set to another.

    Read live so the test stays correct whichever leverage .env carries: whatever
    settings wants, this account is one tick off, which is the whole defect."""

    def get_account_config(self, symbol):
        from config import settings

        return {"leverage": settings.leverage + 1, "margin_mode": "cross",
                "isolated": False, "dual_side": False, "max_notional": 4800000.0}


def test_run_bot_refuses_to_trade_a_misconfigured_account(monkeypatch, tmp_path):
    """verify_account_config being correct is worth nothing if run_bot ignores it.

    Nothing covered the WIRING: the gate could be commented out and every account-config
    test still passed, because they all call the function directly (AUDIT #69)."""
    logs, code = _run_and_capture(monkeypatch, WrongLeverageExchange, tmp_path)

    assert "ACCOUNT NOT SAFE TO TRADE" in logs
    assert "leverage mismatch" in logs
    assert "still open after cleanup" not in logs, "startup continued past the gate"
    assert code == 0, ("a leverage mismatch is a real misconfiguration -- restarting into "
                       "it forever helps nobody, so exit 0 and stay down (AUDIT #126)")


class PreExistingPositionExchange(DirtyBookExchange):
    """A position that is already open the first time the bot sees this account."""

    def get_positions(self, symbol):
        return [{"side": "long", "contracts": 8215.0, "entryPrice": 0.06945,
                 "info": {"positionAmt": "8215"}}]


def test_run_bot_will_not_close_a_position_it_did_not_open(monkeypatch, tmp_path):
    """First run against an account, so any position belongs to whoever opened it --
    most likely the human, right after flipping DEMO_MODE (AUDIT #72)."""
    logs, code = _run_and_capture(monkeypatch, PreExistingPositionExchange, tmp_path)

    assert "PRE-EXISTING POSITION" in logs
    assert "still open after cleanup" not in logs, "startup continued past the guard"
    assert code == 0, ("a position the bot did not open needs a human, not a retry loop "
                       "(AUDIT #126)")


def test_an_orphan_from_a_previous_session_is_still_closed(monkeypatch, tmp_path):
    """The guard must not disarm ordinary crash recovery: with proof the bot ran here
    before, an open position IS an orphan and closing it is the documented behaviour."""
    (tmp_path / f"grid_{main_module.settings.symbol.lower()}_demo.bak.1786498297").write_text("{}")

    logs, code = _run_and_capture(monkeypatch, PreExistingPositionExchange, tmp_path)

    assert "PRE-EXISTING POSITION" not in logs
    assert "still open after cleanup" in logs, "it did not reach the normal cleanup path"
    assert code == 1, ("this run reaches the dirty-book abort, which is transient and "
                       "must ask the supervisor to try again (AUDIT #126)")


def test_run_bot_aborts_on_dirty_book(monkeypatch, caplog):
    """Startup must NOT continue past the cleanup check when the book is still dirty.

    Asserting on the reason, not just on 'it returned': every abort path returns, so a
    bare call proves only that something stopped it -- not that the dirty book did."""
    monkeypatch.setattr(main_module, "Exchange", DirtyBookExchange)
    monkeypatch.setattr(main_module.settings, "telegram_enabled", False)
    monkeypatch.setattr(main_module, "setup_logging", lambda *a, **k: None)

    sink = []
    handle = logger.add(lambda m: sink.append(str(m)), level="INFO")
    code = 0
    try:
        main_module.run_bot()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    finally:
        logger.remove(handle)

    assert code == 1, (
        "a dirty book means the exchange write path is down -- a temporary condition. "
        "Exiting 0 tells supervise.py someone chose to stop and it stays down "
        "(AUDIT #126)"
    )
    logs = "".join(sink)
    assert "still open after cleanup" in logs, (
        "startup aborted somewhere else — this test no longer covers the dirty book"
    )


def test_build_scale_out_orders_returns_nothing_when_no_stop_is_available():
    """AUDIT #31. The live strategy can legitimately have no stop for a side -- a trend
    follower that just closed, or one that is flat while the exchange still reports the
    position for an iteration. Both prices come back None and the arithmetic raised
    TypeError inside the trading loop. Skipping the refresh is recoverable; crashing
    every iteration is not."""
    assert build_scale_out_orders("short", 5000.0, 0.5, trail_price=None, hard_price=None) == []
    assert build_scale_out_orders("long", 5000.0, 0.5, trail_price=0.069, hard_price=None) == []
    assert build_scale_out_orders("long", 5000.0, 0.5, trail_price=None, hard_price=0.069) == []


def test_build_scale_out_orders_still_works_with_real_prices():
    orders = build_scale_out_orders("long", 5000.0, 0.5, trail_price=0.0700, hard_price=0.0680)
    assert [k for k, _, _ in orders] == ["trail", "hard"]


# --- #37: startup must not market-dump the bot's own inventory -------------

def _main_source():
    import pathlib
    return (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")


def test_state_is_read_before_any_position_is_closed():
    """AUDIT #37. Startup cancelled orders and market-closed every position, then read
    the state file 60 lines later. CLOSE_ON_EXIT defaults to false precisely so the grid
    can keep inventory and unwind it through its own levels -- and then housekeeping
    dumped it on the next start. Measured on the 2026-08-12 23:18 restart: 6274 DOGE
    closed at market, verified PnL -1.83 -> -3.26.
    """
    src = _main_source()
    load_at = src.index("saved_state = state_mgr.load()")
    close_at = src.index("exchange.close_all_positions(settings.symbol)")
    assert load_at < close_at, (
        "positions are closed before the state file is read, so the bot cannot tell "
        "its own inventory from an orphan"
    )


def test_positions_are_only_closed_without_saved_state():
    src = _main_source()
    close_at = src.index("exchange.close_all_positions(settings.symbol)")
    guard = src.rindex("if has_saved_grid:", 0, close_at)
    between = src[guard:close_at]
    assert "else:" in between, (
        "close_all_positions is no longer gated on there being no saved grid state"
    )


def test_orders_are_still_cancelled_unconditionally():
    """Untracked resting orders from a dead session are dangerous and the grid re-places
    its own, so the LIMIT sweep stays unconditional. Two things are allowed to be
    conditional: the POSITION decision (AUDIT #37) and, since AUDIT #92, the STOP book --
    stops guarding an inherited position are handed to reconcile_stop_orders instead of
    being cancelled and re-placed 26 seconds later."""
    src = _main_source()
    cancel_at = src.index("cancelled = exchange.cancel_everything(")
    guard_at = src.index("if has_saved_grid:")
    assert cancel_at < guard_at, "order cancellation was moved behind the state check"

    # The sweep itself must not be wrapped in a condition -- only its keep_stops
    # argument may vary.
    line = src[cancel_at:].splitlines()[0]
    assert "keep_stops=" in line, "startup no longer chooses whether to keep stops"
    assert not src[:cancel_at].rstrip().endswith(":"), (
        "the cancel is now inside a conditional block"
    )


# --- AUDIT #155: a restart during an active kill-switch recovery cooldown must not
# silently re-arm the ladder into the very conditions that tripped it --------------

def test_recovery_state_is_restored_before_the_first_post_restart_ladder_placement():
    """risk.load_from_dict used to run one line AFTER place_initial_orders -- so the
    very first ladder placement of every restart always saw the freshly-constructed
    in_recovery=False, never the real saved value, no matter how deep into an active
    cooldown the bot actually was.
    """
    src = _main_source()
    load_at = src.index('risk.load_from_dict(saved_state.get("risk", {}))')
    place_at = src.index("grid.place_initial_orders(exchange.get_balance())")
    assert load_at < place_at, (
        "risk state is restored after the first ladder placement decision, not before"
    )


def test_the_post_cleanup_ladder_placement_is_gated_on_recovery():
    """The ladder placed once the exchange is confirmed flat must not fire while a
    kill-switch recovery cooldown is still active -- that flat state is exactly what
    cleanup.py leaves behind, so it is the trigger condition for the old bug, not a
    guard against it."""
    src = _main_source()
    guard_at = src.index("if not has_exchange_positions:")
    place_at = src.index("grid.place_initial_orders(exchange.get_balance())")
    between = src[guard_at:place_at]
    assert "risk.is_in_recovery()" in between, (
        "the post-cleanup ladder placement no longer checks recovery state at all"
    )
    assert "else:" in between, (
        "place_initial_orders is not gated behind the recovery check's else branch"
    )


def test_the_startup_grid_activation_is_also_gated_on_recovery():
    """The second activation point (the trend-gate block, reached ~20-25 minutes of
    blind re-confirmation later) must independently refuse to switch the grid on
    during an active recovery cooldown -- an inherited active grid would otherwise
    resume trading there regardless of the ladder-placement gate above."""
    src = _main_source()
    activate_at = src.index("grid.activate(exchange.get_balance())")
    recovery_check_at = src.rindex("if risk.is_in_recovery():", 0, activate_at)
    between = src[recovery_check_at:activate_at]
    assert "elif" in between, (
        "grid.activate is not gated behind an elif of the startup recovery check"
    )
