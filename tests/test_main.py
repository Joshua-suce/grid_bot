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


import ast as _ast


def test_the_startup_sequence_is_wrapped_in_its_own_exception_safety_net():
    """AUDIT #157. The ~560-line startup stretch between grid/trend/risk construction
    and the main loop used to have no enclosing try/except of its own -- a crash there
    propagated straight out of run_bot() with zero cleanup, unlike the main loop which
    always gets emergency_stop() via its own finally. Structural check, not a full
    run_bot() integration test -- "run_bot is not unit-testable", per this file's own
    established precedent (see test_startup_cleanup_asks_to_keep_stops).
    """
    tree = _ast.parse(_main_source())
    run_bot = next(n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef) and n.name == "run_bot")
    top_level_tries = [n for n in run_bot.body if isinstance(n, _ast.Try)]
    assert len(top_level_tries) >= 2, "expected the startup wrapper plus the loop's own try"

    startup_try, loop_try = top_level_tries[-2], top_level_tries[-1]
    assert startup_try.lineno < loop_try.lineno, (
        "the startup wrapper must sit before the main loop's own try, not after or "
        "nested inside it"
    )

    handler_names = [h.type.id for h in startup_try.handlers if isinstance(h.type, _ast.Name)]
    assert handler_names == ["Exception"], (
        "the startup wrapper must catch exactly Exception, not a bare except or "
        "BaseException -- SystemExit (abort_startup's mechanism, and the deliberate "
        "sys.exit on a failed balance read) must keep propagating through untouched"
    )


def test_the_startup_exception_handler_attempts_cleanup_and_reraises():
    src = _main_source()
    handler_at = src.index("    except Exception as e:\n        # The main loop already gets this")
    cleanup_at = src.index("grid.emergency_stop(", handler_at)
    reraise_at = src.index("raise", cleanup_at)
    assert handler_at < cleanup_at < reraise_at, (
        "the startup handler does not attempt cleanup before re-raising"
    )
    # A bare re-raise -- the original exception must reach supervise.py unaltered so
    # its logging and exit-code handling stay correct, not get swallowed or replaced.
    line = src[reraise_at:].splitlines()[0]
    assert line.strip() == "raise", f"expected a bare re-raise, got: {line!r}"


def test_the_cleanup_attempt_itself_cannot_mask_the_original_exception():
    """A failure inside the cleanup attempt (e.g. the exchange is ALSO unreachable for
    the emergency_stop call) must not replace or hide the original exception that
    triggered this handler in the first place."""
    src = _main_source()
    handler_at = src.index("    except Exception as e:\n        # The main loop already gets this")
    reraise_at = src.index("raise", handler_at)
    between = src[handler_at:reraise_at]
    assert "except Exception as cleanup_err:" in between, (
        "the cleanup attempt is not itself guarded -- a failed cleanup would replace "
        "the original exception instead of the startup failure reaching supervise.py"
    )


def test_the_startup_wrapper_actually_encloses_ladder_placement_and_the_trend_gate():
    """Not just present somewhere in the file -- it has to actually contain the two
    places a ladder gets placed at startup, or a crash there is still unmanaged."""
    src = _main_source()
    try_at = src.index("    grid = None\n    try:\n        trend = TrendFilter(")
    except_at = src.index("    except Exception as e:\n        # The main loop already gets this", try_at)
    zone = src[try_at:except_at]
    assert "grid.place_initial_orders(exchange.get_balance())" in zone
    assert "grid.activate(exchange.get_balance())" in zone


def test_grid_is_defined_before_the_try_so_the_handler_never_name_errors():
    """The handler reads `grid` unconditionally (`if grid is not None`) -- if an
    exception fired before grid.load_from_dict's branch even ran, `grid` must still
    be bound, or the cleanup attempt itself throws NameError and masks the original
    failure instead of handling it."""
    src = _main_source()
    grid_none_at = src.index("    grid = None\n    try:")
    try_at = src.index("    try:\n        trend = TrendFilter(")
    assert grid_none_at < try_at, "grid = None must be set before the try begins"


def test_trend_state_is_restored_right_after_construction():
    """AUDIT #155/#156. TrendFilter used to have no persistence at all -- every
    restart started at UNCERTAIN regardless of what was confirmed right before the
    stop. The restore must run before the very first live update() overwrites
    whatever a fresh construction defaulted to."""
    src = _main_source()
    construct_at = src.index("trend = TrendFilter(")
    restore_at = src.index('trend.load_from_dict((saved_state or {}).get("trend", {}))')
    first_update_at = src.index("trend.update(ohlcv_tf, settings.trend_timeframe)")
    assert construct_at < restore_at < first_update_at, (
        "trend state is not restored between construction and the first live update"
    )


def test_trend_state_is_included_in_every_state_save():
    """A restore is worthless if nothing ever wrote it -- every state_data dict main.py
    builds must carry the trend snapshot alongside grid/risk/pnl_reconciler."""
    src = _main_source()
    save_sites = [
        i for i in range(len(src))
        if src.startswith("state_data = {", i)
    ]
    assert len(save_sites) >= 4, "expected multiple state_data construction sites"
    for site in save_sites:
        # Brace-depth scan, not the first "}" -- dict values like
        # "grid.to_dict() if grid is not None else {}" contain their own braces.
        depth = 0
        end = site
        for idx in range(site, len(src)):
            if src[idx] == "{":
                depth += 1
            elif src[idx] == "}":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        block = src[site:end]
        assert '"trend": trend.to_dict()' in block, (
            f"a state_data dict at offset {site} does not persist trend state"
        )


def test_lone_trend_alert_is_wired_into_the_periodic_regime_check():
    """AUDIT #156. The alert must be driven from the same periodic recheck that
    already re-evaluates the regime, not left uncalled."""
    src = _main_source()
    check_at = src.index("if trend.time_to_check():")
    regime_log_at = src.index('logger.info("REGIME | {} -> {}", trend.explain(), trend.regime.value)')
    lone_call_at = src.index("trend.lone_trend_duration()")
    alert_call_at = src.index("notifier.on_lone_trend(")
    assert check_at < regime_log_at < lone_call_at < alert_call_at, (
        "the lone-trend alert is not wired into the periodic regime recheck in order"
    )


def test_optional_timeframe_fetch_failures_are_logged_not_silently_swallowed():
    """AUDIT #163. These two used to be bare `except Exception: pass` -- unlike every
    other narrow-purpose catch in this file, which logs at least at debug. A sustained
    feed outage degraded regime confirmation with nothing in the log to explain it."""
    src = _main_source()
    fast_sites = [
        i for i in range(len(src))
        if src.startswith('trend.add_timeframe(ohlcv_fast, settings.trend_timeframe_fast)', i)
    ]
    day_sites = [
        i for i in range(len(src))
        if src.startswith('trend.add_timeframe(ohlcv_1d, "1d")', i)
    ]
    assert len(fast_sites) >= 2 and len(day_sites) >= 2, "expected both call sites (startup + loop)"
    for site in fast_sites + day_sites:
        following = src[site:site + 500]
        assert "except Exception" in following
        except_at = following.index("except Exception")
        after_except = following[except_at:except_at + 400]
        assert "logger.debug(" in after_except, (
            f"a fetch at offset {site} still swallows its failure without logging it"
        )


def test_the_recovery_rebuild_only_swaps_grid_after_everything_succeeds():
    """AUDIT #161. `grid` used to be reassigned to the fresh, zeroed GridEngine BEFORE
    its stats were restored and BEFORE activate() ran -- a throw anywhere in between
    left the outer `grid` pointing at a zeroed engine, with the bot's whole cumulative
    fill/PnL/fee/cycle history reading zero even though nothing happened on the
    exchange. The rebuild must happen on a local `new_grid` and only replace the outer
    `grid` once construction, sizing, stat restoration, and activation have all
    already succeeded.
    """
    src = _main_source()
    section_at = src.index('logger.info("RECOVERY READY | recalculating grid around current price {}", price)')
    except_at = src.index("except Exception as e:\n                            if new_grid is not None:")
    section = src[section_at:except_at]

    construct_at = section.index("new_grid = GridEngine(")
    stats_at = section.index("new_grid.total_fills = old_fills")
    activate_at = section.index("new_grid.activate(exchange.get_balance())")
    swap_at = section.index("\n                            grid = new_grid\n")
    # Search from swap_at on: an explanatory comment above also mentions
    # "risk.exit_recovery()" in prose, which an unanchored search would match first.
    exit_recovery_at = section.index("risk.exit_recovery()", swap_at)
    notify_at = section.index("notifier.on_grid_start(settings.symbol, grid.grid_lower", swap_at)

    assert (construct_at < stats_at < activate_at < swap_at
            < exit_recovery_at < notify_at), (
        "the recovery rebuild does not defer swapping `grid` (and exiting recovery) "
        "until every throwable step has already succeeded"
    )


def test_a_failed_recovery_rebuild_leaves_the_old_grid_and_recovery_state_untouched():
    """Reads as: `grid = new_grid` and `risk.exit_recovery()` must be UNREACHABLE if
    anything above them throws -- i.e. inside the same try, before the except."""
    src = _main_source()
    section_at = src.index('logger.info("RECOVERY READY | recalculating grid around current price {}", price)')
    try_at = src.index("new_grid = None\n                        try:", section_at)
    except_at = src.index("except Exception as e:\n                            if new_grid is not None:", try_at)
    swap_at = src.index("grid = new_grid", try_at)
    assert try_at < swap_at < except_at, (
        "the grid swap is not inside the recovery rebuild's own try block"
    )


def test_a_failed_recovery_rebuild_cleans_up_whatever_new_grid_managed_to_place():
    src = _main_source()
    except_at = src.index("except Exception as e:\n                            if new_grid is not None:")
    block = src[except_at:except_at + 800]
    cleanup_at = block.index("new_grid.emergency_stop(")
    guard_at = block.index("except Exception as cleanup_err:")
    assert 0 < cleanup_at < guard_at, (
        "a failed recovery rebuild does not attempt to clean up whatever new_grid "
        "already placed on the exchange, or does not guard that cleanup attempt itself"
    )


def test_a_genuine_code_defect_in_recovery_rebuild_gets_the_same_treatment_as_the_loop():
    """AUDIT #161. Without this, a real bug in the recovery path retried silently
    forever at poll_interval with no traceback, no dedup, no alert -- unlike every
    other code defect in the loop (AUDIT #31's BUG_ERRORS handling)."""
    src = _main_source()
    except_at = src.index("except Exception as e:\n                            if new_grid is not None:")
    block = src[except_at:except_at + 2000]
    assert "isinstance(e, BUG_ERRORS)" in block
    assert "seen_bug_errors" in block


def test_the_startup_stop_loss_check_probes_positions_readability_first():
    """AUDIT #159. get_net_position() swallows a failed read and returns ("", 0.0) --
    indistinguishable from genuinely flat. Skipping the stop-loss block on that value
    let grid.activate() add fresh exposure on top of a real, silently-unprotected
    inherited position. seed_position_limit() already probes explicitly for exactly
    this reason (AUDIT #125); the startup stop-loss check must too."""
    src = _main_source()
    probe_at = src.index("exchange.get_positions(settings.symbol)\n            positions_readable = True")
    net_position_at = src.index("position_side, position_qty = get_net_position(exchange, settings.symbol)")
    assert probe_at < net_position_at, (
        "positions readability is not probed before the swallowed-failure "
        "get_net_position() call"
    )


def test_an_unreadable_book_blocks_both_sides_at_startup():
    src = _main_source()
    probe_at = src.index("positions_readable = False")
    block_buy_at = src.index('grid.block_side("buy", "positions unreadable at startup")', probe_at)
    block_sell_at = src.index('grid.block_side("sell", "positions unreadable at startup")', probe_at)
    assert probe_at < block_buy_at < block_sell_at, (
        "an unreadable book at startup does not block both sides"
    )


def test_the_stop_loss_placement_is_gated_on_positions_being_readable():
    src = _main_source()
    gate_at = src.index("if positions_readable and position_side in")
    assert gate_at > 0, (
        "the startup stop-loss placement no longer checks positions_readable -- a "
        "swallowed failed read (position_side='') would silently skip it either way, "
        "but so would a successful read of a real position, indistinguishably"
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


def test_fills_today_is_recorded_once_per_fill_before_it_is_reported():
    """`risk.record_fill()` must run inside the `for fill in fills:` loop, once per
    iteration, and before the notifier/events/journal calls in that same iteration --
    otherwise those calls would report yesterday's (or the previous fill's) count
    instead of the tally that already includes the fill just processed."""
    src = _main_source()
    loop_at = src.index("for fill in fills:")
    notify_at = src.index("notifier.on_fill(", loop_at)
    events_at = src.index("events.fill(", loop_at)
    journal_at = src.index("journal.record(", loop_at)
    record_fill_at = src.index("risk.record_fill()", loop_at)

    assert loop_at < record_fill_at < notify_at < events_at < journal_at, (
        "risk.record_fill() is not called inside the fills loop before the calls "
        "that report risk.state.fills_today"
    )
    # Only one call site -- record_fill() must not also be invoked elsewhere (e.g.
    # once per batch instead of once per fill).
    assert src.count("risk.record_fill()") == 1


def test_fills_today_is_threaded_into_the_reporting_calls():
    src = _main_source()
    loop_at = src.index("for fill in fills:")
    notify_at = src.index("notifier.on_fill(", loop_at)
    events_at = src.index("events.fill(", loop_at)
    journal_at = src.index("journal.record(", loop_at)

    notify_block = src[notify_at:events_at]
    events_block = src[events_at:journal_at]
    journal_block = src[journal_at:journal_at + 800]

    assert "daily_fill_count=risk.state.fills_today" in notify_block
    assert "fills_today=risk.state.fills_today" in events_block
    assert "fills_today=risk.state.fills_today" in journal_block


# --- AUDIT #143 (paused branch): an exchange-side close while the grid is paused
# must still be detected, booked, and reported -- not silently adopted as flat on
# the next activate() with zero P&L. ---------------------------------------------

def test_the_paused_branch_also_detects_an_external_close():
    """detect_external_close is only useful here if it runs BEFORE the
    get_net_position() read that decides whether there is anything left to do --
    otherwise a close this same iteration is invisible to both."""
    src = _main_source()
    paused_at = src.index("if not grid.active:")
    detect_at = src.index("grid.detect_external_close(price)", paused_at)
    net_position_at = src.index(
        "held_side, held_qty = get_net_position(exchange, settings.symbol)", paused_at,
    )
    assert paused_at < detect_at < net_position_at, (
        "the paused branch does not check for an external close before deciding "
        "there is nothing held to act on"
    )


def test_the_paused_branch_external_close_has_its_own_call_site():
    """Distinct from the active branch's detect_external_close (AUDIT #143 original) --
    exactly two call sites total, one per branch."""
    src = _main_source()
    assert src.count("grid.detect_external_close(price)") == 2


def test_a_paused_external_close_is_booked_into_risk_with_the_verified_figure():
    """Mirrors the active branch's own pattern (AUDIT #43): sync the reconciler
    first, then hand risk.record_cycles the account's own realized delta, not the
    engine's estimate -- so consecutive_losses and the daily/profit-lock figures
    see this close exactly as they would have seen any other."""
    src = _main_source()
    paused_at = src.index("if not grid.active:")
    detect_at = src.index("grid.detect_external_close(price)", paused_at)
    net_position_at = src.index(
        "held_side, held_qty = get_net_position(exchange, settings.symbol)", paused_at,
    )
    block = src[detect_at:net_position_at]
    sync_at = block.index("pnl_reconciler.sync(exchange, settings.symbol)")
    record_at = block.index("risk.record_cycles(")
    assert sync_at < record_at, (
        "the paused-branch close books risk.record_cycles before syncing the "
        "reconciler, so it would use a stale/pre-close verified figure"
    )


def test_a_paused_external_close_notifies():
    src = _main_source()
    paused_at = src.index("if not grid.active:")
    detect_at = src.index("grid.detect_external_close(price)", paused_at)
    net_position_at = src.index(
        "held_side, held_qty = get_net_position(exchange, settings.symbol)", paused_at,
    )
    block = src[detect_at:net_position_at]
    assert "notifier.send(" in block and "while paused" in block


def test_a_paused_external_close_does_not_add_a_second_record_fill_call_site():
    """risk.record_fill() has exactly one call site by design (see
    test_fills_today_is_recorded_once_per_fill_before_it_is_reported) -- the paused
    branch's own close must not add a second one."""
    src = _main_source()
    assert src.count("risk.record_fill()") == 1
