import time
import pytest

from risk import RiskManager


def test_risk_manager_init():
    rm = RiskManager(stop_loss_pct=0.03, daily_loss_limit_pct=0.05, max_drawdown_pct=0.10)
    rm.initialize(1000)
    assert rm.state.starting_balance == 1000
    assert rm.state.peak_balance == 1000


def test_check_all_ok():
    rm = RiskManager(stop_loss_pct=0.03, daily_loss_limit_pct=0.05, max_drawdown_pct=0.10)
    rm.initialize(1000)
    is_safe, is_fatal = rm.check_all(1000, 78000, 80000)
    assert is_safe is True


def test_drawdown_kills():
    rm = RiskManager(stop_loss_pct=0.03, daily_loss_limit_pct=0.05, max_drawdown_pct=0.10)
    rm.initialize(1000)
    rm.state.peak_balance = 1000
    is_safe, is_fatal = rm.check_all(890, 78000, 80000)
    assert is_safe is False
    assert is_fatal is True


def test_daily_loss_kills():
    rm = RiskManager(stop_loss_pct=0.03, daily_loss_limit_pct=0.05, max_drawdown_pct=0.10)
    rm.initialize(1000)
    rm.state.daily_realized_pnl = -60
    is_safe, is_fatal = rm.check_all(1000, 78000, 80000)
    assert is_safe is False
    assert is_fatal is True


def test_daily_loss_override_uses_reconciled_figure_instead_of_state():
    """daily_realized_pnl override (the exchange-reconciled PnLReconciler.daily_net_pnl
    figure) must drive the kill switch even when state.daily_realized_pnl (the grid's
    own drifting per-level estimate) disagrees -- see AUDIT.md "Daily PnL is still
    unreconciled"."""
    rm = RiskManager(stop_loss_pct=0.03, daily_loss_limit_pct=0.05, max_drawdown_pct=0.10)
    rm.initialize(1000)
    rm.state.daily_realized_pnl = 500  # grid's estimate says healthy profit...
    is_safe, is_fatal = rm.check_all(1000, 78000, 80000, daily_realized_pnl=-60)  # ...reconciled says a real loss
    assert is_safe is False
    assert is_fatal is True


def test_daily_loss_override_none_falls_back_to_state():
    rm = RiskManager(stop_loss_pct=0.03, daily_loss_limit_pct=0.05, max_drawdown_pct=0.10)
    rm.initialize(1000)
    rm.state.daily_realized_pnl = -60
    is_safe, is_fatal = rm.check_all(1000, 78000, 80000, daily_realized_pnl=None)
    assert is_safe is False
    assert is_fatal is True


def test_daily_loss_override_does_not_mutate_state():
    rm = RiskManager(stop_loss_pct=0.03, daily_loss_limit_pct=0.05, max_drawdown_pct=0.10)
    rm.initialize(1000)
    rm.check_all(1000, 78000, 80000, daily_realized_pnl=-999)
    assert rm.state.daily_realized_pnl == 0.0  # untouched -- still drives consecutive_losses separately


def test_grid_stop_loss_kills():
    rm = RiskManager(stop_loss_pct=0.03, daily_loss_limit_pct=0.05, max_drawdown_pct=0.10)
    rm.initialize(1000)
    sl_price = 78000 * 0.97
    is_safe, is_fatal = rm.check_all(1000, 78000, sl_price - 100)
    assert is_safe is False
    assert is_fatal is True


def test_grid_stop_loss_kills_short_when_price_rises_above_ceiling():
    """Regression: the grid stop-loss check used to always test the long-side
    floor direction (current_price <= stop_loss_price) regardless of position
    side, so it never tripped for a short position -- price rising into danger
    doesn't satisfy '<=' a floor computed for the opposite direction. side='short'
    must check the ceiling direction instead (current_price >= stop_loss_price).
    """
    rm = RiskManager(stop_loss_pct=0.03, daily_loss_limit_pct=0.05, max_drawdown_pct=0.10)
    rm.initialize(1000)
    short_sl_price = 82000 * 1.03  # ceiling: danger is price rising above this
    is_safe, is_fatal = rm.check_all(1000, short_sl_price, short_sl_price + 100, side="short")
    assert is_safe is False
    assert is_fatal is True


def test_grid_stop_loss_does_not_kill_short_using_long_direction():
    """Same short-side ceiling, but price is still comfortably below it -- must
    not trigger (this would have false-positived under the old always-'<=' check
    since a short's ceiling sits well above any long-style floor comparison)."""
    rm = RiskManager(stop_loss_pct=0.03, daily_loss_limit_pct=0.05, max_drawdown_pct=0.10)
    rm.initialize(1000)
    short_sl_price = 82000 * 1.03
    is_safe, is_fatal = rm.check_all(1000, short_sl_price, short_sl_price - 500, side="short")
    assert is_safe is True


def test_grid_stop_loss_skipped_when_flat():
    """stop_loss_price=0 (what main.py now passes while flat) must skip the grid
    stop-loss check entirely rather than risk a false kill switch off a stale
    long-side band with no position to protect."""
    rm = RiskManager(stop_loss_pct=0.03, daily_loss_limit_pct=0.05, max_drawdown_pct=0.10)
    rm.initialize(1000)
    is_safe, is_fatal = rm.check_all(1000, 0.0, 50000.0)
    assert is_safe is True


def test_cooldown():
    rm = RiskManager(cooldown_seconds=60)
    rm.initialize(1000)
    rm.trigger_kill_switch()
    is_safe, is_fatal = rm.check_all(1000, 78000, 80000)
    assert is_safe is False
    assert is_fatal is True


# test_record_trade is gone with the method it tested. record_trade was superseded by
# record_cycles in AUDIT #43 -- it was fed the grid engine's per-level cycle profit,
# which is ~20x the account's realised change and can differ in SIGN -- and had no
# production caller left. The two tests below used it only to put P&L on the books,
# and now use the method that actually runs (AUDIT #142).

def test_record_cycles_books_a_trade():
    rm = RiskManager()
    rm.initialize(1000)
    rm.record_cycles(1, 5.0, 5.0)
    assert rm.state.daily_realized_pnl == 5.0
    assert rm.state.trades_today == 1


def test_reset_daily():
    rm = RiskManager()
    rm.initialize(1000)
    rm.record_cycles(1, -10.0, -10.0)
    rm.record_cycles(1, 5.0, 5.0)
    rm.reset_daily()
    assert rm.state.daily_realized_pnl == 0.0
    assert rm.state.trades_today == 0


def test_to_dict_and_load():
    rm = RiskManager()
    rm.initialize(1000)
    rm.record_cycles(1, 10.0, 10.0)
    d = rm.to_dict()
    rm2 = RiskManager()
    rm2.load_from_dict(d)
    assert rm2.state.daily_realized_pnl == 10.0
    assert rm2.state.trades_today == 1


def test_recovery_size_multiplier_aggressive():
    rm = RiskManager(max_recovery_count=5)
    rm.initialize(1000)
    rm.trigger_kill_switch()
    assert rm.get_recovery_size_multiplier() == 0.65
    rm.state.recovery_count = 2
    assert rm.get_recovery_size_multiplier() == 0.40
    rm.state.recovery_count = 3
    assert rm.get_recovery_size_multiplier() == 0.25
    rm.state.recovery_count = 5
    assert rm.get_recovery_size_multiplier() == 0.0
