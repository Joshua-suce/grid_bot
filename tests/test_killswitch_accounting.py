"""The kill switch must count the ACCOUNT's losses, not the grid's estimate. AUDIT #43.

`record_trade()` was fed the grid engine's per-level cycle profit. That number credits
each sell against the particular buy level it was paired with; Binance nets everything
into one position at one blended average entry. They agree only by coincidence -- in the
2026-08-13 run fills #24-#26 booked +2.37, +1.29 and +3.89 (=+7.55) while the reconciler
moved -2.568140 -> -2.247978, a realized **+0.32**.

Magnitude is the lesser problem. The estimate can carry the wrong SIGN, and it does so
exactly when it matters: a grid bleeding into a downtrend reports a run of wins, so
`consecutive_losses` -- the kill switch for "this strategy is repeatedly wrong" -- could
not fire in the one situation it exists for.
"""

import pytest

from risk import RiskManager


def _rm(max_consecutive_losses=3):
    rm = RiskManager(
        max_consecutive_losses=max_consecutive_losses,
        daily_loss_limit_pct=0.03,
        max_drawdown_pct=0.10,
    )
    rm.initialize(5000.0)
    return rm


def test_a_losing_streak_the_grid_calls_wins_still_trips_the_switch():
    """The regression. Buy 1,000 at 0.0690 and 1,000 at 0.0710 -> blended entry 0.0700.
    Sell 1,000 at 0.0695 paired with the 0.0690 level and the grid books +5 while the
    account realises -5. A falling market fills both levels, so this is the ordinary
    shape of a grid bleeding into a downtrend -- not a contrived case."""
    rm = _rm(max_consecutive_losses=3)

    for _ in range(3):
        rm.record_cycles(completed=1, verified_pnl=-5.0, estimated_pnl=+5.0)

    assert rm.state.consecutive_losses == 3, (
        f"three genuinely losing batches counted as {rm.state.consecutive_losses} -- "
        f"the grid's +5 estimate is masking the account's -5"
    )
    is_safe, is_fatal = rm.check_all(5000.0, 0.0, 50000.0)
    assert (is_safe, is_fatal) == (False, True), "kill switch did not fire on a real streak"


def test_the_daily_total_tracks_the_exchange_not_the_estimate():
    """state.daily_realized_pnl is persisted and logged at the daily reset; feeding it
    the estimate made every report disagree with the account by ~20x."""
    rm = _rm()

    rm.record_cycles(completed=3, verified_pnl=+0.32, estimated_pnl=+7.55)

    assert rm.state.daily_realized_pnl == pytest.approx(0.32), (
        f"daily total is {rm.state.daily_realized_pnl}, the grid's estimate, not the "
        f"account's +0.32"
    )
    assert rm.state.trades_today == 3, "completed cycles must still be counted individually"


def test_a_real_win_clears_the_streak():
    """#43 must not make the switch trigger-happy: a verified profit still resets."""
    rm = _rm(max_consecutive_losses=3)
    rm.record_cycles(1, verified_pnl=-5.0, estimated_pnl=+5.0)
    rm.record_cycles(1, verified_pnl=-5.0, estimated_pnl=+5.0)
    assert rm.state.consecutive_losses == 2

    rm.record_cycles(1, verified_pnl=+4.0, estimated_pnl=-1.0)

    assert rm.state.consecutive_losses == 0
    is_safe, _ = rm.check_all(5000.0, 0.0, 50000.0)
    assert is_safe


def test_ledger_lag_holds_the_streak_instead_of_clearing_it():
    """A verified delta of exactly 0.0 means Binance's income ledger has not caught up,
    not that the batch broke even. Treating it as a win would clear a real losing
    streak on nothing more than API lag -- so the streak is held."""
    rm = _rm(max_consecutive_losses=3)
    rm.record_cycles(1, verified_pnl=-5.0, estimated_pnl=+5.0)
    rm.record_cycles(1, verified_pnl=-5.0, estimated_pnl=+5.0)
    assert rm.state.consecutive_losses == 2

    rm.record_cycles(1, verified_pnl=0.0, estimated_pnl=+5.0)      # ledger not settled

    assert rm.state.consecutive_losses == 2, (
        "an unsettled ledger reset the losing streak -- API lag must not look like a win"
    )
    assert rm.state.trades_today == 3, "the cycle still happened and still counts"


def test_without_a_reconciler_it_falls_back_to_the_estimate():
    """The estimate is worse than the account, but it beats nothing at all."""
    rm = _rm(max_consecutive_losses=2)

    rm.record_cycles(1, verified_pnl=None, estimated_pnl=-3.0)
    rm.record_cycles(1, verified_pnl=None, estimated_pnl=-3.0)

    assert rm.state.consecutive_losses == 2
    assert rm.state.daily_realized_pnl == pytest.approx(-6.0)


def test_a_batch_with_no_completed_cycles_records_nothing():
    """Opening fills are not trades yet; they must not touch the streak or the count."""
    rm = _rm()
    rm.record_cycles(1, verified_pnl=-5.0, estimated_pnl=-5.0)
    before = (rm.state.consecutive_losses, rm.state.trades_today, rm.state.daily_realized_pnl)

    rm.record_cycles(0, verified_pnl=-99.0, estimated_pnl=-99.0)

    assert (rm.state.consecutive_losses, rm.state.trades_today,
            rm.state.daily_realized_pnl) == before


def test_main_feeds_the_verified_delta_not_the_grid_sum():
    """Structural: the wiring is the whole point, and a correct risk.py wired to the old
    number fixes nothing. Reading the source keeps this honest without a live exchange."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "main.py"
    text = src.read_text(encoding="utf-8")

    assert "risk.record_cycles(" in text, "main.py no longer records cycles at all"
    assert "pnl_reconciler.net_realized_pnl - verified_before" in text, (
        "main.py is not passing the exchange-verified delta to the risk manager"
    )
    assert "risk.record_trade(profit)" not in text, (
        "main.py still feeds the grid's per-level estimate to the risk manager (#43)"
    )
