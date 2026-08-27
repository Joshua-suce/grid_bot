from config import settings
from pnl_tracker import PnLReconciler
from reset_state import _fresh_reconciler_state


def test_fresh_reconciler_state_survives_reset_for_epoch_when_pnl_epoch_is_fixed(monkeypatch):
    """AUDIT #165. With PNL_EPOCH set to a fixed date, a reconciler reset that omits
    epoch_ms gets silently re-bootstrapped from that same historical date on the very
    next startup: reset_for_epoch() reads a saved None against the real configured
    epoch as "the epoch changed" and re-pulls the full history from that date,
    recomputing the exact total the reset was meant to clear.
    """
    monkeypatch.setattr(settings, "pnl_epoch", "2026-08-14")

    state = _fresh_reconciler_state()
    rec = PnLReconciler.from_dict(state)

    changed = rec.reset_for_epoch(settings.pnl_epoch_ms)

    assert changed is False, (
        "the fresh baseline was discarded on the very next startup -- the reset does "
        "not actually stick when PNL_EPOCH is a fixed date"
    )
    assert rec.bootstrapped is True
    assert rec.realized_pnl == 0.0


def test_fresh_reconciler_state_also_works_with_a_rolling_window(monkeypatch):
    """PNL_EPOCH unset (rolling BOOTSTRAP_LOOKBACK_DAYS window) means pnl_epoch_ms is
    None -- confirm the fix doesn't regress that configuration, where this bug was
    invisible (None already matched None)."""
    monkeypatch.setattr(settings, "pnl_epoch", "")

    state = _fresh_reconciler_state()
    rec = PnLReconciler.from_dict(state)

    changed = rec.reset_for_epoch(settings.pnl_epoch_ms)

    assert changed is False
    assert rec.bootstrapped is True


def test_fresh_reconciler_state_zeroes_every_total():
    state = _fresh_reconciler_state()
    assert state["realized_pnl"] == 0.0
    assert state["commission"] == 0.0
    assert state["funding_fee"] == 0.0
    assert state["bootstrapped"] is True
