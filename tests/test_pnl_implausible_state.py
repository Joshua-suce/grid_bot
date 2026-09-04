"""A loaded pnl_reconciler can claim bootstrapped=True and still be worthless. AUDIT #171.

2026-08-25 21:36:48: a restart loaded a pnl_reconciler whose epoch_ms and
last_income_time_ms both looked intact, but realized_pnl/commission/funding_fee had
been silently hollowed out to near-zero. bootstrapped stayed True, so sync() just
resumed on top of the hollow totals — a -74.84 loss that had been correctly
reconciled five days earlier (and every other dollar of the account's real history
back to epoch) quietly vanished from every number the bot reported from then on.
window_label kept reading "since 2026-08-14" for the next ten days; the figure
behind it only reflected the ten days since the reset. No log line ever announced
it — PNL EPOCH CHANGED only fires when epoch_ms itself changes, and it hadn't.

commission is the tell: it is charged on every fill, maker or taker, so an account
that has been bootstrapped and live for a multi-day span cannot legitimately show
exactly zero. A quiet, flat market can genuinely produce zero REALIZED_PNL, and an
account that is never holding a position at a funding timestamp can genuinely
produce zero FUNDING_FEE — but no order fills for free.
"""

import pytest

from pnl_tracker import PnLReconciler, IMPLAUSIBLE_STATE_MIN_SPAN_MS


THREE_DAYS_MS = 3 * 86_400_000
EPOCH = 1_786_665_600_000  # 2026-08-14 00:00:00 UTC


def _hollowed_out(span_ms=THREE_DAYS_MS + 60_000):
    """Exactly what came back from state on 2026-08-25: epoch and cursor both
    intact, totals reset to (near-)zero, bootstrapped never noticed."""
    return PnLReconciler(
        realized_pnl=-0.9270, commission=0.0, funding_fee=0.0,
        last_income_time_ms=EPOCH + span_ms,
        bootstrapped=True, epoch_ms=EPOCH,
    )


class _FakeExchange:
    """Records every fetch it's asked for; returns no entries."""

    def __init__(self):
        self.calls = []

    def get_income_history(self, symbol, since_ms=None, limit=None):
        self.calls.append(since_ms)
        return []


def test_a_hollowed_out_state_is_flagged_implausible():
    r = _hollowed_out()

    reason = r._looks_implausible()

    assert reason is not None
    assert "3.0" in reason or "days" in reason


def test_sync_discards_and_rebootstraps_from_epoch_when_implausible():
    r = _hollowed_out()
    ex = _FakeExchange()

    ok = r.sync(ex, "ADAUSDT")

    assert ok is True
    # bootstrap() re-fetches starting at epoch_ms, not the hollowed-out cursor —
    # the whole point is to re-derive the true history from scratch.
    assert ex.calls == [EPOCH]
    assert r.bootstrapped is True
    assert r.realized_pnl == 0.0
    assert r.commission == 0.0
    assert r.funding_fee == 0.0
    assert r.last_income_time_ms == 0, (
        "bootstrap() found nothing (the fake returns no entries) so the cursor "
        "stays at its reset value, not the hollowed-out one"
    )


def test_a_freshly_bootstrapped_account_with_no_fills_yet_is_not_flagged():
    """A bot minutes old and genuinely idle must not have its own true, empty
    history discarded as corrupt."""
    r = PnLReconciler(
        realized_pnl=0.0, commission=0.0, funding_fee=0.0,
        last_income_time_ms=EPOCH + 60_000,
        bootstrapped=True, epoch_ms=EPOCH,
    )

    assert r._looks_implausible() is None


def test_the_span_threshold_is_exclusive_at_the_boundary():
    just_under = _hollowed_out(span_ms=IMPLAUSIBLE_STATE_MIN_SPAN_MS - 1)
    just_over = _hollowed_out(span_ms=IMPLAUSIBLE_STATE_MIN_SPAN_MS)

    assert just_under._looks_implausible() is None
    assert just_over._looks_implausible() is not None


@pytest.mark.parametrize("commission", [-0.0494, 0.0494, 1e-6, -12.3])
def test_any_nonzero_commission_is_never_flagged_regardless_of_span(commission):
    """The account that has actually been trading for ten days since the reset —
    commission has long since climbed off zero through ordinary continued use —
    is not what this check is for. It only catches the state in the narrow window
    right after a corrupting event, before new trading papers over it."""
    r = _hollowed_out()
    r.commission = commission

    assert r._looks_implausible() is None


def test_float_dust_at_zero_still_counts_as_zero():
    r = _hollowed_out()
    r.commission = 1e-15

    assert r._looks_implausible() is not None


def test_an_unbootstrapped_state_is_never_flagged():
    """sync()'s existing bootstrap-on-first-run path already handles this —
    _looks_implausible must not fire ahead of it and change the log line."""
    r = PnLReconciler(
        realized_pnl=0.0, commission=0.0, funding_fee=0.0,
        last_income_time_ms=0, bootstrapped=False, epoch_ms=EPOCH,
    )

    assert r._looks_implausible() is None


def test_no_epoch_configured_is_never_flagged():
    """The rolling-window mode has no anchor to measure a span from."""
    r = PnLReconciler(
        realized_pnl=0.0, commission=0.0, funding_fee=0.0,
        last_income_time_ms=1_800_000_000_000,
        bootstrapped=True, epoch_ms=None,
    )

    assert r._looks_implausible() is None


def test_zero_cursor_is_never_flagged():
    """bootstrapped=True with a cursor that never actually advanced (a bootstrap
    that legitimately found zero entries) is not the failure this guards against —
    reset_for_epoch already produces exactly this shape and it must stay quiet."""
    r = PnLReconciler(
        realized_pnl=0.0, commission=0.0, funding_fee=0.0,
        last_income_time_ms=0, bootstrapped=True, epoch_ms=EPOCH,
    )

    assert r._looks_implausible() is None


def test_a_healthy_state_is_left_alone_by_sync():
    """The overwhelmingly common case: real trading history, real commission —
    sync() must not discard it or re-fetch the whole account from epoch again."""
    r = PnLReconciler(
        realized_pnl=6.4776, commission=-3.2016, funding_fee=-0.1373,
        last_income_time_ms=EPOCH + 10 * 86_400_000,
        last_seen_keys={"REALIZED_PNL:123"},
        bootstrapped=True, epoch_ms=EPOCH,
    )
    ex = _FakeExchange()

    cursor_before = r.last_income_time_ms
    r.sync(ex, "ADAUSDT")

    # the incremental fetch started at the real cursor, not epoch — nothing was discarded
    assert ex.calls == [cursor_before]
    assert ex.calls[0] == EPOCH + 10 * 86_400_000
    assert r.realized_pnl == 6.4776
    assert r.commission == -3.2016


def test_the_reason_explains_itself_in_days():
    r = _hollowed_out(span_ms=11 * 86_400_000)

    reason = r._looks_implausible()

    assert "11.0" in reason
    assert "commission" in reason
