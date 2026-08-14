"""The bot could not say how the current run was doing. AUDIT #59.

Every figure it reported was either the 89-day ACCOUNT lifetime or today. So a fresh
start -- after the state had deliberately been cleared -- opened with:

    Total PnL (verified): -30.1986 USDT

That number is correct. Verified against the raw ledger to the cent: over 89 days the
account is -29.08, because July made +63.58 and August lost 92.66. But it is the
account's history, not this run's, and it includes a -50.49 day caused by defects that
are now fixed. Labelled "Total PnL" on a "Bot Started" message, it reads as though the
bot begins in the red -- and it hides the only number that answers "is it working now?"
"""

import pytest

from pnl_tracker import PnLReconciler


def _rec(realized=0.0, commission=0.0, funding=0.0):
    return PnLReconciler(realized_pnl=realized, commission=commission, funding_fee=funding)


def test_session_pnl_starts_at_zero_however_bad_the_account_history_is():
    r = _rec(realized=14.55, commission=-46.71, funding=3.08)
    assert r.net_realized_pnl == pytest.approx(-29.08, abs=0.01)

    r.begin_session()

    assert r.session_pnl == 0.0, (
        "a fresh run opens showing the account's 89-day loss as if it were its own"
    )
    assert r.net_realized_pnl == pytest.approx(-29.08, abs=0.01), (
        "anchoring the session must not alter the account figure"
    )


def test_session_pnl_tracks_only_what_happened_after_the_anchor():
    r = _rec(realized=-100.0)
    r.begin_session()

    r.realized_pnl += 4.0
    r.commission -= 1.0

    assert r.session_pnl == pytest.approx(3.0)
    assert r.net_realized_pnl == pytest.approx(-97.0)


def test_a_profitable_run_reads_positive_on_a_negative_account():
    """The case that matters: the fixes work, the run makes money, and the operator can
    see that even though the account is still underwater from before."""
    r = _rec(realized=-500.0)
    r.begin_session()
    r.realized_pnl += 12.5

    assert r.session_pnl > 0
    assert r.net_realized_pnl < 0


def test_session_baseline_is_not_persisted():
    """It means "since this process started", so a restart must reset it. Persisting it
    would quietly turn it into a second lifetime counter."""
    r = _rec(realized=50.0)
    r.begin_session()
    r.realized_pnl += 10.0
    assert r.session_pnl == pytest.approx(10.0)

    assert "session_start_net" not in r.to_dict()

    revived = PnLReconciler.from_dict(r.to_dict())
    assert revived.session_start_net is None
    assert revived.session_pnl == 0.0, "session PnL survived a restart"


def test_session_pnl_is_zero_before_the_anchor_is_set():
    """Never None, so callers and format strings cannot trip over it."""
    r = _rec(realized=7.0)
    assert r.session_start_net is None
    assert r.session_pnl == 0.0


def test_main_anchors_the_session_before_trading():
    """Structural: anchoring after the first fill would silently swallow it."""
    import pathlib
    import re

    source = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert "pnl_reconciler.begin_session()" in source, (
        "the session baseline is never anchored, so session PnL stays 0 forever"
    )
    anchor = source.index("pnl_reconciler.begin_session()")
    first_order = re.search(r"place_initial_orders|grid\.activate", source)
    assert first_order and anchor < first_order.start(), (
        "the session is anchored after trading starts -- early fills would be lost"
    )


# --- the message the user actually sees -------------------------------------

def test_the_footer_leads_with_this_run_and_labels_the_account_figure():
    from telegram_notifier import _pnl_lines

    text = _pnl_lines(session_pnl=0.0, account_pnl=-30.1986)

    assert "This run" in text
    assert text.index("This run") < text.index("Account"), "session must come first"
    assert "Account" in text and "all activity" in text, (
        "the 89-day figure is still presented as though it were the bot's own"
    )
    assert "Total PnL" not in text, "the misleading label survived"
    assert "+0.0000" in text and "-30.1986" in text


def test_the_footer_degrades_gracefully_when_a_figure_is_missing():
    from telegram_notifier import _pnl_lines

    assert _pnl_lines(None, None) == ""
    assert "This run" in _pnl_lines(1.25, None)
    assert "Account" in _pnl_lines(None, -5.0)
