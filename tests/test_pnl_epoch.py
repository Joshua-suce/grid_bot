"""Cumulative PnL is reported from a date you choose. AUDIT #60.

The rolling BOOTSTRAP_LOOKBACK_DAYS window kept dragging old history into the headline.
On 2026-08-14 it read -29.08, of which -50.49 was a single day (08-08) caused by the
position-cap and stop-loss defects fixed in #49/#50. That day stays inside an 89-day
window until early November, so the bot would have reported a loss for months no matter
how well it actually traded.

PNL_EPOCH pins the start date. The trap it has to avoid: saved state carries
bootstrapped=True and a cursor far ahead of a newly-set epoch, so without an explicit
reset, changing the setting does nothing at all and the operator reads stale totals
believing they changed them.
"""

import pytest

from pnl_tracker import PnLReconciler


def _loaded(epoch_ms=None, realized=100.0):
    """A reconciler as it comes back from saved state: already bootstrapped, cursor set."""
    return PnLReconciler(
        realized_pnl=realized, commission=-10.0, funding_fee=1.0,
        last_income_time_ms=1_780_000_000_000, last_seen_keys={"REALIZED_PNL:1"},
        bootstrapped=True, epoch_ms=epoch_ms,
    )


def test_changing_the_epoch_discards_stale_totals():
    r = _loaded(epoch_ms=None)
    assert r.net_realized_pnl == pytest.approx(91.0)

    changed = r.reset_for_epoch(1_786_665_600_000)

    assert changed is True
    assert r.realized_pnl == 0.0 and r.commission == 0.0 and r.funding_fee == 0.0
    assert r.bootstrapped is False, "a stale bootstrap flag means sync() never re-pulls"
    assert r.last_income_time_ms == 0, "a stale cursor skips everything before it"
    assert r.last_seen_keys == set()
    assert r.epoch_ms == 1_786_665_600_000


def test_an_unchanged_epoch_leaves_everything_alone():
    """Restarts are routine. They must not re-pull history every time."""
    r = _loaded(epoch_ms=1_786_665_600_000)

    assert r.reset_for_epoch(1_786_665_600_000) is False
    assert r.realized_pnl == 100.0
    assert r.bootstrapped is True
    assert r.last_income_time_ms == 1_780_000_000_000


def test_clearing_the_epoch_also_resets():
    """Going back to the rolling window is a change like any other."""
    r = _loaded(epoch_ms=1_786_665_600_000)

    assert r.reset_for_epoch(None) is True
    assert r.epoch_ms is None
    assert r.bootstrapped is False


def test_the_epoch_survives_a_restart():
    r = _loaded(epoch_ms=1_786_665_600_000)
    revived = PnLReconciler.from_dict(r.to_dict())

    assert revived.epoch_ms == 1_786_665_600_000
    assert revived.reset_for_epoch(1_786_665_600_000) is False, (
        "the epoch did not round-trip, so every restart would re-bootstrap"
    )


def test_bootstrap_starts_at_the_epoch_not_the_rolling_window():
    """The whole point: the fetch must begin at the configured date."""
    seen = {}

    class _Ex:
        def get_income_history(self, symbol, since_ms=None, limit=None):
            seen.setdefault("since", since_ms)
            return []

    epoch = 1_786_665_600_000
    r = PnLReconciler(epoch_ms=epoch)
    r.bootstrap(_Ex(), "DOGEUSDT")

    assert seen["since"] == epoch


def test_bootstrap_without_an_epoch_uses_the_rolling_window():
    import time

    from pnl_tracker import BOOTSTRAP_LOOKBACK_DAYS

    seen = {}

    class _Ex:
        def get_income_history(self, symbol, since_ms=None, limit=None):
            seen.setdefault("since", since_ms)
            return []

    r = PnLReconciler()
    r.bootstrap(_Ex(), "DOGEUSDT")

    expected = int(time.time() * 1000) - BOOTSTRAP_LOOKBACK_DAYS * 86400 * 1000
    assert abs(seen["since"] - expected) < 60_000


# --- config parsing ---------------------------------------------------------

def test_epoch_parses_to_utc_midnight():
    from config import Settings

    s = Settings(api_key="k", api_secret="s", pnl_epoch="2026-08-14")

    assert s.pnl_epoch_ms == 1_786_665_600_000


def test_blank_epoch_means_rolling_window():
    from config import Settings

    assert Settings(api_key="k", api_secret="s", pnl_epoch="").pnl_epoch_ms is None
    assert Settings(api_key="k", api_secret="s", pnl_epoch="   ").pnl_epoch_ms is None


def test_a_malformed_epoch_raises_instead_of_silently_reverting():
    """Falling back to the rolling window would leave the operator reading a figure they
    thought they had changed -- the exact failure this feature exists to end."""
    from config import Settings

    for bad in ("14-08-2026", "2026/08/14", "yesterday", "2026-13-01"):
        with pytest.raises(ValueError, match="PNL_EPOCH"):
            Settings(api_key="k", api_secret="s", pnl_epoch=bad).pnl_epoch_ms


def test_the_label_comes_from_the_reconciler_not_from_config():
    """The label must describe the value that produced the number. Reading it off
    settings while the number comes off the reconciler lets them disagree -- a preview
    rendered "since 2026-08-14" above the full 89-day total exactly that way."""
    from pnl_tracker import BOOTSTRAP_LOOKBACK_DAYS

    assert PnLReconciler().window_label == f"{BOOTSTRAP_LOOKBACK_DAYS}d rolling"
    assert PnLReconciler(epoch_ms=1_786_665_600_000).window_label == "since 2026-08-14"


def test_main_takes_the_window_label_from_the_reconciler():
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert "pnl_window = pnl_reconciler.window_label" in source, (
        "the status line label is derived from config again -- it can disagree with the "
        "number beside it"
    )
    assert 'f"since {settings.pnl_epoch}"' not in source


def test_the_footer_uses_the_label_it_is_given():
    from telegram_notifier import _pnl_lines

    text = _pnl_lines(0.0, 1.1069, "since 2026-08-14")
    assert "since 2026-08-14" in text
    assert "89d" not in text


def test_main_resets_before_it_syncs():
    """Order matters: syncing first would advance the cursor past the new epoch."""
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    reset = source.index("pnl_reconciler.reset_for_epoch(")
    sync = source.index("pnl_reconciler.sync(")
    assert reset < sync, "the epoch reset must run before the first sync"
