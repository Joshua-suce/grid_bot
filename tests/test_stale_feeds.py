"""A risk control reading a frozen number is not a risk control. AUDIT #52.

`last_sync_time` was recorded from the very first version of the reconciler and read by
nobody. The income endpoint is separate from the order endpoints and fails on its own;
when it does, `sync()` logs a warning, returns False, and all three callers in main.py
discard that answer. `daily_net_pnl` then simply stops moving -- and it is the input to
the daily-loss kill switch, so the account can bleed while the switch reads a healthy
figure from an hour ago.

This is AUDIT #39's frozen-equity defect in a second location, which is why the fix
follows the same rule as #50: an unmeasurable position may be closed and may be held,
but it may not grow.
"""

import time

import pytest

from pnl_tracker import PnLReconciler


def test_a_reconciler_that_has_never_synced_is_stale():
    r = PnLReconciler()
    assert r.seconds_since_sync() == float("inf")
    assert r.is_stale(1800.0), "a reconciler with no data at all reported itself fresh"


def test_a_frozen_feed_goes_stale():
    r = PnLReconciler()
    r.last_sync_time = time.time() - 3600
    assert r.is_stale(1800.0)
    assert not r.is_stale(7200.0), "the threshold is not being honoured"


def test_a_recent_sync_is_fresh():
    """The guard must not fire in normal operation -- it blocks new exposure."""
    r = PnLReconciler()
    r.last_sync_time = time.time() - 30
    assert not r.is_stale(1800.0)
    assert r.seconds_since_sync() == pytest.approx(30, abs=5)


def test_daily_pnl_can_be_arbitrarily_wrong_while_stale():
    """The point of the guard: staleness is invisible in the value itself."""
    r = PnLReconciler()
    r.daily_net_pnl = 42.0                 # a healthy figure...
    r.last_sync_time = time.time() - 86400  # ...recorded a day ago

    assert r.daily_net_pnl == 42.0, "the number looks perfectly fine"
    assert r.is_stale(1800.0), "and only the age reveals it means nothing"


# --- main.py must act on it -------------------------------------------------

def _main_source():
    import pathlib
    return (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")


def test_main_blocks_new_exposure_on_a_stale_feed():
    text = _main_source()
    assert "pnl_reconciler.is_stale(PNL_STALE_SECONDS)" in text, (
        "main.py feeds daily_net_pnl to the kill switch without checking how old it is"
    )
    i = text.index("pnl_reconciler.is_stale(PNL_STALE_SECONDS)")
    block = text[i : i + 1200]
    assert 'grid.block_side("buy"' in block and 'grid.block_side("sell"' in block, (
        "the stale feed is detected but nothing stops the position growing"
    )
    assert text.index("pnl_reconciler.is_stale(PNL_STALE_SECONDS)") < text.index(
        "daily_realized_pnl=pnl_reconciler.daily_net_pnl"
    ), "the staleness check must run before the kill switch consumes the figure"


def test_startup_blocks_when_it_cannot_establish_a_stop():
    """The third _refresh_sl_stops caller. It runs at STARTUP against an inherited
    position, before the loop has done anything -- and it discarded the answer, so a
    restart that could not re-establish stops went on to activate the grid."""
    text = _main_source()
    assert "if not _refresh_sl_stops(position_side, position_qty):" in text, (
        "the startup stop-loss placement still ignores whether it succeeded"
    )
    i = text.index("if not _refresh_sl_stops(position_side, position_qty):")
    block = text[i : i + 600]
    assert "block_side" in block, "startup detects the failure but still lets the position grow"
    assert "stop-loss missing at startup" in block
