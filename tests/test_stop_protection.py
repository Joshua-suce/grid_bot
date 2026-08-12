"""Regression tests for the two stop-loss defects found in the 2026-08-12 live run.

Both surfaced at the same moment -- the 14:06 recenter -- and both had the same shape:
an event that was not a stop firing nonetheless reduced the protection on a 7108 DOGE
long that stayed open. They are the same class as AUDIT #14/#15, one layer down.

  14:05:26  trail SELL 3554 @ 0.0693647   hard SELL 3554 @ 0.06825994
  14:06:00  recenter -> pause() -> cancel_everything()   (cancels stops too)
  14:06:16  "SCALE-OUT STOP FIRED"  <- nothing fired; price was 0.07035
  14:06:17  hard SELL 7108 @ 0.06696984  <- trail leg gone, hard stop 1.9% wider
"""

import pytest

from grid import GridEngine
from main import trail_stop_fired


class _StubExchange:
    class exchange:
        @staticmethod
        def amount_to_precision(symbol, amount):
            return f"{float(amount):.0f}"

        @staticmethod
        def price_to_precision(symbol, price):
            return f"{float(price):.5f}"


def make_engine(**kw):
    defaults = dict(
        exchange=_StubExchange(), symbol="DOGEUSDT",
        grid_lower=0.07037107, grid_upper=0.07298893, grid_count=10,
        capital_per_grid_pct=0.018, stop_loss_pct=0.03,
    )
    defaults.update(kw)
    return GridEngine(**defaults)


# --- #25: the hard stop must ratchet too -----------------------------------

def test_hard_stop_does_not_widen_when_the_grid_recenters_lower():
    """The exact 14:06 regression, in numbers taken from the log."""
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)

    before = grid.get_hard_stop_loss_price()
    assert before == pytest.approx(0.06825994, abs=1e-8)

    # recenter() moved the band down while the long was still open
    grid.grid_lower, grid.grid_upper = 0.06904107, 0.07165893
    after = grid.get_hard_stop_loss_price()

    assert after >= before - 1e-12, (
        f"hard stop loosened from {before} to {after} with 7108 DOGE still open"
    )
    assert after == pytest.approx(before, abs=1e-8)


def test_hard_stop_still_tightens_when_the_grid_recenters_higher():
    """A ratchet only blocks loosening -- tightening must still work."""
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)
    before = grid.get_hard_stop_loss_price()

    grid.grid_lower, grid.grid_upper = 0.07200000, 0.07460000
    assert grid.get_hard_stop_loss_price() > before


def test_hard_stop_tracks_the_grid_again_once_flat():
    """With nothing open there is nothing to protect, so the level is free to move."""
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)
    grid.get_hard_stop_loss_price()          # arm the ratchet

    grid.set_position_limit(long_position=0.0, short_position=0.0, max_position_qty=8400.0)
    grid.grid_lower, grid.grid_upper = 0.06904107, 0.07165893
    assert grid.get_hard_stop_loss_price() == pytest.approx(0.06904107 * 0.97)


def test_short_hard_stop_never_rises_while_short_is_open():
    grid = make_engine()
    grid.set_position_limit(long_position=0.0, short_position=5000.0, max_position_qty=8400.0)
    before = grid.get_short_hard_stop_loss_price()

    grid.grid_lower, grid.grid_upper = 0.07300000, 0.07560000
    after = grid.get_short_hard_stop_loss_price()
    assert after <= before + 1e-12, f"short hard stop loosened from {before} to {after}"


def test_reset_trailing_releases_the_hard_ratchet():
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)
    grid.get_hard_stop_loss_price()
    assert grid._hard_sl_price is not None

    grid.reset_trailing()
    assert grid._hard_sl_price is None
    assert grid._hard_sl_price_short is None


def test_default_stop_getter_uses_the_ratcheted_hard_level():
    """get_stop_loss_price falls back to the hard level when no trail is set, so it
    must inherit the ratchet rather than recomputing from grid_lower."""
    grid = make_engine()
    grid.set_position_limit(long_position=7108.0, short_position=0.0, max_position_qty=8400.0)
    before = grid.get_stop_loss_price()

    grid.grid_lower, grid.grid_upper = 0.06904107, 0.07165893
    assert grid.get_stop_loss_price() >= before - 1e-12


# --- #26: a cancelled stop is not a fired stop -----------------------------

@pytest.mark.parametrize("status", ["closed", "filled"])
def test_completed_stop_counts_as_fired(status):
    assert trail_stop_fired({"id": "x", "status": status}) is True


@pytest.mark.parametrize("status", ["canceled", "cancelled", "expired", "open", "rejected"])
def test_uncompleted_stop_does_not_count_as_fired(status):
    """The 14:06 case: recenter cancelled it, so it must not latch the scale-out."""
    assert trail_stop_fired({"id": "x", "status": status}) is False


def test_unknown_order_does_not_count_as_fired():
    """fetch_order returning None (unreachable, purged) must read as 'did not fire' --
    wrongly latching strips the trailing leg for the position's whole life."""
    assert trail_stop_fired(None) is False


def test_missing_status_does_not_count_as_fired():
    assert trail_stop_fired({"id": "x"}) is False
