"""Does one side of the ladder fit inside the position cap? AUDIT #66.

config.validate() answers this for PERCENT sizing. It cannot for the
CAPITAL_PER_GRID_USDT path: that size is an absolute USDT figure while
MAX_POSITION_PCT is a fraction of equity, so the comparison needs a balance that config
time does not have.

#63 correctly stopped validating a number that no longer decides anything -- and left
nothing in its place. So the only guard against the failure below disappeared at exactly
the moment the USDT path became the live one.

The failure is documented in this bot's own history: a grid wider than its cap goes
permanently one-sided, the cap blocks that side partway through, and it lives in the
capped state that makes recentering destructive. 89 recenters in one session.
"""

import pytest

from grid import GridEngine


class _Ex:
    class exchange:
        @staticmethod
        def amount_to_precision(s, a): return f"{float(a):.0f}"
        @staticmethod
        def price_to_precision(s, p): return f"{float(p):.5f}"

    def get_price(self, s): return 0.06987
    def get_balance(self, s="USDT"): return 4931.0
    def get_positions(self, s): return []
    def get_open_orders(self, s): return []
    def get_open_order_ids(self, s): return set()
    def can_place_order(self, s): return True
    def place_limit_order(self, *a, **k): return {"id": "x"}


def _engine(count=8, leverage=25, usdt=5.0):
    return GridEngine(
        exchange=_Ex(), symbol="DOGEUSDT", grid_lower=0.0689, grid_upper=0.0709,
        grid_count=count, capital_per_grid_pct=0.018, capital_per_grid_usdt=usdt,
        stop_loss_pct=0.03, max_exposure_pct=0.50, leverage=leverage,
    )


BALANCE = 4931.0
CAP_PCT = 0.12
CAP = BALANCE * CAP_PCT          # 591.72


def test_the_shipped_configuration_fits():
    """25x, 8 rungs, 5 USDT per trade -- 4 a side at 125 is 500 against a 591.72 cap."""
    g = _engine(count=8, leverage=25)

    one_side = g.one_side_notional(BALANCE)

    assert one_side == pytest.approx(500.0)
    assert one_side <= CAP
    assert (CAP - one_side) / 125.0 > 0.5, "less than half a rung of headroom"


def test_ten_rungs_at_25x_would_not_fit():
    """The configuration that made GRID_COUNT=8 necessary rather than cosmetic."""
    g = _engine(count=10, leverage=25)

    assert g.one_side_notional(BALANCE) == pytest.approx(625.0)
    assert g.one_side_notional(BALANCE) > CAP


def test_the_previous_15x_setting_also_fit():
    g = _engine(count=10, leverage=15)

    assert g.one_side_notional(BALANCE) == pytest.approx(375.0)
    assert g.one_side_notional(BALANCE) <= CAP


def test_one_side_is_half_the_ladder_not_all_of_it():
    """The arithmetic error that started this: comparing ALL rungs against a cap that
    only ever sees one direction. Buys below and sells above cannot stack together."""
    g = _engine(count=8, leverage=25)

    per_order = g._calc_usdt_per_grid(BALANCE)

    assert g.one_side_notional(BALANCE) == pytest.approx(per_order * 4)
    assert g.one_side_notional(BALANCE) < per_order * 8


def test_it_tracks_the_volatility_taper():
    g = _engine(count=8, leverage=25)
    g.update_volatility(0.05)                     # violent -> shrinks orders

    assert g._volatility_mult < 1.0
    assert g.one_side_notional(BALANCE) < 500.0


def test_main_warns_rather_than_aborting():
    """Structural. The cap and the taper keep an oversized ladder SAFE, only degraded,
    and equity moves -- a restart after a drawdown must not refuse to start."""
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert "LADDER OUTGROWS THE CAP" in source, "the guard is gone"
    assert "one_side_notional(" in source

    guard = source.index("LADDER OUTGROWS THE CAP")
    window = source[guard - 400:guard + 400]
    assert "logger.warning" in window
    assert "return" not in window.split("LADDER OUTGROWS THE CAP")[0][-200:], (
        "the check aborts startup instead of warning"
    )
