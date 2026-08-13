"""Safety nets that silently were not there. AUDIT #50.

Three defects of one family, all on the risk path, all invisible in the logs:

  1. `_refresh_sl_stops` cancels every stop FIRST, then places replacements inside a
     try/except that only logs. Any failure leaves an open position with no stop, and
     the caller could not tell -- it returned None either way. That is the 2026-08-08
     sequence: cancel succeeded, four placements raised TypeError (#47), each was
     swallowed, and a position 1.8x through its cap (#49) ran unprotected. -50.49.

  2. `_hard_sl_price` / `_hard_sl_price_short` are RATCHETS -- a long's stop may only
     rise, a short's only fall -- because recenter() moves the grid bounds the stop is
     derived from. Every other stop anchor was persisted. These two were not, so a
     restart with an open position silently LOOSENED its stop.

  3. `cancel_everything` logged "CLEANUP VERIFIED | book clean" when the verification
     READ failed, because `_fetch_regular()` returns None and `if remaining:` treats
     None as an empty book. Startup runs this before laying a fresh ladder.
"""

import json

import pytest

from grid import GridEngine


class _Ex:
    class exchange:
        @staticmethod
        def amount_to_precision(s, a): return f"{float(a):.0f}"
        @staticmethod
        def price_to_precision(s, p): return f"{float(p):.5f}"

    def get_price(self, s): return 0.0700
    def get_balance(self, s="USDT"): return 5000.0
    def get_positions(self, s): return []
    def get_open_orders(self, s): return []
    def get_open_order_ids(self, s): return set()
    def can_place_order(self, s): return True
    def place_limit_order(self, *a, **k): return {"id": "x"}


def _engine(lower=0.0690, upper=0.0710):
    g = GridEngine(exchange=_Ex(), symbol="DOGEUSDT", grid_lower=lower, grid_upper=upper,
                   grid_count=10, capital_per_grid_pct=0.018, stop_loss_pct=0.03,
                   min_profit_multiplier=3.0, max_exposure_pct=0.5, leverage=5)
    g.initialize(0.0700, balance=5000.0)
    return g


# --- 2. the hard-stop ratchet must survive a restart ------------------------

def test_a_long_hard_stop_does_not_loosen_across_a_restart():
    """The ratchet held a stop at the tighter level while the grid recentred away.
    Losing it on restart re-derives the stop from the NEW bounds -- looser."""
    g = _engine(lower=0.0690, upper=0.0710)
    g._net_long_qty = 5000.0
    tight = g.get_hard_stop_loss_price()          # ratchets to 0.0690 * (1-0.03)

    g.grid_lower, g.grid_upper = 0.0660, 0.0680   # recenter drops the grid
    held = g.get_hard_stop_loss_price()
    assert held == tight, "precondition: the ratchet should refuse to lower the stop"

    revived = _engine(lower=0.0660, upper=0.0680)
    revived.load_from_dict(json.loads(json.dumps(g.to_dict())), current_price=0.0700)
    revived._net_long_qty = 5000.0

    assert revived.get_hard_stop_loss_price() == pytest.approx(tight), (
        f"stop loosened across the restart: {revived.get_hard_stop_loss_price():.6f} "
        f"vs {tight:.6f} -- the ratchet was not persisted"
    )


def test_a_short_hard_stop_does_not_loosen_across_a_restart():
    """Mirror: a short's hard stop may only fall."""
    g = _engine(lower=0.0690, upper=0.0710)
    g._net_short_qty = 5000.0
    tight = g.get_short_hard_stop_loss_price()

    g.grid_lower, g.grid_upper = 0.0720, 0.0740
    assert g.get_short_hard_stop_loss_price() == tight

    revived = _engine(lower=0.0720, upper=0.0740)
    revived.load_from_dict(json.loads(json.dumps(g.to_dict())), current_price=0.0730)
    revived._net_short_qty = 5000.0

    assert revived.get_short_hard_stop_loss_price() == pytest.approx(tight), (
        "short stop loosened across the restart -- the ratchet was not persisted"
    )


# --- 1. an unprotected position must not also be a growing one --------------

def test_block_side_stops_new_exposure_but_not_exits():
    """block_side() gates only orders that ADD exposure; reduce-only exits are
    untouched, because trapping inventory is #42."""
    g = _engine()
    assert not g._block_buys

    g.block_side("buy", "stop-loss missing")

    assert g._block_buys and not g._block_sells
    # exits are decided by _exit_order_params, which block_side never consults
    g._net_short_qty = 1000.0
    params, qty = g._exit_order_params("buy", 500.0)
    assert params and params.get("reduceOnly"), "the exit path was affected by a block"


def test_block_side_is_idempotent_and_side_specific():
    g = _engine()
    g.block_side("sell", "stop-loss missing")
    g.block_side("sell", "stop-loss missing")
    assert g._block_sells and not g._block_buys


def test_the_trend_follower_withholds_a_blocked_entry():
    """main.py talks to whatever strategy is live. In router mode that is a trend
    follower, so the block has to mean something there too, not just on the grid."""
    from trend_follower import TrendFollower

    tf = TrendFollower.__new__(TrendFollower)
    tf._blocked_sides = {}
    tf.set_position_limit = TrendFollower.set_position_limit.__get__(tf)
    TrendFollower.block_side(tf, "buy", "stop-loss missing")

    assert tf._blocked_sides == {"buy": "stop-loss missing"}

    # ...and lifts on its own once the position is protected again
    tf._net_long_qty = tf._net_short_qty = tf._max_position_qty = 0.0
    tf.set_position_limit(0.0, 0.0, 0.0)
    assert tf._blocked_sides == {}, (
        "the block latched permanently -- a recovered stop would never re-enable entries"
    )


def test_the_router_broadcasts_a_block_to_every_strategy():
    """The position is NET and shared; blocking only the live strategy leaves the
    dormant one free to add to the same unprotected side when it takes over."""
    from router import StrategyRouter

    class _S:
        def __init__(self): self.blocks = []
        def block_side(self, side, reason): self.blocks.append((side, reason))

    r = StrategyRouter.__new__(StrategyRouter)
    r.strategies = {"grid": _S(), "trend": _S()}

    StrategyRouter.block_side(r, "sell", "stop-loss missing")

    for name, s in r.strategies.items():
        assert s.blocks == [("sell", "stop-loss missing")], f"{name} never heard the block"


def test_main_uses_the_return_value_to_block_exposure():
    """Structural: a correct _refresh_sl_stops wired to a caller that ignores its
    result fixes nothing -- which is exactly how #47 survived."""
    import pathlib

    text = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert "def _refresh_sl_stops(side: str, qty: float) -> bool:" in text, (
        "_refresh_sl_stops no longer reports whether the position ended up covered"
    )
    assert "sl_covered = _refresh_sl_stops(" in text, (
        "main.py calls _refresh_sl_stops but discards whether it succeeded"
    )
    assert 'grid.block_side("buy"' in text and 'grid.block_side("sell"' in text, (
        "main.py does not block new exposure when the position is unprotected"
    )


# --- 3. unknown is not clean ------------------------------------------------

def _cancel_everything_logs(get_open_orders):
    """Run the real cancel_everything against a stubbed book and capture its logs."""
    import sys

    from loguru import logger

    from exchange import Exchange

    ex = Exchange.__new__(Exchange)
    ex.get_open_orders = get_open_orders
    ex.get_stop_orders = lambda symbol: []
    ex.cancel_order = lambda oid, symbol: True
    ex.exchange = type("X", (), {"cancel_all_orders": staticmethod(lambda s: None)})()

    sink = []
    logger.remove()
    h = logger.add(lambda m: sink.append(str(m)), level="INFO")
    try:
        ex.cancel_everything("DOGEUSDT", timeout_seconds=0)
    except Exception:
        pass                       # only the logging behaviour is under test
    finally:
        logger.remove(h)
        logger.add(sys.stderr, level="INFO")
    return "".join(sink)


def test_an_unreadable_book_is_not_reported_as_clean():
    """`if remaining:` treated a failed read (None) as an empty book, so a cleanup that
    verified nothing announced 'CLEANUP VERIFIED | book clean'. Startup runs this before
    laying a fresh ladder -- a false all-clear means a new grid on top of live orders."""
    def unreadable(symbol):
        raise ConnectionError("backend unreachable")

    out = _cancel_everything_logs(unreadable)

    assert "CLEANUP VERIFIED" not in out, (
        "a cleanup that could not read the book still reported it clean:\n" + out
    )
    assert "CLEANUP UNVERIFIED" in out, (
        "the unknown state is not surfaced at all:\n" + out
    )


def test_a_genuinely_empty_book_is_still_reported_clean():
    """The fix must not cry wolf: an empty book is verified, not unknown."""
    out = _cancel_everything_logs(lambda symbol: [])

    assert "CLEANUP VERIFIED" in out, out
    assert "CLEANUP UNVERIFIED" not in out, out
