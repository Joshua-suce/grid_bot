"""Tests for the historical replay harness.

A backtester that is subtly wrong is worse than none at all: it produces confident
numbers that justify real trades. These tests pin the parts a wrong result would hinge
on -- fee arithmetic, position accounting through flat, fill ordering within a candle,
and the exchange rejections the live account actually enforces.
"""

import pandas as pd
import pytest

import grid as grid_module
from backtest import SimulatedExchange, VirtualClock, _clock_installed, run_backtest
from exchange import PostOnlyWouldCross


def make_ex(**kw):
    defaults = dict(starting_balance=1000.0, maker_fee=0.0002, taker_fee=0.0004)
    defaults.update(kw)
    ex = SimulatedExchange(**defaults)
    ex.price = 0.0710
    return ex


# --- position accounting ---------------------------------------------------

def test_long_round_trip_realizes_exact_spread():
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "buy", 0.0700, 1000)
    ex.place_limit_order("DOGEUSDT", "sell", 0.0720, 1000)
    ex.step(0.0710, 0.0715, 0.0695, 0.0705)   # trades down through the buy
    assert ex.position_qty == 1000
    assert ex.position_entry == pytest.approx(0.0700)

    ex.step(0.0705, 0.0725, 0.0700, 0.0720)   # trades up through the sell
    assert ex.position_qty == 0
    assert ex.realized_pnl == pytest.approx((0.0720 - 0.0700) * 1000)


def test_short_round_trip_realizes_exact_spread():
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "sell", 0.0720, 1000)
    ex.step(0.0710, 0.0725, 0.0705, 0.0720)
    assert ex.position_qty == -1000

    ex.place_limit_order("DOGEUSDT", "buy", 0.0700, 1000)
    ex.step(0.0720, 0.0722, 0.0695, 0.0700)
    assert ex.position_qty == 0
    assert ex.realized_pnl == pytest.approx((0.0720 - 0.0700) * 1000)


def test_adding_to_a_position_blends_the_average_entry():
    """One-way mode keeps a single blended entry, exactly as the live account does.
    Getting this wrong is what made the engine's own PnL drift (AUDIT #7)."""
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "buy", 0.0700, 1000)
    ex.place_limit_order("DOGEUSDT", "buy", 0.0690, 1000)
    ex.step(0.0710, 0.0712, 0.0685, 0.0690)
    assert ex.position_qty == 2000
    assert ex.position_entry == pytest.approx(0.0695)


def test_flipping_through_flat_resets_entry_to_fill_price():
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "buy", 0.0700, 1000)
    ex.step(0.0710, 0.0712, 0.0695, 0.0700)
    assert ex.position_qty == 1000

    # Sell 2500: closes 1000 long, opens 1500 short.
    ex.price = 0.0700
    ex.place_limit_order("DOGEUSDT", "sell", 0.0720, 2500)
    ex.step(0.0700, 0.0725, 0.0699, 0.0720)
    assert ex.position_qty == -1500
    assert ex.position_entry == pytest.approx(0.0720)
    assert ex.realized_pnl == pytest.approx((0.0720 - 0.0700) * 1000)


def test_fees_are_charged_on_both_legs_at_the_maker_rate():
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "buy", 0.0700, 1000)
    ex.place_limit_order("DOGEUSDT", "sell", 0.0720, 1000)
    ex.step(0.0710, 0.0715, 0.0695, 0.0705)
    ex.step(0.0705, 0.0725, 0.0700, 0.0720)
    expected = 1000 * 0.0700 * 0.0002 + 1000 * 0.0720 * 0.0002
    assert ex.fees_paid == pytest.approx(expected)


def test_equity_equals_cash_plus_unrealized():
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "buy", 0.0700, 1000)
    ex.step(0.0710, 0.0712, 0.0695, 0.0705)
    assert ex.equity == pytest.approx(ex.cash + (0.0705 - 0.0700) * 1000)


# --- fill matching ---------------------------------------------------------

def test_buy_fills_only_when_price_trades_down_to_it():
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "buy", 0.0690, 1000)
    ex.step(0.0710, 0.0715, 0.0700, 0.0705)   # low 0.0700 never reaches 0.0690
    assert ex.position_qty == 0
    ex.step(0.0705, 0.0710, 0.0685, 0.0700)   # low 0.0685 does
    assert ex.position_qty == 1000


def test_sell_fills_only_when_price_trades_up_to_it():
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "sell", 0.0730, 1000)
    ex.step(0.0710, 0.0720, 0.0705, 0.0715)
    assert ex.position_qty == 0
    ex.step(0.0715, 0.0735, 0.0710, 0.0730)
    assert ex.position_qty == -1000


def test_down_candle_fills_the_sell_side_first():
    """Only OHLC is known, so intra-candle ordering is a convention: a down candle is
    assumed to reach its high before its low. If both sides could fill in one candle,
    the order decides which position the account ends up holding."""
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "sell", 0.0715, 1000)
    ex.place_limit_order("DOGEUSDT", "buy", 0.0705, 1000)
    ex.step(0.0710, 0.0720, 0.0700, 0.0702)   # down candle
    assert [f["side"] for f in ex.fill_log] == ["sell", "buy"]


def test_up_candle_fills_the_buy_side_first():
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "sell", 0.0715, 1000)
    ex.place_limit_order("DOGEUSDT", "buy", 0.0705, 1000)
    ex.step(0.0710, 0.0720, 0.0700, 0.0718)   # up candle
    assert [f["side"] for f in ex.fill_log] == ["buy", "sell"]


def test_highest_buy_is_hit_first_on_the_way_down():
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "buy", 0.0700, 1000)
    ex.place_limit_order("DOGEUSDT", "buy", 0.0690, 1000)
    ex.step(0.0710, 0.0712, 0.0685, 0.0688)
    assert [f["price"] for f in ex.fill_log] == [0.0700, 0.0690]


# --- exchange rejections the live account enforces -------------------------

def test_post_only_order_that_would_cross_is_refused():
    ex = make_ex()
    with pytest.raises(PostOnlyWouldCross):
        ex.place_limit_order("DOGEUSDT", "buy", 0.0715, 1000)   # above market
    assert ex.rejected_crossing == 1
    assert ex.get_open_orders("DOGEUSDT") == []


def test_taker_fallback_crosses_and_pays_the_taker_rate():
    ex = make_ex()
    ex.place_limit_order(
        "DOGEUSDT", "buy", 0.0715, 1000, post_only=True, allow_taker_fallback=True,
    )
    assert ex.position_qty == 1000
    assert ex.fees_paid == pytest.approx(1000 * 0.0715 * 0.0004)


def test_reduce_only_beyond_the_position_is_rejected_as_2022():
    ex = make_ex()
    ex.place_limit_order("DOGEUSDT", "buy", 0.0700, 1000)
    ex.step(0.0710, 0.0712, 0.0695, 0.0705)
    with pytest.raises(ValueError, match="-2022"):
        ex.place_limit_order("DOGEUSDT", "sell", 0.0720, 1500, params={"reduceOnly": True})
    assert ex.rejected_reduce_only == 1


def test_reduce_only_while_flat_is_rejected():
    ex = make_ex()
    with pytest.raises(ValueError, match="-2022"):
        ex.place_limit_order("DOGEUSDT", "sell", 0.0720, 100, params={"reduceOnly": True})


def test_sub_minimum_notional_is_rejected_as_4164():
    ex = make_ex()
    with pytest.raises(ValueError, match="-4164"):
        ex.place_limit_order("DOGEUSDT", "buy", 0.0700, 10)   # 0.70 USDT
    assert ex.rejected_min_notional == 1


# --- virtual clock ---------------------------------------------------------

def test_clock_is_installed_and_restored():
    real = grid_module.time
    clock = VirtualClock()
    with _clock_installed(clock):
        assert grid_module.time is clock
    assert grid_module.time is real


def test_clock_restored_even_if_the_body_raises():
    real = grid_module.time
    with pytest.raises(RuntimeError):
        with _clock_installed(VirtualClock()):
            raise RuntimeError("boom")
    assert grid_module.time is real


def test_sleep_advances_simulated_time_instead_of_blocking():
    """Order pacing must not really sleep -- 2000 candles x 0.6s would be 20 minutes."""
    clock = VirtualClock()
    clock.sleep(30.0)
    assert clock.time() == 30.0


# --- end to end ------------------------------------------------------------

def _oscillating(n=400, mid=0.0720, amp=0.004):
    import numpy as np
    t = np.arange(n)
    close = mid + amp * np.sin(t / 6.0)
    op = np.roll(close, 1)
    op[0] = close[0]
    high = np.maximum(op, close) * 1.001
    low = np.minimum(op, close) * 0.999
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close})


def test_backtest_runs_end_to_end_and_conserves_accounting():
    r = run_backtest(_oscillating(), starting_balance=5000, grid_count=10)
    assert r.candles > 0
    # net must equal gross minus fees, and equity must equal start plus net.
    assert r.net_pnl == pytest.approx(r.gross_realized - r.fees_paid)
    assert r.ending_equity == pytest.approx(r.starting_balance + r.net_pnl, abs=1e-6)


def test_backtest_refuses_data_too_short_to_warm_up():
    with pytest.raises(ValueError, match="need more than"):
        run_backtest(_oscillating(n=30), warmup=50)


def test_no_grid_order_ever_fills_as_taker():
    """Grid levels must all rest. A taker fill from a *limit* placement is the leak
    AUDIT #17 closed.

    Stop-loss closes and the final flatten are market orders and legitimately pay the
    taker rate, so those are the only taker fills a clean run may contain.
    """
    r = run_backtest(_oscillating(), starting_balance=5000, grid_count=10)
    market_closes = r.stop_loss_hits + 1          # +1 for the end-of-run flatten
    assert r.taker_fills <= market_closes, (
        f"{r.taker_fills} taker fills but only {market_closes} market closes -- "
        "a grid limit order crossed the spread"
    )
