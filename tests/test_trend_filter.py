import time

import numpy as np
import pandas as pd
import pytest

from trend_filter import TrendFilter, MarketRegime, ema, adx, atr


def make_trend_data(n=200, trend="up"):
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    if trend == "up":
        close = 80000 + np.cumsum(np.random.randn(n) * 50 + 10)
    elif trend == "down":
        close = 80000 + np.cumsum(np.random.randn(n) * 50 - 10)
    else:
        close = 80000 + np.cumsum(np.random.randn(n) * 50)

    high = close + np.abs(np.random.randn(n) * 30)
    low = close - np.abs(np.random.randn(n) * 30)
    opn = close + np.random.randn(n) * 20
    volume = np.random.randint(100, 1000, n).astype(float)

    return pd.DataFrame({
        "timestamp": dates,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_ema_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = ema(s, 3)
    assert len(result) == 5
    assert result.iloc[0] == 1.0
    assert result.iloc[-1] > result.iloc[-2]


def test_adx_trending():
    df = make_trend_data(200, "up")
    result = adx(df["high"], df["low"], df["close"], 14)
    assert not np.isnan(result.iloc[-1])


def test_adx_ranging():
    df = make_trend_data(200, "range")
    result = adx(df["high"], df["low"], df["close"], 14)
    assert isinstance(result.iloc[-1], float)


def test_atr():
    df = make_trend_data(200)
    result = atr(df["high"], df["low"], df["close"], 14)
    assert len(result) == 200
    assert result.iloc[-1] > 0


def test_trend_filter_uptrend():
    tf = TrendFilter(ema_fast=20, ema_slow=50, adx_period=14)
    df = make_trend_data(200, "up")
    regime = tf.update(df)
    assert regime in (MarketRegime.UPTREND, MarketRegime.RANGING, MarketRegime.UNCERTAIN)


def test_trend_filter_ranging():
    tf = TrendFilter(ema_fast=20, ema_slow=50, adx_period=14, trend_threshold=30, range_threshold=15)
    df = make_trend_data(200, "range")
    regime = tf.update(df)
    assert isinstance(regime, MarketRegime)


def test_trend_filter_time_to_check():
    tf = TrendFilter(check_interval=0)
    assert tf.time_to_check() is True


def test_trend_filter_insufficient_data():
    tf = TrendFilter(ema_fast=20, ema_slow=50, adx_period=14)
    df = make_trend_data(10)
    regime = tf.update(df)
    assert regime == MarketRegime.UNCERTAIN


def test_merge_minority_trend_does_not_block():
    tf = TrendFilter()
    tf._timeframes = {
        "1h": MarketRegime.DOWNTREND,
        "30m": MarketRegime.UNCERTAIN,
        "1d": MarketRegime.UNCERTAIN,
    }
    assert tf._merge_timeframes() == MarketRegime.UNCERTAIN


def test_merge_majority_trend_blocks():
    tf = TrendFilter()
    tf._timeframes = {
        "1h": MarketRegime.DOWNTREND,
        "30m": MarketRegime.DOWNTREND,
        "1d": MarketRegime.UNCERTAIN,
    }
    assert tf._merge_timeframes() == MarketRegime.DOWNTREND


def test_merge_conflicting_trends_is_uncertain():
    tf = TrendFilter()
    tf._timeframes = {
        "1h": MarketRegime.DOWNTREND,
        "30m": MarketRegime.UPTREND,
        "1d": MarketRegime.DOWNTREND,
    }
    assert tf._merge_timeframes() == MarketRegime.UNCERTAIN


def test_merge_majority_ranging():
    tf = TrendFilter()
    tf._timeframes = {
        "1h": MarketRegime.RANGING,
        "30m": MarketRegime.RANGING,
        "1d": MarketRegime.UNCERTAIN,
    }
    assert tf._merge_timeframes() == MarketRegime.RANGING


def test_merge_single_timeframe():
    tf = TrendFilter()
    tf._timeframes = {"1h": MarketRegime.DOWNTREND}
    assert tf._merge_timeframes() == MarketRegime.DOWNTREND


def test_is_ranging_includes_uncertain():
    tf = TrendFilter()
    tf.regime = MarketRegime.UNCERTAIN
    assert tf.is_ranging() is True
    tf.regime = MarketRegime.DOWNTREND
    assert tf.is_ranging() is False


def make_flat_then_rally():
    """Flat base then a strong rally, mimicking the 08-08 pause (flat after a move)."""
    np.random.seed(7)
    n = 200
    base = 0.0700 + np.random.randn(n) * 0.0001
    # flat for 120 candles, then steady uptrend for the last 80
    close = base.copy()
    rally = np.linspace(0.0, 0.004, 80)
    close[120:] = close[120:] + rally
    high = close + np.abs(np.random.randn(n) * 0.00008)
    low = close - np.abs(np.random.randn(n) * 0.00008)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="1h"),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100, 1000, n).astype(float),
    })


def test_adx_wilder_uses_rma_not_rolling_mean():
    """Wilder RMA smoothing must be used, not simple rolling means (inflated ADX)."""
    df = make_flat_then_rally()
    result = adx(df["high"], df["low"], df["close"], 14)
    assert not np.isnan(result.iloc[-1])
    # A strong pure trend should not push ADX near the ~77 values the SMA variant produced
    assert 20.0 <= result.iloc[-1] <= 60.0


def test_flat_override_forces_ranging_when_recent_range_tight():
    tf = TrendFilter(confirmation_seconds=0, flat_range_window=6, flat_range_pct=0.01)
    n = 60
    close = pd.Series([0.0710] * n, dtype=float)
    df = pd.DataFrame({
        "high": close + 0.0003,
        "low": close - 0.0003,
        "close": close,
    })
    tf._last_ohlcv = df
    assert tf._is_flat_range() is True
    assert tf._apply_flat_override(MarketRegime.UPTREND) == MarketRegime.RANGING


def test_flat_override_does_not_force_when_range_wide():
    tf = TrendFilter(confirmation_seconds=0, flat_range_window=6, flat_range_pct=0.01)
    n = 60
    close = pd.Series([0.0710] * n, dtype=float)
    df = pd.DataFrame({
        "high": close + 0.002,  # ~2.8% range, above 1% threshold
        "low": close - 0.002,
        "close": close,
    })
    tf._last_ohlcv = df
    assert tf._is_flat_range() is False
    assert tf._apply_flat_override(MarketRegime.UPTREND) == MarketRegime.UPTREND


def test_flat_override_resumes_grid_after_pause_scenario():
    """Flat market with stale high ADX should flip back to RANGING so grid resumes."""
    tf = TrendFilter(
        confirmation_seconds=0,
        flat_range_window=6,
        flat_range_pct=0.01,
    )
    # Last 6 candles flat, earlier rally still inflates ADX
    closes = list(np.linspace(0.0690, 0.0712, 74))
    closes += [0.0710, 0.07098, 0.07102, 0.0710, 0.07101, 0.0710]
    close = pd.Series(closes, dtype=float)
    df = pd.DataFrame({
        "high": close + 0.0003,
        "low": close - 0.0003,
        "close": close,
    })
    tf.update(df, "1h")
    assert tf.regime == MarketRegime.RANGING


def _frame(high, low, close, n=60):
    return pd.DataFrame({
        "high": [high] * n,
        "low": [low] * n,
        "close": [close] * n,
    })


FLAT_FRAME = _frame(0.0713, 0.0707, 0.0710)
WIDE_FRAME = _frame(0.0730, 0.0690, 0.0710)


def test_flat_override_uses_fastest_timeframe():
    """Flat check must use the fastest TF, not just the primary (1h-only) one."""
    # 1h wide (stale trend) but 30m quiet right now -> override fires, grid resumes
    tf = TrendFilter(confirmation_seconds=0, flat_range_window=6, flat_range_pct=0.01)
    tf.update(WIDE_FRAME, "1h")
    tf.add_timeframe(FLAT_FRAME, "30m")
    assert tf._is_flat_range() is True
    assert tf._apply_flat_override(MarketRegime.UPTREND) == MarketRegime.RANGING

    # 1h flat but 30m wide (fresh move just started) -> do NOT override the trend
    tf2 = TrendFilter(confirmation_seconds=0, flat_range_window=6, flat_range_pct=0.01)
    tf2.update(FLAT_FRAME, "1h")
    tf2.add_timeframe(WIDE_FRAME, "30m")
    assert tf2._is_flat_range() is False
    assert tf2._apply_flat_override(MarketRegime.UPTREND) == MarketRegime.UPTREND


def test_flat_override_uses_1d_when_only_primary_available():
    """With only the slow TF stored, the override still works off it (fallback)."""
    tf = TrendFilter(confirmation_seconds=0, flat_range_window=6, flat_range_pct=0.01)
    tf.update(FLAT_FRAME, "1h")
    assert tf._is_flat_range() is True


def test_flat_override_skips_too_short_fastest_frame():
    """A fastest TF with fewer candles than the window must fall back to the next."""
    tf = TrendFilter(confirmation_seconds=0, flat_range_window=6, flat_range_pct=0.01)
    short = _frame(0.0713, 0.0707, 0.0710, n=3)
    tf.update(WIDE_FRAME, "1h")
    tf.add_timeframe(short, "30m")
    assert tf._is_flat_range() is False  # 1h is wide, so NOT flat


def test_adx_value_not_overwritten_by_add_timeframe():
    """Primary timeframe ADX must survive add_timeframe calls (Telegram reporting bug)."""
    tf = TrendFilter(confirmation_seconds=0)
    df1 = make_trend_data(200, "up")
    df2 = make_trend_data(200, "range")
    tf.update(df1, "1h")
    primary_adx = tf.adx_value
    tf.add_timeframe(df2, "1d")
    assert tf.adx_value == primary_adx


def _pending_harness(tf, states, monkeypatch):
    fake_ohlcv = pd.DataFrame({"high": [1.0], "low": [1.0], "close": [1.0]})
    iterator = iter(states)
    monkeypatch.setattr(tf, "_evaluate_timeframe", lambda ohlcv, timeframe="4h": next(iterator))
    monkeypatch.setattr(tf, "_apply_flat_override", lambda r: r)
    return fake_ohlcv


def test_pending_confirmation_cleared_when_regime_reverts(monkeypatch):
    """Stale pending confirmation must clear if the regime returns to the current one."""
    tf = TrendFilter(confirmation_seconds=600)
    tf.regime = MarketRegime.UPTREND
    fake_ohlcv = _pending_harness(
        tf,
        [(MarketRegime.RANGING, 5.0), (MarketRegime.UPTREND, 40.0)],
        monkeypatch,
    )
    tf.update(fake_ohlcv, "1h")
    assert tf._pending_regime == MarketRegime.RANGING
    assert tf.regime == MarketRegime.UPTREND

    tf.update(fake_ohlcv, "1h")
    assert tf._pending_regime is None
    assert tf._pending_since == 0.0


def test_pending_confirmation_cleared_in_add_timeframe(monkeypatch):
    tf = TrendFilter(confirmation_seconds=600)
    tf.regime = MarketRegime.UPTREND
    fake_ohlcv = _pending_harness(
        tf,
        [(MarketRegime.RANGING, 5.0), (MarketRegime.UPTREND, 40.0)],
        monkeypatch,
    )
    tf.add_timeframe(fake_ohlcv, "1d")
    assert tf._pending_regime == MarketRegime.RANGING

    tf.add_timeframe(fake_ohlcv, "1d")
    assert tf._pending_regime is None
    assert tf._pending_since == 0.0


def test_reverted_signal_needs_fresh_confirmation_window(monkeypatch):
    """A reverted signal must not flip instantly using the stale elapsed time."""
    tf = TrendFilter(confirmation_seconds=600)
    tf.regime = MarketRegime.UPTREND
    fake_ohlcv = _pending_harness(
        tf,
        [
            (MarketRegime.UPTREND, 40.0),
            (MarketRegime.RANGING, 5.0),
            (MarketRegime.UPTREND, 40.0),
        ],
        monkeypatch,
    )
    # Simulate a stale pending that would otherwise look long-"confirmed"
    tf._pending_regime = MarketRegime.RANGING
    tf._pending_since = 0.0

    tf.update(fake_ohlcv, "1h")
    assert tf._pending_regime is None
    assert tf.regime == MarketRegime.UPTREND

    tf.update(fake_ohlcv, "1h")
    assert tf._pending_regime == MarketRegime.RANGING
    assert tf.regime == MarketRegime.UPTREND
    assert time.time() - tf._pending_since < 5


def test_legit_signal_still_confirms_after_window(monkeypatch):
    """A persistent signal must still confirm after the confirmation window."""
    tf = TrendFilter(confirmation_seconds=600)
    tf.regime = MarketRegime.UPTREND
    fake_ohlcv = _pending_harness(
        tf,
        [(MarketRegime.RANGING, 5.0), (MarketRegime.RANGING, 5.0)],
        monkeypatch,
    )
    tf.update(fake_ohlcv, "1h")
    assert tf._pending_regime == MarketRegime.RANGING
    tf._pending_since = time.time() - 601

    tf.update(fake_ohlcv, "1h")
    assert tf.regime == MarketRegime.RANGING
    assert tf._pending_regime is None


# --- #40: the flat override must explain itself ----------------------------

def _flat_filter():
    from trend_filter import TrendFilter

    return TrendFilter(trend_threshold=30.0, range_threshold=15.0,
                       flat_range_window=6, flat_range_pct=0.01,
                       confirmation_seconds=0)


def _flat_candles(n=60, price=0.0700):
    """Candles whose last 6 span well under 1% -- a genuinely flat market."""
    import pandas as pd

    return pd.DataFrame({
        "open": [price] * n,
        "high": [price * 1.001] * n,
        "low": [price * 0.999] * n,
        "close": [price] * n,
    })


def test_the_flat_override_says_so_in_the_explanation():
    """AUDIT #40. On 2026-08-13 the log read

        1h=downtrend(31.1) 30m=downtrend(45.2) 1d=uncertain(27.6)
        | needs 2 of 3 to agree -> ranging

    Two timeframes agreed on downtrend, the merge returned downtrend, and the flat
    override silently turned it into ranging. The override is correct -- ADX can read
    as trending while price goes nowhere -- but a log line that contradicts itself with
    no reason given is exactly what #33 existed to stop.
    """
    from trend_filter import MarketRegime

    tf = _flat_filter()
    tf._last_ohlcv = _flat_candles()
    tf._ohlcv["1h"] = tf._last_ohlcv

    out = tf._apply_flat_override(MarketRegime.DOWNTREND)

    assert out is MarketRegime.RANGING
    assert tf._flat_override_active is True
    tf._timeframes["1h"] = MarketRegime.DOWNTREND
    tf._adx_by_timeframe["1h"] = 31.1
    assert "FLAT OVERRIDE" in tf.explain()


def test_no_override_note_when_the_market_is_actually_moving():
    from trend_filter import MarketRegime
    import pandas as pd

    tf = _flat_filter()
    moving = pd.DataFrame({
        "open": [0.070, 0.071, 0.072, 0.073, 0.074, 0.075],
        "high": [0.0705, 0.0715, 0.0725, 0.0735, 0.0745, 0.0755],
        "low": [0.0695, 0.0705, 0.0715, 0.0725, 0.0735, 0.0745],
        "close": [0.0701, 0.0711, 0.0721, 0.0731, 0.0741, 0.0751],
    })
    tf._last_ohlcv = moving
    tf._ohlcv["1h"] = moving

    out = tf._apply_flat_override(MarketRegime.DOWNTREND)

    assert out is MarketRegime.DOWNTREND
    assert tf._flat_override_active is False
    tf._timeframes["1h"] = MarketRegime.DOWNTREND
    tf._adx_by_timeframe["1h"] = 31.1
    assert "FLAT OVERRIDE" not in tf.explain()


# --- AUDIT #155/#156: regime + confirmation clocks survive a restart, and a lone
# persistent timeframe is tracked and can be alerted on ---------------------------

def test_to_dict_round_trips_the_regime_and_confirmation_clocks():
    tf = TrendFilter()
    tf.regime = MarketRegime.DOWNTREND
    tf.last_check = time.time() - 30
    tf._pending_regime = MarketRegime.UPTREND
    tf._pending_since = time.time() - 10
    tf._timeframes = {"30m": MarketRegime.DOWNTREND, "1h": MarketRegime.RANGING}
    tf._timeframe_since = {"30m": time.time() - 400, "1h": time.time() - 50}

    restored = TrendFilter()
    restored.load_from_dict(tf.to_dict())

    assert restored.regime == MarketRegime.DOWNTREND
    assert restored._pending_regime == MarketRegime.UPTREND
    assert restored._pending_since == pytest.approx(tf._pending_since)
    assert restored._timeframes == tf._timeframes
    assert restored._timeframe_since["30m"] == pytest.approx(tf._timeframe_since["30m"])


def test_load_from_dict_is_a_noop_on_empty_or_missing_data():
    tf = TrendFilter()
    tf.load_from_dict({})
    assert tf.regime == MarketRegime.UNCERTAIN
    tf.load_from_dict(None)
    assert tf.regime == MarketRegime.UNCERTAIN


def test_load_from_dict_rejects_a_snapshot_older_than_stale_after_seconds():
    """A restart down for longer than STALE_AFTER_SECONDS must not trust a saved
    regime -- the gap could span a real reversal the bot never observed."""
    tf = TrendFilter()
    saved = tf.to_dict()
    saved["regime"] = MarketRegime.DOWNTREND.value
    saved["last_check"] = time.time() - (TrendFilter.STALE_AFTER_SECONDS + 60)

    restored = TrendFilter()
    restored.load_from_dict(saved)

    assert restored.regime == MarketRegime.UNCERTAIN, (
        "a stale saved regime was trusted instead of discarded"
    )


def test_load_from_dict_accepts_a_snapshot_just_inside_the_staleness_window():
    tf = TrendFilter()
    saved = tf.to_dict()
    saved["regime"] = MarketRegime.DOWNTREND.value
    saved["last_check"] = time.time() - (TrendFilter.STALE_AFTER_SECONDS - 60)

    restored = TrendFilter()
    restored.load_from_dict(saved)

    assert restored.regime == MarketRegime.DOWNTREND


def test_load_from_dict_rejects_a_future_timestamp():
    """A hand-edited or corrupt state file reading a future last_check must not be
    trusted -- same plausibility guard as router.py's handoff_started."""
    tf = TrendFilter()
    saved = tf.to_dict()
    saved["regime"] = MarketRegime.DOWNTREND.value
    saved["last_check"] = time.time() + 3600

    restored = TrendFilter()
    restored.load_from_dict(saved)

    assert restored.regime == MarketRegime.UNCERTAIN


def test_load_from_dict_drops_a_timeframe_since_with_no_matching_regime():
    tf = TrendFilter()
    saved = tf.to_dict()
    saved["last_check"] = time.time()
    saved["timeframes"] = {"30m": MarketRegime.DOWNTREND.value}
    saved["timeframe_since"] = {"30m": time.time() - 100, "1h": time.time() - 100}

    restored = TrendFilter()
    restored.load_from_dict(saved)

    assert set(restored._timeframe_since) == {"30m"}


def test_timeframe_since_only_stamps_on_a_real_regime_change(monkeypatch):
    """Repeated identical readings must accumulate duration, not reset it -- the whole
    point of lone_trend_duration() is measuring how long a disagreement has held."""
    tf = TrendFilter()
    monkeypatch.setattr(tf, "_evaluate_timeframe", lambda ohlcv: (MarketRegime.DOWNTREND, 28.0))
    df = make_trend_data(200, "down")

    tf.update(df, "30m")
    first_stamp = tf._timeframe_since["30m"]
    tf.update(df, "30m")
    second_stamp = tf._timeframe_since["30m"]

    assert first_stamp == second_stamp, "an unchanged regime reading reset the duration clock"


def test_timeframe_since_restamps_when_the_regime_actually_changes(monkeypatch):
    tf = TrendFilter()
    monkeypatch.setattr(tf, "_evaluate_timeframe", lambda ohlcv: (MarketRegime.DOWNTREND, 28.0))
    df = make_trend_data(200, "down")
    tf.update(df, "30m")
    first_stamp = tf._timeframe_since["30m"]

    monkeypatch.setattr(tf, "_evaluate_timeframe", lambda ohlcv: (MarketRegime.UPTREND, 28.0))
    tf.update(df, "30m")

    assert tf._timeframe_since["30m"] != first_stamp


def test_lone_trend_duration_is_none_when_nothing_is_trending():
    tf = TrendFilter()
    tf._timeframes = {"30m": MarketRegime.RANGING, "1h": MarketRegime.UNCERTAIN}
    assert tf.lone_trend_duration() is None


def test_lone_trend_duration_is_none_when_the_trend_agrees_with_the_merged_verdict():
    """Two timeframes agreeing on a direction IS the merged verdict (2-of-3) -- that
    is a normal, acted-on trend, not a lone one."""
    tf = TrendFilter()
    tf.regime = MarketRegime.DOWNTREND
    tf._timeframes = {"30m": MarketRegime.DOWNTREND, "1h": MarketRegime.DOWNTREND}
    tf._timeframe_since = {"30m": time.time() - 5000, "1h": time.time() - 5000}
    assert tf.lone_trend_duration() is None


def test_lone_trend_duration_reports_a_timeframe_outvoting_the_merged_verdict():
    """The exact live scenario: 30m has been trending for hours, 1h/1d never agree,
    so the merged verdict stays uncertain and the grid never sees anything different
    on hour 3 than it saw on minute 3."""
    tf = TrendFilter()
    tf.regime = MarketRegime.UNCERTAIN
    tf._timeframes = {
        "30m": MarketRegime.DOWNTREND, "1h": MarketRegime.RANGING, "1d": MarketRegime.UPTREND,
    }
    tf._timeframe_since = {"30m": time.time() - 10800, "1h": time.time() - 100, "1d": time.time() - 50}

    result = tf.lone_trend_duration()

    assert result is not None
    tf_name, regime, duration = result
    assert tf_name == "30m"
    assert regime == MarketRegime.DOWNTREND
    assert duration == pytest.approx(10800, abs=5)


def test_lone_trend_duration_picks_the_longest_running_when_more_than_one_dissents():
    tf = TrendFilter()
    tf.regime = MarketRegime.UNCERTAIN
    tf._timeframes = {"30m": MarketRegime.DOWNTREND, "1d": MarketRegime.UPTREND}
    tf._timeframe_since = {"30m": time.time() - 200, "1d": time.time() - 9000}

    tf_name, _, _ = tf.lone_trend_duration()

    assert tf_name == "1d"
