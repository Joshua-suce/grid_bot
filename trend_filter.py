from __future__ import annotations

import time
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class MarketRegime(Enum):
    RANGING = "ranging"
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    UNCERTAIN = "uncertain"


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder smoothing (RMA). The rolling-mean variant inflated ADX and slowed its
    # decay, keeping the trend filter stuck in a regime long after a move stalled.
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


_TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def timeframe_minutes(tf: str) -> int:
    return _TIMEFRAME_MINUTES.get(tf, 60)


class TrendFilter:
    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        adx_period: int = 14,
        trend_threshold: float = 25.0,
        range_threshold: float = 20.0,
        check_interval: int = 300,
        confirmation_seconds: int = 600,
        flat_range_window: int = 6,
        flat_range_pct: float = 0.01,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        self.check_interval = check_interval
        self.confirmation_seconds = confirmation_seconds
        self.flat_range_window = flat_range_window
        self.flat_range_pct = flat_range_pct

        self.regime = MarketRegime.UNCERTAIN
        self.last_check = 0.0
        self.adx_value = 0.0
        self._timeframes: dict[str, MarketRegime] = {}
        self._adx_by_timeframe: dict[str, float] = {}
        self._ohlcv: dict[str, pd.DataFrame] = {}
        self._pending_regime: MarketRegime | None = None
        self._pending_since: float = 0.0
        self._last_ohlcv: pd.DataFrame | None = None

    def time_to_check(self) -> bool:
        return (time.time() - self.last_check) >= self.check_interval

    def _evaluate_timeframe(self, ohlcv: pd.DataFrame) -> tuple[MarketRegime, float]:
        if len(ohlcv) < self.ema_slow + self.adx_period:
            return MarketRegime.UNCERTAIN, 0.0

        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]

        ema_fast_val = ema(close, self.ema_fast).iloc[-1]
        ema_slow_val = ema(close, self.ema_slow).iloc[-1]
        adx_val = adx(high, low, close, self.adx_period).iloc[-1]

        adx_float = float(adx_val) if not np.isnan(adx_val) else 0.0
        fast_above_slow = ema_fast_val > ema_slow_val

        if adx_float >= self.trend_threshold:
            regime = MarketRegime.UPTREND if fast_above_slow else MarketRegime.DOWNTREND
        elif adx_float <= self.range_threshold:
            regime = MarketRegime.RANGING
        else:
            regime = MarketRegime.UNCERTAIN

        return regime, adx_float

    def _recent_ohlcv(self) -> pd.DataFrame | None:
        """Pick the fastest timeframe with enough candles for the flat-range check.

        The flat-range override decides whether price is quiet *right now*. The
        fastest available timeframe is the most representative: a fresh trend
        always shows movement there, so it won't be wrongly overridden, and a
        flattening (stale ADX) is caught there first. Falls back to the primary
        dataframe when no per-timeframe data is stored.
        """
        if self._ohlcv:
            for tf in sorted(self._ohlcv, key=timeframe_minutes):
                df = self._ohlcv[tf]
                if df is not None and len(df) >= self.flat_range_window:
                    return df
        return self._last_ohlcv

    def _recent_range_pct(self) -> float | None:
        """Recent (high-low)/close over the flat_range_window, as a percent, or None."""
        ohlcv = self._recent_ohlcv()
        if ohlcv is None or len(ohlcv) < self.flat_range_window:
            return None
        window = ohlcv.tail(self.flat_range_window)
        high = float(window["high"].max())
        low = float(window["low"].min())
        close = float(ohlcv["close"].iloc[-1])
        if close <= 0:
            return None
        return (high - low) / close

    def _is_flat_range(self) -> bool:
        """True if the most recent candles span a tight range (flat market)."""
        pct = self._recent_range_pct()
        return pct is not None and pct <= self.flat_range_pct

    def _apply_flat_override(self, regime: MarketRegime) -> MarketRegime:
        if regime in (MarketRegime.UPTREND, MarketRegime.DOWNTREND) and self._is_flat_range():
            if self.regime in (MarketRegime.UPTREND, MarketRegime.DOWNTREND):
                pct = self._recent_range_pct()
                logger.info(
                    "FLAT OVERRIDE | ADX says {} but recent range {:.2f}% <= {:.2f}% — forcing RANGING",
                    regime.value, (pct or 0.0) * 100, self.flat_range_pct * 100,
                )
            return MarketRegime.RANGING
        return regime

    def update(self, ohlcv: pd.DataFrame, timeframe: str = "4h") -> MarketRegime:
        regime, adx_val = self._evaluate_timeframe(ohlcv)
        self._timeframes[timeframe] = regime
        self._adx_by_timeframe[timeframe] = adx_val
        self.adx_value = adx_val
        self._last_ohlcv = ohlcv
        self._ohlcv[timeframe] = ohlcv

        new_regime = self._apply_flat_override(self._merge_timeframes())

        if new_regime != self.regime:
            if self.confirmation_seconds <= 0:
                logger.info(
                    "REGIME CHANGE | {} -> {} | ADX={:.1f} | timeframes={}",
                    self.regime.value, new_regime.value, self.adx_value,
                    {k: v.value for k, v in self._timeframes.items()},
                )
                self.regime = new_regime
                self._pending_regime = None
                self._pending_since = 0.0
            else:
                now = time.time()
                if self._pending_regime == new_regime:
                    elapsed = now - self._pending_since
                    if elapsed >= self.confirmation_seconds:
                        logger.info(
                            "REGIME CHANGE (confirmed after {:.0f}s) | {} -> {} | ADX={:.1f}",
                            elapsed, self.regime.value, new_regime.value, self.adx_value,
                        )
                        self.regime = new_regime
                        self._pending_regime = None
                        self._pending_since = 0.0
                else:
                    logger.info(
                        "REGIME PENDING | {} -> {} | ADX={:.1f} | confirming in {}s",
                        self.regime.value, new_regime.value, self.adx_value,
                        self.confirmation_seconds,
                    )
                    self._pending_regime = new_regime
                    self._pending_since = now
        elif self._pending_regime is not None:
            logger.info(
                "REGIME PENDING DISCARDED | {} reverted to current {} — stale confirmation cleared",
                self._pending_regime.value, new_regime.value,
            )
            self._pending_regime = None
            self._pending_since = 0.0

        self.last_check = time.time()
        return self.regime

    def explain(self) -> str:
        """One line saying why the regime is what it is.

        'uncertain' is the default outcome here, not an edge case, and nothing in the
        logs said so. Two independent gates have to pass: each timeframe needs ADX
        outside the band (>= trend_threshold to trend, <= range_threshold to range --
        everything between is uncertain by definition), and then two of the three
        timeframes have to agree. A whole session can sit in 'uncertain' with the trend
        follower never once eligible, which looks like a dormant bot rather than an
        undecided market (AUDIT #33).
        """
        if not self._timeframes:
            return "no timeframes evaluated yet"
        parts = [
            f"{tf}={regime.value}(adx={self._adx_by_timeframe.get(tf, 0.0):.1f})"
            for tf, regime in self._timeframes.items()
        ]
        return (
            f"{' '.join(parts)} | bands: range<={self.range_threshold:g} "
            f"trend>={self.trend_threshold:g} | needs 2 of {len(self._timeframes)} to agree"
        )

    def _merge_timeframes(self) -> MarketRegime:
        if not self._timeframes:
            return MarketRegime.UNCERTAIN

        regimes = list(self._timeframes.values())
        total = len(regimes)

        if total == 1:
            return regimes[0]

        trending = [r for r in regimes if r in (MarketRegime.UPTREND, MarketRegime.DOWNTREND)]
        ranging_count = regimes.count(MarketRegime.RANGING)

        if len(trending) >= 2:
            directions = set(trending)
            if len(directions) == 1:
                return directions.pop()
            return MarketRegime.UNCERTAIN

        if ranging_count >= 2:
            return MarketRegime.RANGING

        return MarketRegime.UNCERTAIN

    def add_timeframe(self, ohlcv: pd.DataFrame, timeframe: str) -> None:
        regime, adx_val = self._evaluate_timeframe(ohlcv)
        self._timeframes[timeframe] = regime
        self._adx_by_timeframe[timeframe] = adx_val
        self._ohlcv[timeframe] = ohlcv
        if self._last_ohlcv is None:
            self._last_ohlcv = ohlcv

        new_regime = self._apply_flat_override(self._merge_timeframes())

        if new_regime != self.regime:
            if self.confirmation_seconds <= 0:
                logger.info(
                    "REGIME CHANGE (add_timeframe {}) | {} -> {} | ADX={:.1f}",
                    timeframe, self.regime.value, new_regime.value, adx_val,
                )
                self.regime = new_regime
                self._pending_regime = None
                self._pending_since = 0.0
            else:
                now = time.time()
                if self._pending_regime == new_regime:
                    elapsed = now - self._pending_since
                    if elapsed >= self.confirmation_seconds:
                        logger.info(
                            "REGIME CHANGE (add_timeframe {} confirmed after {:.0f}s) | {} -> {} | ADX={:.1f}",
                            timeframe, elapsed, self.regime.value, new_regime.value, adx_val,
                        )
                        self.regime = new_regime
                        self._pending_regime = None
                        self._pending_since = 0.0
                else:
                    self._pending_regime = new_regime
                    self._pending_since = now
        elif self._pending_regime is not None:
            logger.info(
                "REGIME PENDING DISCARDED (add_timeframe {}) | {} reverted to current {} — stale confirmation cleared",
                timeframe, self._pending_regime.value, new_regime.value,
            )
            self._pending_regime = None
            self._pending_since = 0.0

        logger.debug("TF {} | regime={} | ADX={:.1f}", timeframe, regime.value, adx_val)

    def is_ranging(self) -> bool:
        return self.regime in (MarketRegime.RANGING, MarketRegime.UNCERTAIN)

    def is_trending(self) -> bool:
        return self.regime in (MarketRegime.UPTREND, MarketRegime.DOWNTREND)
