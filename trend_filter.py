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
        trend_min_votes: int = 2,
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
        self.trend_min_votes = max(1, int(trend_min_votes))

        self.regime = MarketRegime.UNCERTAIN
        self.last_check = 0.0
        self.adx_value = 0.0
        self._timeframes: dict[str, MarketRegime] = {}
        self._adx_by_timeframe: dict[str, float] = {}
        self._flat_override_active = False
        self._flat_override_pct = 0.0
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
        """ADX can read as trending while price goes nowhere. Range beats ADX.

        The override itself is right and stays. What changed is that it used to be
        SILENT unless the previous regime was already trending -- so on 2026-08-13 the
        log read `1h=downtrend 30m=downtrend 1d=uncertain | needs 2 of 3 to agree ->
        ranging`, which is self-contradicting nonsense to anyone reading it. Two
        timeframes agreed on downtrend, the merge returned downtrend, and the override
        quietly turned it into ranging with no line explaining why (AUDIT #40).
        """
        self._flat_override_active = False
        if regime in (MarketRegime.UPTREND, MarketRegime.DOWNTREND) and self._is_flat_range():
            pct = self._recent_range_pct()
            self._flat_override_active = True
            self._flat_override_pct = pct or 0.0
            logger.info(
                "FLAT OVERRIDE | ADX says {} but the last {} candles span {:.2f}% "
                "<= {:.2f}% — treating as RANGING",
                regime.value, self.flat_range_window,
                (pct or 0.0) * 100, self.flat_range_pct * 100,
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
        informative = sum(
            1 for r in self._timeframes.values() if r != MarketRegime.UNCERTAIN
        )
        line = (
            f"{' '.join(parts)} | bands: range<={self.range_threshold:g} "
            f"trend>={self.trend_threshold:g} | {informative} of {len(self._timeframes)} "
            f"timeframe(s) voting, trend needs {self.trend_min_votes}"
        )
        trending = sum(
            1 for r in self._timeframes.values()
            if r in (MarketRegime.UPTREND, MarketRegime.DOWNTREND)
        )
        if trending and trending < self.trend_min_votes:
            line += (
                f" — {trending} trending timeframe(s), {self.trend_min_votes} needed, so "
                f"the follower stays ineligible"
            )
        if not informative:
            line += " — ALL ABSTAINED (every ADX inside the dead band)"
        if self._flat_override_active:
            line += (
                f" | FLAT OVERRIDE: last {self.flat_range_window} candles span "
                f"{self._flat_override_pct * 100:.2f}% <= {self.flat_range_pct * 100:.2f}%, "
                f"so a trending ADX reads as ranging"
            )
        return line

    def _merge_timeframes(self) -> MarketRegime:
        if not self._timeframes:
            return MarketRegime.UNCERTAIN

        regimes = list(self._timeframes.values())

        if len(regimes) == 1:
            return regimes[0]

        # UNCERTAIN is an ABSTENTION, not a vote against. Requiring an absolute 2 of 3
        # meant two abstaining timeframes could veto a verdict the third was sure of --
        # and on DOGEUSDT that was not an edge case but the permanent state. Measured
        # 2026-08-15 over 27 evaluations: 1h ADX 13.6-15.8, 30m 19.9-25.9, 1d 25.0-25.1.
        # The 30m and 1d readings sat inside the 15-30 dead band 100% of the time, so
        # only one timeframe could ever vote, two was unreachable, and the filter
        # returned "uncertain" 27 times out of 27 while price ground +1.1% into the grid.
        #
        # A guard that cannot reach a verdict is worse than no guard: it reads as an
        # undecided market rather than a broken vote (AUDIT #78).
        informative = [r for r in regimes if r != MarketRegime.UNCERTAIN]
        if not informative:
            return MarketRegime.UNCERTAIN

        trending = [r for r in informative if r in (MarketRegime.UPTREND, MarketRegime.DOWNTREND)]
        ranging_count = informative.count(MarketRegime.RANGING)

        # TRENDING needs `trend_min_votes` agreeing timeframes, default 2. That default
        # is deliberate -- a trend verdict PAUSES the grid, and one timeframe's opinion
        # is not much to stop trading on -- but it was a hardcoded 2, which made the
        # trade-off invisible and unadjustable.
        #
        # On DOGEUSDT the 1h ADX sits in the dead band for hours at a time, so the only
        # timeframe with an opinion is often a single one. Measured 2026-08-19 00:19-03:00:
        # every REGIME line read `1h=ranging(17.x) 30m=downtrend(25-26) 1d=uncertain(20.3)`.
        # 30m was trending and alone, len(trending) was 1, and the follower was never once
        # eligible across 2h41m -- the AUDIT #33 "dormant bot" shape, reached by the vote
        # rather than by the dead band.
        #
        # Setting this to 1 lets a lone trending timeframe hand over. That is a real
        # trade-off, not a free win: it also pauses the grid on one timeframe's say-so,
        # and the follower's measured edge over 70 trades was -0.14% per trade with a 95%
        # interval of [-0.50%, +0.21%] -- consistent with zero. More trades at that
        # expectancy is more churn, not more profit. Configurable so the choice is the
        # operator's and is visible in the log, rather than buried here (AUDIT #117).
        if len(trending) >= self.trend_min_votes:
            directions = set(trending)
            return directions.pop() if len(directions) == 1 else MarketRegime.UNCERTAIN

        # RANGING is the permissive verdict -- it changes nothing the bot does, it only
        # stops the log claiming the market is undecided when a timeframe was sure. One
        # voter is enough for it, and abstentions no longer veto it.
        if ranging_count and ranging_count >= len(trending):
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
