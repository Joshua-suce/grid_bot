"""How often would the router actually hand over to the trend follower?

STRATEGY_MODE=router is a switch with no cost and, on the evidence, no effect. The
trend follower only ever runs when TrendFilter returns UPTREND or DOWNTREND, and that
verdict needs TWO of the three timeframes to independently read ADX >= trend_threshold
AND agree on direction (trend_filter._merge_timeframes -- one voter is deliberately not
enough, because a trend verdict pauses the grid).

Five days of the live signals observer recorded FOUR regime changes, all between
ranging and uncertain, and not one trend. So flipping the switch would have changed
nothing. This module answers the question that matters before flipping it: over real
history, how often does the two-of-three condition actually hold, for how long at a
stretch, and how far does price travel while it does?

It uses the bot's OWN adx/ema and the real merge rule rather than a reimplementation,
so what it measures is what the bot would have done. Read-only: public market data,
no credentials, no orders.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import pandas as pd
import requests

from trend_filter import MarketRegime, TrendFilter, adx, ema

PUBLIC_HOSTS = ("https://fapi.binance.com", "https://demo-fapi.binance.com")
COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def fetch_klines(symbol: str, interval: str, limit: int = 1500,
                 hosts: tuple[str, ...] = PUBLIC_HOSTS) -> pd.DataFrame:
    """Public klines. No key, no signature -- this endpoint stayed up through the
    outage that took every signed endpoint down."""
    last_err: Exception | None = None
    for host in hosts:
        try:
            r = requests.get(f"{host}/fapi/v1/klines",
                             params={"symbol": symbol, "interval": interval,
                                     "limit": limit},
                             timeout=30)
            r.raise_for_status()
            rows = r.json()
            break
        except Exception as e:                                  # try the next host
            last_err = e
    else:
        raise RuntimeError(f"no public host answered: {last_err}")

    df = pd.DataFrame(
        [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
         for k in rows], columns=COLUMNS)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def verdicts(df: pd.DataFrame, ema_fast: int, ema_slow: int, adx_period: int,
             trend_threshold: float, range_threshold: float) -> pd.DataFrame:
    """Per-candle regime for ONE timeframe, vectorised over the whole history.

    Mirrors trend_filter._evaluate_timeframe exactly, including its warm-up rule: it
    returns UNCERTAIN until there are ema_slow + adx_period candles.
    """
    out = pd.DataFrame(index=df.index)
    out["time"] = df["time"]
    out["adx"] = adx(df["high"], df["low"], df["close"], adx_period)
    out["fast_above"] = ema(df["close"], ema_fast) > ema(df["close"], ema_slow)
    out["close"] = df["close"]

    warm = ema_slow + adx_period
    regimes = []
    for i, (a, up) in enumerate(zip(out["adx"], out["fast_above"])):
        if i < warm or pd.isna(a):
            regimes.append(MarketRegime.UNCERTAIN)
        elif a >= trend_threshold:
            regimes.append(MarketRegime.UPTREND if up else MarketRegime.DOWNTREND)
        elif a <= range_threshold:
            regimes.append(MarketRegime.RANGING)
        else:
            regimes.append(MarketRegime.UNCERTAIN)
    out["regime"] = regimes
    return out


def merge(regimes: list[MarketRegime]) -> MarketRegime:
    """The real vote. Delegates to TrendFilter so the rule cannot drift from the bot's."""
    tf = TrendFilter()
    tf._timeframes = {str(i): r for i, r in enumerate(regimes)}
    return tf._merge_timeframes()


@dataclass
class Episode:
    """One uninterrupted stretch of a trend verdict."""
    regime: MarketRegime
    start: pd.Timestamp
    end: pd.Timestamp
    entry: float
    exit: float
    bars: int = 1

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0

    @property
    def move_pct(self) -> float:
        """Signed in the direction the trade would have been taken."""
        raw = (self.exit - self.entry) / self.entry
        return raw * 100 if self.regime is MarketRegime.UPTREND else -raw * 100


@dataclass
class Study:
    counts: dict = field(default_factory=dict)
    episodes: list = field(default_factory=list)
    bars: int = 0
    span_days: float = 0.0

    def pct(self, regime: MarketRegime) -> float:
        return 100.0 * self.counts.get(regime, 0) / self.bars if self.bars else 0.0


def run(symbol: str, base_tf: str, timeframes: dict[str, str], *, ema_fast: int,
        ema_slow: int, adx_period: int, trend_threshold: float,
        range_threshold: float) -> Study:
    """Walk the base timeframe candle by candle, asking every timeframe what it thinks
    AS OF that moment, then merging. Each timeframe is looked up by timestamp rather
    than by row, so a 1d verdict is held across the 24 hourly bars it covers -- which
    is what the live filter sees between its own refreshes."""
    frames = {name: fetch_klines(symbol, tf) for name, tf in timeframes.items()}
    verd = {
        name: verdicts(df, ema_fast, ema_slow, adx_period, trend_threshold,
                       range_threshold).set_index("time")
        for name, df in frames.items()
    }

    base = verd[base_tf]
    study = Study()
    current: Episode | None = None

    for ts, row in base.iterrows():
        votes = []
        for name, v in verd.items():
            prior = v.index[v.index <= ts]
            votes.append(v.loc[prior[-1], "regime"] if len(prior) else MarketRegime.UNCERTAIN)

        verdict = merge(votes)
        study.counts[verdict] = study.counts.get(verdict, 0) + 1
        study.bars += 1

        trending = verdict in (MarketRegime.UPTREND, MarketRegime.DOWNTREND)
        if trending and current and current.regime is verdict:
            current.end, current.exit, current.bars = ts, row["close"], current.bars + 1
        else:
            if current:
                study.episodes.append(current)
                current = None
            if trending:
                current = Episode(verdict, ts, ts, row["close"], row["close"])
    if current:
        study.episodes.append(current)

    if study.bars:
        study.span_days = (base.index[-1] - base.index[0]).total_seconds() / 86400.0
    return study


def report(study: Study, label: str) -> None:
    print(f"\n{label}")
    print(f"  {study.bars} bars over {study.span_days:.0f} days")
    for regime in (MarketRegime.RANGING, MarketRegime.UNCERTAIN,
                   MarketRegime.UPTREND, MarketRegime.DOWNTREND):
        print(f"    {regime.value:<10} {study.pct(regime):>6.2f}%  "
              f"({study.counts.get(regime, 0)} bars)")

    eps = study.episodes
    if not eps:
        print("\n  TREND EPISODES: none. The trend follower would never have opened "
              "a position.")
        return

    wins = [e for e in eps if e.move_pct > 0]
    print(f"\n  TREND EPISODES: {len(eps)} in {study.span_days:.0f} days "
          f"(one per {study.span_days / len(eps):.1f} days)")
    print(f"    duration   median {sorted(e.hours for e in eps)[len(eps) // 2]:.1f}h  "
          f"max {max(e.hours for e in eps):.1f}h")
    print(f"    move       median {sorted(e.move_pct for e in eps)[len(eps) // 2]:+.2f}%  "
          f"best {max(e.move_pct for e in eps):+.2f}%  "
          f"worst {min(e.move_pct for e in eps):+.2f}%")
    print(f"    favourable {len(wins)}/{len(eps)} ({100 * len(wins) / len(eps):.0f}%)")
    print(f"    mean move  {sum(e.move_pct for e in eps) / len(eps):+.3f}%")


if __name__ == "__main__":
    from config import settings

    sym = settings.symbol.replace("/", "").split(":")[0]
    tfs = {settings.trend_timeframe_fast: settings.trend_timeframe_fast,
           settings.trend_timeframe: settings.trend_timeframe,
           "1d": "1d"}
    print(f"{sym} | timeframes {list(tfs)} | ADX{settings.adx_period} "
          f"trend>={settings.adx_trend_threshold} range<={settings.adx_range_threshold} "
          f"| EMA {settings.ema_fast}/{settings.ema_slow}")

    s = run(sym, settings.trend_timeframe, tfs,
            ema_fast=settings.ema_fast, ema_slow=settings.ema_slow,
            adx_period=settings.adx_period,
            trend_threshold=settings.adx_trend_threshold,
            range_threshold=settings.adx_range_threshold)
    report(s, f"AS CONFIGURED (trend>={settings.adx_trend_threshold})")

    for threshold in (20.0, 22.0, 30.0):
        if threshold <= settings.adx_range_threshold:
            continue
        time.sleep(0.3)
        s2 = run(sym, settings.trend_timeframe, tfs,
                 ema_fast=settings.ema_fast, ema_slow=settings.ema_slow,
                 adx_period=settings.adx_period, trend_threshold=threshold,
                 range_threshold=settings.adx_range_threshold)
        report(s2, f"IF trend>={threshold:g}")
    sys.exit(0)


def merge_vectorised(frame: pd.DataFrame) -> pd.Series:
    """merge() for a whole column-per-timeframe frame at once.

    A row-by-row merge builds a TrendFilter per bar, which costs minutes over a 5m walk.
    This reproduces _merge_timeframes with counts instead. It is verified against the
    authoritative implementation by verify_merge() rather than trusted.
    """
    up = (frame == MarketRegime.UPTREND).sum(axis=1)
    down = (frame == MarketRegime.DOWNTREND).sum(axis=1)
    rang = (frame == MarketRegime.RANGING).sum(axis=1)
    trending = up + down

    out = pd.Series(MarketRegime.UNCERTAIN, index=frame.index, dtype=object)
    out[(trending >= 2) & (down == 0)] = MarketRegime.UPTREND
    out[(trending >= 2) & (up == 0)] = MarketRegime.DOWNTREND
    # mixed directions with >=2 trending stay UNCERTAIN, as the set-of-one test requires
    ranging_wins = (trending < 2) & (rang > 0) & (rang >= trending)
    out[ranging_wins] = MarketRegime.RANGING
    return out


def verify_merge() -> int:
    """Every reachable combination of three timeframe verdicts, fast path against slow.
    Returns the number of combinations checked; raises if any disagree."""
    import itertools

    combos = list(itertools.product(list(MarketRegime), repeat=3))
    frame = pd.DataFrame(combos, columns=["a", "b", "c"])
    fast = merge_vectorised(frame)
    for i, combo in enumerate(combos):
        slow = merge(list(combo))
        if slow is not fast.iloc[i]:
            raise AssertionError(
                f"merge mismatch on {[c.value for c in combo]}: "
                f"authoritative={slow.value} vectorised={fast.iloc[i].value}")
    return len(combos)
