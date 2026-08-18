"""What would the trend follower actually have made? AUDIT #105.

regime_study.py answers how often the router hands over. This answers the question that
decides whether handing over is worth doing: taking every one of those handoffs, with
the follower's real stop, real ratchet and real target, what comes back?

It is not a reimplementation. The stop distance, the trailing ratchet and the take-profit
are lifted from trend_follower.py:

    stop distance   max(atr_pct * atr_multiplier, stop_loss_pct) * price, and the
                    ratchet caps it at trailing_sl_trigger_pct of the extreme
    long trail      candidate = peak - distance(peak), floored by peak*(1-trigger),
                    then monotonic: it may rise, never fall
    short trail     mirrored on the trough, monotonic downward
    1R              |entry - the stop the trade OPENED with|
    target          entry +/- 1R * take_profit_r, frozen at entry
    ties            the stop wins (trend_follower.check_fills)

Deliberately optimistic where it cannot be exact, so the answer is an upper bound:
fills are at the candle close with no slippage, and the stop fills exactly at its price
rather than gapping through it. A real stop-market exit is a taker order into whatever
liquidity is there. If the optimistic version does not pay, the real one does not either.

Read-only: public market data, no credentials, no orders.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import requests

from regime_study import (COLUMNS, PUBLIC_HOSTS, fetch_klines, merge,
                          merge_vectorised, verdicts)
from trend_filter import MarketRegime, atr


def fetch_klines_since(symbol: str, interval: str, start_ms: int) -> pd.DataFrame:
    """Paginated. One request caps at 1500 candles, and a 5m walk over two months
    needs about twelve of them."""
    rows: list = []
    cursor = start_ms
    while True:
        for host in PUBLIC_HOSTS:
            try:
                r = requests.get(f"{host}/fapi/v1/klines",
                                 params={"symbol": symbol, "interval": interval,
                                         "startTime": cursor, "limit": 1500},
                                 timeout=30)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception:
                batch = None
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1500:
            break
        cursor = int(batch[-1][0]) + 1

    df = pd.DataFrame(
        [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
         for k in rows], columns=COLUMNS).drop_duplicates(subset="ts")
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.reset_index(drop=True)


@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    entry: float
    stop0: float
    target: float | None
    exit_time: pd.Timestamp | None = None
    exit: float | None = None
    reason: str = "open"

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop0)

    @property
    def pnl_pct(self) -> float:
        if self.exit is None:
            return 0.0
        raw = (self.exit - self.entry) / self.entry
        return 100 * (raw if self.side == "long" else -raw)

    @property
    def r_multiple(self) -> float:
        """The only unit that makes a reward:risk claim checkable."""
        if self.exit is None or self.risk <= 0:
            return 0.0
        move = (self.exit - self.entry) if self.side == "long" else (self.entry - self.exit)
        return move / self.risk

    @property
    def hours(self) -> float:
        if self.exit_time is None:
            return 0.0
        return (self.exit_time - self.entry_time).total_seconds() / 3600


def stop_distance(price: float, atr_pct: float, mult: float, floor_pct: float) -> float:
    return max(price * atr_pct * mult, price * floor_pct)


def simulate(symbol: str, *, take_profit_r: float, atr_mult: float, floor_pct: float,
             trigger_pct: float, ema_fast: int, ema_slow: int, adx_period: int,
             trend_threshold: float, range_threshold: float,
             round_trip_fee_pct: float, frames=None, walk=None,
             trail_mult: float | None = None,
             verdict_series=None) -> list[Trade]:
    # trend_follower uses ONE distance for both the opening stop and the trail. That
    # couples them: the target sits at N x the opening stop while the trail rides one
    # stop-width behind the peak, so price must run N widths without ever giving back
    # one. Separating them here measures what decoupling them would buy.
    trail_mult = atr_mult if trail_mult is None else trail_mult
    frames = frames or {tf: fetch_klines(symbol, tf) for tf in ("30m", "1h", "1d")}
    verd = {tf: verdicts(df, ema_fast, ema_slow, adx_period, trend_threshold,
                         range_threshold).set_index("time")
            for tf, df in frames.items()}

    # The regime is decided on 30m/1h/1d, but the position is walked on a much finer
    # series. At hourly resolution a single bar's high and low are both "inside" the
    # same step, so ratcheting the trail to the high and then testing the low against
    # it invents a stop-out that could not have happened in that order -- and it
    # pre-empted every take-profit, which is why every TREND_TAKE_PROFIT_R returned
    # byte-identical results with `target hit 0/66`.
    fine = (walk if walk is not None else frames["1h"]).copy()
    hourly = frames["1h"].copy()
    hourly["atr"] = atr(hourly["high"], hourly["low"], hourly["close"], adx_period)
    hourly = hourly.set_index("time")
    atr_series = hourly["atr"]
    fine = fine.set_index("time")

    trades: list[Trade] = []
    open_trade: Trade | None = None
    peak = trough = 0.0
    trail: float | None = None

    def atr_pct_at(ts, price):
        prior = atr_series.index[atr_series.index <= ts]
        if not len(prior) or price <= 0:
            return 0.0
        a = atr_series.loc[prior[-1]]
        return 0.0 if pd.isna(a) else a / price

    # Each timeframe's verdict is held forward onto the walk index -- a 1d call stands
    # for the 288 five-minute bars it covers, which is what the live filter sees between
    # refreshes. Computed once; it does not depend on the stop or the target.
    if verdict_series is None:
        aligned = pd.DataFrame({
            tf: v["regime"].reindex(fine.index, method="ffill")
            for tf, v in verd.items()
        }).fillna(MarketRegime.UNCERTAIN)
        verdict_series = merge_vectorised(aligned)

    for ts, bar in fine.iterrows():
        price, high, low = bar["close"], bar["high"], bar["low"]
        atr_pct = atr_pct_at(ts, price)

        if open_trade is not None:
            # Test against the trail as it stood ENTERING this bar. Only after the bar
            # has been survived does its extreme feed the ratchet, which is the order
            # the live loop sees: it polls, compares, and only then updates the stop.
            if open_trade.side == "long":
                hit_stop = low <= trail
                hit_tp = open_trade.target is not None and high >= open_trade.target
            else:
                hit_stop = high >= trail
                hit_tp = open_trade.target is not None and low <= open_trade.target

            if hit_stop or hit_tp:
                open_trade.exit_time = ts
                open_trade.exit = trail if hit_stop else open_trade.target
                open_trade.reason = "trailing_stop" if hit_stop else "take_profit"
                trades.append(open_trade)
                open_trade, trail = None, None
                peak = trough = 0.0
                continue

            if open_trade.side == "long":
                peak = max(peak, high)
                cand = max(peak - stop_distance(peak, atr_pct, trail_mult, floor_pct),
                           peak * (1 - trigger_pct))
                trail = cand if trail is None else max(trail, cand)
            else:
                trough = min(trough, low) if trough > 0 else low
                cand = min(trough + stop_distance(trough, atr_pct, trail_mult, floor_pct),
                           trough * (1 + trigger_pct))
                trail = cand if trail is None else min(trail, cand)
            continue

        verdict = verdict_series.loc[ts]
        if verdict in (MarketRegime.UPTREND, MarketRegime.DOWNTREND):
            side = "long" if verdict is MarketRegime.UPTREND else "short"
            dist = stop_distance(price, atr_pct, atr_mult, floor_pct)
            capped = min(dist, price * trigger_pct)
            stop0 = price - capped if side == "long" else price + capped
            target = None
            if take_profit_r > 0:
                risk = abs(price - stop0)
                target = price + risk * take_profit_r if side == "long" else price - risk * take_profit_r
            open_trade = Trade(side, ts, price, stop0, target)
            peak = trough = price
            trail = stop0

    # Fees are charged per trade in R terms by the caller; store gross here.
    for t in trades:
        t.fee_pct = round_trip_fee_pct
    return trades


def report(trades: list[Trade], label: str, fee_pct: float) -> dict:
    if not trades:
        print(f"\n{label}\n  no trades")
        return {}

    net = [t.pnl_pct - fee_pct for t in trades]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x <= 0]
    rs = [t.r_multiple for t in trades]
    tps = [t for t in trades if t.reason == "take_profit"]

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    print(f"\n{label}")
    print(f"  {len(trades)} trades | win rate {100*len(wins)/len(trades):.0f}% "
          f"| target hit {len(tps)}/{len(trades)}")
    print(f"  net per trade   mean {sum(net)/len(net):+.3f}%   median "
          f"{sorted(net)[len(net)//2]:+.3f}%")
    print(f"  avg win {avg_win:+.3f}%   avg loss {avg_loss:+.3f}%   "
          f"ratio {abs(avg_win/avg_loss) if avg_loss else float('nan'):.2f}:1")
    print(f"  R multiple      mean {sum(rs)/len(rs):+.2f}R   "
          f"best {max(rs):+.2f}R   worst {min(rs):+.2f}R")
    print(f"  total           {sum(net):+.2f}% of the traded stake, "
          f"median hold {sorted(t.hours for t in trades)[len(trades)//2]:.0f}h")
    return {"n": len(trades), "total": sum(net), "winrate": 100*len(wins)/len(trades),
            "ratio": abs(avg_win/avg_loss) if avg_loss else 0.0}


if __name__ == "__main__":
    from config import settings

    sym = settings.symbol.replace("/", "").split(":")[0]
    rt_fee = 2 * (settings.maker_fee_pct * 0 + settings.taker_fee_pct)  # both legs taker
    print(f"{sym} | ATR stop {settings.trend_atr_stop_multiplier}x "
          f"(floor {settings.stop_loss_pct:.2%}, cap {settings.trailing_sl_trigger_pct:.0%}) "
          f"| round-trip fee {rt_fee:.3f}%")
    print("entries at the close, stops fill exactly at price -- an upper bound")

    frames = {tf: fetch_klines(sym, tf) for tf in ("30m", "1h", "1d")}
    start_ms = int(frames["1h"]["ts"].iloc[0])
    walk = fetch_klines_since(sym, "5m", start_ms)
    print(f"walking {len(walk)} 5m bars for the position, regime on 30m/1h/1d")
    for tp_r in (0.0, 2.0, 3.0, 5.0):
        trades = simulate(
            sym, take_profit_r=tp_r, atr_mult=settings.trend_atr_stop_multiplier,
            floor_pct=settings.stop_loss_pct,
            trigger_pct=settings.trailing_sl_trigger_pct,
            ema_fast=settings.ema_fast, ema_slow=settings.ema_slow,
            adx_period=settings.adx_period,
            trend_threshold=settings.adx_trend_threshold,
            range_threshold=settings.adx_range_threshold,
            round_trip_fee_pct=rt_fee, frames=frames, walk=walk)
        label = f"TREND_TAKE_PROFIT_R={tp_r:g}" + (" (no target, ride the trail)" if tp_r == 0 else "")
        report(trades, label, rt_fee)
