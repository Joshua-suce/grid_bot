"""Command-line front end for the historical replay harness.

  python run_backtest.py --days 90                     single run, current .env config
  python run_backtest.py --sweep grid_count=8,10,12    compare configurations
  python run_backtest.py --robustness                  measure the noise floor

READ THIS BEFORE ACTING ON A RESULT
-----------------------------------
A single backtest number is close to meaningless for this strategy. Holding the
configuration fixed and shifting only the starting candle moves net PnL across a range
wider than the difference between "good" and "bad" parameter values -- so a sweep can
easily rank noise. --robustness exists to measure that directly: it reruns one
configuration across many start offsets and reports mean, standard deviation, and the
share of runs that finished positive.

The number that matters is mean/stdev. Below roughly 0.5 the result is indistinguishable
from chance regardless of how good the headline figure looks.
"""

from __future__ import annotations

import argparse
import statistics
import sys

from backtest import BacktestResult, fetch_ohlcv, load_ohlcv, run_backtest

# Parameters exposed to --sweep, with the type to coerce each to.
SWEEPABLE = {
    "grid_count": int,
    "min_profit_multiplier": float,
    "capital_per_grid_pct": float,
    "max_position_pct": float,
    "range_atr_multiplier": float,
    "stop_loss_pct": float,
    "trailing_sl_trigger_pct": float,
    "adx_trend_threshold": float,
    "adx_range_threshold": float,
    "maker_fee": float,
    "recenter_margin_pct": float,
    "max_open_loss_usdt": float,
}


def _defaults_from_env() -> dict:
    """Seed the backtest from .env so a bare run reflects what the bot would do."""
    try:
        from config import settings
    except Exception:
        return {}
    return {
        "symbol": settings.symbol,
        "grid_count": settings.grid_count,
        "capital_per_grid_pct": settings.capital_per_grid_pct,
        "max_position_pct": settings.max_position_pct,
        "max_exposure_pct": settings.max_exposure_pct,
        "range_atr_multiplier": settings.range_atr_multiplier,
        "min_profit_multiplier": settings.min_profit_multiplier,
        "maker_fee": settings.maker_fee_pct / 100,
        "taker_fee": settings.taker_fee_pct / 100,
        "stop_loss_pct": settings.stop_loss_pct,
        "trailing_sl_trigger_pct": settings.trailing_sl_trigger_pct,
        "recenter_cooldown": settings.recenter_cooldown,
        "replacement_cooldown": settings.replacement_cooldown,
        "recenter_margin_pct": settings.recenter_margin_pct,
        "max_open_loss_usdt": settings.max_open_loss_usdt,
        "adx_trend_threshold": settings.adx_trend_threshold,
        "adx_range_threshold": settings.adx_range_threshold,
        "ema_fast": settings.ema_fast,
        "ema_slow": settings.ema_slow,
        "adx_period": settings.adx_period,
        # The strategy actually deployed. run_backtest.py never passed this and
        # backtest.run_backtest defaults use_router to False, so EVERY figure this
        # tool has ever printed measured the grid ALONE -- not the router the bot
        # runs. Across four non-overlapping 45-day windows of ADA: grid-only +16.11,
        # router -96.21. The tool was reporting the wrong strategy by 112 USDT, and
        # every tuning decision taken from it was taken on the wrong system
        # (AUDIT #140).
        "use_router": settings.strategy_mode == "router",
        "trend_capital_pct": settings.trend_capital_pct,
        "trend_atr_stop_multiplier": settings.trend_atr_stop_multiplier,
        "trend_min_hold_seconds": settings.trend_min_hold_seconds,
        "router_min_regime_seconds": settings.router_min_regime_seconds,
        "router_handoff_grace_seconds": settings.router_handoff_grace_seconds,
    }


def sizing_mismatch(cfg: dict, balance: float) -> tuple:
    """(backtest notional per order, live notional per order, ratio).

    The harness sizes only by percent of balance. The bot, when CAPITAL_PER_GRID_USDT
    is set, sizes by a fixed stake times leverage -- and there is no parameter to tell
    the harness that, so it silently trades a different size than the account does.

    Live that is 0.018 x 4,866 = 87.6 per order against 5.00 x 5 = 25.0, ~3.5x
    oversized. It does not flip the sign of a result, but it inflates every absolute
    USDT figure and every fee total -- which is how a 45% "fee share" was read off a
    run whose fee bill at live sizing is a few USDT. Say so rather than letting the
    reader assume the numbers are to scale (AUDIT #140).
    """
    from config import settings
    bt = cfg.get("capital_per_grid_pct", 0.018) * balance
    live = getattr(settings, "capital_per_grid_usdt", 0.0) * getattr(settings, "leverage", 1)
    if live <= 0:
        return bt, 0.0, 1.0
    return bt, live, bt / live


# Price/quantity precision, from Binance USDM market metadata. `backtest.py` defaults
# these to DOGEUSDT's (5, 0) and this front end never passed them, so every non-DOGE run
# silently used DOGE's tick and step. That is not a small distortion: one grid level is
# ~1.8% of a 5,000 balance = ~90 USDT, and at amount_decimals=0 that is 0 ETH -- the
# order rounds away entirely and the run reports 0 fills, 0.00 net, no error. SOL rounds
# to whole coins instead, mis-sizing every order.
#
# Every cross-asset result this project has recorded predates this fix. Rather than
# guess a symbol's precision, unknown symbols now stop with an explicit message
# (AUDIT #52).
# Read from Binance USDM market metadata on 2026-08-14 (tick/step converted to decimal
# places). Re-read rather than extend by guesswork -- SOL's amount step is 0.01, not the
# whole coin it looks like it should be.
#
# NOTE the harness also hardcodes MIN_NOTIONAL_USDT = 5.0, which is DOGE's and SOL's.
# ETH's real minimum is 20 and BTC's is 50, so cross-asset runs on those two still model
# a smaller minimum order than the exchange allows.
KNOWN_PRECISION: dict[str, tuple[int, int]] = {
    "DOGEUSDT": (5, 0),
    "ETHUSDT": (2, 3),
    "SOLUSDT": (2, 2),
    "BTCUSDT": (1, 3),
}


def _resolve_precision(symbol: str, args) -> tuple[int, int]:
    if args.price_decimals is not None and args.amount_decimals is not None:
        return args.price_decimals, args.amount_decimals
    known = KNOWN_PRECISION.get(symbol.upper())
    if known is not None:
        return known
    raise SystemExit(
        f"No market precision is known for {symbol}.\n\n"
        f"The harness used to fall back to DOGEUSDT's (5 price / 0 amount decimals) "
        f"without saying so, which rounds every ETH-sized order to zero and every SOL "
        f"order to a whole coin -- so the run completes, reports a number, and means "
        f"nothing.\n\n"
        f"Pass the real values from the exchange's market metadata:\n"
        f"  --price-decimals N --amount-decimals N"
    )


def _parse_sweep(spec: str) -> tuple[str, list]:
    if "=" not in spec:
        raise SystemExit(f"--sweep needs KEY=v1,v2,...  got '{spec}'")
    key, raw = spec.split("=", 1)
    key = key.strip()
    if key not in SWEEPABLE:
        raise SystemExit(f"cannot sweep '{key}'. Options: {', '.join(sorted(SWEEPABLE))}")
    cast = SWEEPABLE[key]
    return key, [cast(v.strip()) for v in raw.split(",") if v.strip()]


def _row(label: str, r: BacktestResult) -> str:
    return (f"{label:<26}{r.net_pnl:>10.2f}{r.return_pct:>8.1%}{r.gross_realized:>10.2f}"
            f"{r.fees_paid:>9.2f}{r.fills:>7}{r.max_drawdown_pct:>7.1%}{r.time_paused_pct:>8.0%}")


def _header() -> str:
    head = (f"{'config':<26}{'net':>10}{'return':>8}{'gross':>10}"
            f"{'fees':>9}{'fills':>7}{'maxDD':>7}{'paused':>8}")
    return head + "\n" + "-" * len(head)


def main() -> None:
    p = argparse.ArgumentParser(description="Replay history through the real grid engine.")
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--csv", default=None, help="Use a local OHLCV csv instead of fetching.")
    p.add_argument("--balance", type=float, default=5000.0)
    p.add_argument("--trend-filter", action="store_true", help="Pause the grid in a confirmed trend.")
    p.add_argument("--router", dest="router", action="store_true", default=None,
                   help="Force the grid<->trend router on (default: follow STRATEGY_MODE).")
    p.add_argument("--no-router", dest="router", action="store_false",
                   help="Force the router off, measuring the grid alone.")
    p.add_argument("--sweep", default=None, metavar="KEY=v1,v2",
                   help=f"Compare values of one parameter. Sweepable: {', '.join(sorted(SWEEPABLE))}")
    p.add_argument("--robustness", action="store_true",
                   help="Rerun across start offsets and report mean/stdev. Do this before "
                        "believing any single result.")
    p.add_argument("--offsets", type=int, default=12, help="Number of start offsets for --robustness.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override any sweepable parameter, repeatable.")
    p.add_argument("--price-decimals", type=int, default=None,
                   help="Price precision for the symbol, from the exchange's market "
                        "metadata. Required for any symbol not in KNOWN_PRECISION.")
    p.add_argument("--amount-decimals", type=int, default=None,
                   help="Quantity precision for the symbol. See --price-decimals.")
    args = p.parse_args()

    cfg = _defaults_from_env()
    symbol = args.symbol or cfg.get("symbol", "DOGEUSDT")
    cfg["symbol"] = symbol
    cfg["use_trend_filter"] = args.trend_filter
    if args.router is not None:
        cfg["use_router"] = args.router
    cfg["price_decimals"], cfg["amount_decimals"] = _resolve_precision(symbol, args)

    for override in args.set:
        key, values = _parse_sweep(override)
        cfg[key] = values[0]

    df = load_ohlcv(args.csv) if args.csv else fetch_ohlcv(symbol, args.timeframe, args.days)
    drift = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"\n{symbol} {args.timeframe} | {len(df)} candles | net drift {drift:+.1f}%")
    _bt_size, _live_size, _ratio = sizing_mismatch(cfg, args.balance)
    print(f"strategy: {'ROUTER (grid+trend)' if cfg.get('use_router') else 'grid only'}"
          f" | trend filter: {'on' if args.trend_filter else 'off'}"
          f" | starting balance {args.balance:,.0f}")
    if _live_size > 0 and (_ratio > 1.2 or _ratio < 0.83):
        print(f"  !! SIZING NOT TO SCALE: this run trades {_bt_size:,.1f} USDT/order, "
              f"the bot trades {_live_size:,.1f} ({_ratio:.1f}x). Absolute USDT and fee "
              f"figures are inflated by about that factor; the sign and the win/loss "
              f"structure are not.")
    print()

    if args.robustness:
        offsets = [50 + i * 6 for i in range(args.offsets)]
        nets = []
        for w in offsets:
            nets.append(run_backtest(df, starting_balance=args.balance, warmup=w, **cfg).net_pnl)
        mean = statistics.mean(nets)
        sd = statistics.stdev(nets) if len(nets) > 1 else 0.0
        ratio = mean / sd if sd else float("inf")
        print(f"ROBUSTNESS over {len(offsets)} start offsets (config otherwise identical)")
        print(f"  mean net       {mean:+10.2f}")
        print(f"  stdev          {sd:10.2f}")
        print(f"  range          {min(nets):+.2f} .. {max(nets):+.2f}")
        print(f"  positive runs  {sum(1 for x in nets if x > 0)}/{len(nets)}")
        print(f"  mean / stdev   {ratio:10.2f}")
        if abs(ratio) < 0.5:
            print("\n  VERDICT: indistinguishable from chance. Shifting the start candle moves")
            print("  the result more than this configuration's supposed edge. Do not tune on it.")
        else:
            print("\n  VERDICT: signal exceeds the noise floor on this data. Still confirm on a")
            print("  different symbol and period before trusting it.")
        return

    if args.sweep:
        key, values = _parse_sweep(args.sweep)
        print(_header())
        for v in values:
            r = run_backtest(df, starting_balance=args.balance, **{**cfg, key: v})
            print(_row(f"{key}={v}", r))
        print("\nRanking these is only meaningful if the spread between them exceeds the")
        print("noise floor. Check with:  python run_backtest.py --robustness")
        return

    r = run_backtest(df, starting_balance=args.balance, **cfg)
    print(r.summary())
    if r.engine_reported_pnl != 0:
        drift_pnl = r.engine_reported_pnl - r.net_pnl
        print(f"\n  engine self-reported {r.engine_reported_pnl:+,.2f} "
              f"(drift vs position accounting: {drift_pnl:+,.2f})")


if __name__ == "__main__":
    sys.exit(main())
