# grid-bot

Grid trading bot for Binance USDM perpetual futures. Trades a symbol inside
an auto-calculated price range, backed by a multi-timeframe trend filter
(only trades in ranging markets), layered risk management (drawdown/daily
loss kill switch, trailing + hard stop-loss, position/exposure caps,
backoff-sized recovery mode), and crash-safe state reconciliation.

Currently configured for `DOGEUSDT` in demo/testnet mode (see `.env`).

## Layout

```
main.py               entry point / trading loop
strategy.py            Strategy protocol -- the interface main.py needs to trade
grid.py                grid engine: level placement, fills, recentering, trailing SL
trend_follower.py      trend strategy: one position, ratcheted ATR trailing stop
router.py              regime router: grid in chop, trend follower in trends
exchange.py            Binance USDM wrapper (retries, circuit breaker, demo/live via API creds)
risk.py                kill switch / drawdown / daily loss / recovery sizing
trend_filter.py        ADX/EMA multi-timeframe regime detection
config.py               settings (pydantic-settings, reads .env)
state.py               atomic JSON state save/load (survives restarts/crashes)
trade_journal.py       logs/trades.csv -- one row per fill
event_journal.py       logs/events.jsonl -- structured event log
telegram_notifier.py   optional Telegram alerts
logger.py              loguru setup (console + rotating file logs)
cleanup.py             one-shot: cancel everything + close all positions
reset_state.py         one-shot: clear saved grid/risk state, start fresh
backtest.py            historical replay through the real grid engine
run_backtest.py        backtest CLI (single run / sweep / robustness)

configs/                tuning profiles, see below
tools/analyze_performance.py   evidence-based tuning report, see below
tests/                  pytest suite
```

Packaging is `pyproject.toml` + flat modules (not a `src/` package) on
purpose: this is a live-deployed bot with real state files and log history,
and moving/renaming the modules would risk breaking however it's currently
being launched (cron, systemd, Task Scheduler, etc). `python main.py` keeps
working exactly as before.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env   # then fill in API_KEY/API_SECRET/TELEGRAM_*
```

## Running

```bash
python main.py
```

Stop with Ctrl+C -- the bot cancels open orders and saves state on exit
(unless interrupted harder than that; state is also saved every loop, so a
restart recovers via `state/*.json`).

```bash
python cleanup.py   # cancel everything + close all positions; run before a manual restart if orders piled up
```

## Testing

```bash
pytest
```

## Config profiles

`config.py` always loads `.env` first (your API keys / Telegram token stay
there). Set `GRID_BOT_ENV_FILE` to layer one of the tuning presets in
`configs/` on top of it -- the profile only needs to list the parameters it
tunes:

```bash
GRID_BOT_ENV_FILE=configs/high_frequency.env python main.py
```

- **`configs/balanced.env`** -- the parameters the bot has actually been
  running with (current defaults). Kept as a documented baseline to
  compare other profiles against.
- **`configs/high_frequency.env`** -- more completed cycles/day and less
  trend-lag drag, same risk envelope. See the comments in the file for the
  specific evidence behind each change.
- **`configs/defensive.env`** -- lower leverage/exposure/position size, for
  before moving off demo mode onto real funds.

## Tuning: `tools/analyze_performance.py`

```bash
python tools/analyze_performance.py
```

Reads `logs/trades.csv` (every fill the bot has ever made -- no exchange
connection needed) and reports, broken down by market regime:

- win rate, gross/fee/net PnL, net-per-trade
- the worst individual completed cycles
- flags any regime that's net negative, with a concrete config suggestion

This is what caught the finding behind `high_frequency.env`: over the
2026-07-28 -> 2026-08-10 window, every regime was profitable except
`downtrend` (73 trades, net -$9.51, driven by a couple of large outlier
losses right at a regime transition) -- a lag artifact of
`TREND_CHECK_INTERVAL`/`TREND_CONFIRMATION_SECONDS` both being 300s (up to
~10 minutes between a trend actually forming and the grid pausing).

Re-run it after a week or two on a new profile and compare the summary line
(`net/day`, `fee % of gross`, `cycles/day`) against the baseline in
`configs/balanced.env`'s header comment to see whether the change actually
helped, rather than guessing.

## Strategies: `STRATEGY_MODE`

```bash
STRATEGY_MODE=grid     # default -- the grid engine alone
STRATEGY_MODE=router   # grid in ranging markets, trend follower in trends
```

A grid earns in chop and bleeds in sustained direction. In `grid` mode a confirmed
trend simply *pauses* the bot, so capital sits idle through it. In `router` mode the
same regime signal hands over to `trend_follower.py` instead, which holds one position
in the trend's direction behind a ratcheted ATR trailing stop.

The router satisfies the same `Strategy` protocol and delegates everything, so the
trading loop is identical either way.

**Handoffs are strict.** Binance one-way mode keeps a single net position per symbol,
so the router pauses the outgoing strategy, flattens, re-reads the exchange to confirm
flat, and only then activates the incoming one. If it cannot confirm flat -- including
when the API call fails -- it stays paused and retries. It never runs two strategies
against one position.

Because each switch costs a taker fee to flatten plus the spread to re-enter,
`ROUTER_MIN_REGIME_SECONDS` (default 900) makes a regime prove itself first.

**`grid` is the default deliberately.** The router's switching thresholds are not yet
validated: the backtest noise floor exceeds the effect sizes involved, and one
out-of-sample symbol already reversed a ranking. The mechanism is built and tested; the
evidence that it helps is not there yet. See `AUDIT.md` issues #23-#24.

## Backtesting: `run_backtest.py`

```bash
python run_backtest.py --days 90                    # current .env config
python run_backtest.py --sweep grid_count=8,10,12   # compare parameter values
python run_backtest.py --robustness                 # measure the noise floor
```

Replays historical candles through the **real** `GridEngine` -- `SimulatedExchange`
implements the same surface the engine calls on the live `Exchange`, so the replay
exercises actual order placement, fill handling, recentering and post-only logic
rather than a separate model of them. Candles come from Binance's public klines
endpoint and are cached under `data/`; no API credentials needed.

**Run `--robustness` before believing any result.** It reruns one configuration
across many start offsets and reports `mean/stdev`. This strategy is strongly
path-dependent: on DOGE 1h/90d, holding the config fixed and shifting only the
starting candle moved net PnL across a ~150 USDT range. A sweep can easily rank
noise. Below `mean/stdev ≈ 0.5` a result is indistinguishable from chance no
matter how good the headline number looks.

Known limits, all of which make results **optimistic**: no slippage or order-book
depth, no partial fills, no funding fees, one candle per loop iteration (the live
bot polls every 10s), and `main.py`'s kill switches are not simulated. Use it to
compare configurations against each other -- that comparison is fair, since every
configuration gets the same optimistic treatment -- not to predict returns.

See `AUDIT.md` "Backtesting (issue #21)" for what the first run revealed.

## Recent changes

- **Per-level replacement cooldown** (`grid.py`): `REPLACEMENT_COOLDOWN` used
  to be a single engine-wide timer -- a fill on any level blocked replacement
  of orphaned/cancelled orders on *every other* level for the cooldown
  window, which throttled the grid hardest exactly when it was most active
  (a burst of fills). It's now tracked per level, so lowering the cooldown
  (see `high_frequency.env`) is safe and only ever throttles a level that
  was itself just replaced.
