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
grid.py                grid engine: level placement, fills, recentering, trailing SL
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

## Recent changes

- **Per-level replacement cooldown** (`grid.py`): `REPLACEMENT_COOLDOWN` used
  to be a single engine-wide timer -- a fill on any level blocked replacement
  of orphaned/cancelled orders on *every other* level for the cooldown
  window, which throttled the grid hardest exactly when it was most active
  (a burst of fills). It's now tracked per level, so lowering the cooldown
  (see `high_frequency.env`) is safe and only ever throttles a level that
  was itself just replaced.
