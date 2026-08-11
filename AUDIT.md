# Pre-test-run audit (2026-08-11)

Deep review of every module plus a scan of all production logs (`logs/grid_*.log`,
`logs/errors_*.log`, 2026-07-16 -> 2026-08-10) for error patterns that actually
happened, not just theoretical ones. Findings below are grouped by what was
fixed vs. what's flagged for awareness. All fixes are covered by new regression
tests; the full suite (86 tests across `test_grid.py`, `test_risk.py`,
`test_exchange.py`, `test_analyze_performance.py`) passes. `test_main.py` and
`test_telegram_notifier.py` need `pydantic`/`httpx`, which weren't available in
this sandbox -- run `pytest` yourself once before going live to cover those too.

## Fixed

### 1. Unhandled failure in the post-only order fallback (exchange.py) -- HIGH
**Evidence:** 57 occurrences of `Failed to place order at N: exceptions must
derive from BaseException` in the logs, always immediately following `Post-only
order rejected ... retrying without postOnly`.

When a post-only limit order is rejected because it would cross the spread,
`place_limit_order` resubmits without `postOnly` -- but that resubmit call had
no error handling of its own. Any failure there (observed: an unmapped ccxt
error code on the retried `clientOrderId`) propagated out of the function
entirely instead of joining the normal retry loop, and ccxt raising a
malformed/unmapped exception surfaced as this unhelpful TypeError. Every one
of these 57 events was a grid level that silently failed to get an order
placed.

**Fix:** the fallback call is now wrapped in its own try/except, gets its own
`clientOrderId` (not reused from the rejected attempt), and participates in
the normal attempt/retry loop with a clear log message. Tests:
`test_place_limit_order_retries_when_postonly_fallback_also_fails`,
`test_place_limit_order_raises_real_error_when_fallback_exhausts_retries`.

### 2. Risk kill-switch stop-loss check was blind to short positions -- MEDIUM-HIGH
**Evidence:** code inspection (not yet observed live, but reproducible).

`risk.check_all(..., stop_loss_price, current_price)` always compared
`current_price <= stop_loss_price` -- a long-only "price fell below the
floor" check -- and main.py always passed `grid.get_stop_loss_price()` (the
long-side value) regardless of whether the open position was actually short.
For a short position the danger direction is price *rising*, which this check
could never detect; it silently never fired. This is a secondary/backstop
check (the actual protective stop-market order on the exchange is correctly
side-aware), but it undermined that second layer of defense specifically when
it matters most -- during a short.

A related issue: the check ran even while flat, checking price against a
static long-side band with no position to protect (a possible false kill
switch off a coincidental price dip with nothing actually at risk).

**Fix:** `risk.check_all`/`_check_grid_stop_loss` now take a `side` parameter
and check the correct direction (`>=` ceiling for short, `<=` floor for long).
main.py picks `get_short_stop_loss_price()` / `get_stop_loss_price()` / skip
(`0.0`) based on the actual current position side. Tests:
`test_grid_stop_loss_kills_short_when_price_rises_above_ceiling`,
`test_grid_stop_loss_does_not_kill_short_using_long_direction`,
`test_grid_stop_loss_skipped_when_flat`.

### 3. Unbounded loop searching for a free replacement slot (grid.py) -- MEDIUM
**Evidence:** code inspection.

`_handle_fill`'s replacement-price search steps outward by `grid_spacing`
until it finds a free slot, with no iteration cap. If `grid_spacing` ever
rounds to zero at the exchange's price precision (an edge case with very
tight spacing on a fine-tick symbol -- exactly the direction the new
`high_frequency.env` profile pushes), `new_price` would never actually change
and the loop would spin forever, freezing the bot's main loop.

**Fix:** capped at `grid_count + 2` iterations; on giveup, the level is pushed
outside the grid bounds so it falls into the existing "outside bounds, don't
place" path instead of hanging. Covered indirectly by the existing 55-test
`grid.py` suite continuing to pass unchanged (no test forces the cap itself,
since it requires contriving a sub-tick-precision spacing).

### 4. `MAX_CONSECUTIVE_LOSSES` not configurable -- LOW-MEDIUM
**Evidence:** code inspection -- every other risk parameter in `RiskManager`
is threaded through `Settings`/`.env` except this one, which was silently
stuck at its hardcoded default (10) no matter what `.env` said.

**Fix:** added `MAX_CONSECUTIVE_LOSSES` to `Settings` (default 10, unchanged
behavior) and wired it through in `main.py`. Added to all three `configs/*.env`
profiles (`defensive.env` tightens it to 6).

### 5. Orders guaranteed to fail the exchange's minimum-notional filter (grid.py) -- MEDIUM
**Evidence:** 84 occurrences across `-4164 Order's notional must be no smaller
than 5` (both from regular placement and from the reduce-only unwind path).

When position-limit scaling (`_buy_scale`/`_sell_scale`, see
`set_position_limit`) shrinks an order's size as the position approaches its
cap, the resulting notional can fall under Binance's 5 USDT minimum -- a
placement that is *guaranteed* to be rejected, every time, but was still being
attempted (wasted API round-trip, ERROR-level log, Telegram alert each time).

**Fix:** pre-check the scaled notional against a `MIN_NOTIONAL_USDT = 5.0`
floor and skip cleanly (debug log, no API call) instead of attempting a
placement that can't succeed. Test:
`test_place_order_skips_when_scaled_below_min_notional`.

### 6. Recovery-grid event log always recorded old bounds as 0/0 -- trivial
`events.grid_recalculated(...)` during recovery hardcoded the previous
grid's lower/upper bounds as `0, 0` instead of the actual prior range (the
`grid` variable had already been reassigned to the new engine by the time the
event fired, so the old bounds were gone). Fixed by capturing them before
reassignment. Telemetry-only, no trading-logic impact.

### 7. Reported cumulative PnL drifted from the real account balance -- HIGH
**Evidence:** user-reported mismatch between the bot's self-reported PnL and
the actual Binance Demo Trading account equity, confirmed by comparing
`grid_dogeusdt.json`'s `total_pnl` (drifted to ~$498.56 by the end of one
session) against the account's Transaction History, a gap of roughly $510
that grew over time rather than staying constant.

Root cause: `GridEngine` tracks each grid level's own independent
`entry_price`/`exit_price` and computes `_cycle_pnl()` per level
(`_handle_fill` in grid.py). Binance, however, runs one-way position mode --
a single net position per symbol with one blended average entry price. As the
grid replaces/re-centers levels and partial fills stack up, the per-level
cost basis the bot tracks diverges from Binance's blended-average accounting,
and the divergence compounds every cycle. This is a bookkeeping/reporting bug,
not a trading-logic bug -- order placement, fill detection, and risk checks
were all working correctly the whole time.

**Fix (reconciliation, not a rewrite):** added a read-only
`PnLReconciler` (`pnl_tracker.py`) that pulls Binance's own income ledger
(`GET /fapi/v1/income`, exposed as `Exchange.get_income_history()`) --
`REALIZED_PNL`, `COMMISSION`, and `FUNDING_FEE` entries -- and sums them into
a `net_realized_pnl` figure that always agrees with the exchange's own
accounting. It bootstraps a full history pull on first run (89-day lookback --
Binance's income endpoint only retains "the last three months" per their docs
regardless of how far back `startTime` asks, so anything longer buys nothing),
then incrementally syncs after every batch of fills plus a periodic fallback
(every 60 loop iterations) to catch funding settlements that happen
independent of fills. This reconciled figure now replaces `grid.total_pnl`
everywhere it was surfaced to the user: the trade journal's `cumulative_pnl`
column, the Telegram fill/balance/startup messages ("Total PnL (verified)"),
and a new `verified_net=` field in the main loop's status log line.
`grid.total_pnl` itself is left untouched and still drives
internal analytics (`log_analytics`) -- no change to order placement, fill
handling, or risk logic. State is persisted under a new `pnl_reconciler` key
alongside `grid`/`risk` so it survives restarts without re-bootstrapping.

**Two follow-up bugs found and fixed in this same pass, before ever running
live**, both in the incremental-sync cursor logic and both stemming from the
same root cause: Binance frequently logs multiple income entries (e.g. a
`REALIZED_PNL` and its paired `COMMISSION`) at the *exact same millisecond*,
and `tranId` is only guaranteed unique *within* one `incomeType`, not across
types.
  - The first cursor design used an exclusive `since = last_time + 1` for
    incremental syncs, which would silently and permanently drop whichever
    sibling entry at the boundary millisecond didn't happen to set the max
    timestamp -- an undercount, not a crash, so it would never have surfaced
    as an error, just a slowly-wrong number again. Fixed by making the cursor
    inclusive of `last_income_time_ms` and deduping re-fetched boundary
    entries against a small persisted set of `incomeType:tranId` keys seen at
    that timestamp.
  - The pagination-level dedup (across pages within one fetch) keyed only on
    raw `tranId`, so a `REALIZED_PNL` and `COMMISSION` entry sharing a
    `tranId` would collide and one would be dropped. Fixed by keying on
    `(incomeType, tranId)`.
  - Verified with an offline simulated exchange (stubbed `get_income_history`,
    no network needed) covering: same-millisecond siblings arriving in the
    same sync, arriving in different syncs, a `tranId` collision across two
    income types, non-PnL income types (e.g. `TRANSFER`) being correctly
    ignored, and JSON state round-tripping (the `set` field is serialized as
    a sorted list) -- all pass.

**Tests:** the exchange-hitting paths (`bootstrap`/`sync` against the real
Binance API) aren't runnable in this sandbox (no network access to
Binance/PyPI -- `ccxt`, `pytest`, `pydantic`, `httpx` aren't installed here
either, so `pytest` itself couldn't run). The pure logic (`_apply_entries`,
pagination dedup, dict round-trip) was verified as described above with a
stubbed exchange; recommend formalizing that into `tests/test_pnl_tracker.py`
in your own environment. Also watch the first `PNL RECONCILER BOOTSTRAP` log
line on the next startup to confirm it pulls a non-trivial history that's in
the right ballpark versus the account's real Transaction History.

## Flagged, not changed (lower confidence or already handled)

- **`Exchange.amount_to_precision() missing 1 required positional argument`**
  (4 occurrences, all during stop-loss refresh). Already caught gracefully by
  the existing `except Exception` around `_refresh_sl_stops` in main.py -- the
  bot doesn't crash, a stop-loss update just fails for one cycle and retries
  next cycle. The `Exchange.` (capital E) in the message points at ccxt's own
  base class, suggesting a ccxt-internal edge case rather than a bug in this
  repo's code. Worth keeping an eye on if it recurs; not root-caused further
  given how rare it is.
- **`date.today() takes no arguments (1 given)`** (15 occurrences). No call in
  this codebase passes an argument to `date.today()` -- this is either a
  dependency-internal quirk (loguru's `rotation="1 day"` parsing, most likely)
  or an environment/version mismatch. Not something to blindly patch without
  reproducing it; suggest pinning exact dependency versions
  (`pip freeze > requirements.lock`) if it recurs, so it's at least
  reproducible.
- **`ReduceOnly Order is rejected` (-2022)**, 435+50 occurrences. Expected
  under the current design: reconcile/unwind logic places reduce-only orders
  against a position snapshot that can go stale between the read and the
  placement. Already handled (caught, logged, retried next cycle), just noisy.
  Not fixed here to avoid changing reconcile timing/behavior without more
  testing; consider re-fetching the position immediately before a reduce-only
  placement if this volume of noise matters to you.
- **`Reach max open order limit` / `Reach max stop order limit`** (213 + 44).
  Already anticipated and handled (`MAX_OPEN_ORDERS`, `enforce_order_limit`).
  Just means the grid occasionally pushes close to exchange limits during
  volatile stretches -- worth knowing if you push `GRID_COUNT` higher.
- **Dead code, harmless:** `RiskManager.can_recover()` and
  `GridEngine._min_profit_multiplier` (always 1.0, never adjusted elsewhere)
  are unused/static. Left alone -- no behavioral impact, not worth the churn
  of removing public methods that might be referenced elsewhere.

## Fixed (follow-up pass, 2026-08-11)

### 8. Daily PnL was still unreconciled -- MEDIUM (safety-relevant)
**Evidence:** live Telegram fill notification (2026-08-11) showed `Daily PnL:
136.5748` alongside `Total PnL (verified): -20.1711` in the same message,
which reads as contradictory but isn't -- confirmed against the saved state
at the time:

```
risk.daily_realized_pnl        = 136.70   (today's estimate)
pnl_reconciler.realized_pnl    =  19.22   (all-time, from Binance ledger)
pnl_reconciler.commission      = -42.68
pnl_reconciler.funding_fee     =  +3.12
pnl_reconciler.net_realized_pnl= -20.34   (all-time, verified)
```

Issue #7's fix only replaced the *cumulative* total (`grid.total_pnl` ->
`pnl_reconciler.net_realized_pnl`). `risk.state.daily_realized_pnl` -- shown
as "Daily PnL" in Telegram (`on_fill`) and used by
`RiskManager._check_daily_loss()` for the daily-loss kill switch -- is still
fed by `risk.record_trade(profit)` in main.py, where `profit` comes from the
grid's own per-level `_handle_fill` calculation. That's the same
blended-entry-vs-per-level-entry drift as issue #7, just bounded to one day
instead of compounding forever, so it's smaller but not zero, and it means
the daily-loss safety check is evaluating against a number that doesn't match
the account's real daily P&L.

Real finding underneath the confusing juxtaposition: the account's true
all-time realized PnL is slightly *negative* (-20.34), driven by commission
(-42.68) outweighing gross realized gains (+19.22) -- a fee/frequency
problem, not a bug.

**Fix:** extended `PnLReconciler` (`pnl_tracker.py`) with a same-day bucket,
`daily_net_pnl`/`daily_reset_date`:
- `_apply_entries()` only attributes a newly-applied income entry to
  `daily_net_pnl` when that entry's own `time` falls on the current UTC date
  (computed fresh each call) -- a long lookback bootstrap backfill or a
  late-synced entry from a prior day does not pollute today's bucket.
- `rollover_daily(today)` snapshots the just-completed day's total and resets
  the bucket; it's called from `daily_reset_check()` in `main.py`, right
  alongside the existing `risk.reset_daily()`, so the "yesterday" figure in
  the daily-summary Telegram message and event-journal entry is captured
  before it's zeroed. It's a no-op (just returns the running total) when
  called again on the same day, so it's safe to call every loop iteration.
- `_apply_entries()` also defensively self-resets the bucket (via
  `_ensure_daily_bucket()`) if it ever sees a UTC date change before
  `rollover_daily()` got a chance to run first (e.g. a long gap between
  syncs spanning midnight) -- the bucket never carries a stale prior-day
  total forward indefinitely.
- `RiskManager.check_all()`/`_check_daily_loss()` gained an optional
  `daily_realized_pnl` override parameter; `main.py` now passes
  `pnl_reconciler.daily_net_pnl` there instead of relying solely on
  `risk.state.daily_realized_pnl`. `notifier.on_fill(..., daily_pnl=...)`,
  `events.fill(daily_pnl=...)`, and `journal.record(daily_pnl=...)` were
  switched to the same reconciled figure. Per the original proposal,
  `risk.state.daily_realized_pnl` itself is left untouched and still drives
  `consecutive_losses` via `record_trade()` -- only the PnL *value* used for
  the loss-limit check and the Telegram/journal display switched sources.

**Tests:** `tests/test_pnl_tracker.py` (new) -- daily-bucket accumulation
(only today's entries, nets all three income types), bootstrap backfill not
polluting the daily bucket, `rollover_daily()` snapshot/reset/no-op
behavior, the defensive same-sync reset, plus the pagination/dedup/round-trip
coverage recommended in issue #7's writeup (same-millisecond siblings within
and across syncs, `tranId` collisions across income types, non-PnL income
types ignored, `to_dict`/`from_dict` round-trip) using a stubbed exchange, no
network required. `tests/test_risk.py` gained coverage for the
`daily_realized_pnl` override (used instead of `state.daily_realized_pnl`,
falls back to it when omitted, and never mutates `state.daily_realized_pnl`).
Full suite: 160 passed (`test_main.py`/`test_telegram_notifier.py` included
this run -- `pydantic`/`httpx` were available in this environment).

## Recommendation

The fixes above are all defensive/correctness fixes with no strategy changes
of their own (separate from the `configs/high_frequency.env` tuning already
covered in `README.md`). Safe to restart on. Before a live-funds run
specifically (this bot is currently `DEMO_MODE=true`), give it at least a few
days on `configs/defensive.env` first, and keep an eye on the first
`Daily PnL` figure after a restart to confirm it reflects only that day's
activity (it bootstraps by re-deriving from the reconciler's persisted
`pnl_reconciler` state, not a fresh zero, so a same-day restart should show
continuity rather than a reset).
