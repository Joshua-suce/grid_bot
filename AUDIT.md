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

### 9. Unrealized PnL was locally re-derived instead of using the exchange's own figure -- LOW-MEDIUM (one latent sign bug found along the way)
**Evidence:** code inspection, prompted by wanting every PnL figure shown to
the user (or fed into the risk kill switch) to be exchange-verified rather
than re-computed locally -- the same principle behind issues #7/#8.

Three places recomputed unrealized PnL manually from `entry_price`/`qty`/last
price instead of using ccxt's unified `unrealizedPnl` field (Binance's own
mark-price-based number, already present in every `fetch_positions()`
response):
- `_notify_status()`'s Telegram position update,
- the main loop's `events.position_snapshot()` call -- and this one had an
  actual bug, not just an approximation: it applied `(price - entry) * qty`
  unconditionally regardless of side, which silently inverts the sign for any
  short position (a profitable short would be logged as a loss, and vice
  versa),
- `risk.update_unrealized()`, fed from `grid.get_unrealized_pnl(price)` --
  the grid's own per-level estimate, subject to the same
  blended-entry-vs-per-level-entry drift as issues #7/#8, and it's an input
  to the daily-loss kill switch (`_check_daily_loss`'s `total_daily =
  daily_realized_pnl + daily_unrealized_pnl`).

**Fix:** `get_position_details()` (main.py) now passes through the
exchange's `unrealizedPnl` as `unrealized_pnl` on each position dict (`None`
when an exchange/mock doesn't provide it). A new `_position_unrealized_pnl()`
helper prefers that real figure and only falls back to the manual
side-aware calculation when it's unavailable; all three call sites above now
go through this one helper, fixing the short-side sign bug as a side effect
of removing the duplicated inline logic. The main loop fetches
`pos_details` once and reuses it for both the unrealized-PnL sum (now real,
fed into `risk.update_unrealized()`) and the position-snapshot loop, instead
of fetching positions twice per iteration.
`grid.get_unrealized_pnl()` has no remaining callers -- left in place as
harmless dead code, consistent with how `RiskManager.can_recover()` and
`GridEngine._min_profit_multiplier` were handled above.

**Tests:** `tests/test_main.py` gained coverage for `unrealized_pnl`
passthrough (present and `None`-when-omitted), `_position_unrealized_pnl`
preferring the exchange figure over a contradicting manual calculation, the
long-side fallback, and a regression test pinning the short-side sign fix.

### 10. Removed the mock/simulated trading fallback entirely -- MEDIUM (safety-relevant)
**Evidence:** user request -- the bot should only ever trade against a real
Binance account (Demo Trading or live), never fabricate balances, positions,
or fills.

`exchange.py` had a `demo=True and not has_credentials` fallback path used
throughout: a hardcoded `DEMO_MOCK_BALANCE = 10000.0` returned from every
balance/equity getter, orders faked with `MOCK-*` client IDs tracked in
in-memory dicts, fills simulated by comparing the mock order's price against
the real ticker, and `get_positions()`/`get_income_history()` unconditionally
returning empty. This was intended as a zero-setup way to try the bot without
API keys, but it meant a bot silently missing credentials (a blank `.env`, a
typo'd key name, etc.) would start up, log as if trading, and never trade
anything real or error out -- exactly the kind of "made up" behavior flagged
in issues #7-#9.

**Fix:** the fallback is gone. `Settings.validate()` (`config.py`) now
requires `API_KEY`/`API_SECRET` in **both** DEMO and LIVE mode (previously
only LIVE); `Exchange.__init__` (`exchange.py`) raises `ValueError`
immediately if constructed without credentials, as a defense-in-depth check
independent of `validate()`. Every `if self.demo and not self.has_credentials`
branch, the `_mock_orders`/`_mock_filled`/`_mock_positions`/`_mock_balance`
state, and the `DEMO_MOCK_BALANCE` constant were deleted -- `demo=True` now
only toggles `ccxt`'s `enable_demo_trading(True)` (Binance's real Demo
Trading endpoint), no local simulation layer.

**Tests:** `tests/test_config.py` (new) covers `validate()` rejecting missing
credentials in both modes and passing once they're set.
`tests/test_exchange.py` gained a construction-time rejection test for both
modes. 165/165 tests passing overall (existing exchange tests already
constructed `Exchange` via `Exchange.__new__` with `has_credentials` set
directly, bypassing `__init__`, so they were unaffected by this change).

## Technical diagnosis of the 2026-08-11 session (issues #11-#16)

Traced from `logs/grid_2026-08-11.log` after the run "got messy". The headline
numbers for that single session: **89 dead-grid recenters at a median 196s
apart** (the recenter cooldown is 180s -- the bot was recentering as fast as it
was allowed to, continuously), **740 `-2022 ReduceOnly Order is rejected`
errors**, and a 9148 DOGE long held for over an hour whose stop-loss drifted
*down* from 0.07003 to 0.06905 while it was open.

These are not six independent bugs. #11 and #13 form a feedback loop that #12,
#14 and #15 then amplify:

```
net short  --> every replacement SELL sent reduceOnly (#11) --> -2022, always
           --> sell side can never re-arm --> grid decays to one-sided book
           --> "DEAD GRID INSIDE BAND" fires (#13) --> recenter
           --> recenter pauses grid == cancels the resting exit orders
           --> rebuild + re-unwind (over-sized, #12) --> more -2022
           --> state is immediately "dead" again --> wait out 180s cooldown
           --> recenter ... (89x)
```

Each turn of that loop also reset the trailing stop anchor (#15), which is why
the stop walked downward all night instead of ratcheting.

### 11. `reduceOnly` hard-coded on every replacement sell -- CRITICAL
**Evidence:** 265 rejections from `grid:_handle_fill`; every one a SELL placed
while the account was net short.

`_handle_fill` set `params = {"reduceOnly": True}` for *every* sell replacement.
In Binance one-way position mode the sell side of a grid is an **exit while
long** but an **entry while short** -- `reduceOnly` is only legal in the first
case. While the bot was short, every single sell replacement was rejected, so
the sell side could never re-arm; the grid bled sell levels until it was
one-sided, which is what tripped #13. (It partially self-healed one cycle later
because `check_fills`'s orphan path re-places the same level *without*
`reduceOnly` -- two code paths placing the same order with different params.)

**Fix:** `_reduce_only_qty()` / `_exit_order_params()` derive the flag from the
live net position (cached in `set_position_limit`, which already receives
`long_position`/`short_position` every loop): `reduceOnly` only when the order
actually closes something, and the quantity clamped to the remaining position
(Binance also rejects a `reduceOnly` order *larger* than the position).

### 12. Unwind sized every exit level at full grid notional -- HIGH
**Evidence:** 105 rejections from `grid:_unwind_position_through_grid`; the
`placed=` counter in the unwind summary reads 6 or 7 out of 10 whenever the
position is smaller than `10 x grid_qty`.

`_unwind_position_through_grid` placed one order per free exit level, each sized
`_calc_usdt_per_grid(balance) / price` -- the *normal grid size*, with no
reference to how big the position actually was. Against a 9148 position with
~1450 per level it asked to close ~14,500: the exchange accepted orders until
the cumulative reduce-only quantity reached the position and rejected the rest.
Guaranteed-fail API calls on every recenter.

**Fix:** slice the actual position (`per_level = amt / len(levels)`), track
`remaining`, stop when it is exhausted, and skip slices below `MIN_NOTIONAL_USDT`.

### 13. "Dead grid" false positive drove the recenter loop -- CRITICAL
**Evidence:** 89 `DEAD GRID INSIDE BAND` recenters, median 196s apart, while the
position sat unchanged at 9148 for over an hour.

`recenter()` treated "no active buys + all sells above price" as a grid that can
never fill. But that is the **normal, healthy state of a capped long unwinding**:
the position limit deliberately blocks the buy side (`POSITION LIMIT | long
9148 >= 8197 -- buy orders blocked`) and the exit sells rest above price
precisely so they fill when price ticks up. Declaring it dead made recenter fire
on every cooldown expiry -- and because `recenter()` calls `pause()`, which
cancels every open order, **it destroyed the exit orders that were about to
fill**. The position could not unwind; it was cancelled and re-posted every 196
seconds, ~20 order writes per cycle, all night.

**Fix:** the dead-grid test now requires the missing side to be missing *for no
reason* -- not blocked by the position cap (`_block_buys`/`_block_sells`) and
with no inventory whose exit orders explain the imbalance. The genuine failure
it was written for (idle grid, no position, price stranded past every resting
order) still triggers, and is covered by
`test_genuinely_dead_grid_still_recenters`.

### 14. Trailing stop-loss was not a ratchet -- HIGH (safety-relevant)
**Evidence:** long open continuously from 22:36; `SL STATUS` shows
sl=0.07003181 -> 0.06965351 -> 0.06921701 -> 0.06912001 -> **0.06905211**.

`update_trailing_sl` recomputed the level from scratch each call as
`max(grid_lower * (1 - stop_loss_pct), peak * (1 - trigger))`. Both inputs move
down when the grid recenters downward, so the "stop" followed price down. A
trailing stop must be monotonic while a position is open -- one that slides down
in a downtrend can never be hit, which is exactly when it is the only protection
left.

**Fix:** both directions now ratchet (`max(...)` for long, `min(...)` for short)
against the previously published level. `reset_trailing()` on a side flip
remains the one sanctioned release.

### 15. `recenter()` wiped the trailing anchor mid-position -- HIGH
`recenter()` unconditionally set `_peak_price = current_price` and
`_trailing_sl_price = None`. With #13 firing every 196s the high-water mark was
continuously handed back to the market, which is the mechanism behind #14.

**Fix:** only re-anchor when flat (`_net_long_qty <= 0 and _net_short_qty <= 0`).

### 16. Stop-loss rebuilt on every partial fill -- MEDIUM
`_sl_needs_update` returned True on any quantity delta down to 1e-6 relative, and
`_refresh_sl_stops` is cancel-then-place -- so **every partial fill opened a
~1s window with no stop on the book**, dozens of times per hour.

Stops are `reduceOnly`, so a stop *larger* than the position is harmless (it
closes whatever remains); only under-coverage is real exposure. The check now
always refreshes when the stop no longer covers the position or the trigger
price moved, and tolerates over-coverage up to 10% before resizing.

**Tests:** `tests/test_grid_diagnosis.py` (new, 16 tests) pins all six. Verified
adversarially: with the fixes stashed, 9 of the 16 fail against the original
code -- including the exact production states (`capped long with exit sells is
not a dead grid`, `unwind slices the position not the grid notional`,
`replacement sell is not reduce only while net short`).

### Not changed (diagnosed, no defect found)
- **Fee units.** `maker_fee_pct=0.02` in `.env` is *percent*; `main.py` converts
  with `/100` at every `GridEngine` construction site. Verified against the log:
  fill #1341 charged 0.047841 on (0.07091+0.07168) x 1681, implying a 0.0002
  rate. Correct.
- **`_cycle_pnl` sign convention.** `-delta * qty` for shorts is right.
- **Balance/equity figures.** Already sourced from `fetch_balance()` (issues
  #9/#10); the numbers in the log are the real account's.
- **`regime=uncertain` with `grid=ON`.** By design: only `is_trending()` pauses
  the grid, `uncertain` does not. Flagged as a tuning question, not a bug -- but
  see the recommendation below.

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

### What to watch on the next run (issues #11-#16)

These six are correctness fixes to existing mechanisms, not strategy changes,
but they change behaviour visibly. Confirm on the next session:

1. `DEAD GRID INSIDE BAND` should become **rare**. If it still fires on a
   cadence close to `RECENTER_COOLDOWN` (180s), the loop is not fully closed --
   capture the surrounding 200 lines before restarting.
2. `ReduceOnly Order is rejected` should approach **zero**. A residual few are
   expected from genuine races (a position closing between the position read and
   the placement); a steady stream is not.
3. `SL STATUS` for a continuously-open position must show `sl=` **monotonic** --
   non-decreasing for a long, non-increasing for a short. Any reversal while the
   position stays open means the ratchet is still leaking somewhere.
4. `UNWIND | ... rides through N levels (placed=M)` should show `M == N` (or M
   short only because slices fell under the 5 USDT minimum), never M < N due to
   rejections.

The underlying economics are unchanged and still unfavourable: the account is
net negative because commission (-42.68 over the reconciled window) exceeds
gross realized gains, at ~370 fills/day on a 1.95%-wide grid. These fixes stop
the bot from *destroying its own exit orders* and from trading with a stop that
slides downward — they do not make a fee-losing configuration profitable. Once
the run is clean, the next lever is trade frequency vs. spacing (widen
`RANGE_MIN_SPACING_PCT` / reduce `GRID_COUNT`), not more plumbing.

---

## Fee economics: the losing configuration (issues #17-#20)

The economics deferred above, now addressed. Issues #11-#16 were the bot
sabotaging its own mechanisms; these four are the reason a *correctly working*
bot still lost money. Evidence from the reconciled ledger:

```
gross realized  +13.21
commission      -45.26     <- 3.4x gross
funding          +3.05
net             -28.99
```

~700 round trips returned **0.026% average capture against a 0.102% theoretical
spacing** — the grid realized about a quarter of its own edge while paying full
freight. Two mechanisms leaked the difference, and two settings guaranteed the
edge was too thin to survive either.

### 17. Crossing orders were silently downgraded to taker fills -- CRITICAL

`exchange.py` caught Binance `-2019` ("post only order would be immediately
matched") and resubmitted the same order with `postOnly=False`. The order then
filled instantly at the **taker** rate.

For a grid level this is the worst possible outcome. The level exists to earn one
grid spacing by resting passively; crossing means it pays double the fee (taker,
not maker) *and* captures none of the spread it was placed to collect. It was
logged as `ORDER PLACED`, so it looked like a success.

Observed live at 01:22:40 and 01:22:43 on 2026-08-12: two sells placed below
market with `postOnly=False`, both filled 22 seconds later, both at a loss
(`-0.254569`, `-0.148980`) — the first two fills of a freshly started session.

**Fix:** `place_limit_order` gained `allow_taker_fallback` (default `False`) and
raises the new `PostOnlyWouldCross` instead of crossing. The grid engine treats
it as a normal transient condition: level left unplaced and retryable, no error
log, no journal entry, no Telegram alert. Exit paths (reduce-only unwinds,
hedges) already pass `postOnly=False` explicitly and never reach this branch —
for them the taker fee is worth paying to get out.

### 18. Grid levels were placed without checking the book side -- HIGH

`_place_order_for_level` placed at `level.price` with `level.side` and never
re-checked that side against the current market. A restored or stale level whose
side no longer matched the market was placed anyway and crossed immediately.
This is what fed #17 its victims. Now that crossing orders are refused, such a
level is skipped and retried instead of filling at a loss.

### 19. Profitability gate was pinned at break-even -- HIGH

`_is_level_profitable` already compared spacing against round-trip fees, but
`_min_profit_multiplier` was **hardcoded to 1.0** — break-even plus epsilon. A
level clearing its own fees by a hair passed the gate. With spacing at only ~2.6x
the round-trip fee, every level passed and ~39% of gross went straight back to
the exchange before slippage or any adverse move.

**Fix:** exposed as `MIN_PROFIT_MULTIPLIER`, default **3.0** (keeps ~2/3 of gross
after fees). The old behaviour remains reachable by setting it to 1.0, but now
that is an explicit choice rather than a buried constant.

### 20. Two settings made the losing configuration unreachable to fix -- HIGH

Both are now validated at startup, so an incoherent config fails loudly instead
of trading:

**Spacing below the fee floor.** `RANGE_MIN_SPACING_PCT=0.001` (0.1%) against a
0.04% maker round trip. `validate()` now requires
`range_min_spacing_pct >= 2 * maker_fee * min_profit_multiplier` and names all
three ways out. Raised to `0.002` — a 5.0x ratio, keeping 80% of gross.

**Grid wider than the position cap.** Per-level size is
`CAPITAL_PER_GRID_PCT` (1.8% of equity) and the cap is `MAX_POSITION_PCT` (12%),
so only `0.12 / 0.018 = 6.7` levels per side could ever fill — but `GRID_COUNT=20`
put **10** on each side. Levels 7-20 were decorative. Because both scale with
equity, **the ratio is fixed and adding capital does not help.**

This is also *why* #13's recenter loop was so destructive: the position hit the
cap after ~6 fills, that side was blocked, the book went permanently one-sided,
and the bot lived in exactly the state that made recentering cancel its own
exits. The live log shows it plainly — `exposure=13.8%` against a 12% cap,
`filled=10/20`, `POSITION LIMIT | buy orders blocked` repeating.

`validate()` now rejects `grid_count / 2 > max_position_pct / capital_per_grid_pct`
and suggests a count that works. `GRID_COUNT` lowered to **10** (5 per side vs
6.7 affordable), which also widens spacing to ~0.22% — both problems, one change.

### Resulting economics

| | before | after |
|---|---|---|
| Grid levels | 20 | 10 |
| Usable levels per side | 5 of 10 | 5 of 5 |
| Spacing | 0.102% | 0.222% |
| Spacing / round-trip fee | 2.6x | 5.0x |
| Gross retained after fees | 61% (all-maker) | 80% |
| Crossing orders | filled as taker | refused |

Verified across ATR regimes from 0.5% to 8%: the runtime profitability gate
clears by at least 2.3x in every case, so the new floor cannot silently produce
a grid that validates and then places nothing.

### What to watch on the next run (issues #17-#20)

1. **Every** initial order should log `postOnly=True`. A `postOnly=False` on a
   grid level (as opposed to an unwind or hedge) means a crossing order got
   through somewhere.
2. `POST-ONLY WOULD CROSS` at DEBUG is normal and healthy — it is the leak being
   refused. A *continuous* stream of it for the same level means the grid is
   mis-centred and should recenter rather than retry.
3. `SKIP ORDER @ ... < min_profit` should be rare at `GRID_COUNT=10`. Frequent
   hits mean ATR collapsed and the range narrowed below the fee floor; the grid
   is correctly declining to trade, not broken.
4. `POSITION LIMIT | buy orders blocked` should now be **occasional**, not
   permanent. If the grid still lives at its cap, `MAX_POSITION_PCT` and
   `CAPITAL_PER_GRID_PCT` are still mismatched for the realized volatility.
5. Commission versus gross in `PNL RECONCILER SYNC`. The target is commission
   well under gross realized. It was 3.4x over.

### Still not addressed

**Leverage.** `LEVERAGE=5` amplifies the directional-inventory loss that a grid
takes in a trend. It is deliberately left alone: it is a risk-appetite choice,
not a config incoherence, and belongs to the account owner rather than the audit.

**No backtester.** Every value above is derived from arithmetic and one live
session. Nothing here has been validated against historical price action, and
this remains the single largest gap in the project — a fee-sensitive strategy
whose parameters can only be tested by spending real days and real fees.

**Trend exposure.** A grid earns in chop and bleeds in trend. The ADX/EMA filter
pauses the grid in a confirmed trend, but the loss booked on 2026-08-11 was
largely directional: the bot accumulated into a falling market and unwound at a
loss (`-1.168`, `-1.060`, `-0.952` on consecutive fills). No amount of fee tuning
addresses that; it is the strategy's inherent exposure.

---

## Backtesting (issue #21) and what it revealed

`backtest.py` replays historical candles through the **real** `GridEngine` --
`SimulatedExchange` implements the same 13-method surface the engine calls on the
live `Exchange`, so the replay exercises actual order placement, fill handling,
recentering, position caps and post-only logic rather than a second model that
could agree with the first while both are wrong. `run_backtest.py` is the CLI.
Verified by 22 tests covering fee arithmetic, position accounting through flat,
intra-candle fill ordering, and the `-2019` / `-2022` / `-4164` rejections.

Two implementation notes worth keeping:

- **Virtual clock.** `GridEngine` gates replacement (20s) and recentering (180s)
  on `time.time()`. A replay covering months finishes in seconds, so under the
  real clock those cooldowns would never expire and the run would be meaningless.
  `VirtualClock` is installed as the engine's `time` module for the duration.
- **PnL is computed from position accounting, not from the engine.** The engine's
  own figure is reported alongside for comparison, and the gap is large -- see
  below.

### Result 1: the #17-#20 fixes are real and measurable

DOGEUSDT 1h, 90 days (2160 candles), a period in which DOGE fell **-35.9%**:

| configuration | net | gross | fees | fills | maxDD |
|---|---|---|---|---|---|
| OLD `gc=20 mult=1.0` | -165.88 | -114.70 | 51.18 | 2831 | 4.7% |
| NEW `gc=10 mult=3.0` | -161.01 | -124.05 | 36.96 | 1758 | 4.1% |
| OLD + trend filter | -154.73 | -116.10 | 38.63 | 2041 | 4.9% |
| NEW + trend filter | **+28.28** | +53.87 | 25.59 | 1132 | 2.2% |

Fees fell 50% and fills 60%. Those are genuine improvements and they hold up.

### Result 2: the direction of the effect is real; the exact numbers are not

A single run is close to meaningless here. Repeating each configuration across 15
start offsets (identical data, only the first candle differs) separates what holds
from what is noise:

| configuration | mean | stdev | positive | mean/sd |
|---|---|---|---|---|
| OLD `gc=20 mult=1.0` | -83.17 | 106.45 | 4/15 | -0.78 |
| NEW `gc=10 mult=3.0` | -54.06 | 96.63 | 4/15 | -0.56 |
| OLD + trend filter | -51.42 | 85.12 | 4/15 | -0.60 |
| NEW + trend filter | **+28.07** | 51.66 | **11/15** | +0.54 |

**What holds:** the ordering. Every fix improves the mean monotonically, the spread
narrows as the configuration improves, and only the fully-fixed configuration is
positive in a majority of offsets. That direction is consistent across all 15 runs,
so it is an effect rather than a lucky draw.

**What does not hold:** any specific parameter value. Neighbouring `GRID_COUNT`
values swing wildly on identical data — the signature of fitting noise:

```
gc=6   +10.06     gc=12  -140.92
gc=8   -34.43     gc=14   -29.24
gc=10  +28.28     gc=16    -0.08
                  gc=20  -154.73
```

The same applies to `RANGE_ATR_MULTIPLIER`: 2.5 (the `.env` value) scored +6.52 and
1.5 scored +93.49 on single runs — a gap well inside a 51-point standard deviation,
so it is not evidence that either is better.

> **Verdict: the #11-#20 fixes demonstrably improve the strategy. They do not
> demonstrably make it profitable.** `mean/sd = 0.54` is suggestive, not conclusive
> — the threshold for acting on a result should be nearer 1.0.

Two caveats that make even `0.54` optimistic:

1. **The offsets are not independent samples.** All 15 replay the same 90-day window
   shifted by at most 84 candles out of 2160 — they overlap ~96%. The standard
   deviation therefore measures start-anchor sensitivity, not sampling uncertainty of
   the edge. Effective sample size is far closer to 1 than to 15.
2. **One symbol, one period, one regime** (DOGE, -35.9%).

**Out-of-sample check — and it does not fully replicate.** Two instruments the
configuration was never derived from, 5 start offsets each:

| symbol | drift | configuration | mean | stdev | positive |
|---|---|---|---|---|---|
| DOGE (in-sample) | -35.9% | OLD | -83.17 | 106.45 | 4/15 |
| | | NEW + filter | **+28.07** | 51.66 | 11/15 |
| ETH | -16.5% | OLD | -47.24 | 73.23 | 1/5 |
| | | NEW + filter | **-3.19** | 85.70 | 3/5 |
| SOL | -15.7% | OLD | **+54.21** | 125.60 | 3/5 |
| | | NEW + filter | +12.20 | 74.69 | 4/5 |

**On SOL the old configuration beat the new one** (+54.21 vs +12.20). Mean PnL
improves on two instruments out of three, not three out of three. An earlier draft of
this section claimed the direction replicated; that was written from DOGE and ETH
before SOL finished, and it was wrong.

What *does* hold on all three is the share of runs finishing positive — DOGE 27% ->
73%, ETH 20% -> 60%, SOL 60% -> 80%. The fixed configuration loses less often
everywhere, even where its mean is lower.

Both out-of-sample samples are n=5 against standard deviations of 75-126, so neither
is individually distinguishable from zero or from the other. SOL's +54.21 +/- 125.60
is not evidence the old configuration is better; it is evidence the sample is too
small to tell.

> **Honest overall verdict: the #11-#20 changes are sound engineering — they remove
> real defects, halve fee drag, and reduce the frequency of losing runs on every
> instrument tested. A consistent improvement in expected PnL is NOT established, and
> profitability is not established on any instrument.**

This is the harness earning its keep on day one: without it, `gc=10` would have been
adopted as "the profitable configuration" on the strength of a single `+28.28`, and
`GRID_COUNT` would have been tuned on pure noise.

### Result 3: the engine's self-reported PnL still drifts badly

On the 90-day replay the engine reported **+249.20** while true position
accounting gave **+6.52** — a drift of **+242.68**. This is AUDIT #7 reproduced
under controlled conditions, and it independently justifies sourcing every
displayed figure from `PnLReconciler` and the exchange rather than from
`grid.total_pnl`. The engine's internal figure should be treated as diagnostic
only, never reported to the user as PnL.

### Result 4: `MIN_PROFIT_MULTIPLIER` is currently inactive

Sweeping it across 1.0 / 3.0 / 5.0 on the live `.env` config produced **identical**
results (net +6.52, 570 fills, every time). With `RANGE_ATR_MULTIPLIER=2.5` and
`GRID_COUNT=10`, ATR-derived spacing sits far above even five round-trip fees, so
the gate never binds.

That does not make it useless -- it is the guard that stops a volatility collapse or
a future config change from silently reproducing the 2.6x-fee grid that lost money.
But it is a safety net, not an active constraint today, and it should not be credited
with any of the improvement measured above. The improvement came from `GRID_COUNT`,
the crossing refusal, and the trend filter.

### How to use this

```
python run_backtest.py --days 90                    # current .env config
python run_backtest.py --sweep grid_count=8,10,12   # compare
python run_backtest.py --robustness                 # measure the noise floor
```

`--robustness` reports mean/stdev across start offsets. **Below ~0.5 the result is
indistinguishable from chance** and must not be tuned on, however good the headline
number looks. Run it before believing any sweep.

---

## Strategy protocol (issue #22) -- step 2 of the multi-strategy plan

`strategy.py` defines `Strategy`, the interface main.py needs from anything that
trades. **Zero behaviour change:** `GridEngine` already satisfies it as written,
because the protocol was derived from what main.py actually calls rather than
invented and imposed.

A structural `Protocol` was chosen over an ABC deliberately. `GridEngine` is 1600
lines of live-tested logic with real state files behind it; reparenting it would mean
touching its constructor and MRO for no behavioural gain. Duck-typed conformance
asserts the same contract while modifying nothing.

The protocol covers 19 members -- lifecycle (`initialize`/`activate`/`pause`/
`emergency_stop`), trading (`place_initial_orders`/`check_fills`), the risk interface
(`set_position_limit`/`get_exposure_pct`/`update_volatility`), stops, reconciliation,
persistence and metrics.

**Ten members are deliberately excluded** as ladder-of-orders specific, and they are
the remaining step 3/4 work queue:

```
recenter  grid_lower  grid_upper  grid_count  grid_spacing
levels    log_sl_status  log_analytics  get_scale_out_trail_price  update_orderbook
```

`tests/test_strategy.py` (25 tests) keeps the boundary honest. One test parses main.py
and asserts every `grid.<member>` it calls is classified as either protocol or
grid-specific -- so adding a new coupling to main.py fails the suite until someone
decides which side of the line it belongs on. That is the mechanism that stops this
abstraction rotting the way an undocumented interface would.

---

## Stop-loss protection defects from the 2026-08-12 live run (issues #25-#26)

Both surfaced at the same instant -- the 14:06 recenter -- and share a shape: an event
that was *not* a stop firing nonetheless reduced protection on a 7108 DOGE long that
stayed open. Same class as #14/#15, one layer down.

```
14:05:26  trail SELL 3554 @ 0.0693647   hard SELL 3554 @ 0.06825994
14:06:00  recenter -> pause() -> cancel_everything()   <- cancels the stops too
14:06:16  "SCALE-OUT STOP FIRED"                       <- nothing fired; price was 0.07035
14:06:17  hard SELL 7108 @ 0.06696984                  <- trail leg gone, hard stop 1.9% wider
```

### 25. The hard stop followed grid_lower away from an open position -- HIGH

`hard_price` was recomputed as `grid_lower * (1 - stop_loss_pct)` on every refresh, and
`recenter()` moves `grid_lower`. With a long open, the 14:06 recenter dropped the hard
stop from `0.06825994` to `0.06696984`.

AUDIT #15 stopped `recenter()` resetting the *trailing* anchor while a position was
open. The *hard* leg was still free to track the band. **Fix:** `get_hard_stop_loss_price`
/ `get_short_hard_stop_loss_price` ratchet the static level -- it may tighten while a
position is open, never loosen -- released by `reset_trailing()` once flat. main.py now
asks for the ratcheted level instead of recomputing it.

### 26. A cancelled stop was read as a fired stop -- HIGH

`_detect_trail_fill()` inferred a trigger purely from the trail order's absence from
the open-stop list:

```python
if sl_orders["trail"]["id"] not in open_ids:
    _scale_out_done = True
```

It could not distinguish *triggered* from *cancelled by us* -- and we cancel stops on
every refresh, on `pause()`, and inside `recenter()`. Worse, `_scale_out_done` latches:
the trailing leg was never re-placed, so the position spent the rest of its life on the
hard stop alone.

**Fix:** confirm with the exchange. `trail_stop_fired(order)` (module-level, so it is
testable) treats only a completed fill as a fire; cancelled, expired, unknown or
unreachable all read as "did not fire", and the leg is re-placed on the next pass. The
conservative default matters -- a false positive strips protection permanently, a false
negative merely re-places an order.

### Caught by the step-2 boundary test

Adding the two getters to main.py immediately failed
`test_protocol_covers_what_main_actually_calls`, which parses main.py and rejects any
`grid.<member>` that is classified neither as protocol nor grid-specific. They are
universal -- any strategy holding a position needs a last-line stop -- so they went into
the `Strategy` protocol, with implementations on `TrendFollower` and `StrategyRouter`.
That is the abstraction doing its job on its first real test.

`tests/test_stop_protection.py` (15 tests) pins both, using the numbers from the log.
Verified adversarially: with the ratchet removed and absence restored as a fire,
**11 of the 15 fail**.

---

## Trend follower (issue #23) and regime router (issue #24) -- steps 3 and 4

`trend_follower.py` is the grid's complement: it holds one position in the direction of
a confirmed trend, exits on a ratcheted ATR-scaled trailing stop or when the regime
stops supporting the side. It exists because the bot previously *detected* trends and
responded by switching off, leaving capital idle through every trend.

Its fee rule is the opposite of the grid's, for the opposite reason. Grid levels must
never cross (AUDIT #17) because a level earns one grid spacing and a taker fee consumes
a large share of it. A trend position targets multiples of ATR, so 0.04% is noise and
missing the entry costs far more -- entries and exits therefore cross deliberately.

Deliberately absent: pyramiding, partial exits, re-entry after a stop. Each adds
parameters, and the noise floor measured above already exceeds the effect sizes being
chased, so extra knobs would be unfalsifiable rather than useful.

`router.py` presents a single `Strategy` to main.py and delegates to whichever strategy
the regime selects. Anything off-protocol (`recenter`, `grid_lower`, `levels`, ...)
falls through `__getattr__`, which is why TrendFollower implements harmless equivalents
of `GRID_SPECIFIC_MEMBERS` -- main.py needs no changes to drive either.

### The handoff is the dangerous part

In one-way position mode there is one net position per symbol. Activating a strategy
while another still holds one means both write to the same position: they pay fees to
cancel each other out, and the reduce-only bookkeeping AUDIT #11 fixed goes stale with
two writers. The sequence is strict and stops on failure:

```
pause outgoing    cancel its resting orders
flatten           close the position it held
verify flat       re-read the exchange -- the exchange is the truth
hand over         only now activate the incoming strategy
```

If verification still sees a position the router stays paused and retries next tick. An
API failure during verification counts as *not flat*: an unknown position must never
read as no position. `test_incoming_strategy_is_not_activated_while_a_position_remains`
pins this.

Switching costs a taker fee to flatten plus the spread to re-enter, so
`ROUTER_MIN_REGIME_SECONDS` (default 900) requires a regime to persist before it is
acted on, on top of TrendFilter's own confirmation window.

### Off by default, and why

`STRATEGY_MODE` defaults to `grid`, reproducing the previous behaviour exactly. The
router is opt-in because **its switching thresholds remain unvalidated** -- the
measured noise floor (sd 50-126) exceeds the effect sizes (~40) on a single symbol and
period, and SOL already showed a ranking reversing out of sample. The mechanism is
built and tested; the evidence that it *helps* is not there yet, and turning it on by
default would be asserting something the data does not support.

The remaining work is evidential, not structural: walk-forward evaluation across
several instruments, and backtest support for the router so the thresholds can be
tuned against history rather than a live account.

### What this changes about the roadmap

The strategy-protocol and trend-follower work sketched earlier is still the right
structural direction — regime coverage is a real gap, and the trend filter was the
single largest measured effect (gross -116 → +54 in the table above). But the
router's thresholds cannot be tuned on a single symbol and period, because the noise
floor here exceeds the effect size. Before that work is worth doing, this harness
needs walk-forward evaluation and more than one instrument.

Known limits of the harness, all of which make results **optimistic**: no slippage
or depth model, no partial fills, no funding, one candle per loop iteration, and
main.py's kill switches are not simulated.
