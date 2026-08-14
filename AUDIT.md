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

The protocol covers 20 members -- lifecycle (`initialize`/`activate`/`pause`/
`emergency_stop`), trading (`place_initial_orders`/`check_fills`), the risk interface
(`set_position_limit`/`block_side`/`get_exposure_pct`/`update_volatility`), stops,
reconciliation, persistence and metrics.

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

## Router backtest support (#27) and the deadlock it found (#28)

`backtest.py` gained `use_router=True`, which wraps the engine alongside a
`TrendFollower` in a `StrategyRouter` exactly as main.py's `_install_strategy` does,
and drives the loop through `strategy` rather than `engine`. Two mechanical
prerequisites:

- **`SimulatedExchange.get_price()`** -- TrendFollower sizes and stops off it.
- **The virtual clock now patches three modules**, not one. `grid.py`, `router.py` and
  `trend_follower.py` all gate on `time.time()`: replacement (20s), recentering (180s),
  the router's regime hold (900s), the follower's minimum hold (300s). Patching only
  `grid` would leave the others on the real clock, which never advances relative to
  simulated candles -- the router would switch on the first reading and the follower
  could never satisfy its hold.

### 28. The handoff could never complete in live trading -- CRITICAL

The first router replay returned `strategy switches: 0` and `time paused: 99.3%`
despite `handoff grid -> trend starting` appearing in the log. The handoff began and
never finished.

`_continue_handoff` was driven only from `place_initial_orders`. In main.py:

```
L471   grid.place_initial_orders(...)     <- once, at startup, BEFORE the loop
L731   while True:
L903       if grid.active:                <- everything else lives in here
L957           grid.check_fills(...)
L1036          grid.set_position_limit(...)
```

`place_initial_orders` is never called inside the loop, and every other strategy call
sits behind `if grid.active:`. Since `_begin_handoff` pauses the outgoing strategy,
`active` goes False and none of them run. **The bot would have paused the grid on the
first confirmed trend and stopped trading permanently** -- no orders, no fills, no
recovery, holding whatever position it had.

`update_regime` is the only strategy call main.py makes unconditionally every
iteration, so the handoff is now driven from there as well and completes in one call
in the normal case. `_current_balance()` reads the balance the router already has an
exchange reference for, falling back to 0.0 rather than aborting.

The flat-before-handover rule is unchanged: `test_update_regime_driven_handoff_still_
refuses_to_hand_over_dirty` pins that driving it from a new place did not weaken it.

**This is the backtester paying for itself a second time.** The bug was unreachable by
unit tests -- every router test called `place_initial_orders`, because that is how the
router was designed to be driven. Only replaying against main.py's actual call pattern
exposed it, and `STRATEGY_MODE=router` was already enabled in `.env` at the time.

Verified adversarially: with the fix reverted, 2 of the 29 router tests fail.

---

## 51. A failed cancel is not a cancel, and the fees were never all-maker -- CRITICAL

`Exchange.cancel_order` has always returned an honest `bool`: `True` only when the order
is confirmed gone (`OrderNotFound` counts as gone), `False` for "status unknown". It
never falls off the end -- every path returns explicitly.

**Three of the four production callers discarded it.** Only the reconcile path (#41)
checked. And because `cancel_order` swallows the ccxt exception internally and returns
`False`, the `except` blocks wrapped around these calls almost never fire: the normal
failure is a quiet `False` that nobody read.

### 51a. The position cap freed levels it had not cancelled -- CRITICAL

`_cancel_resting_orders` is the last line of defence: it runs when the position is **at
its cap**. On a failed cancel it fell straight through to journalling the order as
cancelled, notifying the operator it was cancelled, setting `order_id = None`, `status =
"pending"`, and counting it in the total.

So the order was still live, the engine had forgotten it, and the level was marked
free -- meaning the next `place_initial_orders` laid a **second order at the same
price**. At the cap, cancels fail, the engine believes it shrank exposure, and it grows
it instead. That is #49's failure with the guard's own hands.

It now keeps `order_id` on any unconfirmed cancel (retried next iteration, self-healing),
counts only confirmed cancels, and logs the failures at ERROR instead of reporting a
clean sweep.

### 51b. `pause()` did the same -- HIGH

Pause deliberately does not flatten. A level it could not cancel must stay claimed, or
resuming re-places on top of the survivor.

### 51c. An unconfirmed entry cancel became a double entry -- HIGH

`TrendFollower._cancel_entry` cleared `_order_id` regardless. That is the exact variable
`place_initial_orders` guards on (`if self._order_id is not None: return 0`), so the next
call opened a **second entry at full size** while the first was still live -- and the
survivor filled outside `check_fills`, untracked, with no stop. It now returns `bool` and
keeps the order claimed unless the cancel is confirmed.

### 51d. The round trip was never 2 x maker -- MEDIUM-HIGH

Eight gates priced a completed cycle at `2 * maker_fee_pct`, documented as "the true
round-trip cost, not an optimistic floor". The income ledger disagrees: **11.8% of fill
volume paid the taker rate**, so the real round trip is 0.0447% against the 0.0400%
assumed -- 1.12x.

The gap is structural, not drift. Reduce-only exits are placed `postOnly=False` on
purpose (a queued exit that never fills is worse than a crossed one), stop-losses always
cross, and reconcile/unwind close at market. It will never be zero.

Understating it hurt twice:

- `MIN_PROFIT_MULTIPLIER=3.0` bought a real **2.68x** margin, not 3.0x.
- Worse, **every break-even price was 12% short.** `_position_break_even` returns
  `entry * (1 +/- fees)`, and that number is what #32's guard clamps exits to. Clamping
  to a break-even that understates fees books a small *genuine* loss on every exit that
  hits it -- a guard whose entire purpose is to not lose money.

All eight now route through one `round_trip_fee_pct` property blending both rates by
`TAKER_FILL_SHARE_PCT` (default 11.8, measured). Config validation prices the required
spacing the same way. Setting the share to 0 collapses it to the old figure exactly.

### What this exposed in the dormancy tests, and what it did not

The wider floor tipped `test_dormancy` red, which looked like #42 reopening. It was not.

The move being refused was 0.00009 from a neighbour, against a floor that grew from
0.00008425 to 0.00009419 -- **the old test passed by 7%.** The first attempt at a fix
was to ignore *pending* neighbours when checking crowding; that is wrong, and
`test_no_move_when_it_would_deform_the_ladder` caught it immediately. `place_initial_
orders` walks the whole ladder every iteration, so a pending level is not an empty
price, it is an order about to exist.

The guard is right as written, and it bounds its own dormancy risk: a neighbour inside
the fee floor is a neighbour within ~0.13%, and it quotes. The #42 incident was the
opposite case -- the blocked level was the **only one within 1.2% of the price**. That
is the condition that makes refusing catastrophic, and it is now the condition the
dormancy tests actually set up, rather than relying on a 7% margin in an unrelated
constant.

### Verified adversarially

`tests/test_unconfirmed_cancels.py` -- 8 tests. With all four fixes reverted, 6 fail.
The two that hold are the positive controls: a *confirmed* cancel still frees its level
(the fix must not strand inventory the way #42 did), and an all-maker book still
collapses to the old fee number.

`test_pause_keeps_a_level_that_would_not_cancel` passed against broken code on the first
adversarial run -- `pause()` returns immediately when `active` is False, so it asserted
nothing. Fixed, then re-verified. Two vacuous tests have now been caught this way in
three audits; the adversarial pass is not optional.

Full suite: 488 passed, 5 skipped.

---

## 50. Three safety nets that were not there -- CRITICAL

#49 closed the mechanism that let the position reach 31,761 DOGE against a 17,467 cap.
This entry covers why that position was also running **naked** -- and two other places
where a safety net reported success it had not achieved. One family, one failure mode:
**a risk control that cannot tell you it failed is not a risk control.**

### 50a. `_refresh_sl_stops` cancelled first and hoped -- CRITICAL

The sequence was: cancel every existing stop, then place the replacements inside a
`try/except` that only logged. It returned `None` either way, so no caller could
distinguish "protected" from "the cancels went through and every placement raised".

That is exactly the 2026-08-08 sequence. The cancels succeeded, the placements hit the
`TypeError` that #47's contract test now catches, each was swallowed one line at a
time, and a position 1.8x through its cap sat with no stop for the rest of the day.
-50.49, 60% of the entire 15-day loss, on six closes.

It now returns coverage as a `bool`. On total failure it logs `POSITION UNPROTECTED`
and emits a `risk_check` event, and main.py acts on the answer:

```python
sl_covered = _refresh_sl_stops("long", position_qty)
...
if not sl_covered:
    grid.block_side("buy", "stop-loss missing")
```

**An unprotected position may be closed and may be held. It may not grow.** That is the
rule the -50.49 day needed and did not have. `block_side` gates only orders that ADD
exposure -- `_exit_order_params` never consults it, because trapping inventory behind a
safety check is its own bug (#42).

The block is deliberately **not sticky**: `set_position_limit` clears it at the top of
every iteration and main.py re-blocks below if the position is still uncovered, so it
lasts exactly as long as the condition and lifts by itself when the stops come back.

### The gap this opened, and the test that caught it

Adding `block_side` broke `test_strategy.py` -- **as designed**. main.py calls whatever
strategy is live, and in `STRATEGY_MODE=router` that is a `StrategyRouter` delegating to
a `TrendFollower`, neither of which had the method. A fix that protects only the grid
would have been an `AttributeError` every iteration in router mode.

This is the boundary test from #22 doing precisely the job it was written for: it fails
the suite until someone decides which side of the line a new coupling belongs on. So
`block_side` is now on the `Strategy` protocol, the router **broadcasts** it to every
strategy (the position is NET and shared -- an unprotected long is unprotected no matter
which strategy would add to it), and `TrendFollower` withholds blocked entries while
leaving `_close_position` untouched.

### 50b. The hard-stop ratchets were not persisted -- HIGH

`_hard_sl_price` and `_hard_sl_price_short` are ratchets: a long's stop may only rise, a
short's only fall. They have to be, because `recenter()` moves the bounds the stop is
derived from, so without the ratchet a recenter downward would quietly widen the stop on
an open position (#15, again).

Checked against a real state backup: `_peak_price`, `_trough_price`,
`_trailing_sl_price` and `_trailing_sl_price_short` are all saved. **These two were
not.** A restart with an open position re-derived the stop from the new bounds and
silently loosened it -- the ratchet held perfectly right up until the process bounced.

Both are now in `to_dict`/`load_from_dict`.

### 50c. "CLEANUP VERIFIED" was printed when nothing had been verified -- MEDIUM-HIGH

`cancel_everything`'s `_fetch_regular()` returns `None` when the read fails. The check
was `if remaining:` ... `else: "CLEANUP VERIFIED | book clean"`. `None` is falsy, so a
verification that could not read the book announced the book was clean.

Startup runs this before laying a fresh ladder. A false all-clear means a new grid on
top of live orders -- the duplicate-level pile-up `cancel_everything` was written to
prevent in the first place.

Now three-way: incomplete / **UNVERIFIED** / verified.

### Verified adversarially

`tests/test_safety_nets.py` -- 9 tests. With all three fixes reverted, 4 fail, and the
cleanup test reproduces the defect verbatim: after the book read raises
`ConnectionError`, the log still reads `CLEANUP VERIFIED | book clean for DOGEUSDT`.

The cleanup test drives the real `cancel_everything` against a stubbed book and asserts
on captured log output, not on source text -- the first version asserted on the order of
two string literals in the source and was simply wrong, because there is an earlier
`CLEANUP VERIFIED` in the same function. A companion test pins that a genuinely empty
book is still reported clean, so the fix cannot pass by crying wolf.

Full suite: 480 passed, 5 skipped.

---

## 49. The position cap was advisory -- the replacement path bypassed it -- CRITICAL

The mechanism behind #47's -50.49 day, now found and closed.

`_handle_fill` places the paired replacement order by calling `place_limit_order`
**directly**. Every guard lives in `_place_order_for_level`, which that path skips
entirely:

- `_block_buys` / `_block_sells` -- the position cap
- `_buy_scale` / `_sell_scale` -- the size taper as the cap is approached
- the minimum-notional check

So the cap never capped. Each fill immediately armed another order regardless of it;
that order filled; its replacement did the same. In a trend the position ratchets
upward with nothing to stop it. On 2026-08-08 it reached **31,761 DOGE against a 17,467
cap** -- 1.8x through a limit the log was simultaneously reporting as enforced:

```
23:00:14  BUY SCALE | long=10465.0/17467.8 | scale=0.80
23:03:12  BUY SCALE | long=11812.0/17476.7 | scale=0.65
...
18:18:11  Closed existing SHORT position: 31,761 DOGEUSDT
```

Combined with the stop-loss TypeError from #47, that is the whole disaster: an
unbounded position running unprotected. -48.92 realised on six closes, 60% of the
fortnight's loss.

### The fix, and the line it must not cross

`_exit_order_params` already distinguishes the two cases: `params is None` means it
found nothing to reduce, i.e. **this order opens exposure**. Opening orders now respect
the block flags and the taper; **reduce-only exits stay unconditional**, because
refusing those traps inventory -- that is #42, and re-introducing it here would trade
one failure mode for a worse one. The min-notional check was added for the same reason
it exists in the other path: a sub-minimum order is a guaranteed -4164.

Reverting the guard fails 4 of the 5 tests in `tests/test_position_cap.py`; the fifth
covers exits, which the revert does not affect. 470 tests pass.

### A note on how this was nearly missed twice

The first version of the regression test blocked buys and then filled a *buy* level.
It passed against the broken code, because a buy fill produces a **sell** replacement --
the sides flip across a fill. A second test looked correct and was vacuous: level
quantities start at 0.0, so the assertion compared against a fallback size and could
never fail. Both were caught by running the tests against the *unfixed* code first,
which is the only reason they mean anything.

---

## 48. The parameter search does not generalise -- and the trend filter does

32 configurations (grid_count 4/6/8/10 x range_atr 3.0/4.0/5.0/6.0 x trend filter
on/off), selected on the EARLIER 90 days, validated on the RECENT 90 days which the
search never saw. Every previous sweep in this file picked its winner after looking at
both windows; this one did not, and the difference is the whole point.

### The winner failed out of sample, decisively

```
SELECTED on earlier window: 8 levels / atr 6.0 / filter on  ->  +41.66

VALIDATION -- RECENT 90d, never used for selection
  old default        10 / 2.5 / on      +35.49   sem 11.68   455 cycles
  currently deployed 10 / 3.5 / on      +46.12   sem 17.29   283 cycles
  SELECTED            8 / 6.0 / on      -13.59   sem  4.14    57 cycles

selected vs deployed, paired on the holdout: -59.71 (sem 16.11, t=3.7)
```

**t=3.7 is the largest statistic anywhere in this audit, and it points against the
optimised configuration.** +41.66 in-sample became -13.59 out-of-sample. Had the winner
been shipped -- which is exactly what the earlier sweeps in #44 and #46 invited -- the
account would have been materially worse off.

The practical conclusion is not "8/6.0 is a bad cell". It is that **this parameter
surface is noise**: optimising against it produces configurations that do not survive
contact with an unseen period. Further tuning is not a route to profitability, and the
strength of this result is the best evidence in the file for stopping.

The currently deployed 10 / 3.5 was positive in both windows (+22.13 earlier, +46.12
recent) and is left alone.

### The trend filter, by contrast, is completely robust

```
                     mean      min       max
  filter on       +21.77    +5.33    +41.66
  filter off       -9.84   -50.25    +25.30
```

**Filter on beats filter off in 16 of 16 geometries**, mean difference +31.6, and it is
the only variable tested all session that behaves consistently across the whole surface.
It is already enabled (`STRATEGY_MODE=grid` runs the grid behind the filter), so this
changes nothing -- but it settles a question never previously isolated, and it says the
single most valuable component of the system is the one that decides *not* to trade.

Per-cycle capture does keep rising with width (0.085 at atr 3.0 to 0.277 at atr 6.0),
confirming the #46 mechanism. It simply stops converting into PnL past ~3.5, because the
cycles become too few to matter.

---

## 47. The single worst day, and the guard that was meant to prevent it -- CRITICAL

Pulling Binance's income ledger directly (not the reconciler, which had been
re-baselined and was reporting a truncated window) gives the real 15-day scoreboard:

```
date          realized  commission  funding      NET   closes
2026-07-31      +13.74      -4.51    -0.00    +9.23     192
2026-08-01       -6.91      -2.74    -0.60   -10.25      22
2026-08-02      -10.18      -1.61    +0.01   -11.77      13
2026-08-03       +3.98      -1.96    +3.62    +5.65      37
2026-08-04       -3.54      -1.78    +0.12    -5.20      92
2026-08-05      -13.25      -1.60    -0.10   -14.96      53
2026-08-06       -2.78      -3.33    +0.17    -5.93      43
2026-08-07       -1.03      -2.69    +0.21    -3.50     205
2026-08-08      -48.92      -2.08    +0.51   -50.49       6   <-- 60% of all losses
2026-08-09      +11.09      -1.73    -0.35    +9.02      43
2026-08-10       +5.52      -3.88    +0.08    +1.71     240
2026-08-11       -1.95      -3.89    +0.02    -5.82      93
2026-08-12       -3.16      -1.09    -0.07    -4.32      29
2026-08-13       +2.42      -0.40    +0.04    +2.07       5
                                            -84.59  = -1.72% of the account
```

**One day is 60% of the total loss, on six closing trades.** -8.15 per close against a
typical -0.06. This corrects #45, which claimed the loss was essentially all commission
-- that was read off a re-baselined reconciler covering only hours.

What happened on 08-08:

```
18:18:11  Closed existing SHORT position: 31,761 DOGEUSDT     (~2,220 USDT notional)
18:20:38  Failed to place/update stop-loss:
          Exchange.amount_to_precision() missing 1 required positional argument: 'amount'
18:20:52  (same)
18:21:07  (same)
18:21:21  (same)
18:21:30  EMERGENCY STOP
```

A position roughly **3.7x MAX_POSITION_PCT** ran with **no stop-loss**, because stop
placement raised a TypeError inside `except Exception` at main.py:837 and logged a
generic failure. Identical in shape to #38 -- a call that could never bind, swallowed --
one file over.

### The guard existed and did not cover it

`tests/test_exchange_contract.py` was written for exactly this after #38. It checked
`grid.py`, `trend_follower.py` and `router.py`. **It did not check `main.py`** -- the
module that owns stop-loss placement. Widening the parametrize list was the obvious fix
and it was not sufficient: injecting the 08-08 call shape still passed, because of

```python
real = getattr(Exchange, name, None)
if not callable(real):
    continue        # ccxt passthrough or helper
```

A call to a method that **does not exist on Exchange at all** is an AttributeError --
strictly worse than the TypeError the file was written to catch -- and this waved it
through silently. `amount_to_precision` lives on the ccxt object
(`exchange.exchange.…`), not on the wrapper, so `getattr` returned None and the check
skipped it.

Both are fixed: the matcher now also recognises main.py's bare `exchange.<method>(...)`
spelling, and a missing or non-callable attribute is a failure with the correction
suggested. All 27 methods currently called on the wrapper resolve correctly, so the
codebase is clean today -- it is the guard that was not.

Verified both ways: passes on current code, and injecting
`exchange.amount_to_precision(q)` into main.py fails with

```
main.py:609 exchange.amount_to_precision(...) does not exist on Exchange
-- AttributeError at runtime (did you mean exchange.exchange.amount_to_precision?)
```

### Fee assumption checked while here

A separate scare turned out to be my own arithmetic error. The `fee` column in
trades.csv is a ROUND-TRIP estimate, `(buy+sell) * qty * rate`; dividing it by a single
side's notional doubles it and makes maker look like taker. Binance's own user-trade
records over 7 days:

```
maker fills 882 (88.2%)   taker fills 118 (11.8%)
actual commission per side: median 0.0200%, mean 0.0224%
real round trip 0.0447% vs the 0.0400% the backtest models -- understated 1.12x
```

Fees are close to modelled. They are not the problem.

---

## 46. The grid is too narrow for the asset, and RECENTER_MARGIN_PCT is a dead knob

Following #45. The question was where 87% of the per-cycle capture goes. It is not fees.

### The mechanism, measured

`recenter()` has four triggers: price outside the margin band, `stranded` (one side
consumed and price outside the grid), `dead_inside`, and `deformed`. Counting which one
actually fires, over 1,570 recenters across four test cells:

```
                          recenters/90d   stranded  deformed  dead-inside  outside-band
current 10 / 2.5 earlier         112        74.5%     16.9%       8.7%          0%
wider    6 / 4.0 earlier          60        60.9%     33.0%       6.1%          0%
current 10 / 2.5 recent           57        67.9%     25.1%       7.0%          0%
wider    6 / 4.0 recent           29        42.4%     50.6%       7.0%          0%
```

**"price outside margin band" never fires -- not once.** `RECENTER_MARGIN_PCT` is wired
correctly (backtest.py:750, main.py:952) and is nonetheless inert: setting it to 2%, 5%
or 10% produced byte-identical PnL, cycle counts and recenter counts. One of the other
three conditions always fires first. It is a knob that looks meaningful and is not.

The dominant trigger is `stranded`: price runs out of the grid, consumes every level on
that side, and the ladder must be rebuilt. That is the chain:

  grid too narrow -> price blows through one side -> forced rebuild -> half-finished
  cycles stranded -> realised capture collapses to 13% of geometry

Widening halves the rebuild rate (112->60, 57->29) and triples per-cycle capture
(0.046->0.151, 0.117->0.419). Those two moving together is what makes it a mechanism
rather than a coincidence.

### The parameter surface

16 configurations, grid_count {6,8,10,12} x range_atr_multiplier {2.5,3.0,3.5,4.0},
two non-overlapping 90-day windows, 12 overlapping starts each. Totals summed over both
windows, then averaged along each axis:

```
range_atr_multiplier          grid_count
  2.5   mean 16.52  <- current    6   mean 40.88
  3.0   mean 48.88              8   mean 36.69
  3.5   mean 47.78             10   mean 51.76  <- current
  4.0   mean 39.92             12   mean 23.78
```

**The grid COUNT is already right at 10. The RANGE is too tight at 2.5**, which is the
worst of the four values by a factor of three, averaged across every count. The best
single cell (10 / 3.5 = 68.25) sits at the intersection of the best row and the best
column -- what a real effect looks like, rather than a spike.

### The honest caveat

**No individual paired comparison reaches significance.** Every |t| is below 2.1 and most
are below 1.0. The ranking is unstable cell to cell: 10/4.0 is the best config in the
earlier window (+30.00 paired) and among the worst in the recent one (-15.31); 8/4.0
flips sign entirely (+27.56 / -28.56). What carries weight here is the marginal
structure plus the independently measured mechanism, not any single number.

And the size of the prize is small:

```
current 10 / 2.5   +40.48 over 180 days   = +0.81% of 5000
best    10 / 3.5   +68.25 over 180 days   = +1.36% of 5000
```

Roughly doubling a very small number. This is a real improvement to a strategy that is
still noise-dominated -- the per-window standard error (12-26) remains comparable to the
means. It does not make the bot profitable in any meaningful sense; it makes it less
badly configured.

No config was changed. `RANGE_ATR_MULTIPLIER` is the user's call.

---

## 45. Why the bot is not profitable -- measurement, not a fix

#42 and #44 were both real defects and both are fixed. Neither makes the bot money, and
saying otherwise would be dishonest. This entry records what the numbers actually say.

Method: `run_backtest` on 180 days of real DOGE 1h data, split into two non-overlapping
90-day windows, 12 overlapping starts each, `STRATEGY_MODE=grid` with the trend filter.
Paired differences on identical windows.

### The centre-gap fix trades more, but does not earn more

```
EARLIER 90d              mean     sem   fills   paired diff vs BEFORE   sem     t
BEFORE  10 levels       -9.33   19.44  1049.3            --
AFTER   10 levels       +4.99   19.03  1244.9         +14.32          25.30   0.6
AFTER   14 levels      -45.34   20.07  1517.6         -36.01          30.00   1.2
AFTER   20 levels      -41.11   43.26  1830.9         -31.78          44.83   0.7

RECENT 90d
BEFORE  10 levels      +42.86   12.31   613.9            --
AFTER   10 levels      +35.49   11.68   721.7          -7.37          19.19   0.4
AFTER   14 levels      +36.53   21.56   959.5          -6.33          24.12   0.3
AFTER   20 levels       +7.93   23.99  1160.5         -34.93          27.79   1.3
```

Fills rise ~18-19% in both windows -- that part is consistent and mechanical. **The PnL
effect is not distinguishable from noise** (t=0.6 and t=0.4, and the two windows
disagree on sign). #44 is justified as a correctness fix, not as an edge.

### Tightening the spacing is measurably WORSE

The obvious inference from "0.382% spacing is 3.2x the 0.120% fee floor, so the same
range could hold 31 levels" is wrong. 14 and 20 levels both lose ground, and 20 levels
loses in *both* windows. The fee floor is a lower bound on viability, not a target.

### Where the money actually goes

```
                              EARLIER 90d    RECENT 90d
  completed cycles                  731.4         454.6
  winners                           91.2%         91.8%
  what the GRID thinks it made     407.94        295.91
  what was actually realized        33.70         53.39
  fees paid                        -28.71        -17.90
  net PnL                            4.99         35.49
                                  (+0.10%)      (+0.71%)   of 5000 over 90 days
```

Two things stand out.

**Fees consume most of the real gross.** 28.71 of 33.70 in the earlier window -- 85%.
17.90 of 53.39 in the recent one -- 34%. This is the direct explanation for why adding
levels loses money: the marginal cycle earns less than it costs to trade. It is not a
tuning oversight, it is the binding constraint.

**The engine's own PnL is inflated roughly twelvefold.** 407.94 booked against 33.70
realized; 0.56 per cycle claimed against 0.046 actually banked. Same root cause as #43 --
the grid credits each sell against the buy level it was paired with, Binance nets
everything at one blended average entry. 91% of cycles are "winners" by the ladder's own
arithmetic while the account is roughly flat. This is why the live status line reads
`net=12.42` next to `verified_net=-1.20`: the left number is fiction and always was.

Note the equity drawdown stays at 0.01-0.02%, and `ex.equity` does include unrealized
PnL, so this is not a hidden bag of inventory quietly bleeding. The account is genuinely
close to flat. There is no large loss to find -- there is barely any profit to begin
with.

### What this means

Over 180 days the configuration nets roughly +0.8% total, with a per-window standard
error larger than the mean. The strategy as configured is noise-dominated: the per-cycle
edge after fees is too thin for the fill rate it achieves. Trading *more* makes it worse;
that has now been measured twice.

Nothing here is a bug to fix. The remaining levers are structural -- fee tier, a wider
spacing with fewer but larger cycles, or a different instrument -- and each needs
measuring before it is believed, not after.

---

## 44. The price sat in the widest hole in the ladder, by construction -- HIGH

The 2026-08-13 16:58 run placed all ten orders cleanly, ran for 77 minutes, and filled
**nothing**. Not a dead level this time (#42 had already fixed that) -- the ladder was
healthy. The geometry was wrong.

```
0.06849  0.06875  0.06901  0.06928  0.06954  |  0.07006  0.07032  0.07059  0.07085  0.07111
   0.380%   0.378%   0.391%   0.375%   0.748%     0.371%   0.384%   0.368%   0.367%
                                        ^^^^^^
                          price traded 0.06973-0.06994 for 77 minutes, entirely inside here
```

Every gap is ~0.38% except the one straddling the price, which is 0.748% -- exactly
double. That is not bad luck, it is the construction:

```python
buy_prices  = linspace(lower, price, n, endpoint=False)   # step (price-lower)/n
sell_prices = linspace(price, upper, m+1)[1:]             # step (upper-price)/m
```

The last buy lands one FULL step below the price and the first sell one full step above
it, so the centre gap is always the sum of two half-ladders' steps -- 2x the spacing
everywhere else, for every `grid_count`. **The widest hole in the ladder is parked
permanently wherever the price is**, which is the one place a grid needs levels most. It
doubled the movement required before the bot could trade at all.

The fix offsets each side by half a step, so the price sits in the middle of one spacing
instead of two:

```python
buy_prices  = [price - buy_step  * (i + 0.5) for i in range(half_count)][::-1]
sell_prices = [price + sell_step * (i + 0.5) for i in range(sell_count)]
```

Rebuilt on the identical inputs, the centre gap becomes 0.373% against 0.377% elsewhere
-- 0.99x, and the nearest level moves from 0.37% away to 0.186%. Against the range the
price actually traded in those 77 minutes:

```
BEFORE  levels inside 0.06973-0.06994 : NONE        -> zero fills, as observed
AFTER   levels inside 0.06973-0.06994 : 0.06993     -> would have filled
```

Two smaller things fell out of writing the tests:

- The old construction put a level exactly at `grid_lower`, and tick-rounding then
  pushed it *below* the configured range (0.06849107 -> 0.06849). Every level now sits
  strictly inside. `test_every_level_stays_inside_the_grid` fails on the old code for
  all six grid counts.
- `grid_count` can shrink to 1 via tick-rounding dedup, which makes `half_count` zero
  and the new step arithmetic a division by zero. It falls back to the uniform ladder.

Reverting the construction fails 14 of the 21 tests in `tests/test_grid_geometry.py`.

### What this does NOT fix

Spacing *width* is a separate question from where the gap sits. At 0.382% the live grid
is 3.2x the 0.120% fee floor, and the same 3.82% range could hold 31 levels rather than
10. Whether tightening actually earns more -- or just pays more fees and hits
`MAX_POSITION_PCT` sooner -- is being measured separately; it is not assumed here.

---

## 43. The kill switch counted the grid's opinion, not the account -- HIGH

Recorded at the end of #42 as "still open", then fixed here. It needed one correction
first: I had said the daily-loss kill switch was fed the inflated number. **It was not.**
`main.py:1134` already passes `daily_realized_pnl=pnl_reconciler.daily_net_pnl`, the
exchange-verified figure, and has done since the reconciler landed. That check was fine.

What was still wrong is a *different* kill switch. `risk.record_trade(profit)` was fed
`fill["profit"]` -- the grid engine's per-level cycle estimate -- and that number drives
`state.consecutive_losses`, which `_check_consecutive_losses` kills on.

The estimate is not the account. The grid credits each sell against the particular buy
level it was paired with; Binance nets everything into one position at one blended
average entry. In the 2026-08-13 run, fills #24-#26 booked +2.37, +1.29 and +3.89
(=+7.55) while the reconciler moved -2.568140 -> -2.247978: a realized **+0.32**. Twenty-
three times out.

Magnitude is the lesser problem. **The estimate can carry the wrong sign, and it does so
exactly when it matters.** Buy 1,000 at 0.0690 and 1,000 at 0.0710 -- blended entry
0.0700. Sell 1,000 at 0.0695 paired with the 0.0690 level: the grid books +5, the account
realises -5. A falling market fills both levels, so this is not a contrived case, it is
the ordinary shape of a grid bleeding into a downtrend.

Which means a grid losing steadily in a trend reports a **run of wins**, and
`consecutive_losses` -- the switch whose entire purpose is "this strategy is repeatedly
wrong" -- never increments. It could not fire in the one situation it exists for.

`record_cycles(completed, verified_pnl, estimated_pnl)` replaces it. `main.py` captures
`net_realized_pnl` before the sync and passes the delta across the batch, so both the
losing streak and `state.daily_realized_pnl` now track the exchange. The grid's estimate
is kept for the log line, where the divergence stays visible instead of driving anything:

```
CYCLES RECORDED | 3 cycle(s) | verified pnl=+0.32 | grid estimated +7.55
                | daily_total=0.32 | trades_today=3 | consec_losses=0
```

One deliberate subtlety: a verified delta of **exactly 0.0** means Binance's income
ledger has not settled yet, not that the batch broke even. Treating that as a win would
clear a genuine losing streak on nothing but API lag, so the streak is *held* -- neither
reset nor incremented -- and the log says so. The cycle still counts toward
`trades_today`.

`verified_pnl=None` falls back to the estimate: worse than the account, better than
nothing, and it keeps the method usable without a reconciler.

Reverting the wiring fails five of the seven new tests in
`tests/test_killswitch_accounting.py`; the two that still pass cover the None-fallback
and the no-completed-cycles no-op, which the revert does not change. 442 pass.

---

## 42. The break-even guard went dormant instead of re-quoting -- HIGH

The 2026-08-13 09:11 run traded **seven times in seven hours**, six of them inside a
single 90-second burst at 10:45. Between 10:46 and 15:29 -- four hours and forty-three
minutes -- there was not one fill, and the log carried this line every fifteen seconds:

```
SKIP BUY @ 0.07034 | below break-even 0.07021909 on the open short - would book a
loss to close inventory the grid is meant to wait out (AUDIT #32)
```

Roughly 1,300 identical lines. The guard was mine, from #32, and its economics are
right: a short entered at 0.07024719 cannot be covered at 0.07034 without booking a
loss. What was wrong is what it did about it -- `return False`, and nothing else.
Nothing re-sited the level, nothing replaced it, so the ladder kept a permanent hole
exactly where trading happens: next to the price.

That hole was the entire strategy. After the 10:46 cascade the nearest sell was 0.07086,
1.2% above a market sitting at 0.0702, and the only level anywhere nearer was the
blocked one. The grid had nothing quoting within reach of the price for the rest of the
session. It was not waiting out a bad position -- it was not trading at all.

**Waiting out inventory does not require refusing to quote. It requires quoting at a
price that does not lose.** A blocked level now moves to break-even instead of dying:

```
MOVED BUY 0.07034 -> 0.0702 | the open short makes the original price a loss;
quoting at break-even instead of leaving the level dead (AUDIT #42)
```

Three constraints on the move, because #42 must not undo the fixes around it:

- **Never into a loss (#32).** The target is break-even, rounded *away* from the loss
  with `_round_price_toward` -- rounding to nearest crosses the line, which is #41.
- **Never across the book.** A buy above the market is post-only rejected and retried
  forever at debug level: the same dormancy, silent. The target is the stricter of "does
  not lose" and "is a valid maker price", so it actually rests.
- **Never onto a neighbour (#34).** If break-even sits inside the fee floor of another
  level, the level stays put rather than deforming the ladder.

When no legal price exists the level does stay idle -- but it says so **once**, not
1,300 times. A log that repeats itself every poll buries the fills between the repeats.

Reverting the change fails five of the six new tests in `tests/test_dormancy.py`.

### What this run does NOT show

The session was not a loss. `net_realized_pnl` went -3.2680 -> -1.1988: **+2.07 for the
day**, on 7 fills. The problem was never that the trades lost -- of the six completed
cycles, every one was positive. The problem is that six trades in seven hours is not a
grid, and the reason was one dead level.

### Still open: the internal PnL is roughly 6x reality

Worth recording because it is not fixed here. The status line reports both numbers and
they disagree badly:

```
gross=13.29 fees=0.87 net=12.42 | verified_net=-2.25 verified_daily=1.02
```

`risk:record_trade` accumulated `daily_total=11.04` while the exchange-verified figure
for the same moment was `1.02`. The grid credits each level's cycle against its own
paired entry, but Binance nets everything into one position at one blended entry, so
several levels claim profit against the same inventory -- fills #24, #25 and #26 booked
+2.37, +1.29 and +3.89 within one second while the account realised a fraction of it.

The `verified_*` figures are exchange truth and are correct. The concern is that
`risk.record_trade` is fed the inflated number, so the daily-loss kill switch is
measuring something that is not the account.

---

## 41. reconcile_positions had its own order path, outside every guard -- HIGH

Found by replaying the real `state/grid_dogeusdt.json` through the restore sequence
after switching to `STRATEGY_MODE=grid`. The account holds a **short of 9,916 DOGE at
0.07024719**, and #37 had just made the restore-with-position branch reachable for the
first time -- so this would have run on the next start.

Three defects, all in the same function:

**It covered the short at a loss.** The hedge price is one grid spacing in the
favourable direction *from the nearest level*, which is not the same as profitable. The
nearest sell level was 0.07059, so the cover landed at 0.07030 -- above the entry, and
covering a short above its entry is a loss. On 9,916 DOGE, **-0.52**. The AUDIT #32
break-even guard lives in `_place_order_for_level` and `_unwind_position_through_grid`;
this path reaches `place_limit_order` directly and never consulted it. The hedge is now
clamped to break-even on both sides.

**The cover was not reduceOnly.** `params = {...} if hedge_side == "sell" else None` --
set only when hedging a long. Covering a *short* went out as a plain buy, so if the
position had closed between reading it and placing the order, that opens a fresh 9,916
long instead of closing anything. Now reduceOnly on both sides.

**It placed reduce-only SELLs against a short.** The trailing orphan-sell loop fires for
any sell level carrying a quantity, regardless of what is actually open. A reduceOnly
sell can only reduce a *long*; with a short open all three were guaranteed -2022
rejections -- the same failure AUDIT #11 fixed elsewhere. Now gated on a long existing.

### And a rounding bug underneath it

The first fix did not work. Break-even for the cover is 0.07021909, which **rounds to
0.07022** at five decimals -- above break-even. The "safe" price was still a loss, by
0.0000009 per unit. Trivial per unit; on 9,916 DOGE it is the difference between a
winning exit and a losing one, and no amount of guard logic helps if the last step
rounds across the line.

`_round_price_toward(value, direction)` rounds to exchange precision without crossing
`value`, backing off by a doubling step until the rounded result lands on the safe side
(the tick size is not exposed, and a fixed epsilon is either too small to move a coarse
tick or needlessly wide on a fine one).

Verified against the real state, before and after:

```
short 9916 @ 0.07024719 | break-even cover = 0.07021909
before:  BUY 9916 @ 0.07030  reduceOnly=False   -> -0.52, and can open a long
after :  BUY 9916 @ 0.07021  reduceOnly=True    -> +0.09 vs break-even
plus 3 reduce-only sells against a short, all -2022, now not placed
```

Reverting all three fixes fails exactly the three new tests. 428 tests pass.

---

## 40. The flat override contradicted the log without explaining itself

From the 2026-08-13 01:49 run:

```
01:54:58 REGIME CHANGE (confirmed after 312s) | uncertain -> ranging | ADX=31.1
01:54:59 REGIME | 1h=downtrend(adx=31.1) 30m=downtrend(adx=45.2) 1d=uncertain(adx=27.6)
                | bands: range<=15 trend>=30 | needs 2 of 3 to agree -> ranging
```

Two timeframes agree on downtrend, the stated rule is "2 of 3 to agree", and the answer
is *ranging*. Read literally it is nonsense.

It was actually correct. `_merge_timeframes` did return DOWNTREND, and then
`_apply_flat_override` replaced it: DOGE's last 6 candles spanned under 1%, so ADX was
reading trend strength in a market that was not going anywhere. Range beats ADX, and
that is the right call -- a grid should keep trading a flat market whatever ADX says.

The defect was that the override only logged when the *previous* regime was already
trending, so in this case it fired silently. #33 added `explain()` precisely so a
regime could not be unexplainable; this was the same hole one layer down.

The override now always logs, and `explain()` appends the reason when it is active:

```
... | needs 2 of 3 to agree | FLAT OVERRIDE: last 6 candles span 0.74% <= 1.00%,
so a trending ADX reads as ranging
```

No behaviour changed -- only whether the log can be believed.

### What else that run showed

Three earlier fixes fired correctly in production for the first time:

- **#34** -- `RESET LEVELS | restored ladder is deformed (2 level pair(s) closer than the
  0.12% fee floor (tightest 0.04%)) — rebuilding it` at startup. The deformed ladder
  from the previous session was detected and rebuilt instead of traded.
- **#37** -- `STARTUP | saved grid state found — keeping any open position for the
  restored grid to unwind rather than closing it at market`. No market dump.
- **#35** -- shutdown logged `SHUTDOWN | cancelling all orders` and
  `TREND FOLLOWER STOPPED | shutdown` at INFO. No phantom ERROR lines.

And #32 earned its keep in the other direction: the 03:19 recenter unwound a 9,564 DOGE
**short** through buy levels at 0.06902-0.07007, all below the 0.07014576 entry -- which
for a short is profit, so the guard correctly allowed them. Both completed cycles that
followed were positive (+0.144847 and +0.994240).

Net over the 6.5-hour session: -0.0036 realised. Effectively flat, on 7 fills.

---

## 39. The kill switch measured a frozen equity, and routine probes tripped the breaker

Two defects in `exchange.py`, both found by the same sweep as #38 and verified by hand
after its verifiers died.

### The equity feeding the drawdown kill switch was fetched once

`get_balance_cached` and `get_total_equity_cached` shared a single `_balance_cache_time`.
Whichever ran first refreshed it for both, so the second saw "fresh" and returned its own
stale value. main.py calls them back to back:

```
main.py:990   balance = exchange.get_balance_cached()
main.py:991   equity  = exchange.get_total_equity_cached()
```

Replaying that exact pattern against the real caching code, with an account losing 25 per
iteration:

```
iter  true equity  equity used by risk       lag
   1      4875.00              4875.00      0.00
   4      4800.00              4875.00    -75.00
   7      4725.00              4875.00   -150.00

real fetches: get_balance=7  get_total_equity=1
```

Equity was fetched **once** and served from cache for the rest of the run, while free
balance updated every iteration. That `equity` is what `risk.check_all` uses for the
drawdown check -- so `MAX_DRAWDOWN_PCT` was being evaluated against a number that could
not fall, and the drawdown kill switch could not fire.

Each cache key now carries its own timestamp. Same replay after the fix: seven fetches,
zero lag.

### "Order does not exist" counted as a circuit-breaker failure

Introduced by #35's own fix. Downgrading the log was right; keeping
`record_failure()` was not. The exchange **answered** -- it just said the order is gone.

The breaker opens after 5 consecutive failures and then refuses **every** request for
120 seconds, stop-loss placement included. A recenter cancels every order and then checks
what it cancelled, which is five `-2013` replies in a row against a threshold of five.
The 22:05 log shows three of them back to back; two more and the bot would have gone
blind for two minutes while holding 10,456 DOGE.

`OrderNotFound`/`InvalidOrder` no longer record a failure. A test pins that genuine
`NetworkError`s still open the breaker -- #39 narrowed what counts, not whether.

---

## 38. The trend follower could never close a position -- CRITICAL

`Exchange.close_position` takes three required arguments:

```python
def close_position(self, symbol: str, side: str, amount: float, max_attempts=None) -> dict:
```

Two call sites passed one:

```
trend_follower.py:352   self.exchange.close_position(self.symbol)
router.py:302           self.exchange.close_position(self.symbol)
```

Both sit inside `except Exception`, so the `TypeError: missing 2 required positional
arguments` was caught, logged as a generic "close failed", and discarded.

Every exit the trend follower has runs through `_close_position`: the trailing stop, and
the regime-change exit. **Neither could ever fire.** And the early return happens before
`_side` is cleared, so the strategy stayed wedged believing it still held the position --
`place_initial_orders` returns 0 while `_side` is set, so it never exited and never
re-entered. One position, then permanently dead.

Reproduced against a stub carrying the real signature:

```
before      : side=long qty=7000  stop=0.06720
price -> 0.0600   (14% below entry, far past the stop)
close calls : []
fills       : []
side after  : long        qty after: 7000
regime -> downtrend, place_initial_orders() -> 0, side still long
```

The router's forced flatten was broken identically, so an expired grace period could
never resolve: it paused the outgoing strategy, failed to close, and deferred forever.

### Why 410 passing tests missed it

Every fake exchange in the suite -- and in `backtest.py` -- declared
`close_position(self, symbol)`. The doubles were **more permissive than the real class**,
so the tests proved the code worked against a signature production does not have. Same
shape as #31, where main.py called a method the router could not forward.

`tests/test_exchange_contract.py` now closes that hole structurally:

- every fake that defines `close_position` must accept what the real one requires;
- every `self.exchange.<method>(...)` call in grid.py, trend_follower.py and router.py is
  bound against the real `Exchange` signature, so an unbindable call fails the suite
  instead of hiding inside an `except Exception` in production.

Verified by binding the pre-fix call against the real class: `missing a required
argument: 'side'`. Both exits now work -- the trailing stop closes 7000 @ 0.0600 and
books -70.00, and the regime-change exit closes and clears the side.

Found by a six-lens agent sweep; four lenses reported it independently. It is the only
finding of that sweep that completed adversarial verification -- the rest lost their
verifiers to a session limit and remain unverified, neither confirmed nor refuted.

---

## 36. The refill deformed the ladder it was repairing -- HIGH

AUDIT #34 detected and rebuilt deformed ladders. This is what was deforming them.

`_refill_missing_grid_lines` re-adds levels after the duplicate merge, and it walked a
**uniform** template -- `grid_lower + i * grid_spacing` -- adding any template price not
already occupied. But `_initialize_dynamic` builds a deliberately **non-uniform** ladder,
concentrated near the price. The two never line up, so every template price lands a few
ticks off a real level and gets inserted beside it.

Reproduced exactly against the 2026-08-12 range:

```
uniform template : 0.06771 0.06800 0.06829 0.06858 0.06887 0.06917 0.06946 0.06975 0.07004 0.07033
actual ladder    : 0.06771 0.06797 0.06823 0.06850 0.06876 0.06928 0.06954 0.06981 0.07007 0.07033

0.06800 -> nearest existing 0.06797, 0.04% apart
0.06829 -> nearest existing 0.06823, 0.09% apart
```

Those are precisely the two sub-fee-floor pairs found in the saved state. Neither pair can
clear the 0.12% round-trip cost, so both levels were dead weight, while the middle of the
range -- where the price actually was -- stayed empty.

The refill now inserts at the midpoint of the ladder's **widest actual gap**, repeatedly,
and refuses to split a gap that would leave either neighbour closer than the fee floor.
When the widest remaining gap is too narrow, it stops and runs with fewer levels: a grid
with nine good levels beats one with ten where two can never profit.

Verified by replaying the real ladder minus its two merged duplicates -- the refill
reconstructs 0.06797 and 0.06981 exactly where they belong, and `ladder_defects` comes
back clean.

## 37. Startup market-dumped the bot's own inventory -- HIGH

`CLOSE_ON_EXIT` defaults to false, deliberately: a position left open at shutdown stays
under exchange-side stop protection so the grid can unwind it through its own levels on
the next run. Then startup did this, unconditionally, before reading anything:

```
STARTUP CLEANUP | cancelling all orders and closing orphan positions...
POSITION CLOSED | SELL 6274.0 DOGEUSDT
Closed 1 orphan positions from previous sessions
```

The state file was not read until 60 lines later. So the bot could not tell an orphan
from its own inventory, and housekeeping liquidated at market exactly what the previous
session had deliberately preserved. Measured on the 23:18 restart: 6274 DOGE closed at
market, verified PnL **-1.83 -> -3.26**. That single restart cost more than the entire
session before it.

It also made the restore-with-position branch unreachable: `has_exchange_positions` is
evaluated after cleanup, so it was always False, and `reconcile_positions()` never ran on
a real position.

The state file is now read first. Orders are still cancelled unconditionally -- untracked
resting orders from a dead session are genuinely dangerous and the grid re-places its own.
The *position* decision became conditional: with saved grid state, keep it and let the
restored ladder work it down; with no state to unwind it, it is a true orphan and still
gets closed.

Same principle as #32, one layer out: do not book a loss to tidy up.

---

## 35. Clean shutdowns logged as crashes -- LOW (but it hid everything else)

Every normal Ctrl+C ended like this:

```
00:04:28 | INFO  | Shutting down...
00:04:28 | ERROR | grid:emergency_stop | EMERGENCY STOP | cancelling all orders
00:04:34 | ERROR | trend_follower:emergency_stop | TREND FOLLOWER EMERGENCY STOP
00:04:34 | INFO  | Bot stopped. State saved.
```

Nothing was wrong. `emergency_stop()` is called from exactly two places -- the
kill-switch trip at main.py:1114, and the `finally:` block at main.py:1243 that runs on
every clean exit -- and it logged ERROR for both. So the routine path produced two red
lines that mean "this bot crashed".

The same miscalibration one layer down: `Exchange._retry` logged every non-retryable
exception at ERROR, including `OrderNotFound`. But "that order does not exist" is a
legitimate *answer* to a probe, not a fault -- `fetch_order` already catches it, logs at
debug, and returns None. The inner ERROR fired first and defeated that. Three of them
appeared right after the 22:04 recenter, which cancels everything and then checks what
it cancelled.

Both are now level-calibrated: `emergency_stop(reason=...)` picks INFO for `"shutdown"`
and keeps ERROR for the kill switch, and `OrderNotFound`/`InvalidOrder` drop to debug in
`_retry` while still raising exactly as before.

`reason` is presentational only, and a test enforces that -- it inspects the function
source and fails if `reason` gates anything but a logger call. A shutdown that logged
quietly while skipping the cancel would be far worse than one that shouts.

This is the lowest-severity entry in this file and it is here for a reason: for four
sessions the log's ERROR lines were mostly noise, which is exactly the condition under
which a real one -- `Loop error (consecutive=1): _last_orderbook`, 27 minutes of it --
gets read as normal.

---

## 34. The ladder stopped being a ladder -- HIGH

45 minutes of the 2026-08-12 23:18 run, zero fills. The restored grid, read back from
`state/grid_dogeusdt.json`, at a price of 0.06945:

```
0.06771  0.06797  0.06800  0.06823  0.06829  0.06850  0.06876   ...   0.06981  0.07007  0.07033
            \____/            \____/                          1.51% hole
            0.04%             0.09%                        (3.6x the spacing)
```

Seven buys crammed into the bottom of the range and three sells at the top, with
nothing at all where the price actually was. Two of those buy pairs sit 0.04% and 0.09%
apart -- below the 0.12% round-trip fee floor -- so neither pair could turn a profit
even if both legs filled.

**How much it cost, concretely.** Nominal spacing is 0.42%. On an even ladder the
nearest sell would have been ~0.42% above price. It was 0.52% away, because the hole
had pushed it out. DOGE travelled 0.446% during the run. That is a fill the bot should
have had and didn't.

The ladder deforms through ordinary operation: fills convert a level to the other side
and move it, replacements land on new prices, recentring rebuilds around a different
centre, and the duplicate merge plus `_refill_missing_grid_lines` re-add levels wherever
there is space. Each step is individually reasonable. The composition is not a grid.

Nothing looked for it. `recenter` has three triggers -- price outside the range, price
outside the margin band, and a one-sided grid with no fillable side -- and all three ask
*is a whole side missing*. This grid had seven buys and three sells and price sat
comfortably inside the range for the entire 45 minutes, so none of them fired.

`ladder_defects(price)` now names the two failure modes: adjacent levels closer than
the fee floor, and a gap straddling the price wider than 2x the nominal spacing. It is
consulted in two places:

- `reset_levels_to_pending` -- the startup path when the exchange reports no position.
  Restoring level *prices* is only worth doing while they still form a ladder; nothing
  is at risk on this path, so a deformed one is rebuilt instead of resurrected. Running
  it against the real state file above turns that ladder into an even one with the
  nearest levels 0.50% and 0.26% from price.
- `recenter` -- **only while flat**. Recentring cancels resting orders, and with
  inventory open those orders are the exits; firing this with a position open is the
  mistake that made recenter run 89 times in one session and stopped a position ever
  unwinding.

Per-level fill counts are discarded by a rebuild. They are statistics. A grid with a
hole where the price sits does not trade.

### Not a defect: the fill rate itself

Worth separating, because it looks like the same problem. Spacing is 0.42% by design --
`MIN_PROFIT_MULTIPLIER=3.0` requires every level to clear 3x the round-trip fee before
it may be placed. DOGE moved 0.446% across those 45 minutes. A handful of fills per day
is the arithmetic of that setting, not a fault, and the sweep in the activity section
above shows what tightening it costs: gc=10 gives 6.5 fills/day at +6.52, gc=28 gives
17 fills/day at -146.11. Activity and margin are the same dial.

---

## Where the bot stands after #29-#33

Full walk-forward at HEAD, driven from the live `.env` (`STRATEGY_MODE=router`,
`GRID_COUNT=10`, `MIN_PROFIT_MULTIPLIER=3.0`, `ADX 15/30`), 12 start offsets per
instrument, 90 days of 1h candles. DOGE is the instrument everything was developed
against; ETH and SOL are out of sample.

| instrument | mode | mean | sem | positive | fills | in position | max dd | stops | return on 5000 |
|---|---|---|---|---|---|---|---|---|---|
| DOGE | grid + filter | **40.00** | 17.30 | 8/12 | 491 | 83% | 1.27% | 18.4 | +0.80% |
| DOGE | router | 4.92 | 10.38 | 7/12 | 490 | 82% | 1.46% | 17.3 | +0.10% |
| ETH | grid + filter | 14.90 | 14.13 | 9/12 | 666 | 84% | 1.67% | 19.5 | +0.30% |
| ETH | router | **28.67** | 14.34 | 8/12 | 708 | 87% | 1.62% | 18.8 | +0.57% |
| SOL | grid + filter | 24.41 | 21.04 | 6/12 | 789 | 84% | 1.58% | 24.9 | +0.49% |
| SOL | router | **26.91** | 17.10 | 8/12 | 812 | 89% | 2.38% | 19.4 | +0.54% |

What this does and does not say:

- **All six are positive.** At the start of this sequence the router averaged -43 on
  DOGE and the grid alone -83 on the same data. That is the whole delta from #29-#33.
- **Router mode is defensible out of sample.** It loses badly to the filter on DOGE
  (+4.92 vs +40.00, the one instrument everything was tuned against) and wins on both
  instruments that were not (+28.67 vs +14.90, +26.91 vs +24.41). Positive-window
  counts are identical across all three: 23 of 36 each. Nothing here justifies changing
  `STRATEGY_MODE` in either direction.
- **The returns are small.** +0.10% to +0.80% of capital over 90 days -- roughly 0.4% to
  3.3% annualised, before any of the costs the harness does not model.
- **A single run can lose.** The standard error is the same size as the mean nearly
  everywhere, and about a third of the 36 windows per mode finished negative. "Positive
  on average across 12 overlapping windows" is not "will not lose over the next month".
- **It carries inventory almost all the time** (82-89% in position) and takes ~18-25
  stop-outs per 90 days. The equity path is shallow (max drawdown 1.3-2.4%) because
  position size is capped, not because it exits early.

Known limits of the harness, unchanged and all of which make these numbers
**optimistic**: no slippage or order-book depth, no partial fills, no funding, one
candle per loop iteration, and main.py's kill switches are not simulated.

---

## 32. The bot sold its own inventory at a loss to keep the ladder tidy -- HIGH

From the 2026-08-12 22:04 demo run. Price fell out of the grid, the ladder was rebuilt
lower, and the unwind spread the open long across the new sell levels:

```
22:04:34 RECENTERING GRID | price 0.06902 outside [0.06904107-0.07165893]
22:05:01 UNWIND | long 10456.0 reduce-only SELL @ 0.06928 (entry=0.06962068)
22:05:01 UNWIND | long 10456.0 reduce-only SELL @ 0.06954 (entry=0.06962068)
22:05:03 UNWIND | long 10456.0 reduce-only SELL @ 0.06981 (entry=0.06962068)
...
22:07:38 FILL #13 | SELL @ 0.06928 | profit=-0.712368
22:07:40 FILL #14 | SELL @ 0.06954 | profit=-0.168708
```

Two of the five exits were priced **below the position's own average entry**. Both
filled. -0.88 realised, against a session total of -1.83 -- roughly half the loss, and
entirely self-inflicted: the hard stop was at 0.06697, nowhere near, and price was back
above 0.0694 within four minutes.

`_unwind_position_through_grid` exists precisely to avoid realising a loss on recenter,
which is why the docstring says it beats market-closing. But it took `level.price`
without ever comparing it to `entry`. Recentring *downward* while holding a long moves
the whole exit ladder below cost, so the function did exactly what it was written to
prevent, one limit order at a time.

The engine's other profitability check does not catch this. `_is_level_profitable`
compares grid *spacing* against fees -- whether a round trip is worth doing at all. It
knows nothing about what the open position cost.

### The rule

Binance nets everything into a single position at one blended average entry (the same
truth AUDIT #7/#8 turned on). So a sell below that average realises a loss no matter
which level the engine has internally paired it with. Two guards now enforce that:

- `_unwind_position_through_grid` filters exit levels to those clearing break-even
  before slicing the position across them, so the position is spread over the levels
  that actually pay rather than the first N in the list.
- `_place_order_for_level` applies the same test, because the unwind is not the only
  route to the mistake: after a downward recenter the *ordinary* ladder has sell levels
  below cost too.

Break-even is the average entry plus one round trip of maker fees, read from the
exchange with a 2-second cache -- internal per-level bookkeeping is exactly the thing
that drifts, and this decides whether an order is allowed to lose money. An unreadable
position returns "no constraint", so an API failure cannot silently freeze the grid.

A grid is allowed to sit on inventory and wait; that is what its levels are for. What
it must not do is book a loss to tidy the ladder. If price never returns, the hard stop
-- not an exit ladder priced below cost -- is what closes the position.

### Measured

DOGE 1h, 90 days, 12 start offsets, current `.env`:

| mode | mean | sd | sem | positive | fills | max dd |
|---|---|---|---|---|---|---|
| grid + filter, sells below cost allowed | 28.64 | 44.17 | 12.75 | 8/12 | 514 | 1.29% |
| grid + filter, break-even guard | **40.00** | 59.93 | 17.30 | 8/12 | 491 | 1.27% |
| router, sells below cost allowed | 14.09 | 48.59 | 14.03 | 8/12 | 500 | 1.46% |
| router, break-even guard | 4.92 | 35.96 | 10.38 | 7/12 | 490 | 1.46% |

The guard helps grid mode by +11.36 and costs router mode -9.17. Both are well under
one standard error on overlapping samples, so **the backtest does not settle this
either way**; what it does show is that refusing ~4% of fills costs nothing in
drawdown (1.29% -> 1.27%, 1.46% -> 1.46%).

The change is kept regardless of that, and the reasoning is not statistical. Selling
below your own cost is not a parameter to tune -- it is the bot paying to make its
bookkeeping neat. The live evidence is one unambiguous event: -0.88 realised in four
minutes on a position that was never in danger. A 12-sample backtest is not grounds for
keeping a mechanism that does that.

### 33. "uncertain" was the normal state, and nothing said so

The same run logged `regime=uncertain` for its entire 90 minutes. Not a fault -- but the
router sends `uncertain` to the grid, so the trend follower was never once eligible, and
nothing in the log explained why.

Two gates have to pass, and both are strict. Each timeframe classifies as trending only
at `ADX >= 30` and ranging only at `ADX <= 15`; **everything between is `uncertain` by
definition**, and that band is half the usable ADX range. Then `_merge_timeframes`
requires two of the three timeframes (30m, 1h, 1d) to agree. Three independent readings
each having to clear the same band, twice over, makes `uncertain` the default outcome
rather than an edge case.

Nothing was changed about the thresholds -- the backtests below say more switching is
not obviously better, and retuning them on one symbol and period would be fitting noise.
What changed is that the state is now legible:

- `TrendFilter.explain()` reports each timeframe's regime, its ADX, the band edges, and
  the agreement rule.
- main.py logs that line every regime check, and the status line carries the live ADX:
  `regime=uncertain(adx=21.4)`.

So "why is the trend follower never running" is now answerable from the log rather than
from the source.

Also fixed here: `BUY SCALE` was logged on every iteration. The position cap moves with
equity, so an exact float comparison always differed; it now compares at the precision
the message prints.

Relaxing the trend threshold so the follower runs more often, same 12 offsets:

| adx_trend_threshold | mean | sd | sem | positive | fills | max dd |
|---|---|---|---|---|---|---|
| 30 (current) | 4.92 | 35.96 | 10.38 | 7/12 | 490 | 1.46% |
| 25 | **-51.75** | 64.70 | 18.68 | 3/12 | 455 | 2.33% |
| 20 | 10.72 | 23.92 | 6.91 | 8/12 | 334 | 1.14% |

30 -> 25 collapses, 25 -> 20 recovers past the start. A monotone knob does not behave
like that; this is the noise floor, measured. Reading a recommendation out of it would
be fitting one symbol and one 90-day window, so the thresholds stay where they are and
the state is merely made visible instead.

---

## 31. Router mode crashed every iteration on a private attribute -- CRITICAL

The 19:22 demo run placed its ten orders, logged `GRID ACTIVATED`, and then produced
this for twenty-seven minutes:

```
19:23:37 | ERROR | Loop error (consecutive=1): _last_orderbook
19:23:52 | ERROR | Loop error (consecutive=2): _last_orderbook
19:24:17 | ERROR | Loop error (consecutive=3): _last_orderbook
19:24:17 | WARNING | Multiple consecutive errors — attempting reconnection
19:24:34 | INFO  | RECONNECTED | exchange connection restored
```

`main.py` logged the spread with `grid._last_orderbook.get("spread_pct", 0)`. In router
mode `grid` is a `StrategyRouter`, whose `__getattr__` refuses to forward underscore
names -- forwarding them would let an internal typo silently read a strategy's
unrelated attribute. So the status log raised `AttributeError` on every iteration.

What it actually cost, precisely: the crash is in the *tail* of the loop, after fills,
stops, recentering and the state save. Trading logic ran. What did not run was the
`PRICE=... fills=... net=...` status line -- so the run was completely unobservable --
and `consecutive_errors = 0`, so the counter climbed to 3 every third iteration and
triggered a reconnect that could not possibly help. Each cycle burned 10-25s of a 10s
poll interval on reconnect and reconcile traffic.

Four separate defects made that possible, and all four are fixed.

**The private access itself.** `get_spread_pct()` and a `peak_price` property are now
public on both strategies and on the protocol, and the router delegates both
explicitly. (`isinstance` against a `runtime_checkable` Protocol uses
`inspect.getattr_static`, which never fires `__getattr__` -- a forwarding proxy only
satisfies the protocol for members its own class actually declares.)

**The test that was supposed to catch it, didn't.** The step-2 boundary test scanned
main.py for `grid.<member>` and then dropped the private ones:

```python
used = {u for u in used if not u.startswith("_")}   # private probes are not contract
```

They are exactly the contract that breaks. That line is gone, replaced by three checks
that fail on the real failure mode rather than a naming convention:

- no `grid._private` access in main.py at all;
- every member main.py touches must *resolve* on a real router with each strategy live;
- every call site main.py makes must *bind* against each strategy's signature.

The last one immediately found a second live crash nobody had hit yet:
`TrendFollower.log_sl_status(self)` against main.py's `grid.log_sl_status(side)` --
`TypeError` on every stop-status log once the follower took over.

**Writes were shadowing too.** `grid.total_fills = old_fills` and
`grid._peak_price = old_peak`, which main.py runs when it rebuilds the engine after
recovery, landed in the router's own `__dict__`. From that point every read finds the
stale shadow first and `__getattr__` is never consulted. The router now has
`__setattr__`, routing writes to the live strategy and keeping only its own bookkeeping
locally; read-only aggregate properties forward writes rather than raising.

**The handler hid all of it.** `logger.error("Loop error ...: {}", e)` prints an
`AttributeError` as the bare attribute name -- no type, no traceback, no line. Bug-class
exceptions (`AttributeError`, `TypeError`, `NameError`, `IndexError`,
`UnboundLocalError`, `ZeroDivisionError`, `AssertionError`) are now logged with type and
traceback, alert once over Telegram, and do **not** trigger the reconnect path, which
can never fix a code defect. There is deliberately no auto-shutdown on them:
`emergency_stop` cancels the resting stop-loss orders too, so killing the bot over a
defect that may sit in the tail of the iteration would leave an open position with
nothing protecting it. Loud and running beats silent and flat.

### Also fixed: the reset that quietly cost a grid line every restart

The same startup path logged this on the same run:

```
DEDUPLICATED 1 levels with duplicate price+side (10 -> 9)
RECOVERED grid state | refilled 1 empty grid line(s)
```

When the exchange reports no position at startup, main.py reverted each replaced sell
to a buy at its entry price -- and that price can already be occupied. The duplicate was
saved to the state file and only merged on the *next* start, which then rebuilt the lost
line from scratch, discarding its fill and cycle bookkeeping. That loop was also raw
`GridLevel` mutation living in main.py, which the router has no business forwarding.

It is now `GridEngine.reset_levels_to_pending()`, which merges and refills in the
session that created the duplicate. `TrendFollower` answers it with a no-op.

### Two more that the same audit turned up

Both were reachable in router mode and neither had been hit yet.

**Every price was "outside" the trend follower's range.** `grid_lower` and `grid_upper`
both returned 0.0, and main.py tests `price > grid.grid_upper` to detect price escaping
the ladder. That is true at every price, so while the follower was live and flat the
loop logged GRID EXIT and journalled an event once per iteration. `grid_upper` is now
`inf`: a strategy with no ladder is never outside it.

**A missing stop price crashed the stop refresh.** `build_scale_out_orders` computed
`abs(trail_price - hard_price)` without checking for None. A flat trend follower returns
None for both -- reachable when the exchange still reports a position the strategy has
already closed, or in the iteration right after a handoff. It now returns no orders and
warns, so the next iteration re-reads the position and places a stop if it is really
there. No stop for one iteration is recoverable; a TypeError every iteration is not.

Verified adversarially: reverting the private access fails 4 tests, reverting the
`log_sl_status` signature fails 2, and reverting either strategy fix fails its own.
390 tests pass.

---

## Aggression without dormancy: the router's two remaining defects (#29-#30)

Both found while chasing one instruction: *be profitable in all trends and never
dormant -- fix the strategy mode rather than turning it off*. Router mode was doing the
opposite of both, for two independent reasons.

### 30. The trend follower could never open a position live -- CRITICAL

`TrendFollower`'s only entry path was `place_initial_orders`. As #28 established,
main.py calls that exactly once, at startup, before the loop. Inside the loop the grid
re-places filled levels from within `check_fills`; nothing ever calls
`place_initial_orders` again.

So in live router mode the sequence was: trend confirmed -> handoff -> trend follower
activated -> **stands flat for the entire move**. The bot would have gone quiet in
exactly the conditions the router exists to trade -- the dormancy that prompted the
work, caused by the fix for it.

The backtester could not see this: its loop calls `strategy.place_initial_orders(balance)`
every candle, so the follower was driven by a path production does not have. That
divergence is now itself a known limit of the harness (below).

Two changes, both idempotent:

- `activate()` opens immediately if the regime already supports a side, matching
  `GridEngine.activate`, which places its ladder rather than waiting to be driven.
- `check_fills` -- called every iteration whenever a strategy is live -- arms the entry
  when there is no position and no resting order.

`place_initial_orders` already no-ops while a position or entry order exists, so being
driven from both paths in one iteration (as the backtester now does) still opens one
position. Verified adversarially: with the fix reverted, 2 of the 5 new tests fail.

### 29. The handoff market-dumped whatever the grid was holding -- HIGH

`_begin_handoff` paused the outgoing strategy and closed its position immediately, on
the reasoning that one-way position mode allows only one writer. The safety reasoning
was right; the timing was not. A grid accumulates inventory *expecting* to unwind it
through its own levels -- flattening it at market realises precisely the loss those
levels exist to avoid, and pays a taker fee for the privilege.

Instrumented over 90 days of DOGE 1h:

```
handoffs completed        : 44
...with an OPEN position  : 18
DOGE force-liquidated     : 64767
realised AT those dumps   : -46.16
router total net          : -92.78
```

Half the router's entire shortfall was the switching mechanism, not either strategy.
The split confirmed it: grid self-pnl +275.41 over 474 fills, trend self-pnl **+0.65**
over 56 fills and 28 cycles. The trend follower is roughly breakeven -- it was not the
thing losing money.

The switch now waits. `_begin_handoff` records the target and a clock; the outgoing
strategy stays live and keeps working its position down; the moment it is genuinely
flat the switch completes for free. Only after `ROUTER_HANDOFF_GRACE_SECONDS`
(default 21600 -- 6h) does the old force-close path run.

Three supporting details, each a bug in its own right if omitted:

- **The grace clock must not restart.** `_begin_handoff` is reached on every iteration
  while a switch is pending; re-stamping the start time would push the deadline away
  faster than time passes and the router would wait forever.
- **The waiting strategy may close but not open.** Left at its normal cap the grid
  keeps refilling the side it is meant to be working down and never reaches flat, so
  the grace expires into the forced dump anyway. During a pending handoff the router
  clamps `max_position_qty` to what is already open: exits still fill, nothing new
  does. It still gets to re-arm those exits -- `place_initial_orders` delegates to it
  while waiting, since refusing would strand the very inventory being unwound.
- **A pending switch is cancelled if the regime comes back.** Otherwise the grace clock
  keeps running against a switch nobody wants, and the next trend inherits a spent one
  and dumps instantly.

The safety property is untouched: the incoming strategy is activated only once flat is
confirmed against the exchange, and an unreadable position still counts as not flat.

### What it measured

DOGE 1h, 90 days, 12 start offsets, `ADX_RANGE_THRESHOLD=15`, everything else at the
current `.env`. `idle` is the share of candles with no strategy trading at all.

| mode | mean | sd | sem | positive | fills | switches | forced dumps | idle |
|---|---|---|---|---|---|---|---|---|
| grid only, no filter | -14.18 | 51.38 | 14.83 | 5/12 | 718 | 0 | 0 | 0% |
| grid + filter (pause) | **+21.79** | 39.56 | 11.42 | 7/12 | 508 | 0 | 0 | **30%** |
| router, grace=0 (old behaviour) | -33.64 | 31.54 | 9.10 | 2/12 | 514 | 40.2 | 15.8 | 0% |
| router, grace=2h | -31.89 | 31.75 | 9.17 | 2/12 | 515 | 38.5 | 14.4 | 0% |
| router, grace=6h | **-2.99** | 57.70 | 16.66 | 7/12 | 502 | 31.8 | 9.8 | **0%** |
| router, grace=12h | -43.53 | 59.57 | 17.20 | 2/12 | 506 | 29.3 | 6.6 | 0% |

What this does and does not establish:

- **The dump was costing real money.** grace=0 -> grace=6h is +30.65 mean and 2/12 ->
  7/12 positive, with forced dumps down from 15.8 to 9.8 per run. That is the same
  order as the -46.16 measured directly at the dump moments.
- **It is not statistically established.** The sem on that difference is roughly 19 on
  overlapping (non-independent) samples -- about 1.6 sigma. And the relationship is not
  monotonic: grace=12h has the *fewest* forced dumps (6.6) and the *worst* mean
  (-43.53), which is not what a clean causal story would look like. Waiting longer also
  means holding clamped inventory through more of a trend. 6h is a plausible middle,
  not a tuned optimum, and tuning it on this one series would be fitting noise.
- **The router still does not beat pausing.** grid+filter is +21.79 against the
  router's -2.99. What the router buys is the thing that was actually asked for:
  it is in the market ~100% of the time against grid+filter's ~70%, at a cost of
  roughly 25 in mean PnL over 90 days on this series -- itself inside the noise band.

So the router is no longer the clear loser it was (-43 against +48 on the earlier
6-offset run), and its remaining gap is within the noise floor. Continuous market
presence is now a defensible trade rather than an expensive one.


### Known limit this exposed

The backtest loop calls `place_initial_orders` every candle; main.py's loop never does.
That divergence hid #30 completely. Strategy code must therefore be self-arming from
`check_fills` and `activate` -- being driven by the backtester is not evidence that
production will drive it at all.

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
