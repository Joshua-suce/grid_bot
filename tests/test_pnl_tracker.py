import time
from datetime import datetime, timezone

from pnl_tracker import PnLReconciler


def _entry(income_type: str, amount: float, time_ms: int, tran_id) -> dict:
    return {
        "incomeType": income_type,
        "income": str(amount),
        "time": time_ms,
        "tranId": tran_id,
        "asset": "USDT",
        "symbol": "DOGEUSDT",
    }


def _now_ms() -> int:
    return int(time.time() * 1000)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# A fixed instant well in the past -- real "now" will never fall on this UTC date,
# so it stands in for "not today" without needing to mock the clock. Only safe for
# tests that call sync()/_apply_entries() directly (cursor starts at 0); bootstrap()
# computes its own ~89-day lookback window, so bootstrap-path tests use
# RECENT_PAST_MS instead (still "not today", but inside that window).
OLD_TIME_MS = 1577836800000  # 2020-01-01T00:00:00Z
RECENT_PAST_MS = _now_ms() - 5 * 86400 * 1000  # 5 days ago


class _FakeExchange:
    """Stubs Exchange.get_income_history() with an in-memory list, no network."""

    def __init__(self, entries=None):
        self.entries = list(entries or [])
        self.calls: list[int] = []

    def get_income_history(self, symbol, since_ms=None, income_type=None, limit=1000):
        self.calls.append(since_ms or 0)
        batch = [e for e in self.entries if e.get("time", 0) >= (since_ms or 0)]
        batch.sort(key=lambda e: e["time"])
        return batch[:limit]


# --- basic accumulation -----------------------------------------------------

def test_apply_entries_sums_by_type():
    fake = _FakeExchange([
        _entry("REALIZED_PNL", 12.5, OLD_TIME_MS, 1),
        _entry("COMMISSION", -0.5, OLD_TIME_MS, 2),
        _entry("FUNDING_FEE", -0.1, OLD_TIME_MS + 1, 3),
    ])
    r = PnLReconciler(bootstrapped=True)
    assert r.sync(fake, "DOGE/USDT") is True
    assert r.realized_pnl == 12.5
    assert r.commission == -0.5
    assert r.funding_fee == -0.1
    assert r.net_realized_pnl == 11.9


def test_non_pnl_income_types_ignored():
    fake = _FakeExchange([
        _entry("REALIZED_PNL", 5.0, OLD_TIME_MS, 1),
        _entry("TRANSFER", 1000.0, OLD_TIME_MS, 2),
        _entry("WELCOME_BONUS", 10.0, OLD_TIME_MS, 3),
    ])
    r = PnLReconciler(bootstrapped=True)
    r.sync(fake, "DOGE/USDT")
    assert r.net_realized_pnl == 5.0


def test_bootstrap_sets_flag_and_applies_history():
    fake = _FakeExchange([_entry("REALIZED_PNL", 3.0, RECENT_PAST_MS, 1)])
    r = PnLReconciler()
    assert r.bootstrapped is False
    r.bootstrap(fake, "DOGE/USDT")
    assert r.bootstrapped is True
    assert r.realized_pnl == 3.0


def test_sync_bootstraps_automatically_on_first_call():
    fake = _FakeExchange([_entry("REALIZED_PNL", 2.0, RECENT_PAST_MS, 1)])
    r = PnLReconciler()
    assert r.sync(fake, "DOGE/USDT") is True
    assert r.bootstrapped is True
    assert r.realized_pnl == 2.0


# --- same-millisecond sibling entries (issue #7 follow-up fixes) -----------

def test_same_millisecond_siblings_both_counted_in_same_sync():
    fake = _FakeExchange([
        _entry("REALIZED_PNL", 10.0, 5000, 1),
        _entry("COMMISSION", -1.0, 5000, 2),
    ])
    r = PnLReconciler(bootstrapped=True)
    r.sync(fake, "DOGE/USDT")
    assert r.realized_pnl == 10.0
    assert r.commission == -1.0
    assert r.last_income_time_ms == 5000
    assert r.last_seen_keys == {"REALIZED_PNL:1", "COMMISSION:2"}


def test_same_millisecond_sibling_arriving_in_a_later_sync_is_not_dropped():
    fake = _FakeExchange([_entry("REALIZED_PNL", 10.0, 5000, 1)])
    r = PnLReconciler(bootstrapped=True)
    r.sync(fake, "DOGE/USDT")
    assert r.realized_pnl == 10.0

    # A sibling entry at the exact same boundary millisecond shows up on Binance's
    # side later (arrives on a subsequent sync call, not the same one).
    fake.entries.append(_entry("COMMISSION", -1.0, 5000, 2))
    r.sync(fake, "DOGE/USDT")
    assert r.commission == -1.0
    assert r.realized_pnl == 10.0  # not double-counted


def test_boundary_resync_does_not_double_count_already_applied_entry():
    fake = _FakeExchange([_entry("REALIZED_PNL", 10.0, 5000, 1)])
    r = PnLReconciler(bootstrapped=True)
    r.sync(fake, "DOGE/USDT")
    assert r.realized_pnl == 10.0

    # Nothing new arrives; an inclusive re-fetch of the boundary millisecond must
    # not re-apply the same entry a second time.
    r.sync(fake, "DOGE/USDT")
    assert r.realized_pnl == 10.0


def test_tranid_collision_across_income_types_both_counted():
    # tranId is only unique *within* one incomeType per Binance's docs -- a
    # REALIZED_PNL and a COMMISSION entry can legitimately share a raw tranId.
    fake = _FakeExchange([
        _entry("REALIZED_PNL", 8.0, OLD_TIME_MS, 42),
        _entry("COMMISSION", -2.0, OLD_TIME_MS, 42),
    ])
    r = PnLReconciler(bootstrapped=True)
    r.sync(fake, "DOGE/USDT")
    assert r.realized_pnl == 8.0
    assert r.commission == -2.0


# --- state round-trip --------------------------------------------------------

def test_to_dict_from_dict_round_trip():
    fake = _FakeExchange([
        _entry("REALIZED_PNL", 4.0, OLD_TIME_MS, 1),
        _entry("REALIZED_PNL", 6.0, _now_ms(), 2),
    ])
    r = PnLReconciler(bootstrapped=True)
    r.sync(fake, "DOGE/USDT")
    d = r.to_dict()
    assert d["last_seen_keys"] == sorted(d["last_seen_keys"])

    r2 = PnLReconciler.from_dict(d)
    assert r2.realized_pnl == r.realized_pnl
    assert r2.commission == r.commission
    assert r2.funding_fee == r.funding_fee
    assert r2.last_income_time_ms == r.last_income_time_ms
    assert r2.last_seen_keys == r.last_seen_keys
    assert r2.bootstrapped == r.bootstrapped
    assert r2.daily_net_pnl == r.daily_net_pnl
    assert r2.daily_reset_date == r.daily_reset_date


def test_from_dict_empty_returns_defaults():
    r = PnLReconciler.from_dict(None)
    assert r.net_realized_pnl == 0.0
    assert r.daily_net_pnl == 0.0
    assert r.daily_reset_date == ""
    assert r.bootstrapped is False


# --- daily bucket (AUDIT.md "Daily PnL is still unreconciled") -------------

def test_daily_net_pnl_only_counts_todays_entries():
    fake = _FakeExchange([
        _entry("REALIZED_PNL", 50.0, OLD_TIME_MS, 1),
        _entry("REALIZED_PNL", 7.0, _now_ms(), 2),
    ])
    r = PnLReconciler(bootstrapped=True)
    r.sync(fake, "DOGE/USDT")
    assert r.net_realized_pnl == 57.0
    assert r.daily_net_pnl == 7.0
    assert r.daily_reset_date == _today()


def test_daily_net_pnl_nets_all_three_income_types():
    now = _now_ms()
    fake = _FakeExchange([
        _entry("REALIZED_PNL", 10.0, now, 1),
        _entry("COMMISSION", -2.0, now, 2),
        _entry("FUNDING_FEE", 0.5, now, 3),
    ])
    r = PnLReconciler(bootstrapped=True)
    r.sync(fake, "DOGE/USDT")
    assert r.daily_net_pnl == 8.5


def test_bootstrap_backfill_does_not_pollute_daily_bucket():
    # A long lookback bootstrap must not attribute historical (non-today) income
    # to "today" just because it was fetched today.
    fake = _FakeExchange([_entry("REALIZED_PNL", 100.0, RECENT_PAST_MS, 1)])
    r = PnLReconciler()
    r.bootstrap(fake, "DOGE/USDT")
    assert r.realized_pnl == 100.0
    assert r.daily_net_pnl == 0.0


def test_rollover_daily_snapshots_and_resets():
    r = PnLReconciler(daily_net_pnl=25.0, daily_reset_date="2020-01-01")
    completed = r.rollover_daily("2020-01-02")
    assert completed == 25.0
    assert r.daily_net_pnl == 0.0
    assert r.daily_reset_date == "2020-01-02"


def test_rollover_daily_is_a_noop_on_the_same_day():
    r = PnLReconciler(daily_net_pnl=25.0, daily_reset_date="2020-01-02")
    completed = r.rollover_daily("2020-01-02")
    assert completed == 25.0
    assert r.daily_net_pnl == 25.0  # not wiped -- still the current day's running total


def test_sync_defensively_resets_a_stale_daily_bucket():
    # Simulates a UTC day boundary crossing without main.py's explicit
    # rollover_daily() call landing first (e.g. a long gap between syncs) --
    # the bucket must still self-correct rather than carry yesterday's total
    # (and a stale date) forward indefinitely.
    r = PnLReconciler(bootstrapped=True, daily_net_pnl=99.0, daily_reset_date="2020-01-01")
    fake = _FakeExchange([_entry("REALIZED_PNL", 4.0, _now_ms(), 1)])
    r.sync(fake, "DOGE/USDT")
    assert r.daily_reset_date == _today()
    assert r.daily_net_pnl == 4.0


def test_sync_with_no_new_entries_still_rolls_the_daily_bucket():
    # Funding-fee-less, fill-less days shouldn't leave daily_net_pnl stuck on
    # a prior date just because _apply_entries had nothing new to add.
    r = PnLReconciler(bootstrapped=True, daily_net_pnl=99.0, daily_reset_date="2020-01-01")
    fake = _FakeExchange([])
    r.sync(fake, "DOGE/USDT")
    assert r.daily_reset_date == _today()
    assert r.daily_net_pnl == 0.0


# --- rollover_daily vs. sync()'s defensive reset race ------------------------
#
# main.py calls daily_reset_check() -> rollover_daily() once at the very top of
# each loop iteration, and pnl_reconciler.sync() later in that same iteration
# (fill-triggered, or the 60-loop periodic fallback). If UTC midnight falls
# between those two calls, daily_reset_check() still sees "yesterday" and
# no-ops, while sync()'s _apply_entries -> _ensure_daily_bucket sees "today"
# and resets the bucket FIRST. Without recovery, the next rollover_daily() call
# would find daily_reset_date already at today's date, treat it as an ordinary
# no-op, and hand back whatever sliver has accumulated since the premature
# reset instead of the real completed-day total.

def test_rollover_daily_recovers_the_true_total_after_a_defensive_reset_races_ahead_of_it():
    r = PnLReconciler(bootstrapped=True, daily_net_pnl=25.0, daily_reset_date="2020-01-01")
    fake = _FakeExchange([_entry("REALIZED_PNL", 3.0, _now_ms(), 1)])

    # sync() races ahead: its defensive _ensure_daily_bucket sees "today" and
    # resets before rollover_daily() has been explicitly called for today at all.
    r.sync(fake, "DOGE/USDT")
    assert r.daily_reset_date == _today()
    assert r.daily_net_pnl == 3.0          # the bucket now holds only today's fresh entry

    # daily_reset_check()'s rollover_daily() now runs for the first time today --
    # it must recover 2020-01-01's real total (25.0), not misreport today's 3.0.
    completed = r.rollover_daily(_today())
    assert completed == 25.0
    # The recovery reads the snapshot; it must not itself disturb the live bucket.
    assert r.daily_net_pnl == 3.0
    assert r.daily_reset_date == _today()


def test_rollover_daily_after_recovering_the_race_behaves_as_an_ordinary_noop():
    r = PnLReconciler(bootstrapped=True, daily_net_pnl=25.0, daily_reset_date="2020-01-01")
    fake = _FakeExchange([_entry("REALIZED_PNL", 3.0, _now_ms(), 1)])
    r.sync(fake, "DOGE/USDT")
    r.rollover_daily(_today())             # first call for today: recovers 25.0

    # A second call the same day is the documented ordinary no-op again: the
    # live running total, not the stale recovered snapshot.
    assert r.rollover_daily(_today()) == 3.0
    assert r.daily_net_pnl == 3.0           # still not reset


def test_rollover_daily_with_no_race_is_unaffected_by_the_recovery_path():
    """The ordinary (non-racing) path -- rollover_daily() itself performs the
    first flip for the day -- must produce identical results to before this fix."""
    r = PnLReconciler(daily_net_pnl=25.0, daily_reset_date="2020-01-01")
    completed = r.rollover_daily("2020-01-02")
    assert completed == 25.0
    assert r.daily_net_pnl == 0.0
    assert r.daily_reset_date == "2020-01-02"
    # Ordinary same-day no-op still returns the live running total.
    r.daily_net_pnl = 7.0
    assert r.rollover_daily("2020-01-02") == 7.0


def test_last_completed_daily_pnl_round_trips_through_to_dict():
    r = PnLReconciler(bootstrapped=True, daily_net_pnl=25.0, daily_reset_date="2020-01-01")
    fake = _FakeExchange([_entry("REALIZED_PNL", 3.0, _now_ms(), 1)])
    r.sync(fake, "DOGE/USDT")               # races the defensive reset, sets the pending flag
    d = r.to_dict()
    assert d["last_completed_daily_pnl"] == 25.0
    assert d["_daily_bucket_reset_pending"] is True

    r2 = PnLReconciler.from_dict(d)
    assert r2.last_completed_daily_pnl == 25.0
    assert r2._daily_bucket_reset_pending is True
    # And the restored object still recovers the race exactly like the original would.
    assert r2.rollover_daily(_today()) == 25.0


def test_a_freshly_restored_object_with_no_pending_flag_is_an_ordinary_noop():
    """Loading state where daily_reset_date already equals today (e.g. a restart
    shortly after a previous session's own rollover_daily() already ran, or state
    persisted mid-day with no race involved) must behave as the documented
    ordinary no-op -- NOT be mistaken for a recoverable race just because the
    date happens to already match."""
    r = PnLReconciler.from_dict({
        "daily_net_pnl": 25.0, "daily_reset_date": "2020-01-02",
        "last_completed_daily_pnl": 999.0,  # stale/irrelevant -- must not be returned
    })
    assert r.rollover_daily("2020-01-02") == 25.0
