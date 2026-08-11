from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger


# Income types that make up "true" realized PnL on a futures account, per Binance's
# own income ledger. TRANSFER/WELCOME_BONUS/INSURANCE_CLEAR etc. are intentionally
# excluded — we only want types that reflect trading activity on this symbol.
PNL_INCOME_TYPES = {"REALIZED_PNL", "COMMISSION", "FUNDING_FEE"}

PAGE_LIMIT = 1000
MAX_PAGES_PER_CALL = 50
# Binance's income history endpoint only retains the last three calendar months
# of data regardless of what startTime is requested (per Binance API docs), so
# asking further back than that buys nothing — 89 days stays safely inside that
# window without relying on an exact 90/91/92-day boundary.
BOOTSTRAP_LOOKBACK_DAYS = 89


@dataclass
class PnLReconciler:
    """Tracks the bot's cumulative PnL using Binance's own income history as the
    source of truth, instead of the grid engine's internal per-level bookkeeping.

    The grid engine (grid.py) tracks each grid level's own independent entry_price,
    which can drift from Binance's single blended-average-entry accounting for the
    net position. This class periodically pulls the account's actual income ledger
    (realized PnL + commission + funding) for the traded symbol and sums it, giving
    a number that always agrees with the real account balance/equity trajectory.

    This does NOT touch order-placement or fill-handling logic — it's a read-only
    reconciliation layer that runs alongside the grid.
    """

    realized_pnl: float = 0.0
    commission: float = 0.0
    funding_fee: float = 0.0
    last_income_time_ms: int = 0
    # tranId+incomeType keys of every applied entry whose timestamp equals
    # last_income_time_ms. Binance frequently logs REALIZED_PNL and COMMISSION
    # for the same fill at the *identical* millisecond, so a plain "since
    # last_time + 1" cursor can silently skip whichever sibling entry didn't
    # happen to set the max timestamp. Re-fetching that boundary millisecond
    # inclusively and deduping against this set avoids both dropping and
    # double-counting entries at the boundary.
    last_seen_keys: set = field(default_factory=set)
    bootstrapped: bool = False
    last_sync_time: float = field(default=0.0, repr=False)
    # Same-day bucket, mirroring RiskManager.state.daily_realized_pnl but built
    # from the exchange's own income ledger instead of the grid's per-level
    # estimate — see AUDIT.md "Daily PnL is still unreconciled". Reset happens
    # both explicitly (rollover_daily(), called from main.py's daily_reset_check
    # alongside RiskManager.reset_daily()) and defensively on every sync() as a
    # safety net in case a UTC day rolls over without that call landing first.
    daily_net_pnl: float = 0.0
    daily_reset_date: str = ""

    @property
    def net_realized_pnl(self) -> float:
        """Realized PnL net of commissions and funding — the true cumulative PnL."""
        return self.realized_pnl + self.commission + self.funding_fee

    def to_dict(self) -> dict:
        return {
            "realized_pnl": self.realized_pnl,
            "commission": self.commission,
            "funding_fee": self.funding_fee,
            "last_income_time_ms": self.last_income_time_ms,
            "last_seen_keys": sorted(self.last_seen_keys),
            "bootstrapped": self.bootstrapped,
            "daily_net_pnl": self.daily_net_pnl,
            "daily_reset_date": self.daily_reset_date,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "PnLReconciler":
        if not d:
            return cls()
        return cls(
            realized_pnl=float(d.get("realized_pnl", 0.0)),
            commission=float(d.get("commission", 0.0)),
            funding_fee=float(d.get("funding_fee", 0.0)),
            last_income_time_ms=int(d.get("last_income_time_ms", 0)),
            last_seen_keys=set(d.get("last_seen_keys", []) or []),
            bootstrapped=bool(d.get("bootstrapped", False)),
            daily_net_pnl=float(d.get("daily_net_pnl", 0.0)),
            daily_reset_date=str(d.get("daily_reset_date", "")),
        )

    @staticmethod
    def _utc_today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _ensure_daily_bucket(self, today: str) -> None:
        """Zero the daily bucket the first time `today` differs from the stored
        `daily_reset_date`. Defensive/idempotent — safe to call every sync."""
        if self.daily_reset_date != today:
            self.daily_reset_date = today
            self.daily_net_pnl = 0.0

    def rollover_daily(self, today: str | None = None) -> float:
        """Snapshot the just-completed day's net PnL and reset the bucket for `today`.

        Call once per UTC day rollover (main.py's daily_reset_check, right
        alongside RiskManager.reset_daily()) so the "yesterday" figure reported
        in the daily summary is captured before it's zeroed. A no-op (returns
        the current running total without resetting) if `today` is already the
        active bucket's date -- callers can invoke this every loop iteration
        without worrying about double-resetting.
        """
        if today is None:
            today = self._utc_today()
        if self.daily_reset_date == today:
            return self.daily_net_pnl
        completed = self.daily_net_pnl
        self.daily_reset_date = today
        self.daily_net_pnl = 0.0
        return completed

    def _apply_entries(self, entries: list[dict]) -> int:
        """Sum new income entries into the running totals. Returns count applied.

        Entries are expected to come from a fetch starting at (and including)
        last_income_time_ms, so anything strictly before it is a re-fetch of
        already-applied history and is skipped; anything exactly at the
        boundary is deduped against last_seen_keys; everything newer advances
        the cursor.
        """
        applied = 0
        new_max_time = self.last_income_time_ms
        new_seen_at_max = set(self.last_seen_keys)
        today = self._utc_today()
        self._ensure_daily_bucket(today)
        for entry in sorted(entries, key=lambda e: int(e.get("time", 0) or 0)):
            income_type = entry.get("incomeType")
            if income_type not in PNL_INCOME_TYPES:
                continue
            try:
                amount = float(entry.get("income", 0.0))
                entry_time = int(entry.get("time", 0))
            except (TypeError, ValueError):
                continue
            tran_id = entry.get("tranId")
            key = f"{income_type}:{tran_id}" if tran_id is not None else f"{income_type}:{entry_time}:{amount}"
            if entry_time < self.last_income_time_ms:
                continue
            if entry_time == self.last_income_time_ms and key in self.last_seen_keys:
                continue
            if income_type == "REALIZED_PNL":
                self.realized_pnl += amount
            elif income_type == "COMMISSION":
                self.commission += amount
            elif income_type == "FUNDING_FEE":
                self.funding_fee += amount
            # Only count entries whose own timestamp falls on today (UTC) —
            # skips bootstrap backfill and anything synced late from a prior
            # day, matching what RiskManager.state.daily_realized_pnl was
            # meant to represent before it drifted (see AUDIT.md).
            entry_date = datetime.fromtimestamp(entry_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            if entry_date == today:
                self.daily_net_pnl += amount
            applied += 1
            if entry_time > new_max_time:
                new_max_time = entry_time
                new_seen_at_max = {key}
            elif entry_time == new_max_time:
                new_seen_at_max.add(key)
        self.last_income_time_ms = new_max_time
        self.last_seen_keys = new_seen_at_max
        return applied

    def _fetch_paginated(self, exchange, symbol: str, start_time_ms: int) -> list[dict]:
        """Page through income history from start_time_ms forward, deduplicating by
        (incomeType, tranId) since Binance's income endpoint is paged by time window
        with a max page size, not by cursor, so consecutive pages can overlap at the
        boundary timestamp. tranId is only unique *within* a given incomeType (per
        Binance's docs), so incomeType must be part of the key too — otherwise a
        REALIZED_PNL and a COMMISSION entry that happen to share a tranId would be
        treated as the same record and one would be silently dropped."""
        all_entries: list[dict] = []
        seen_keys: set = set()
        cursor = start_time_ms
        for _ in range(MAX_PAGES_PER_CALL):
            batch = exchange.get_income_history(symbol, since_ms=cursor, limit=PAGE_LIMIT)
            if not batch:
                break
            new_in_batch = 0
            max_time_in_batch = cursor
            for entry in batch:
                tran_id = entry.get("tranId")
                income_type = entry.get("incomeType")
                if tran_id is not None:
                    key = (income_type, tran_id)
                else:
                    key = (income_type, entry.get("time"), entry.get("income"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_entries.append(entry)
                new_in_batch += 1
                t = int(entry.get("time", 0))
                if t > max_time_in_batch:
                    max_time_in_batch = t
            if len(batch) < PAGE_LIMIT:
                break
            if max_time_in_batch <= cursor:
                # No forward progress — avoid an infinite loop.
                break
            cursor = max_time_in_batch
            if new_in_batch == 0:
                break
        return all_entries

    def bootstrap(self, exchange, symbol: str) -> None:
        """One-time full-history pull, run once on first startup (or if state is lost)."""
        start_time_ms = int(time.time() * 1000) - BOOTSTRAP_LOOKBACK_DAYS * 86400 * 1000
        try:
            entries = self._fetch_paginated(exchange, symbol, start_time_ms)
            applied = self._apply_entries(entries)
            self.bootstrapped = True
            self.last_sync_time = time.time()
            logger.info(
                "PNL RECONCILER BOOTSTRAP | {} income entries applied | realized={:.6f} commission={:.6f} funding={:.6f} net={:.6f} | daily_net={:.6f}",
                applied, self.realized_pnl, self.commission, self.funding_fee, self.net_realized_pnl, self.daily_net_pnl,
            )
        except Exception as e:
            logger.warning("PNL RECONCILER BOOTSTRAP FAILED | {} — will retry on next sync", e)

    def sync(self, exchange, symbol: str) -> bool:
        """Incrementally pull new income entries since the last known timestamp.
        Bootstraps automatically if this is the first run. Returns True on success."""
        if not self.bootstrapped:
            self.bootstrap(exchange, symbol)
            return self.bootstrapped
        try:
            # Inclusive of last_income_time_ms itself: _apply_entries dedupes
            # anything at that exact boundary timestamp against last_seen_keys,
            # which is what lets same-millisecond sibling entries (e.g. a
            # REALIZED_PNL and COMMISSION entry for one fill) both get counted
            # instead of one being silently skipped by an exclusive "+1" cursor.
            entries = self._fetch_paginated(exchange, symbol, self.last_income_time_ms)
            applied = self._apply_entries(entries)
            self.last_sync_time = time.time()
            if applied:
                logger.info(
                    "PNL RECONCILER SYNC | {} new income entries | net_realized_pnl={:.6f} | daily_net={:.6f}",
                    applied, self.net_realized_pnl, self.daily_net_pnl,
                )
            return True
        except Exception as e:
            logger.debug("PNL RECONCILER SYNC FAILED | {}", e)
            return False
