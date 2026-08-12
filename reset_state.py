"""Clear saved bot state so the next run starts from a clean slate.

Stop the bot first, then run:  python reset_state.py

What gets discarded: the grid's levels and per-level entry-price bookkeeping, its
cumulative counters (total_pnl / total_fees / total_fills / total_completed_cycles),
the trailing stop-loss anchors, and the risk manager's baselines (starting_balance,
peak_balance, consecutive_losses, trades_today). Carrying these across a session is
what made a fresh start behave like a continuation: levels restored with entry prices
from a position that startup cleanup had already closed, so the first replacement
fills booked losses against entries that no longer existed.

The PnL reconciler is handled separately because it is not bot bookkeeping -- it is a
running total of Binance's own income ledger. Two choices:

  default          Re-baseline it to now, so reported PnL covers this run forward.
                   Nothing is fabricated; only the accounting epoch moves. Past
                   losses remain real and remain in Binance's ledger.

  --full-history   Delete it. On next start the reconciler re-bootstraps the full
                   89-day lookback, so it reports cumulative PnL across every
                   session in that window.

The previous state file is always backed up alongside itself before anything is
removed. This script never touches the exchange -- run cleanup.py for orders and
positions (main.py also does that automatically on startup).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from config import settings
from logger import setup_logging
from state import StateManager


def _summarize(state: dict) -> None:
    """Log what the outgoing state was carrying, so the reset is auditable."""
    grid = state.get("grid") or {}
    risk = state.get("risk") or {}
    rec = state.get("pnl_reconciler") or {}

    if grid:
        logger.info(
            "  discarding grid    | fills={} cycles={} local_pnl={:.2f} local_fees={:.2f} levels={}",
            grid.get("total_fills", 0), grid.get("total_completed_cycles", 0),
            float(grid.get("total_pnl", 0.0)), float(grid.get("total_fees", 0.0)),
            len(grid.get("levels") or []),
        )
        logger.info(
            "  discarding stops   | trail_long={} trail_short={} peak={} trough={}",
            grid.get("_trailing_sl_price"), grid.get("_trailing_sl_price_short"),
            grid.get("_peak_price"), grid.get("_trough_price"),
        )
    if risk:
        logger.info(
            "  discarding risk    | start_balance={} peak_balance={} consec_losses={} trades_today={}",
            risk.get("starting_balance"), risk.get("peak_balance"),
            risk.get("consecutive_losses"), risk.get("trades_today"),
        )
    if rec:
        net = (float(rec.get("realized_pnl", 0.0)) + float(rec.get("commission", 0.0))
               + float(rec.get("funding_fee", 0.0)))
        logger.info(
            "  reconciler (real)  | net={:.4f} (realized={:.4f} commission={:.4f} funding={:.4f})",
            net, float(rec.get("realized_pnl", 0.0)), float(rec.get("commission", 0.0)),
            float(rec.get("funding_fee", 0.0)),
        )


def _fresh_reconciler_state() -> dict:
    """A reconciler baselined at now: zeroed totals, cursor parked at the current
    time, and bootstrapped=True so sync() does the incremental path instead of
    re-pulling 89 days of history. Income logged from this moment on is counted."""
    now_ms = int(time.time() * 1000)
    return {
        "realized_pnl": 0.0,
        "commission": 0.0,
        "funding_fee": 0.0,
        "last_income_time_ms": now_ms,
        "last_seen_keys": [],
        "bootstrapped": True,
        "daily_net_pnl": 0.0,
        "daily_reset_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear saved grid bot state.")
    parser.add_argument(
        "--full-history", action="store_true",
        help="Also drop the PnL reconciler so it re-bootstraps Binance's full 89-day "
             "income history on next start (default: re-baseline it to now).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing anything.",
    )
    args = parser.parse_args()

    setup_logging(settings.log_dir, "INFO")
    state_mgr = StateManager(settings.state_dir, settings.symbol)
    path: Path = state_mgr.filepath

    if not path.exists():
        logger.info("RESET | no state file at {} - already clean", path)
        return

    try:
        state = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("RESET | could not parse existing state ({}) - backing it up and removing it", e)
        state = {}

    logger.info("RESET | current state at {}", path)
    _summarize(state)

    mode = "delete (reconciler re-bootstraps 89d history)" if args.full_history \
        else "re-baseline reconciler to now (PnL counts from this run forward)"
    logger.info("RESET | mode: {}", mode)

    if args.dry_run:
        logger.info("RESET | dry run - nothing written")
        return

    backup = path.with_suffix(f".bak.{int(time.time())}")
    shutil.copy2(path, backup)
    logger.info("RESET | previous state backed up to {}", backup)

    if args.full_history:
        state_mgr.delete()
        logger.info("RESET | state cleared - next start rebuilds the grid and re-pulls full income history")
    else:
        state_mgr.save({
            "pnl_reconciler": _fresh_reconciler_state(),
            "last_update": datetime.now().isoformat(),
        })
        logger.info("RESET | state cleared - next start rebuilds the grid; PnL counts from now")

    logger.info("RESET | run cleanup.py if the exchange still holds orders or a position")


if __name__ == "__main__":
    main()
