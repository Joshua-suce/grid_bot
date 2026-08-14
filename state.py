from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from loguru import logger


class StateManager:
    """Persisted grid state, scoped to the ACCOUNT it belongs to.

    The filename used to be `grid_{symbol}.json` with no demo/live distinction, so
    flipping DEMO_MODE and restarting loaded one account's state against the other. The
    consequences are not subtle (AUDIT #67):

      - order ids from the other exchange, which check_fills reads as vanished orders
      - hard-stop ratchets anchored to the other account's prices
      - a PnL reconciler carrying the other account's totals, and a cursor far in the
        future of the new account's income, so real income before it is skipped forever
      - `has_saved_grid` True, which tells startup NOT to flatten a pre-existing
        position it knows nothing about (see AUDIT #37)

    Separate files per mode mean the two can never occupy one another's state.
    """

    def __init__(self, state_dir: str = "state", symbol: str = "BTCUSDT", *,
                 demo: bool):
        # `demo` is required and keyword-only on purpose. A default would let the fix
        # above be undone by omission -- a live caller that forgot the argument would
        # silently get the demo file, which is the exact bug (AUDIT #67/#71).
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.mode = "demo" if demo else "live"
        self.filepath = self.state_dir / f"grid_{symbol.lower()}_{self.mode}.json"
        self.consecutive_save_failures = 0

        # A file from before the rename cannot be attributed to either account, so it is
        # left alone rather than adopted or deleted -- but silence would look like a
        # clean start when real state exists on disk.
        legacy = self.state_dir / f"grid_{symbol.lower()}.json"
        if legacy.exists() and not self.filepath.exists():
            logger.warning(
                "LEGACY STATE IGNORED | {} predates per-account state files and cannot be "
                "attributed to demo or live — starting fresh for {}. Delete it, or rename "
                "it to {} if you know it belongs to this account",
                legacy, self.mode, self.filepath.name,
            )

    def has_history(self) -> bool:
        """Has this bot ever run against THIS account and symbol?

        Startup market-closes any position it finds when there is no saved grid to
        unwind it with, on the reasoning that it is an orphan of a dead session. That
        reasoning holds for a crash. It does not hold the first time the bot is pointed
        at an account, where a position it did not open belongs to whoever did.

        Backups, emptied files and corrupt files all count: they are proof of a previous
        session even though none of them can be restored (AUDIT #72).
        """
        stem = self.filepath.stem
        return any(
            p.name == self.filepath.name or p.name.startswith(f"{stem}.")
            for p in self.state_dir.glob(f"{stem}*")
        )

    def save(self, data: dict) -> bool:
        """Atomically persist state. Returns False if it could not be written.

        os.replace is atomic, but without an fsync first the rename can land while the
        data behind it is still in the page cache -- a power loss or hard kill then
        leaves a present-but-empty state file. load() copes with that (it renames the
        empty file aside and starts fresh), but starting fresh means losing the stop
        ratchets, which is #50b's failure mode arriving by a different road.

        The return value exists because a persistent save failure is otherwise invisible
        beyond one log line: the bot keeps trading and every restart silently reloads
        older state (AUDIT #57).
        """
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self.state_dir, suffix=".tmp", prefix=".state_",
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.filepath)
                if self.consecutive_save_failures:
                    logger.info(
                        "STATE SAVE RECOVERED | after {} consecutive failures",
                        self.consecutive_save_failures,
                    )
                self.consecutive_save_failures = 0
                return True
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            self.consecutive_save_failures += 1
            # Every caller discards the return value, so a disk that stopped accepting
            # writes would otherwise show up as one identical line per iteration and
            # nothing else. Escalate instead: the consequence is that a restart silently
            # reloads older state, including looser stops.
            if self.consecutive_save_failures in (1, 5) or self.consecutive_save_failures % 50 == 0:
                logger.error(
                    "STATE NOT PERSISTED | {} consecutive save failures — a restart will "
                    "reload STALE state (stop ratchets included): {}",
                    self.consecutive_save_failures, e,
                )
            return False

    def load(self) -> dict | None:
        if not self.filepath.exists():
            logger.info("No existing state file at {}", self.filepath)
            return None
        try:
            with open(self.filepath) as f:
                content = f.read()
            if not content.strip():
                logger.warning("State file is empty, treating as no state")
                self.filepath.rename(self.filepath.with_suffix(".empty"))
                return None
            data = json.loads(content)
            logger.info("State loaded from {}", self.filepath)
            return data
        except json.JSONDecodeError as e:
            logger.error("Corrupt state file: {} — backing up and starting fresh", e)
            backup = self.filepath.with_suffix(f".corrupt.{int(time.time())}.{os.getpid()}")
            try:
                self.filepath.rename(backup)
            except FileExistsError:
                backup = self.filepath.with_suffix(f".corrupt.{int(time.time()*1000)}.{os.getpid()}")
                self.filepath.rename(backup)
            logger.info("Corrupt state backed up to {}", backup)
            return None
        except Exception as e:
            logger.error("Failed to load state: {}", e)
            return None

    def delete(self) -> None:
        if self.filepath.exists():
            self.filepath.unlink()
            logger.info("State file deleted: {}", self.filepath)
