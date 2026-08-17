"""One exposure budget shared by every bot process on the same account. AUDIT #97.

`max_exposure_pct` is enforced inside a single process against a single symbol. Every
piece of that check is per-process: RiskManager holds the limit, GridEngine.get_exposure_pct
reads positions for ITS symbol only, and risk.check_all compares the two. Run the bot on
DOGEUSDT and again on SOLUSDT against the same Binance account -- which is the obvious way
to trade more, since the ladder used 2.1% of a 50% cap on 2026-08-17 -- and each process
independently permits 50%. The account can reach 100% with both halves reporting healthy,
and the second instance is invisible to the first at exactly the moment it matters.

The registry is a file per account-mode in the state directory. Each process publishes its
own symbol's exposure every poll and reads back what the others published, so the cap is
evaluated against the account rather than against one ladder.

Two deliberate properties:

  * A process always counts its OWN exposure from its own live figure, never from the
    file. Read-modify-write between processes can drop a concurrent update, and dropping
    your own would under-report the account -- the unsafe direction. Others' entries can
    be lost, but every process rewrites its entry every poll, so a lost one is restored
    within a single interval and the worst case is one poll of under-counting.

  * Entries go stale. A process that dies leaves its last exposure behind, and a dead
    process holds no position; treating that file forever would wedge the survivors at a
    cap they cannot get under. Anything older than `stale_after` is ignored and swept.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from loguru import logger

# Comfortably more than a poll interval (10s configured, and the loop can overrun on a
# slow venue) and comfortably less than the time it takes a human to notice a dead bot.
STALE_AFTER_SECONDS = 90.0


class ExposureRegistry:
    """Cross-process view of how much of the account is committed."""

    def __init__(self, state_dir: str | Path = "state", *, demo: bool, symbol: str,
                 stale_after: float = STALE_AFTER_SECONDS) -> None:
        self.mode = "demo" if demo else "live"
        self.symbol = symbol.upper()
        self.stale_after = stale_after
        self.state_dir = Path(state_dir)
        # Demo and live are different accounts with different balances; sharing one budget
        # file between them would have a paper position throttle a real one (AUDIT #67).
        self.filepath = self.state_dir / f"exposure_{self.mode}.json"
        self._degraded = False

    # --- reading -----------------------------------------------------------------

    def _load(self) -> dict[str, dict]:
        try:
            with open(self.filepath, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError, ValueError) as e:
            # A torn write from a concurrent process, or a corrupt file. Treating it as
            # empty costs one poll of other-process visibility; refusing to trade over it
            # would be a worse failure for a file that is rebuilt every interval anyway.
            logger.debug("EXPOSURE REGISTRY | unreadable ({}) — treating as empty", e)
            return {}
        return data if isinstance(data, dict) else {}

    def others(self, now: float | None = None) -> dict[str, dict]:
        """Fresh entries published by OTHER symbols on this account."""
        now = time.time() if now is None else now
        out = {}
        for symbol, entry in self._load().items():
            if symbol == self.symbol or not isinstance(entry, dict):
                continue
            at = float(entry.get("at", 0.0) or 0.0)
            if now - at > self.stale_after:
                continue
            out[symbol] = entry
        return out

    # --- writing -----------------------------------------------------------------

    def publish(self, exposure_pct: float, notional_usdt: float = 0.0,
                now: float | None = None) -> None:
        """Record this process's exposure and sweep entries whose process is gone."""
        now = time.time() if now is None else now
        data = self._load()
        data[self.symbol] = {
            "pct": float(exposure_pct),
            "notional": float(notional_usdt),
            "pid": os.getpid(),
            "at": now,
        }
        for symbol in [s for s, e in data.items()
                       if s != self.symbol
                       and (not isinstance(e, dict)
                            or now - float(e.get("at", 0.0) or 0.0) > self.stale_after)]:
            del data[symbol]

        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            # Atomic swap, so a reader never sees a half-written budget.
            fd, tmp = tempfile.mkstemp(dir=str(self.state_dir), prefix=".exposure", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                os.replace(tmp, self.filepath)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            self._degraded = False
        except OSError as e:
            if not self._degraded:
                # Once, not every poll: this runs on the hot path.
                logger.warning(
                    "EXPOSURE REGISTRY | cannot write {} ({}) — the cap falls back to this "
                    "process's own exposure, which is what it was before the registry "
                    "existed. Another instance on this account would be invisible.",
                    self.filepath, e,
                )
                self._degraded = True

    # --- the number the risk check should use -------------------------------------

    def account_exposure_pct(self, own_pct: float, now: float | None = None) -> float:
        """Total committed across the account: this process's live figure plus the others'.

        `own_pct` is passed in rather than read back so a process can never under-report
        itself through a lost write.
        """
        return float(own_pct) + sum(float(e.get("pct", 0.0) or 0.0)
                                    for e in self.others(now).values())

    def describe_others(self, now: float | None = None) -> str:
        """Short 'SOLUSDT 12.4%, ETHUSDT 3.1%' for logging, or '' when alone."""
        others = self.others(now)
        return ", ".join(f"{s} {float(e.get('pct', 0.0) or 0.0):.1%}"
                         for s, e in sorted(others.items()))
