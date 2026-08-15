"""Keep the bot running unattended, and know when not to. AUDIT #84.

The trading loop already survives almost everything thrown at it: a code defect logs
and keeps polling, -1021 resyncs the clock, three consecutive errors reconnect, and
state is written every poll so a hard kill costs ~15 seconds. What nothing covers is
the PROCESS dying -- an exception outside the loop, the machine rebooting, the
terminal closing, the OOM killer. Then the bot is simply gone, and it is gone quietly:
the position and its stops stay on the exchange with nothing tending them.

This runs main.py and puts it back when it dies. Two things matter more than the
restarting itself:

  * Restarting a trading bot is not free. Startup cancels every resting order and
    reconciles positions, so a crash loop churns the book. Restarts back off, and a
    bot that keeps dying trips a breaker and STAYS down with a loud alert -- a
    predictable stop beats an unpredictable loop.

  * A hung bot is worse than a dead one, because the supervisor cannot see it. The
    loop writes a line every poll, so a log that stops advancing means the process is
    alive and not trading. That is treated as a death.

Usage:
    py supervise.py                     # run until clean exit or the breaker trips
    py supervise.py --max-restarts 3    # trip sooner
    py supervise.py --dry-run           # print the decisions, start nothing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


CLEAN_EXIT = 0


@dataclass
class RestartPolicy:
    """Whether to restart, and how long to wait. Pure decision logic, no I/O.

    Separated from the process plumbing so the part that can strand you at 3am is the
    part that is actually tested.
    """

    max_restarts: int = 5
    window_seconds: float = 600.0
    base_backoff: float = 10.0
    max_backoff: float = 300.0
    _crashes: list[float] = field(default_factory=list)

    def record_crash(self, when: float) -> None:
        self._crashes.append(when)
        cutoff = when - self.window_seconds
        self._crashes = [t for t in self._crashes if t >= cutoff]

    def crashes_in_window(self) -> int:
        return len(self._crashes)

    def should_restart(self, exit_code: int | None) -> bool:
        """A clean exit is a decision someone made; honour it.

        Ctrl-C, a completed shutdown, an operator stopping the bot -- none of those
        should be undone by a supervisor.
        """
        if exit_code == CLEAN_EXIT:
            return False
        return self.crashes_in_window() <= self.max_restarts

    def tripped(self) -> bool:
        return self.crashes_in_window() > self.max_restarts

    def backoff(self) -> float:
        """Double the wait per consecutive crash, capped.

        Immediate restarts would have the bot cancelling and rebuilding its ladder
        every few seconds against an exchange that may be the reason it is failing.
        """
        n = max(self.crashes_in_window() - 1, 0)
        return min(self.base_backoff * (2 ** n), self.max_backoff)


def log_is_stale(log_dir: Path, limit_seconds: float, now: float) -> bool:
    """Has the newest bot log stopped advancing?

    The loop writes a PRICE= line every poll, so a log that has not been touched in
    many polls means the process is alive but not trading. With no log at all this
    says False: a bot that has not started yet is not a hung bot.
    """
    try:
        logs = list(Path(log_dir).glob("grid_*.log"))
    except OSError:
        return False
    if not logs:
        return False
    newest = max(l.stat().st_mtime for l in logs)
    return (now - newest) > limit_seconds


def run(cmd: list[str], policy: RestartPolicy, log_dir: Path,
        stale_after: float, dry_run: bool = False) -> int:
    attempt = 0
    while True:
        attempt += 1
        print(f"[supervise] start #{attempt}: {' '.join(cmd)}", flush=True)
        if dry_run:
            print("[supervise] dry run — not starting anything")
            return 0

        started = time.time()
        try:
            proc = subprocess.Popen(cmd)
        except OSError as e:
            print(f"[supervise] could not start: {e}", flush=True)
            return 1

        exit_code = None
        try:
            while True:
                try:
                    exit_code = proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    if log_is_stale(log_dir, stale_after, time.time()):
                        print(f"[supervise] log has not advanced in {stale_after:.0f}s "
                              f"— treating as hung, stopping it", flush=True)
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        exit_code = -1
                        break
        except KeyboardInterrupt:
            # Ctrl-C is the operator talking. Pass it down, wait, and stay down.
            print("\n[supervise] interrupted — shutting the bot down", flush=True)
            try:
                proc.terminate()
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
            return 0

        ran_for = time.time() - started
        print(f"[supervise] exited {exit_code} after {ran_for:.0f}s", flush=True)

        if not policy.should_restart(exit_code):
            if policy.tripped():
                print(f"[supervise] STOPPING — {policy.crashes_in_window()} crashes in "
                      f"{policy.window_seconds/60:.0f} min. Something is wrong that "
                      f"restarting will not fix. Any position and its stops are still "
                      f"on the exchange.", flush=True)
                _alert(f"Bot supervisor STOPPED after "
                       f"{policy.crashes_in_window()} crashes. Position and stops "
                       f"remain on the exchange and nothing is tending them.")
                return 1
            print("[supervise] clean exit — not restarting", flush=True)
            return 0

        policy.record_crash(time.time())
        if policy.tripped():
            continue                     # loop re-tests and reports the trip above
        wait = policy.backoff()
        print(f"[supervise] restarting in {wait:.0f}s "
              f"({policy.crashes_in_window()} crash(es) in window)", flush=True)
        time.sleep(wait)


def _alert(message: str) -> None:
    """Best effort. A supervisor that dies trying to send a message is useless."""
    try:
        from config import settings
        from telegram_notifier import TelegramNotifier
        TelegramNotifier(settings).send(f"<b>SUPERVISOR</b>\n{message}")
    except Exception as e:
        print(f"[supervise] alert failed: {e}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-restarts", type=int, default=5,
                    help="crashes allowed inside the window before stopping (default 5)")
    ap.add_argument("--window", type=float, default=600.0,
                    help="crash-counting window in seconds (default 600)")
    ap.add_argument("--stale-after", type=float, default=600.0,
                    help="treat the bot as hung if the log has not advanced in this "
                         "many seconds (default 600)")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    policy = RestartPolicy(max_restarts=a.max_restarts, window_seconds=a.window)
    return run([sys.executable, "main.py"], policy, Path(a.log_dir),
               a.stale_after, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
