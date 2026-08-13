from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger


@dataclass
class RiskState:
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0
    peak_balance: float = 0.0
    starting_balance: float = 0.0
    last_kill_switch: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    last_reset_date: str = ""
    in_recovery: bool = False
    recovery_start_time: float = 0.0
    recovery_count: int = 0


class RiskManager:
    def __init__(
        self,
        stop_loss_pct: float = 0.03,
        daily_loss_limit_pct: float = 0.05,
        max_drawdown_pct: float = 0.10,
        cooldown_seconds: int = 3600,
        max_exposure_pct: float = 0.50,
        max_consecutive_losses: int = 10,
        max_recovery_count: int = 5,
        event_journal: object | None = None,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_seconds = cooldown_seconds
        self.max_exposure_pct = max_exposure_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_recovery_count = max_recovery_count
        self.state = RiskState()
        self._event_journal = event_journal

    def initialize(self, starting_balance: float) -> None:
        self.state.starting_balance = starting_balance
        self.state.peak_balance = starting_balance
        logger.info(
            "RISK MANAGER INIT | start_balance={:.2f} | SL={:.1%} | daily_limit={:.1%} | max_dd={:.1%} | max_exposure={:.0%}",
            starting_balance, self.stop_loss_pct, self.daily_loss_limit_pct, self.max_drawdown_pct, self.max_exposure_pct,
        )

    def check_all(
        self, current_balance: float, stop_loss_price: float, current_price: float,
        exposure_pct: float = 0.0, side: str = "long", daily_realized_pnl: float | None = None,
    ) -> tuple[bool, bool]:
        """Returns (is_safe, is_fatal). is_safe=False with is_fatal=True means kill switch.

        `side` must match the direction `stop_loss_price` was computed for ("long" or
        "short") so the breach direction is checked correctly -- see
        `_check_grid_stop_loss`.

        `daily_realized_pnl`, if given, overrides `state.daily_realized_pnl` for the
        daily-loss check only -- pass the exchange-reconciled figure (see
        pnl_tracker.PnLReconciler.daily_net_pnl) so the kill switch evaluates against
        the account's real daily P&L instead of the grid's per-level estimate, which
        can drift from Binance's blended-average accounting (see AUDIT.md). Leaves
        `state.daily_realized_pnl` itself untouched -- it still drives
        `consecutive_losses` via `record_trade()`.
        """
        self.state.peak_balance = max(self.state.peak_balance, current_balance)
        if not self._check_cooldown():
            return False, True
        if not self._check_drawdown(current_balance):
            return False, True
        if not self._check_daily_loss(current_balance, daily_realized_pnl):
            return False, True
        if not self._check_grid_stop_loss(stop_loss_price, current_price, side=side):
            return False, True
        if not self._check_consecutive_losses():
            return False, True
        if not self._check_exposure(exposure_pct):
            return False, False  # warning only, not fatal
        return True, False

    def _check_cooldown(self) -> bool:
        elapsed = time.time() - self.state.last_kill_switch
        if elapsed < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - elapsed)
            if self._event_journal:
                self._event_journal.risk_check("cooldown", elapsed, self.cooldown_seconds, "BLOCKED")
            logger.warning("COOLDOWN ACTIVE | {}s remaining", remaining)
            return False
        return True

    def _check_drawdown(self, current_balance: float) -> bool:
        if self.state.peak_balance <= 0:
            return True
        drawdown = (self.state.peak_balance - current_balance) / self.state.peak_balance
        if drawdown >= self.max_drawdown_pct:
            logger.error(
                "KILL SWITCH: drawdown {:.1%} exceeds limit {:.1%} | peak={:.2f} current={:.2f}",
                drawdown, self.max_drawdown_pct, self.state.peak_balance, current_balance,
            )
            if self._event_journal:
                self._event_journal.risk_check("drawdown", drawdown, self.max_drawdown_pct, "KILL")
                self._event_journal.risk_kill_switch("drawdown", drawdown, 0.0, self.state.peak_balance, current_balance)
            self.trigger_kill_switch()
            return False
        if self._event_journal:
            self._event_journal.risk_check("drawdown", drawdown, self.max_drawdown_pct, "OK")
        return True

    def _check_daily_loss(self, current_balance: float = 0.0, daily_realized_pnl: float | None = None) -> bool:
        realized = self.state.daily_realized_pnl if daily_realized_pnl is None else daily_realized_pnl
        total_daily = realized + self.state.daily_unrealized_pnl
        denominator = self.state.peak_balance if self.state.peak_balance > 0 else self.state.starting_balance
        if denominator <= 0:
            return True
        daily_loss_pct = abs(min(0, total_daily)) / denominator
        if daily_loss_pct >= self.daily_loss_limit_pct:
            logger.error(
                "KILL SWITCH: daily loss {:.1%} exceeds limit {:.1%} | pnl={:.2f}",
                daily_loss_pct, self.daily_loss_limit_pct, total_daily,
            )
            if self._event_journal:
                self._event_journal.risk_check("daily_loss", daily_loss_pct, self.daily_loss_limit_pct, "KILL")
                self._event_journal.risk_kill_switch("daily_loss", 0.0, daily_loss_pct, self.state.peak_balance, current_balance)
            self.trigger_kill_switch()
            return False
        if self._event_journal:
            self._event_journal.risk_check("daily_loss", daily_loss_pct, self.daily_loss_limit_pct, "OK")
        return True

    def _check_grid_stop_loss(self, stop_loss_price: float, current_price: float, side: str = "long") -> bool:
        """Checks the grid-level stop-loss backstop against the current position.

        Direction matters: a long's stop_loss_price is a floor (breached when price
        falls to or below it); a short's is a ceiling (breached when price rises to
        or above it). This used to always check the long/floor direction regardless
        of position side, so it was effectively blind whenever the grid was holding a
        short (price rising into danger never tripped `current_price <= stop_loss_price`).
        Pass stop_loss_price <= 0 (e.g. when flat) to skip this check entirely.
        """
        if stop_loss_price <= 0:
            return True
        breached = current_price >= stop_loss_price if side == "short" else current_price <= stop_loss_price
        if breached:
            logger.error(
                "KILL SWITCH: price {} {} stop loss {} (side={})",
                current_price, "above" if side == "short" else "below", stop_loss_price, side,
            )
            if self._event_journal:
                self._event_journal.risk_check("stop_loss", current_price, stop_loss_price, "KILL")
                self._event_journal.stop_loss_triggered("", stop_loss_price, current_price, 0.0, 0.0)
            self.trigger_kill_switch()
            return False
        return True

    def _check_exposure(self, exposure_pct: float) -> bool:
        if self.max_exposure_pct <= 0:
            return True
        if exposure_pct > self.max_exposure_pct:
            logger.warning(
                "EXPOSURE WARNING: {:.1%} exceeds limit {:.1%}",
                exposure_pct, self.max_exposure_pct,
            )
            if self._event_journal:
                self._event_journal.risk_check("exposure", exposure_pct, self.max_exposure_pct, "WARNING")
            return False
        if self._event_journal:
            self._event_journal.risk_check("exposure", exposure_pct, self.max_exposure_pct, "OK")
        return True

    def _check_consecutive_losses(self) -> bool:
        if self.max_consecutive_losses <= 0:
            return True
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            logger.error(
                "KILL SWITCH: {} consecutive losing fills exceeds limit {}",
                self.state.consecutive_losses, self.max_consecutive_losses,
            )
            if self._event_journal:
                self._event_journal.risk_check("consecutive_losses", self.state.consecutive_losses, self.max_consecutive_losses, "KILL")
                self._event_journal.risk_kill_switch("consecutive_losses", 0.0, 0.0, self.state.peak_balance, 0.0)
            self.trigger_kill_switch()
            return False
        if self._event_journal:
            self._event_journal.risk_check("consecutive_losses", self.state.consecutive_losses, self.max_consecutive_losses, "OK")
        return True

    def trigger_kill_switch(self) -> None:
        self.state.last_kill_switch = time.time()
        self.state.in_recovery = True
        self.state.recovery_start_time = time.time()
        self.state.recovery_count += 1
        logger.error("KILL SWITCH TRIGGERED | cooldown={}s | recovery_count={}", self.cooldown_seconds, self.state.recovery_count)

    def is_in_recovery(self) -> bool:
        return self.state.in_recovery

    def recovery_cooldown_remaining(self) -> int:
        if not self.state.in_recovery:
            return 0
        elapsed = time.time() - self.state.recovery_start_time
        backoff_multiplier = min(self.state.recovery_count, 4)
        effective_cooldown = self.cooldown_seconds * max(1, backoff_multiplier)
        remaining = int(effective_cooldown - elapsed)
        return max(0, remaining)

    def can_recover(self) -> bool:
        if not self.state.in_recovery:
            return False
        return self.recovery_cooldown_remaining() == 0

    def exit_recovery(self) -> None:
        logger.info(
            "RECOVERY COMPLETE | exiting recovery mode | consec_losses={}",
            self.state.consecutive_losses,
        )
        self.state.in_recovery = False
        self.state.recovery_start_time = 0.0
        self.state.consecutive_losses = 0

    def get_recovery_size_multiplier(self) -> float:
        if not self.state.in_recovery:
            return 1.0
        count = self.state.recovery_count
        if count >= self.max_recovery_count:
            logger.error(
                "MAX RECOVERY REACHED | {} attempts — bot will shut down",
                self.state.recovery_count,
            )
            return 0.0
        if count <= 1:
            return 0.65
        elif count == 2:
            return 0.40
        else:
            return 0.25

    def reset_daily(self) -> None:
        logger.info(
            "DAILY RESET | yesterday pnl={:.2f} | trades={} | consec_losses={}",
            self.state.daily_realized_pnl, self.state.trades_today, self.state.consecutive_losses,
        )
        self.state.daily_realized_pnl = 0.0
        self.state.daily_unrealized_pnl = 0.0
        self.state.trades_today = 0
        if not self.state.in_recovery:
            self.state.consecutive_losses = 0
        self.state.last_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def record_trade(self, pnl: float) -> None:
        self.state.daily_realized_pnl += pnl
        self.state.trades_today += 1
        if pnl >= 0:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
        logger.info(
            "TRADE RECORDED | pnl={:.2f} | daily_total={:.2f} | trades_today={} | consec_losses={}",
            pnl, self.state.daily_realized_pnl, self.state.trades_today, self.state.consecutive_losses,
        )

    def record_cycles(
        self, completed: int, verified_pnl: float | None, estimated_pnl: float,
    ) -> None:
        """Record a batch of completed grid cycles against the EXCHANGE's realized PnL.

        AUDIT #43. `record_trade()` was fed the grid engine's own per-level cycle
        estimate, and that estimate is not the account. The grid credits each sell
        against the particular buy level it is paired with; Binance nets everything into
        one position at one blended average entry. The two only agree by coincidence --
        in the 2026-08-13 run, fills #24-#26 booked +2.37, +1.29 and +3.89 (=+7.55)
        while the reconciler moved -2.568140 -> -2.247978, a realized **+0.32**.

        Magnitude is the lesser problem. The estimate can carry the wrong SIGN, and it
        does so exactly when it matters. Buy 1,000 at 0.0690 and 1,000 at 0.0710 and the
        blended entry is 0.0700; sell 1,000 at 0.0695 paired with the 0.0690 level and
        the grid books +5 while the account realises -5. A falling market fills both
        levels, so a grid bleeding into a downtrend reports a *run of wins*.

        `consecutive_losses` is the kill switch meant to catch "this strategy is
        repeatedly wrong". Feeding it the per-level estimate meant it could not fire in
        the one situation it exists for.

        A verified delta of exactly 0.0 means Binance's income ledger has not caught up
        yet, not that the batch broke even -- so the streak is left ALONE rather than
        reset. Guessing "win" there would clear a real losing streak on ledger lag.
        """
        if completed <= 0:
            return
        self.state.trades_today += completed
        pnl = estimated_pnl if verified_pnl is None else verified_pnl
        self.state.daily_realized_pnl += pnl

        if verified_pnl is None or verified_pnl != 0.0:
            if pnl >= 0:
                self.state.consecutive_losses = 0
            else:
                self.state.consecutive_losses += 1
            streak = str(self.state.consecutive_losses)
        else:
            streak = f"{self.state.consecutive_losses} (held: ledger not settled)"

        drift = ""
        if verified_pnl is not None and abs(estimated_pnl - verified_pnl) > 0.01:
            drift = f" | grid estimated {estimated_pnl:+.2f}"
        logger.info(
            "CYCLES RECORDED | {} cycle(s) | verified pnl={:+.2f}{} | daily_total={:.2f} "
            "| trades_today={} | consec_losses={}",
            completed, pnl, drift, self.state.daily_realized_pnl,
            self.state.trades_today, streak,
        )

    def update_unrealized(self, unrealized: float) -> None:
        self.state.daily_unrealized_pnl = unrealized

    def to_dict(self) -> dict:
        return {
            "daily_realized_pnl": self.state.daily_realized_pnl,
            "daily_unrealized_pnl": self.state.daily_unrealized_pnl,
            "peak_balance": self.state.peak_balance,
            "starting_balance": self.state.starting_balance,
            "last_kill_switch": self.state.last_kill_switch,
            "trades_today": self.state.trades_today,
            "consecutive_losses": self.state.consecutive_losses,
            "last_reset_date": self.state.last_reset_date,
            "in_recovery": self.state.in_recovery,
            "recovery_start_time": self.state.recovery_start_time,
            "recovery_count": self.state.recovery_count,
            "max_recovery_count": self.max_recovery_count,
        }

    def load_from_dict(self, data: dict) -> None:
        if not data:
            logger.warning("Empty risk state data, keeping current state")
            return
        valid_keys = {f.name for f in RiskState.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        self.state = RiskState(**filtered)
        self.max_recovery_count = data.get("max_recovery_count", self.max_recovery_count)
