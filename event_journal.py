from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


class EventJournal:
    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / "events.jsonl"

    def _emit(self, event: str, **data: object) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        try:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error("Event journal write failed: {}", e)

    def fill(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        fee: float,
        cycle_pnl: float,
        completed_cycle: bool,
        fill_number: int,
        balance: float,
        equity: float,
        exposure_pct: float,
        daily_pnl: float,
        regime: str,
    ) -> None:
        self._emit(
            "fill",
            symbol=symbol, side=side, price=price, qty=qty, fee=fee,
            cycle_pnl=cycle_pnl, completed_cycle=completed_cycle,
            fill_number=fill_number, balance=balance, equity=equity,
            exposure_pct=exposure_pct, daily_pnl=daily_pnl, regime=regime,
        )

    def order_placed(self, symbol: str, side: str, price: float, qty: float, order_id: str) -> None:
        self._emit("order_placed", symbol=symbol, side=side, price=price, qty=qty, order_id=order_id)

    def order_failed(self, symbol: str, side: str, price: float, qty: float, error: str) -> None:
        self._emit("order_failed", symbol=symbol, side=side, price=price, qty=qty, error=error)

    def order_cancelled(self, symbol: str, side: str, price: float, order_id: str, reason: str) -> None:
        self._emit("order_cancelled", symbol=symbol, side=side, price=price, order_id=order_id, reason=reason)

    def grid_exit(self, symbol: str, reason: str, current_price: float, positions_held: float) -> None:
        self._emit("grid_exit", symbol=symbol, reason=reason, current_price=current_price, positions_held=positions_held)

    def grid_recalculated(
        self, symbol: str, old_lower: float, old_upper: float, new_lower: float,
        new_upper: float, grid_count: int, reason: str, atr: float,
    ) -> None:
        self._emit(
            "grid_recalculated",
            symbol=symbol, old_lower=old_lower, old_upper=old_upper,
            new_lower=new_lower, new_upper=new_upper, grid_count=grid_count,
            reason=reason, atr=atr,
        )

    def grid_recentered(self, symbol: str, old_lower: float, old_upper: float, new_lower: float, new_upper: float) -> None:
        self._emit("grid_recentered", symbol=symbol, old_lower=old_lower, old_upper=old_upper, new_lower=new_lower, new_upper=new_upper)

    def position_snapshot(
        self, symbol: str, side: str, entry_price: float, qty: float,
        current_price: float, unrealized_pnl: float,
    ) -> None:
        self._emit(
            "position_snapshot",
            symbol=symbol, side=side, entry_price=entry_price, qty=qty,
            current_price=current_price, unrealized_pnl=unrealized_pnl,
        )

    def balance_snapshot(self, free: float, used: float, total_equity: float, exposure_pct: float) -> None:
        self._emit("balance_snapshot", free=free, used=used, total_equity=total_equity, exposure_pct=exposure_pct)

    def risk_check(self, check_type: str, value: float, threshold: float, result: str) -> None:
        self._emit("risk_event", check_type=check_type, value=value, threshold=threshold, result=result)

    def risk_kill_switch(self, reason: str, drawdown: float, daily_loss: float, peak_balance: float, current_balance: float) -> None:
        self._emit(
            "risk_kill_switch", reason=reason, drawdown=drawdown,
            daily_loss=daily_loss, peak_balance=peak_balance, current_balance=current_balance,
        )

    def stop_loss_triggered(self, symbol: str, sl_price: float, current_price: float, qty: float, unrealized_loss: float) -> None:
        self._emit(
            "stop_loss_triggered", symbol=symbol, sl_price=sl_price,
            current_price=current_price, qty=qty, unrealized_loss=unrealized_loss,
        )

    def trend_change(self, old_regime: str, new_regime: str, adx: float, timeframe: str) -> None:
        self._emit("trend_change", old_regime=old_regime, new_regime=new_regime, adx=adx, timeframe=timeframe)

    def recovery_event(self, phase: str, recovery_count: int, cooldown_remaining: int = 0, sizing_pct: float = 0.0) -> None:
        self._emit("recovery_event", phase=phase, recovery_count=recovery_count, cooldown_remaining=cooldown_remaining, sizing_pct=sizing_pct)

    def exposure_update(self, exposure_pct: float, equity: float) -> None:
        self._emit("exposure_update", exposure_pct=exposure_pct, equity=equity)

    def daily_reset(self, yesterday_pnl: float, trades_today: int, balance: float) -> None:
        self._emit("daily_reset", yesterday_pnl=yesterday_pnl, trades_today=trades_today, balance=balance)

    def grid_paused(self, symbol: str, reason: str, orders_cancelled: int) -> None:
        self._emit("grid_paused", symbol=symbol, reason=reason, orders_cancelled=orders_cancelled)

    def grid_activated(self, symbol: str, grid_lower: float, grid_upper: float, grid_count: int, orders_placed: int) -> None:
        self._emit(
            "grid_activated", symbol=symbol, grid_lower=grid_lower,
            grid_upper=grid_upper, grid_count=grid_count, orders_placed=orders_placed,
        )
