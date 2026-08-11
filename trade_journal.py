from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from loguru import logger


class TradeJournal:
    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / "trades.csv"
        self._ensure_header()

    _HEADER = [
        "timestamp", "symbol", "side", "price", "quantity",
        "grid_spacing", "fill_number", "completed_cycle",
        "cycle_pnl", "cumulative_pnl", "daily_pnl",
        "trades_today", "regime", "regime_adx",
        "fee", "balance", "equity", "exposure_pct",
        "unrealized_pnl",
    ]

    def _ensure_header(self) -> None:
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="") as f:
                csv.writer(f).writerow(self._HEADER)
            return
        try:
            with open(self.filepath, "r") as f:
                first_line = f.readline().strip()
            existing = [c.strip() for c in first_line.split(",")]
            if existing != self._HEADER:
                backup = self.filepath.with_suffix(f".csv.{int(self.filepath.stat().st_mtime)}")
                self.filepath.rename(backup)
                logger.warning(
                    "Trades CSV header mismatch — archived old file to {} ({} cols != {} cols)",
                    backup.name, len(existing), len(self._HEADER),
                )
                with open(self.filepath, "w", newline="") as f:
                    csv.writer(f).writerow(self._HEADER)
        except Exception as e:
            logger.error("Failed to validate trades CSV header: {}", e)

    def record(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        grid_spacing: float,
        fill_number: int,
        cumulative_pnl: float,
        daily_pnl: float,
        trades_today: int,
        regime: str,
        fee: float = 0.0,
        completed_cycle: bool = False,
        cycle_pnl: float = 0.0,
        regime_adx: float = 0.0,
        balance: float = 0.0,
        equity: float = 0.0,
        exposure_pct: float = 0.0,
        unrealized_pnl: float = 0.0,
    ) -> None:
        try:
            with open(self.filepath, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    symbol,
                    side.upper(),
                    f"{price:.8f}",
                    f"{quantity:.6f}",
                    f"{grid_spacing:.8f}",
                    fill_number,
                    completed_cycle,
                    f"{cycle_pnl:.6f}",
                    f"{cumulative_pnl:.6f}",
                    f"{daily_pnl:.6f}",
                    trades_today,
                    regime,
                    f"{regime_adx:.1f}",
                    f"{fee:.6f}",
                    f"{balance:.2f}",
                    f"{equity:.2f}",
                    f"{exposure_pct:.4f}",
                    f"{unrealized_pnl:.6f}",
                ])
            logger.info(
                "TRADE JOURNAL | {} {} @ {} | fill#{} | cycle_pnl={:.6f} | cum_pnl={:.6f}",
                side.upper(), symbol, price, fill_number, cycle_pnl, cumulative_pnl,
            )
        except Exception as e:
            logger.error("Failed to write trade journal: {}", e)
