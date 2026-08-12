"""Regime-change signal generator -- a read-only observer alongside the trading loop.

WHAT IT IS
On every confirmed market-regime change the bot already detects (TrendFilter's
ADX/EMA multi-timeframe classification), this records what the regime implies and
optionally sends it to Telegram. It is a *reporter*, not a strategy.

WHY IT IS AN OBSERVER, NOT A Strategy
A `Strategy` is something the router hands control *to* -- installing a signal-only
strategy there would mean nothing trades while it is selected. This has to run
*alongside* whatever is trading, so it is a plain object main.py feeds regime updates
to.

READ-ONLY BY CONSTRUCTION
It never receives an exchange handle. Not "it chooses not to trade" -- it structurally
cannot, because it has no way to reach an order endpoint. A bug in here can produce a
wrong message; it cannot produce a wrong order. `test_signals.py` pins that property.

WHY IT EARNS ITS KEEP
Each signal is scored against what price actually did before the *next* regime change,
so signals.csv accumulates evidence about whether the regime calls are any good --
which is exactly the question blocking the router's thresholds (see AUDIT.md #24).
That evidence costs nothing to gather: no orders, no fees, no risk.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

# What each regime implies for a directional trader.
REGIME_BIAS = {
    "uptrend": "LONG",
    "downtrend": "SHORT",
    "ranging": "RANGE",
    "uncertain": "NONE",
}

# Only these reach Telegram. "uncertain" transitions are still written to the CSV --
# they matter for scoring -- but pushing every flicker to a phone trains you to ignore
# the alerts, which defeats the point.
ACTIONABLE = {"LONG", "SHORT", "RANGE"}

# A RANGE call is judged correct if price stayed inside this band until the next
# regime change. Set from the grid's own reasoning: a range signal is useful if the
# market stayed inside roughly one grid width.
RANGE_TOLERANCE_PCT = 0.02


@dataclass
class Signal:
    """One regime transition, plus how it turned out once the next one arrived."""

    timestamp: str
    symbol: str
    from_regime: str
    to_regime: str
    bias: str
    price: float
    adx: float
    # Filled in retrospectively by score_previous(), when the next signal lands.
    resolved_at: str = ""
    resolved_price: float = 0.0
    move_pct: float = 0.0
    outcome: str = ""          # "hit" | "miss" | "flat" | "" while unresolved
    held_seconds: float = 0.0

    def describe(self) -> str:
        return (
            f"{self.from_regime} -> {self.to_regime} ({self.bias}) "
            f"@ {self.price:.8f} ADX={self.adx:.1f}"
        )


def score(signal: Signal, exit_price: float) -> tuple[str, float]:
    """Judge a signal against where price actually went. Returns (outcome, move_pct).

    Directional calls are judged on sign. A RANGE call is judged on *staying put* --
    it is right when nothing much happened, which is the opposite test, so it cannot
    share the directional branch.
    """
    if signal.price <= 0:
        return "", 0.0
    move = (exit_price - signal.price) / signal.price

    if signal.bias == "LONG":
        return ("hit" if move > 0 else "miss"), move
    if signal.bias == "SHORT":
        return ("hit" if move < 0 else "miss"), move
    if signal.bias == "RANGE":
        return ("hit" if abs(move) <= RANGE_TOLERANCE_PCT else "miss"), move
    return "flat", move


class SignalGenerator:
    """Emits one signal per confirmed regime change. Takes no exchange handle."""

    _HEADER = [
        "timestamp", "symbol", "from_regime", "to_regime", "bias",
        "price", "adx", "resolved_at", "resolved_price",
        "move_pct", "outcome", "held_seconds",
    ]

    def __init__(
        self,
        symbol: str,
        log_dir: str = "logs",
        notifier: object | None = None,
        event_journal: object | None = None,
        notify: bool = True,
    ):
        self.symbol = symbol
        self.notifier = notifier
        self.event_journal = event_journal
        self.notify = notify

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / "signals.csv"
        self._ensure_header()

        self._last_regime: str | None = None
        self.pending: Signal | None = None
        self.history: list[Signal] = []

    def _ensure_header(self) -> None:
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="") as f:
                csv.writer(f).writerow(self._HEADER)

    # --- the one entry point ----------------------------------------------

    def observe(self, regime: str, price: float, adx: float = 0.0,
                now: float | None = None) -> Signal | None:
        """Feed the current regime. Returns a Signal only when the regime changed.

        Called every loop iteration, so the no-change path must be cheap and silent.
        The first call establishes a baseline without emitting -- the bot starting up
        in `ranging` is not a transition into it, and reporting it as one would put a
        phantom signal at the head of every session's scoring.
        """
        regime = (regime or "uncertain").lower()

        if self._last_regime is None:
            self._last_regime = regime
            logger.debug("SIGNAL BASELINE | starting regime={} @ {}", regime, price)
            return None

        if regime == self._last_regime:
            return None

        previous, self._last_regime = self._last_regime, regime
        self._score_pending(price, now=now)

        sig = Signal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=self.symbol,
            from_regime=previous,
            to_regime=regime,
            bias=REGIME_BIAS.get(regime, "NONE"),
            price=float(price),
            adx=float(adx),
        )
        self.pending = sig
        self.history.append(sig)
        self._emit(sig, now=now)
        return sig

    def _score_pending(self, exit_price: float, now: float | None = None) -> None:
        """Resolve the previous signal now that price has had time to move."""
        if self.pending is None:
            return
        outcome, move = score(self.pending, float(exit_price))
        self.pending.resolved_at = datetime.now(timezone.utc).isoformat()
        self.pending.resolved_price = float(exit_price)
        self.pending.move_pct = move
        self.pending.outcome = outcome
        try:
            started = datetime.fromisoformat(self.pending.timestamp)
            self.pending.held_seconds = (
                datetime.now(timezone.utc) - started
            ).total_seconds()
        except (TypeError, ValueError):
            self.pending.held_seconds = 0.0

        self._write(self.pending)
        logger.info(
            "SIGNAL RESOLVED | {} | move={:+.2%} outcome={} held={:.0f}s",
            self.pending.describe(), move, outcome or "n/a", self.pending.held_seconds,
        )
        self.pending = None

    def _emit(self, sig: Signal, now: float | None = None) -> None:
        logger.info("SIGNAL | {}", sig.describe())

        if self.event_journal is not None:
            try:
                self.event_journal.trend_change(
                    sig.from_regime, sig.to_regime, sig.adx, "signal",
                )
            except Exception as e:
                logger.debug("SIGNAL | event journal write failed: {}", e)

        if not (self.notify and self.notifier and sig.bias in ACTIONABLE):
            return
        arrow = {"LONG": "\U0001f7e2", "SHORT": "\U0001f534", "RANGE": "\U0001f7e1"}.get(sig.bias, "")
        try:
            self.notifier.send(
                f"{arrow} <b>SIGNAL {sig.bias}</b>\n"
                f"{sig.symbol}\n"
                f"Regime: {sig.from_regime} -> {sig.to_regime}\n"
                f"Price: {sig.price:.8f}\n"
                f"ADX: {sig.adx:.1f}\n"
                f"<i>Informational — the bot trades on its own rules.</i>"
            )
        except Exception as e:
            logger.debug("SIGNAL | notify failed: {}", e)

    def _write(self, sig: Signal) -> None:
        try:
            with open(self.filepath, "a", newline="") as f:
                csv.writer(f).writerow([
                    sig.timestamp, sig.symbol, sig.from_regime, sig.to_regime,
                    sig.bias, f"{sig.price:.8f}", f"{sig.adx:.1f}",
                    sig.resolved_at, f"{sig.resolved_price:.8f}",
                    f"{sig.move_pct:.6f}", sig.outcome, f"{sig.held_seconds:.0f}",
                ])
        except Exception as e:
            logger.error("Failed to write signal journal: {}", e)

    # --- reporting ---------------------------------------------------------

    def accuracy(self) -> dict:
        """Hit rate over resolved signals, overall and per bias.

        This is the number that matters: if the regime calls cannot beat a coin flip
        here, the router's switching thresholds have no evidence behind them and
        should not be tuned (AUDIT.md #24).
        """
        resolved = [s for s in self.history if s.outcome in ("hit", "miss")]
        out: dict = {
            "resolved": len(resolved),
            "hits": sum(1 for s in resolved if s.outcome == "hit"),
        }
        out["hit_rate"] = out["hits"] / len(resolved) if resolved else 0.0
        for bias in sorted(ACTIONABLE):
            group = [s for s in resolved if s.bias == bias]
            hits = sum(1 for s in group if s.outcome == "hit")
            out[bias] = {
                "n": len(group),
                "hits": hits,
                "hit_rate": hits / len(group) if group else 0.0,
            }
        return out

    def log_accuracy(self) -> None:
        a = self.accuracy()
        if not a["resolved"]:
            return
        logger.info(
            "SIGNAL ACCURACY | {}/{} hit ({:.0%}) | LONG {}/{} SHORT {}/{} RANGE {}/{}",
            a["hits"], a["resolved"], a["hit_rate"],
            a["LONG"]["hits"], a["LONG"]["n"],
            a["SHORT"]["hits"], a["SHORT"]["n"],
            a["RANGE"]["hits"], a["RANGE"]["n"],
        )
