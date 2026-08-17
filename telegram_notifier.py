from __future__ import annotations

import queue
import threading
import time

from loguru import logger

from pnl_tracker import BOOTSTRAP_LOOKBACK_DAYS

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


_DEFAULT_WINDOW = f"{BOOTSTRAP_LOOKBACK_DAYS}d rolling"


def _pnl_lines(
    session_pnl: float | None,
    account_pnl: float | None,
    window_label: str = _DEFAULT_WINDOW,
) -> str:
    """Render the PnL footer shared by the fill, balance and startup messages.

    The account figure used to be sent alone, labelled "Total PnL (verified)". On a
    "Bot Started" message that reads as though the bot begins in the red: it opened at
    -30.20, which is true but is 89 days of ACCOUNT history -- including a -50.49 day
    caused by defects since fixed -- and says nothing about how this run is doing.

    Session first, because it is the number that answers "is it working now?", and the
    account figure explicitly labelled as the account's, not the bot's (AUDIT #59).
    """
    lines = []
    if session_pnl is not None:
        lines.append(f"\nThis run: {session_pnl:+.4f} USDT")
    if account_pnl is not None:
        lines.append(f"\nAccount ({window_label}, all activity): {account_pnl:+.4f} USDT")
    return "".join(lines)


class TelegramNotifier:
    """Fire-and-forget notifications.

    Sends run on a background worker, never on the caller's thread. They used to run
    inline, and a slow Telegram call therefore stopped the trading loop dead: measured
    four times in the 2026-08-17 05:00 session at 16-18 seconds each --

        06:54:39 ORDER PLACED -> 06:54:55 TG OK
        08:20:48 FILL #41     -> 08:21:06 TG OK
        08:50:18 ORDER PLACED -> 08:50:34 TG OK
        09:10:41 ORDER PLACED -> 09:10:57 TG OK

    -- with the next log line in each case arriving only after the send returned. That
    is the whole explanation for a 10s poll interval running at a 15-16s cadence and
    stretching to 22-28s in places, and for a 14-order startup taking 26 seconds. A
    notification is worth nothing if the price it describes has moved on while it sent.
    """

    # Bounded on purpose. If Telegram is wedged, the choice is between dropping status
    # messages and growing an unbounded backlog inside a process that has to keep
    # trading; dropping is obviously right. Sized to hold a busy burst (a fill emits
    # fill + position + balance) many times over.
    QUEUE_LIMIT = 256
    # How long close() waits for the backlog before giving up and shutting down anyway.
    DRAIN_TIMEOUT_SECONDS = 5.0

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = False, max_retries: int = 3):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.max_retries = max_retries
        self.enabled = enabled and HAS_HTTPX and bool(bot_token) and bool(chat_id)
        self._client: httpx.Client | None = None
        self._last_event_time: dict[str, float] = {}
        self.order_event_cooldown: float = 30.0
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=self.QUEUE_LIMIT)
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._dropped = 0
        self._closed = False

        if self.enabled:
            self._client = httpx.Client(timeout=15)
            if self._test_connection():
                logger.info("Telegram notifications enabled and connected")
            else:
                logger.warning("Telegram enabled but connection test failed — notifications may not work")
        elif enabled:
            logger.warning("Telegram enabled but httpx not installed or credentials missing")

    def _test_connection(self) -> bool:
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            resp = self._client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                bot_name = data.get("result", {}).get("username", "unknown")
                logger.info("Telegram bot connected: @{}", bot_name)
                return True
            logger.warning("Telegram getMe failed: status={} body={}", resp.status_code, resp.text[:200])
            return False
        except Exception as e:
            logger.error("Telegram connection test failed: {}", e)
            return False

    def _ensure_worker(self) -> None:
        """Start the sender thread on first use, so a disabled notifier spawns nothing."""
        with self._worker_lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._drain, name="telegram-notifier", daemon=True,
                )
                self._worker.start()

    def _drain(self) -> None:
        while True:
            message = self._queue.get()
            try:
                if message is None:              # shutdown sentinel
                    return
                self._send_now(message)
            except Exception as e:               # a notifier must never kill its thread
                logger.debug("Telegram worker swallowed: {}", e)
            finally:
                self._queue.task_done()

    def send(self, message: str) -> bool:
        """Queue a message. Returns whether it was accepted, NOT whether it was sent.

        The caller is a trading loop; it cannot afford to wait on an HTTP round trip and
        has nothing useful to do with the delivery result either way.
        """
        if not self.enabled or not self._client or self._closed:
            return False
        self._ensure_worker()
        try:
            self._queue.put_nowait(message)
            return True
        except queue.Full:
            self._dropped += 1
            # One line per 50 drops: a wedged Telegram must not itself become the thing
            # flooding the log the trading loop writes to.
            if self._dropped % 50 == 1:
                logger.warning(
                    "TELEGRAM BACKLOG | queue full ({} deep), {} message(s) dropped so "
                    "far — trading is unaffected", self.QUEUE_LIMIT, self._dropped,
                )
            return False

    def _send_now(self, message: str) -> bool:
        """The blocking send. Runs on the worker thread only."""
        if not self.enabled or not self._client:
            return False

        last_err = None
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                resp = self._client.post(url, json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                })
                if resp.status_code == 200:
                    logger.info("TG OK: {}", message[:80])
                    return True
                if resp.status_code == 429:
                    retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                    logger.warning("Telegram rate limited, retrying in {}s", retry_after)
                    time.sleep(retry_after)
                    attempt -= 1
                    continue
                logger.warning("Telegram send failed (attempt {}/{}): status={} {}", attempt, self.max_retries, resp.status_code, resp.text[:200])
                return False
            except httpx.TimeoutException as e:
                last_err = e
                delay = 2 * attempt
                logger.warning("Telegram timeout (attempt {}/{}): {} — retrying in {}s", attempt, self.max_retries, e, delay)
                time.sleep(delay)
            except httpx.HTTPError as e:
                last_err = e
                delay = 2 * attempt
                logger.warning("Telegram HTTP error (attempt {}/{}): {} — retrying in {}s", attempt, self.max_retries, e, delay)
                time.sleep(delay)
            except Exception as e:
                logger.error("Telegram unexpected error: {}", e)
                return False

        logger.error("Telegram send failed after {} retries: {}", self.max_retries, last_err)
        return False

    @staticmethod
    def _esc(text: str) -> str:
        """Escape text for Telegram's HTML parse mode."""
        return (
            text.replace("&", "&#38;")
            .replace("<", "&#60;")
            .replace(">", "&#62;")
            .replace('"', "&#34;")
            .replace("'", "&#39;")
        )

    def _throttled(self, key: str, cooldown: float | None = None) -> bool:
        """Return True if an event of this type was sent recently (within cooldown).

        Used for high-volume per-order events (failures/cancels) so an exchange
        outage burst collapses into a few messages instead of dozens.
        """
        now = time.time()
        cooldown = self.order_event_cooldown if cooldown is None else cooldown
        if now - self._last_event_time.get(key, 0.0) < cooldown:
            return True
        self._last_event_time[key] = now
        return False

    def on_grid_start(self, symbol: str, lower: float, upper: float, count: int) -> None:
        lo = f"{lower:.8f}".rstrip("0").rstrip(".")
        hi = f"{upper:.8f}".rstrip("0").rstrip(".")
        self.send(
            f"<b>Grid Bot Started</b>\n"
            f"Pair: {self._esc(symbol)}\n"
            f"Range: {lo} - {hi}\n"
            f"Levels: {count}"
        )

    def on_fill(
        self, side: str, price: float, pnl: float, fill_count: int, daily_pnl: float = 0.0,
        total_pnl_verified: float | None = None,
        session_pnl: float | None = None,
        pnl_window: str | None = None,
    ) -> None:
        emoji = "&#x1f7e2;" if side == "sell" else "&#x1f534;"
        price_str = f"{price:.8f}".rstrip("0").rstrip(".")
        verified_line = _pnl_lines(session_pnl, total_pnl_verified, pnl_window or _DEFAULT_WINDOW)
        self.send(
            f"{emoji} <b>FILL #{fill_count}</b>\n"
            f"Side: {side.upper()}\n"
            f"Price: {price_str}\n"
            f"Cycle PnL: {pnl:.4f} USDT\n"
            f"Daily PnL: {daily_pnl:.4f} USDT"
            f"{verified_line}"
        )

    def on_trend_pause(self, regime: str, adx: float) -> None:
        self.send(
            f"&#x26a0;&#xfe0f; <b>Grid PAUSED</b>\n"
            f"Reason: Trend detected\n"
            f"Regime: {self._esc(regime)}\n"
            f"ADX: {adx:.1f}"
        )

    def on_grid_resume(self) -> None:
        self.send("&#x2705; <b>Grid RESUMED</b> — Ranging market detected")

    def on_recenter(self, old_lower: float, old_upper: float, new_lower: float, new_upper: float) -> None:
        ol = f"{old_lower:.8f}".rstrip("0").rstrip(".")
        oh = f"{old_upper:.8f}".rstrip("0").rstrip(".")
        nl = f"{new_lower:.8f}".rstrip("0").rstrip(".")
        nh = f"{new_upper:.8f}".rstrip("0").rstrip(".")
        self.send(
            f"&#x1f504; <b>Grid RECENTERED</b>\n"
            f"Old: {ol} - {oh}\n"
            f"New: {nl} - {nh}"
        )

    def on_kill_switch(self, reason: str) -> None:
        self.send(
            f"&#x1f6a8; <b>KILL SWITCH TRIGGERED</b>\n"
            f"Reason: {self._esc(reason)}\n"
            f"All orders cancelled.\n"
            f"Entering recovery mode..."
        )

    def on_recovery_start(self, cooldown_secs: int, recovery_count: int) -> None:
        mins = cooldown_secs // 60
        self.send(
            f"&#x23f3; <b>RECOVERY MODE</b>\n"
            f"Cooldown: {mins} min\n"
            f"Recovery attempt: #{recovery_count}\n"
            f"Will resume with reduced sizing."
        )

    def on_recovery_resume(self, sizing_pct: float) -> None:
        self.send(
            f"&#x2705; <b>GRID RESUMED (RECOVERY)</b>\n"
            f"Size: {sizing_pct:.0%} of normal\n"
            f"Grid recalculated around current price."
        )

    def on_order_placed(self, symbol: str, side: str, price: float, qty: float, order_id: str) -> None:
        emoji = "&#x1f7e2;" if side == "sell" else "&#x1f534;"
        price_str = f"{price:.8f}".rstrip("0").rstrip(".")
        self.send(
            f"{emoji} <b>ORDER PLACED</b>\n"
            f"{side.upper()} {qty:.1f} {self._esc(symbol)}\n"
            f"Price: {price_str}\n"
            f"ID: {self._esc(order_id)}"
        )

    def on_order_failed(self, symbol: str, side: str, price: float, qty: float, error: str) -> None:
        if self._throttled("order_failed"):
            logger.debug("Throttled ORDER FAILED notification (cooldown)")
            return
        price_str = f"{price:.8f}".rstrip("0").rstrip(".")
        self.send(
            f"&#x274c; <b>ORDER FAILED</b>\n"
            f"{side.upper()} {qty:.1f} {self._esc(symbol)}\n"
            f"Price: {price_str}\n"
            f"Error: {self._esc(error[:200])}"
        )

    def on_order_cancelled(self, symbol: str, side: str, price: float, order_id: str, reason: str) -> None:
        if self._throttled("order_cancelled"):
            logger.debug("Throttled ORDER CANCELLED notification (cooldown)")
            return
        price_str = f"{price:.8f}".rstrip("0").rstrip(".")
        self.send(
            f"&#x1f6ab; <b>ORDER CANCELLED</b>\n"
            f"{side.upper()} {self._esc(symbol)}\n"
            f"Price: {price_str}\n"
            f"Reason: {self._esc(reason)}"
        )

    def on_grid_exit(self, symbol: str, reason: str, current_price: float, positions_held: float) -> None:
        price_str = f"{current_price:.8f}".rstrip("0").rstrip(".")
        self.send(
            f"&#x1f6a8; <b>GRID EXIT</b>\n"
            f"{self._esc(symbol)}\n"
            f"Price: {price_str}\n"
            f"Positions held: {positions_held:.1f}\n"
            f"Reason: {self._esc(reason)}"
        )

    def on_position_update(self, symbol: str, side: str, entry_price: float, qty: float, current_price: float, unrealized_pnl: float) -> None:
        emoji = "&#x1f7e2;" if unrealized_pnl >= 0 else "&#x1f534;"
        entry_str = f"{entry_price:.8f}".rstrip("0").rstrip(".")
        cur_str = f"{current_price:.8f}".rstrip("0").rstrip(".")
        self.send(
            f"{emoji} <b>POSITION</b>\n"
            f"{side.upper()} {qty:.1f} {symbol}\n"
            f"Entry: {entry_str}\n"
            f"Current: {cur_str}\n"
            f"Unrealized: {unrealized_pnl:.4f} USDT"
        )

    def on_balance_update(
        self, free: float, used: float, total_equity: float, exposure_pct: float,
        total_pnl_verified: float | None = None,
        session_pnl: float | None = None,
        pnl_window: str | None = None,
    ) -> None:
        verified_line = _pnl_lines(session_pnl, total_pnl_verified, pnl_window or _DEFAULT_WINDOW)
        self.send(
            f"&#x1f4b0; <b>BALANCE</b>\n"
            f"Free: {free:.2f} USDT\n"
            f"Used: {used:.2f} USDT\n"
            f"Equity: {total_equity:.2f} USDT\n"
            f"Exposure: {exposure_pct:.1%}"
            f"{verified_line}"
        )

    def on_risk_check(self, check_type: str, value: float, threshold: float, result: str) -> None:
        emoji = "&#x2705;" if result == "OK" else "&#x26a0;&#xfe0f;"
        self.send(
            f"{emoji} <b>RISK: {check_type.upper()}</b>\n"
            f"Value: {value:.2%}\n"
            f"Limit: {threshold:.2%}\n"
            f"Result: {result}"
        )

    def on_risk_kill_switch(self, reason: str, drawdown: float, daily_loss: float) -> None:
        self.send(
            f"&#x1f6a8; <b>KILL SWITCH</b>\n"
            f"Reason: {self._esc(reason)}\n"
            f"Drawdown: {drawdown:.2%}\n"
            f"Daily loss: {daily_loss:.2%}\n"
            f"Entering recovery mode..."
        )

    def on_stop_loss(self, symbol: str, sl_price: float, current_price: float, qty: float, unrealized_loss: float) -> None:
        sl_str = f"{sl_price:.8f}".rstrip("0").rstrip(".")
        cur_str = f"{current_price:.8f}".rstrip("0").rstrip(".")
        self.send(
            f"&#x1f6a8; <b>STOP LOSS TRIGGERED</b>\n"
            f"{self._esc(symbol)}\n"
            f"SL Price: {sl_str}\n"
            f"Current: {cur_str}\n"
            f"Qty: {qty:.1f}\n"
            f"Loss: {unrealized_loss:.4f} USDT"
        )

    def on_trend_change(self, old_regime: str, new_regime: str, adx: float) -> None:
        self.send(
            f"&#x1f4c8; <b>TREND CHANGE</b>\n"
            f"{self._esc(old_regime)} → {self._esc(new_regime)}\n"
            f"ADX: {adx:.1f}"
        )

    def on_exposure(self, exposure_pct: float, equity: float) -> None:
        self.send(
            f"&#x1f4ca; <b>EXPOSURE</b>\n"
            f"Current: {exposure_pct:.1%}\n"
            f"Equity: {equity:.2f} USDT"
        )

    def on_grid_recalculated(self, symbol: str, new_lower: float, new_upper: float, grid_count: int, reason: str) -> None:
        lo = f"{new_lower:.8f}".rstrip("0").rstrip(".")
        hi = f"{new_upper:.8f}".rstrip("0").rstrip(".")
        self.send(
            f"&#x1f504; <b>GRID RECALCULATED</b>\n"
            f"{self._esc(symbol)}\n"
            f"Range: {lo} - {hi}\n"
            f"Levels: {grid_count}\n"
            f"Reason: {self._esc(reason)}"
        )

    def on_startup_summary(
        self, symbol: str, mode: str, price: float,
        grid_lower: float, grid_upper: float, grid_count: int,
        balance_free: float, equity: float, leverage: int,
        regime: str, adx: float, grid_active: bool,
        total_pnl_verified: float | None = None,
        session_pnl: float | None = None,
        pnl_window: str | None = None,
    ) -> None:
        lo = f"{grid_lower:.8f}".rstrip("0").rstrip(".")
        hi = f"{grid_upper:.8f}".rstrip("0").rstrip(".")
        status = "ACTIVE" if grid_active else "PAUSED (trend)"
        verified_line = _pnl_lines(session_pnl, total_pnl_verified, pnl_window or _DEFAULT_WINDOW)
        self.send(
            f"<b>Bot Started</b> | {self._esc(mode)}\n"
            f"Pair: {self._esc(symbol)} @ {price:.6f}\n"
            f"Grid: {lo} - {hi} ({grid_count} levels)\n"
            f"Status: {status}\n"
            f"Leverage: {leverage}x\n"
            f"Balance: {balance_free:.2f} USDT\n"
            f"Equity: {equity:.2f} USDT\n"
            f"Regime: {self._esc(regime)} (ADX {adx:.1f})"
            f"{verified_line}"
        )

    def on_daily_summary(self, pnl: float, fills: int, balance: float) -> None:
        emoji = "&#x1f4c8;" if pnl >= 0 else "&#x1f4c9;"
        self.send(
            f"{emoji} <b>Daily Summary</b>\n"
            f"PnL: {pnl:.4f} USDT\n"
            f"Fills: {fills}\n"
            f"Balance: {balance:.2f} USDT"
        )

    def close(self) -> None:
        """Stop accepting messages, give the backlog a bounded chance, then shut down.

        Bounded because this runs on the shutdown path: the bot has orders to cancel and
        stops to leave armed, and none of that should wait on a status message.
        """
        self._closed = True
        worker = self._worker
        if worker is not None and worker.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            worker.join(timeout=self.DRAIN_TIMEOUT_SECONDS)
            if worker.is_alive():
                logger.debug("Telegram worker still draining at shutdown — abandoning it")
        self._worker = None
        if self._client:
            self._client.close()
            self._client = None
