from telegram_notifier import TelegramNotifier


class CountingNotifier(TelegramNotifier):
    def __init__(self):
        super().__init__("", "", False)
        self.sent = 0

    def send(self, message):
        self.sent += 1
        return True


class RecordingNotifier(TelegramNotifier):
    def __init__(self):
        super().__init__("", "", False)
        self.messages = []

    def send(self, message):
        self.messages.append(message)
        return True


def test_order_failed_burst_is_throttled():
    n = CountingNotifier()
    n.on_order_failed("DOGEUSDT", "buy", 0.069, 100, "timeout")
    n.on_order_failed("DOGEUSDT", "buy", 0.070, 100, "timeout")
    n.on_order_failed("DOGEUSDT", "buy", 0.071, 100, "timeout")
    assert n.sent == 1


def test_order_failed_throttle_expires():
    n = CountingNotifier()
    n.on_order_failed("DOGEUSDT", "buy", 0.069, 100, "timeout")
    n.order_event_cooldown = 0.0
    n.on_order_failed("DOGEUSDT", "buy", 0.070, 100, "timeout")
    assert n.sent == 2


def test_order_cancelled_burst_is_throttled():
    n = CountingNotifier()
    n.on_order_cancelled("DOGEUSDT", "buy", 0.069, "id-1", "pause")
    n.on_order_cancelled("DOGEUSDT", "buy", 0.070, "id-2", "pause")
    n.on_order_cancelled("DOGEUSDT", "buy", 0.071, "id-3", "pause")
    assert n.sent == 1


def test_critical_events_are_not_throttled():
    n = CountingNotifier()
    n.on_kill_switch("max drawdown")
    n.on_kill_switch("daily loss")
    n.on_grid_start("DOGEUSDT", 0.06, 0.08, 20)
    assert n.sent == 3


def test_balance_update_carries_no_pnl_figure():
    """AUDIT #147. A message titled BALANCE that also carries the account's
    cumulative realized PnL (which is very often negative -- it includes every
    loss ever taken, not just the current session) reads as the balance itself
    being negative, no matter how the fine print underneath is labelled. Live,
    2026-08-24/25: a real ~4867 USDT equity sent alongside "Account (89d rolling,
    all activity): -70.76 USDT" in the same message was mistaken for a negative
    balance three separate times in one session.

    Balance must stand alone: free/used/equity/exposure, nothing else.
    """
    n = RecordingNotifier()
    n.on_balance_update(free=4800.0, used=67.0, total_equity=4867.0, exposure_pct=0.014)
    assert len(n.messages) == 1
    message = n.messages[0]
    assert "BALANCE" in message
    assert "Free: 4800.00 USDT" in message
    assert "Used: 67.00 USDT" in message
    assert "Equity: 4867.00 USDT" in message
    assert "Exposure: 1.4%" in message
    assert "Account (" not in message, "cumulative account PnL has no place in a BALANCE message"
    assert "This run:" not in message


def test_balance_update_reads_unambiguous_even_when_account_pnl_is_negative():
    """The exact scenario that motivated AUDIT #147: account-wide realized PnL is
    deeply negative while equity itself is healthy and positive. The message must
    not contain a negative-looking figure at all in that case.
    """
    n = RecordingNotifier()
    n.on_balance_update(free=4800.0, used=67.0, total_equity=4867.0, exposure_pct=0.014)
    message = n.messages[0]
    assert "-70" not in message
    assert "+" not in message
