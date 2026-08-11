from telegram_notifier import TelegramNotifier


class CountingNotifier(TelegramNotifier):
    def __init__(self):
        super().__init__("", "", False)
        self.sent = 0

    def send(self, message):
        self.sent += 1
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
