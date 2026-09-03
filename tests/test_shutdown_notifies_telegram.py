"""No notification told Telegram the bot had gone down. AUDIT #170.

Startup sends "Bot Started" (TelegramNotifier.on_startup_summary). Shutdown -- a plain
Ctrl+C, a signal, a give-up-and-abort after a failed reconnect, or an unhandled
exception -- reaches main.py's `finally:` block either way, which calls
GridEngine.emergency_stop(reason="shutdown") and then notifier.close(). Nothing in that
path ever told the chat the bot had stopped trading.

Observed live: a plain Ctrl+C, and the terminal showed a clean shutdown ("Bot stopped.
State saved.") while Telegram stayed silent -- the only way to know the bot was down
was to be watching the terminal.
"""

from unittest.mock import MagicMock

from grid import GridEngine


def _engine(positions, notifier=None):
    ex = MagicMock()
    ex.exchange.amount_to_precision.side_effect = lambda s, a: str(int(float(a)))
    ex.get_positions.return_value = positions
    g = GridEngine(ex, "DOGEUSDT", grid_lower=0.068, grid_upper=0.072, grid_count=8,
                   capital_per_grid_pct=0.018, stop_loss_pct=0.005,
                   capital_per_grid_usdt=5.0, leverage=25, notifier=notifier)
    return g, ex


def test_a_normal_shutdown_notifies_telegram_the_bot_stopped():
    notifier = MagicMock()
    g, ex = _engine([], notifier=notifier)

    g.emergency_stop("shutdown")

    notifier.on_bot_stopped.assert_called_once_with("shutdown", False)


def test_a_shutdown_holding_a_position_reports_it_as_still_protected():
    notifier = MagicMock()
    g, ex = _engine([{"contracts": 5350.0}], notifier=notifier)

    g.emergency_stop("shutdown")

    notifier.on_bot_stopped.assert_called_once_with("shutdown", True)


def test_the_kill_switch_does_not_also_fire_bot_stopped():
    """emergency_stop runs on the kill switch too (reason="emergency" there), which
    already sends its own on_kill_switch notification via the caller -- this must not
    double up with a second, misleading "Bot Stopped" while the process keeps running
    in recovery mode rather than exiting."""
    notifier = MagicMock()
    g, ex = _engine([], notifier=notifier)

    g.emergency_stop("emergency")

    notifier.on_bot_stopped.assert_not_called()


def test_a_recovery_rebuild_failure_does_not_fire_bot_stopped_either():
    """The process survives this one and retries next cycle -- it never actually
    stops, so a "Bot Stopped" here would be a false alarm."""
    notifier = MagicMock()
    g, ex = _engine([], notifier=notifier)

    g.emergency_stop("recovery_rebuild_failed")

    notifier.on_bot_stopped.assert_not_called()


def test_no_notifier_configured_does_not_crash_shutdown():
    g, ex = _engine([], notifier=None)

    g.emergency_stop("shutdown")  # must not raise


def test_telegram_notifier_renders_a_bot_stopped_message_for_each_case():
    """Behavioural check on the real notifier, not a mock -- confirms the method
    exists with this exact signature and calls send() (rather than, say, silently
    swallowing the call because of a typo'd method name that a MagicMock would hide)."""
    from telegram_notifier import TelegramNotifier

    sent = []
    notifier = TelegramNotifier.__new__(TelegramNotifier)
    notifier.send = lambda text: sent.append(text)
    notifier._esc = lambda s: s

    notifier.on_bot_stopped("shutdown", True)
    notifier.on_bot_stopped("shutdown", False)

    assert len(sent) == 2
    assert "Bot Stopped" in sent[0]
    assert "still open" in sent[0]
    assert "flat" in sent[1]
