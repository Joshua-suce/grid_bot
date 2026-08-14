"""A paused strategy still holds a position. AUDIT #53.

`pause()` deliberately does not flatten -- "the position stays under stop protection",
per the Strategy protocol. So a paused grid routinely holds inventory.

Every risk control lived inside `if grid.active:`. While paused, therefore:

  * no drawdown check
  * no daily-loss check
  * no grid stop-loss backstop
  * no consecutive-loss check
  * `daily_unrealized_pnl` frozen at whatever it held when the grid went quiet
  * and `_refresh_sl_stops` never ran, so a stop that failed to place (#50) or was
    cancelled stayed missing for the whole pause

The trend filter pauses on a confirmed TREND -- which is exactly when a held position
runs away -- and measurement puts the grid at ~60% paused on DOGE (AUDIT #52). The risk
layer was off duty for the majority of the bot's life, and specifically during the
regime that hurts a held position most.
"""

import pathlib
import re

MAIN = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _paused_block() -> str:
    """The body of the `if not grid.active:` branch in the trading loop."""
    lines = MAIN.splitlines()
    start = next(
        i for i, l in enumerate(lines)
        if l.strip() == "if not grid.active:" and _indent(l) >= 12
    )
    base = _indent(lines[start])
    out = []
    for line in lines[start + 1:]:
        if line.strip() and _indent(line) <= base:
            break
        out.append(line)
    return "\n".join(out)


def test_the_paused_branch_evaluates_the_kill_switches():
    block = _paused_block()
    assert "risk.check_all(" in block, (
        "a paused grid still holds its position, but no kill switch is evaluated while "
        "it is paused -- drawdown, daily loss and the stop-loss backstop are all off duty"
    )
    assert "daily_realized_pnl=pnl_reconciler.daily_net_pnl" in block, (
        "the paused check must use the exchange-verified daily figure, like the active "
        "path does (AUDIT #43)"
    )


def test_the_paused_branch_keeps_the_stop_loss_alive():
    block = _paused_block()
    assert "_refresh_sl_stops(" in block, (
        "stops are never refreshed while paused, so a stop that failed to place stays "
        "missing for the entire pause (AUDIT #50)"
    )
    assert "_sl_needs_update(" in block, (
        "the paused path should reuse the same needs-update rule, not re-place blindly "
        "every iteration"
    )


def test_the_paused_branch_refreshes_unrealized_pnl():
    """A frozen unrealized figure makes the daily-loss check meaningless: the position
    can run against you all pause and the total never moves."""
    block = _paused_block()
    assert "risk.update_unrealized(" in block


def test_the_paused_branch_uses_the_side_correct_stop():
    """Passing a long's floor for a short is AUDIT #25's blindness."""
    block = _paused_block()
    assert "get_short_stop_loss_price()" in block and "get_stop_loss_price()" in block, (
        "the paused backstop must pick the stop matching the held side"
    )


def test_the_paused_branch_only_acts_on_a_real_position():
    """It must not fire kill switches, or re-place stops, while genuinely flat."""
    block = _paused_block()
    assert re.search(r'held_side in \("long", "short"\) and held_qty > 0', block), (
        "the paused risk work is not gated on actually holding something"
    )


def test_a_paused_kill_switch_does_not_re_enter_recovery_every_iteration():
    """check_all returns (False, True) for the whole cooldown, so an ungated caller
    would re-trigger emergency_stop and re-notify on every single loop."""
    block = _paused_block()
    assert "was_paused_recovery" in block, "no recovery latch on the paused path"
    assert "not was_paused_recovery" in block
