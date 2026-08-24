"""Account settings can move under a running bot. Nothing was watching.

verify_account_config's own docstring says every money figure the bot computes --
notional per order, margin consumed, the stop distance -- depends on what it checks.
It ran once, at startup, and never again. So leverage, margin mode, position mode or
fee rates changed mid-session went unnoticed until a restart, and everything computed
after the change was wrong with nothing saying so.

This account really did go 25x -> 5x between sessions on 2026-08-19/20. Mid-session
that would have been silent. AUDIT #137.

Deliberately alert-only: startup refuses to trade on a mismatched account because
nothing is open yet, but killing a running bot that holds a position is a bigger risk
than the mis-sizing it would avoid.
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap

import pytest

import main as main_module
from main import account_recheck_due


# ------------------------------------------------------------------ the timer
def test_nothing_is_owed_before_the_interval_elapses():
    assert account_recheck_due(1000.0, 1000.0 + 899.0, 900.0) is False


def test_a_check_is_owed_once_the_interval_elapses():
    assert account_recheck_due(1000.0, 1000.0 + 900.0, 900.0) is True


def test_it_keeps_being_owed_afterwards():
    """The alert must repeat while the account stays wrong. A once-only report is how
    a correct detection went unheard for three hours on 2026-08-19 (AUDIT #127)."""
    assert account_recheck_due(1000.0, 1000.0 + 5000.0, 900.0) is True


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_a_non_positive_interval_disables_the_check(interval):
    assert account_recheck_due(0.0, 1e9, interval) is False


def test_the_two_verdicts_are_distinguishable():
    """Guard on the guard: a function stuck on one answer satisfies half of the above
    and looks fine in isolation."""
    assert account_recheck_due(0.0, 1e9, 900.0) != account_recheck_due(0.0, 0.0, 900.0)


# ----------------------------------------------------------------- the default
def test_the_shipped_interval_is_on_and_sane():
    """Off by default would make this fix decorative; a very short interval would
    spend API calls on account endpoints every poll."""
    from config import Settings
    default = Settings.model_fields["account_recheck_seconds"].default
    assert 60.0 <= default <= 3600.0


# ------------------------------------------------------------ wired to the loop
def _loop_source() -> str:
    src = inspect.getsource(main_module.run_bot)
    return src[src.index("while True:"):]


def _recheck_guard() -> "ast.If | None":
    """The `if` statement whose condition calls account_recheck_due.

    Found by parsing, not by string search. A substring test passes against
    `if False and account_recheck_due(...)` -- the call is still written, it just
    never runs. That mutant survived the first version of this file, which is the
    fourth time in this repo a source scan has been satisfied by text that does not
    execute.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(main_module.run_bot)))

    # A guard may hold its verdict in a variable -- `_due = account_recheck_due(...)`
    # then `if _due:` -- which is perfectly legitimate. Resolve those, or this test
    # fails on a refactor while still passing on a genuinely disabled guard.
    from_call = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "account_recheck_due"):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    from_call.add(t.id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        called = {c.func.id for c in ast.walk(node.test)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if "account_recheck_due" in called or (names & from_call):
            return node
    return None


def test_the_recheck_actually_runs_in_the_trading_loop():
    """A helper nothing calls is not a check. This is the wiring, and it is the part
    that has silently gone missing before -- ladder_cap_room needed the same pin."""
    loop = _loop_source()
    assert "account_recheck_due(" in loop
    assert "verify_account_config(" in loop, "the timer fires but nothing is verified"


def test_the_guard_is_the_call_itself_not_a_disabled_expression():
    """Kills `if False and account_recheck_due(...)`: the condition must BE the call,
    not something that merely mentions it."""
    node = _recheck_guard()
    assert node is not None, "no `if` statement guards on account_recheck_due"
    assert isinstance(node.test, (ast.Call, ast.Name)), (
        "the guard condition is neither a bare account_recheck_due(...) call nor a "
        "variable holding one -- something wraps it, and a wrapper is how a "
        "permanently-false guard hides"
    )
    if isinstance(node.test, ast.Call):
        assert node.test.func.id == "account_recheck_due"


def test_the_guarded_body_is_what_verifies_the_account():
    """The timer could fire and check nothing. Pin that verify_account_config is
    called INSIDE the guarded block, not merely somewhere in the loop."""
    node = _recheck_guard()
    assert node is not None
    called = {c.func.id for stmt in node.body for c in ast.walk(stmt)
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "verify_account_config" in called


def test_the_ast_scan_can_still_fail():
    """Guard on the guard. Feed the finder a body where the call is wrapped and prove
    it reports a non-Call condition rather than quietly finding nothing."""
    wrapped = ast.parse(textwrap.dedent('''
        def f():
            while True:
                if False and account_recheck_due(a, b, c):
                    verify_account_config(x, y, z)
    '''))
    found = None
    for node in ast.walk(wrapped):
        if isinstance(node, ast.If):
            called = {c.func.id for c in ast.walk(node.test)
                      if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            if "account_recheck_due" in called:
                found = node
    assert found is not None, "the finder cannot see a wrapped call at all"
    assert not isinstance(found.test, ast.Call), (
        "the finder would call a disabled guard healthy"
    )


def test_the_loop_slice_is_not_empty():
    """If `while True:` moves, the slice could go empty and every assertion on it
    would pass vacuously."""
    assert len(_loop_source()) > 5000


def _guard_body() -> str:
    """Source of the block the recheck guards.

    Read from the AST rather than by slicing N characters after the first mention of
    account_recheck_due. That slice broke the moment another feature was wired in
    ahead of it: the first mention became an assignment, and the window no longer
    reached the block these tests are about.
    """
    node = _recheck_guard()
    assert node is not None, "no recheck guard found"
    return "\n".join(ast.unparse(stmt) for stmt in node.body)


def test_it_alerts_rather_than_aborting_the_running_bot():
    """The deliberate choice. A running bot holds a position; stopping it over a
    settings mismatch trades a known risk for a larger one."""
    body = _guard_body()
    assert "ACCOUNT DRIFTED" in body
    assert "abort_startup" not in body, "a mid-session recheck must not exit"
    assert "SystemExit" not in body


def test_an_unreadable_account_is_not_reported_as_drift():
    """An outage is not a settings change, and the loop has its own machinery for
    outages. Reporting it as drift would cry wolf every interval of a venue problem.
    Same UNKNOWN-is-not-a-verdict discipline as AUDIT #128/#132/#134."""
    assert "ACCOUNT_UNREADABLE" in _guard_body()


def test_the_timer_advances_even_when_the_check_fails():
    """Otherwise a raising endpoint turns the recheck into a per-poll retry against
    the account endpoints."""
    assert re.search(r"account_checked_at\s*=\s*time\.time\(\)", _guard_body()), (
        "the timestamp is not reset before the check runs"
    )


def test_the_startup_check_seeds_the_timer():
    """Starting the clock at zero would fire a redundant recheck on the first poll,
    seconds after startup already verified the account."""
    src = inspect.getsource(main_module.run_bot)
    assert re.search(r"account_checked_at\s*=\s*time\.time\(\)", src)
    assert "account_checked_at = 0" not in src
