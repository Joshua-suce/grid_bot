"""The engine's P&L and the account's already sat side by side. Nothing compared them.

Every status line prints both:

    net=2.11 | session=-1.12 account(since 2026-08-14)=-70.46
    ^^^ engine ledger                    ^^^ Binance income

On 2026-08-20 the engine claimed +2.11 while the account was down 71.43, printed
every ten seconds for days. `py attribute_pnl.py 9` puts the truth at -71.17, of
which a single -74.84 stop is 105%.

The engine cannot see it. Forced closes -- the hard stop-market leg, reconcile
closes, emergency_stop -- go through exchange.close_position() and never reach
_handle_fill, so grid.total_pnl never books them. cycle_pnl showed 47 wins, 1 loss,
+10.96 on that account, and the -74.84 produced no journal row at all.

The backtest harness already reports this ("drift vs position accounting: +374.17").
Live had no equivalent. AUDIT #139.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

import main as main_module
from main import pnl_divergence


# ------------------------------------------------------------------- the maths
def test_agreement_reports_no_gap():
    assert pnl_divergence(1.50, 1.50, 2.0) == 0.0


def test_small_disagreement_is_within_tolerance():
    """Fee rounding and fill-price granularity make the two differ slightly always.
    Alerting on that would train the operator to ignore the alert."""
    assert pnl_divergence(1.50, 1.00, 2.0) == 0.0


def test_the_incident_shape_is_flagged():
    """Engine books a small win while the account takes a large loss -- exactly what
    an unbooked stop-out looks like."""
    gap = pnl_divergence(+2.11, -71.43, 2.0)
    assert gap == pytest.approx(73.54)


def test_the_sign_says_which_way_it_lies():
    """Positive gap = the engine is overstating. That is the dangerous direction and
    the one actually observed."""
    assert pnl_divergence(+5.0, -5.0, 2.0) > 0
    assert pnl_divergence(-5.0, +5.0, 2.0) < 0


def test_a_zero_tolerance_disables_the_check():
    assert pnl_divergence(+100.0, -100.0, 0.0) == 0.0


def test_the_boundary_is_inclusive():
    """Exactly at tolerance is not yet a divergence."""
    assert pnl_divergence(2.0, 0.0, 2.0) == 0.0
    assert pnl_divergence(2.01, 0.0, 2.0) != 0.0


def test_the_two_verdicts_are_distinguishable():
    """Guard on the guard: a function stuck on one answer satisfies half the above."""
    assert pnl_divergence(0.0, 0.0, 2.0) != pnl_divergence(100.0, 0.0, 2.0)


# ----------------------------------------------------------------- the default
def test_the_shipped_tolerance_is_on_and_tight_enough_to_catch_a_stop():
    """A single forced close on this account was -74.84. A tolerance anywhere near
    that would miss the event the detector exists for."""
    from config import Settings
    default = Settings.model_fields["pnl_divergence_alert_usdt"].default
    assert 0 < default <= 10.0


# ------------------------------------------------------------ wired to the loop
def _loop_source() -> str:
    src = inspect.getsource(main_module.run_bot)
    return src[src.index("while True:"):]


def _divergence_call() -> "ast.Call | None":
    """The pnl_divergence call inside run_bot, found by parsing rather than by
    string search -- a substring test passes against a call that never executes,
    which is how the account-recheck wiring mutant survived its first test."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(main_module.run_bot)))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "pnl_divergence"):
            return node
    return None


def test_the_detector_actually_runs_in_the_loop():
    assert _divergence_call() is not None, "pnl_divergence is never called"


def _origin_of(name: str) -> "ast.AST | None":
    """What `name` was last assigned from inside run_bot."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(main_module.run_bot)))
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    found = node.value
    return found


def test_it_compares_deltas_not_totals():
    """The engine's total spans every session ever run; the reconciler's session
    figure resets on restart. Comparing the totals would report a divergence that is
    only a difference of window, and cry wolf on every restart.

    Resolves each argument to what it was assigned from rather than demanding a
    literal shape at the call site -- the first version of this asserted the args
    were BinOps and broke the moment the deltas were extracted into variables, which
    is a test pinning formatting instead of behaviour.
    """
    call = _divergence_call()
    assert call is not None
    for arg in call.args[:2]:
        expr = arg if not isinstance(arg, ast.Name) else _origin_of(arg.id)
        assert expr is not None, f"cannot resolve {ast.dump(arg)[:60]}"
        assert isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Sub), (
            "an argument is a raw total, not a change since the last check"
        )
        src = ast.dump(expr)
        assert "pnl_marks" in src, (
            "the subtraction is not against the previous mark, so it is not a delta "
            "over the interval"
        )


def test_the_marks_are_advanced_so_a_gap_is_not_re_reported_forever():
    """Must be advanced in the branch that DOES the comparison, not merely somewhere.

    The first version searched the loop text for the assignment. A mutant that deleted
    the advance from the comparing branch survived it, because the seeding assignment
    in `if pnl_marks is None:` still matched the same string -- the fifth time in this
    repo a substring test has been satisfied by a different occurrence than the one
    that matters.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(main_module.run_bot)))
    branch = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for stmt in node.body:
            for c in ast.walk(stmt):
                if (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                        and c.func.id == "pnl_divergence"):
                    branch = node
    assert branch is not None, "no branch computes pnl_divergence"

    advanced = any(
        isinstance(t, ast.Name) and t.id == "pnl_marks"
        for stmt in branch.body for n in ast.walk(stmt)
        if isinstance(n, ast.Assign) for t in n.targets
    )
    assert advanced, (
        "the branch that compares does not advance pnl_marks -- the same gap would be "
        "re-reported every interval forever"
    )


def test_the_first_pass_only_takes_a_mark_and_does_not_alert():
    """With no prior mark there is no interval to compare, and treating the opening
    totals as a delta would fire on the restored state of every restart."""
    loop = _loop_source()
    assert "if pnl_marks is None:" in loop


def test_it_tells_the_operator_which_number_to_believe():
    """The whole failure was two numbers on one line with no guidance. An alert that
    does not say which one is right repeats it."""
    loop = _loop_source()
    idx = loop.index("PNL DIVERGED")
    window = loop[idx:idx + 800]
    assert "account=" in window and "net=" in window
