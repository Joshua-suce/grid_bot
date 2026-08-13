"""Every stand-in for Exchange must be callable exactly like the real one.

AUDIT #38. `TrendFollower._close_position` and `StrategyRouter._continue_handoff` both
called `exchange.close_position(self.symbol)`. The real method is

    close_position(self, symbol, side, amount, max_attempts=None)

with no defaults on `side` or `amount`, so both calls were a TypeError against the live
exchange -- swallowed by a surrounding `except Exception`, logged as a generic "close
failed", and returning None. That made every exit path of the trend follower dead: the
trailing stop could not close, the regime-change exit could not close, and because the
early return happens before `_side` is cleared, the strategy stayed wedged believing it
still held the position and never re-entered.

410 tests passed throughout, because every fake exchange in the suite -- and in
backtest.py -- declared `close_position(self, symbol)`. The doubles were more permissive
than the real thing, so the tests proved the code worked against a signature production
does not have. That is the defect this file exists to prevent, and it is the same shape
as AUDIT #31, where main.py called a method the router could not forward.

The check is deliberately signature-only: it does not care what a fake DOES, only that
every call the codebase makes would bind against the real class too.
"""

import ast
import inspect
import pathlib

import pytest

from exchange import Exchange

REPO = pathlib.Path(__file__).resolve().parents[1]

# Methods a strategy or the router may call on whatever exchange object it is handed.
SHARED_SURFACE = [
    "close_position",
    "get_positions",
    "get_price",
    "cancel_order",
    "cancel_everything",
    "get_open_order_ids",
    "fetch_order",
    "place_limit_order",
]


def _fake_exchange_classes():
    """Every class in tests/ and backtest.py that defines close_position.

    Identified structurally rather than by name: a stand-in is anything that implements
    the method whose signature drifted.
    """
    found = []
    paths = sorted(REPO.glob("tests/*.py")) + [REPO / "backtest.py"]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "close_position":
                    found.append((path.name, node.name, item))
    return found


def _params(fn_node):
    a = fn_node.args
    names = [p.arg for p in a.posonlyargs + a.args if p.arg != "self"]
    return names, a.vararg is not None, a.kwarg is not None


FAKES = _fake_exchange_classes()


def test_the_scan_actually_finds_the_fakes():
    """A guard on the guard: if the discovery breaks, every test below passes vacuously."""
    assert len(FAKES) >= 4, f"expected several fake exchanges, found {FAKES}"


@pytest.mark.parametrize(
    "case", FAKES, ids=[f"{f}::{c}" for f, c, _ in FAKES]
)
def test_fake_close_position_accepts_the_real_call(case):
    """A fake must accept every argument the real Exchange requires.

    A fake that takes fewer makes production TypeErrors invisible; that is exactly how
    #38 survived three audits and 410 passing tests.
    """
    filename, classname, node = case
    names, has_varargs, has_kwargs = _params(node)
    if has_varargs and has_kwargs:
        return

    real = [
        p for p in inspect.signature(Exchange.close_position).parameters
        if p != "self"
    ]
    required = [
        p.name for p in inspect.signature(Exchange.close_position).parameters.values()
        if p.name != "self" and p.default is inspect.Parameter.empty
    ]
    missing = [p for p in required if p not in names]
    assert missing == [], (
        f"{filename}::{classname}.close_position{tuple(names)} cannot accept the call "
        f"the real Exchange.close_position{tuple(real)} requires -- missing {missing}. "
        f"A permissive fake hides production TypeErrors (AUDIT #38)."
    )


def _calls_on_exchange_attr(path: pathlib.Path):
    """Every `self.exchange.<method>(...)` call in a module, with its arity."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        v = f.value
        if not (isinstance(v, ast.Attribute) and v.attr == "exchange"
                and isinstance(v.value, ast.Name) and v.value.id == "self"):
            continue
        out.append((f.attr, len(node.args), tuple(sorted(k.arg for k in node.keywords if k.arg)),
                    node.lineno))
    return out


@pytest.mark.parametrize("module", ["grid.py", "trend_follower.py", "router.py"])
def test_every_exchange_call_binds_against_the_real_class(module):
    """The check that would have caught #38 on the day it was written.

    Each `self.exchange.<method>(...)` in the strategy layer is bound against the real
    Exchange signature. A call that cannot bind is a guaranteed runtime TypeError --
    and both offenders sat inside `except Exception`, so nothing would have surfaced it.
    """
    failures = []
    for name, nargs, kwnames, lineno in _calls_on_exchange_attr(REPO / module):
        real = getattr(Exchange, name, None)
        if not callable(real):
            continue        # ccxt passthrough (self.exchange.exchange...) or helper
        try:
            inspect.signature(real).bind(
                None, *[object()] * nargs, **{k: object() for k in kwnames}
            )
        except TypeError as e:
            failures.append(f"{module}:{lineno} exchange.{name}({nargs} args, {list(kwnames)}): {e}")

    assert failures == [], (
        "these calls cannot bind against the real Exchange and are runtime TypeErrors:\n"
        + "\n".join(failures)
    )
