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
    """Every call on the Exchange wrapper in a module, with its arity.

    Matches two spellings, because the modules differ:
      `self.exchange.<method>(...)`  -- the strategy layer
      `exchange.<method>(...)`       -- main.py, which holds a bare local

    AUDIT #47. Only the first was checked, so main.py -- which owns stop-loss placement
    -- was never covered. On 2026-08-08 that gap cost more than every other day of this
    account's history combined:

        18:18:11  Closed existing SHORT position: 31,761 DOGEUSDT   (~3.7x MAX_POSITION_PCT)
        18:20:38  Failed to place/update stop-loss:
                  Exchange.amount_to_precision() missing 1 required positional argument
        (x4, then EMERGENCY STOP)

    A TypeError inside `except Exception` at main.py:837 meant the stop-loss silently
    never existed, and an oversized position ran unprotected. Realised -48.92 on six
    closes -- 60% of the 15-day loss, in one day. Same shape as #38, one file over.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        v = f.value
        is_self_exchange = (isinstance(v, ast.Attribute) and v.attr == "exchange"
                            and isinstance(v.value, ast.Name) and v.value.id == "self")
        is_bare_exchange = isinstance(v, ast.Name) and v.id == "exchange"
        if not (is_self_exchange or is_bare_exchange):
            continue
        out.append((f.attr, len(node.args), tuple(sorted(k.arg for k in node.keywords if k.arg)),
                    node.lineno))
    return out


@pytest.mark.parametrize(
    "module", ["grid.py", "trend_follower.py", "router.py", "main.py"]
)
def test_every_exchange_call_binds_against_the_real_class(module):
    """The check that would have caught #38 on the day it was written -- and #47.

    Each call on the Exchange wrapper is bound against the real signature. A call that
    cannot bind is a guaranteed runtime TypeError, and these all sit inside
    `except Exception`, so nothing surfaces them until money is gone.
    """
    failures = []
    for name, nargs, kwnames, lineno in _calls_on_exchange_attr(REPO / module):
        real = getattr(Exchange, name, None)
        if real is None:
            # The original hole. `if not callable(real): continue` skipped calls to
            # methods that DO NOT EXIST on Exchange -- an AttributeError at runtime,
            # strictly worse than the TypeError this file was written to catch, and
            # silently waved through. That is precisely the 08-08 shape:
            # `exchange.amount_to_precision(q)` where the method lives on the ccxt
            # object (`exchange.exchange.…`), not on the wrapper.
            failures.append(
                f"{module}:{lineno} exchange.{name}(...) does not exist on Exchange "
                f"-- AttributeError at runtime (did you mean exchange.exchange.{name}?)"
            )
            continue
        if not callable(real):
            failures.append(
                f"{module}:{lineno} exchange.{name} is not callable ({type(real).__name__})"
            )
            continue
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


# --- #39: cache freshness and breaker accounting ---------------------------

def _bare_exchange():
    """An Exchange with cache state but no network, for pure cache-logic tests."""
    ex = Exchange.__new__(Exchange)
    ex._balance_cache = {}
    ex._balance_cache_at = {}
    ex._balance_cache_time = 0.0
    ex._balance_cache_ttl = 5.0
    return ex


def test_equity_is_not_kept_stale_by_the_free_balance_getter():
    """AUDIT #39. Both caches shared one timestamp, so whichever getter ran first
    refreshed it for all of them and the next one returned its own stale value.

    main.py calls get_balance_cached() at :990 and get_total_equity_cached() at :991,
    back to back -- so equity was fetched once at startup and served from cache for the
    rest of the run. That equity feeds risk.check_all's drawdown check, i.e. the kill
    switch, so a falling account was measured against a number that never fell.
    """
    import types

    ex = _bare_exchange()
    truth = {"free": 4800.0, "total": 4900.0}
    fetches = {"total": 0}
    ex.get_balance = types.MethodType(lambda self, a="USDT": truth["free"], ex)

    def total(self, a="USDT"):
        fetches["total"] += 1
        return truth["total"]

    ex.get_total_equity = types.MethodType(total, ex)

    seen = []
    for i in range(1, 5):
        truth["total"] = 4900.0 - i * 25
        truth["free"] = 4800.0 - i * 25
        ex.get_balance_cached()          # main.py:990 -- must not mask the next line
        seen.append(ex.get_total_equity_cached())
        ex._balance_cache_at = {k: v - 10.0 for k, v in ex._balance_cache_at.items()}

    assert fetches["total"] == 4, (
        f"equity fetched {fetches['total']} times in 4 iterations -- the free-balance "
        f"getter is still refreshing equity's freshness for it"
    )
    assert seen == [4875.0, 4850.0, 4825.0, 4800.0], seen


def test_a_missing_order_does_not_trip_the_circuit_breaker():
    """AUDIT #39. "Order does not exist" means the exchange ANSWERED. Counting it as a
    failure let routine probing open the breaker: a recenter cancels every order and
    then checks what it cancelled, which is five -2013 replies against a threshold of
    five -- and an open breaker refuses every request for 120s, stop-loss placement
    included."""
    import ccxt

    from exchange import CircuitBreaker

    ex = Exchange.__new__(Exchange)
    ex._circuit_breaker = CircuitBreaker()
    ex.max_retries = 1
    ex.retry_delay = 0
    ex._is_timestamp_error = lambda e: False

    def missing(*a, **k):
        raise ccxt.OrderNotFound("-2013 Order does not exist")

    for _ in range(ex._circuit_breaker.failure_threshold + 1):
        with pytest.raises(ccxt.OrderNotFound):
            ex._retry(missing, label="fetch_order")

    assert not ex._circuit_breaker.open, "routine order probes opened the circuit breaker"
    assert ex._circuit_breaker.failures == 0


def test_a_real_network_failure_still_trips_the_breaker():
    """The breaker must still do its job -- #39 narrowed what counts, not whether."""
    import ccxt

    from exchange import CircuitBreaker

    ex = Exchange.__new__(Exchange)
    ex._circuit_breaker = CircuitBreaker()
    ex.max_retries = 1
    ex.retry_delay = 0
    ex._is_timestamp_error = lambda e: False

    def down(*a, **k):
        raise ccxt.NetworkError("connection reset")

    for _ in range(ex._circuit_breaker.failure_threshold):
        with pytest.raises(Exception):
            ex._retry(down, label="fetch_positions")

    assert ex._circuit_breaker.open, "a genuinely unreachable exchange no longer trips the breaker"
