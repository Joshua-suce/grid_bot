"""Conformance tests for the Strategy protocol (step 2 of the multi-strategy plan).

These exist so the abstraction cannot silently rot. The protocol was derived from what
main.py actually calls on GridEngine, so if someone renames a method on the engine, the
first test here fails and names it -- rather than main.py failing at runtime against a
live account.
"""

import inspect

import pytest

from grid import GridEngine
from strategy import GRID_SPECIFIC_MEMBERS, Strategy
from trend_follower import TrendFollower


class _StubExchange:
    """Minimal stand-in so a GridEngine can be constructed for attribute checks."""

    class exchange:
        @staticmethod
        def amount_to_precision(symbol, amount):
            return f"{float(amount):.0f}"

        @staticmethod
        def price_to_precision(symbol, price):
            return f"{float(price):.5f}"


def _engine() -> GridEngine:
    return GridEngine(
        exchange=_StubExchange(), symbol="DOGEUSDT",
        grid_lower=0.0710, grid_upper=0.0730, grid_count=10,
        capital_per_grid_pct=0.018, stop_loss_pct=0.03,
    )


def _main_code() -> str:
    """main.py with comment text blanked out, every other offset preserved.

    The scrapers below ask which strategy members main.py USES. A regex over raw file
    text also reads prose: the phrase "grid.py" written in a comment yielded a member
    named 'py' and failed three tests with a message about adding it to the Strategy
    protocol, which is not remotely where the problem was. Tokenising finds comments
    properly -- including a '#' inside a string literal, which naive splitting gets
    wrong -- and blanking them in place leaves the regexes otherwise untouched.
    """
    import io
    import pathlib
    import tokenize

    path = pathlib.Path(__file__).resolve().parents[1] / "main.py"
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            (row, c0), (_, c1) = tok.start, tok.end
            line = lines[row - 1]
            lines[row - 1] = line[:c0] + " " * (c1 - c0) + line[c1:]
    return "\n".join(lines)


def test_the_scraper_reads_code_and_not_prose():
    """Guards _main_code itself. Writing "grid.py" in a main.py comment used to invent
    a member called 'py' and fail three tests with a message about the Strategy
    protocol -- an hour of looking in the wrong place. A real grid.<member> call must
    still be seen, or the fix would have disarmed the very check it was protecting.
    """
    import re

    code = _main_code()
    assert "grid.py" not in code, "a comment mentioning grid.py still reaches the regex"

    found = set(re.findall(r"\bgrid\.([a-zA-Z_][a-zA-Z0-9_]*)", code))
    assert "py" not in found
    assert found, "scraper found no members at all -- it is no longer reading main.py"
    assert "active" in found, "a real attribute access was lost along with the comments"


def _protocol_methods() -> list[str]:
    return [
        name for name in dir(Strategy)
        if not name.startswith("_") and callable(getattr(Strategy, name, None))
    ]


def _trend_follower() -> TrendFollower:
    return TrendFollower(exchange=_StubExchange(), symbol="DOGEUSDT")


IMPLEMENTATIONS = [GridEngine, TrendFollower]


@pytest.mark.parametrize("impl", IMPLEMENTATIONS, ids=lambda c: c.__name__)
def test_implements_every_protocol_method(impl):
    """Every strategy must satisfy Strategy. GridEngine does so unchanged, which is
    the point of step 2; TrendFollower was written against it in step 3."""
    missing = [m for m in _protocol_methods() if not hasattr(impl, m)]
    assert missing == [], f"{impl.__name__} is missing protocol methods: {missing}"


@pytest.mark.parametrize("impl", IMPLEMENTATIONS, ids=lambda c: c.__name__)
def test_implements_the_grid_specific_surface_too(impl):
    """main.py still calls these, so anything the router can install must answer them
    -- otherwise switching strategies would crash the trading loop on the first
    `grid.recenter()`. TrendFollower supplies harmless equivalents."""
    obj = _engine() if impl is GridEngine else _trend_follower()
    missing = [m for m in GRID_SPECIFIC_MEMBERS if not hasattr(obj, m)]
    assert missing == [], f"{impl.__name__} cannot stand in for main.py: missing {missing}"


@pytest.mark.parametrize("name", _protocol_methods())
def test_signatures_are_compatible(name):
    """Parameter names must line up, so main.py can call either by keyword."""
    proto = inspect.signature(getattr(Strategy, name))
    impl = inspect.signature(getattr(GridEngine, name))
    proto_params = [p for p in proto.parameters if p != "self"]
    impl_params = [p for p in impl.parameters if p != "self"]
    for p in proto_params:
        assert p in impl_params, (
            f"GridEngine.{name} has no '{p}' parameter (protocol declares {proto_params})"
        )


def test_protocol_covers_what_main_actually_calls():
    """Every GridEngine member main.py touches must be either in the protocol or
    explicitly listed as grid-specific. Nothing may be silently uncovered.

    This is the test that keeps the remaining coupling honest: adding a new
    `grid.something()` call to main.py fails here until it is classified.
    """
    import pathlib
    import re

    used = set(re.findall(r"\bgrid\.([a-zA-Z_][a-zA-Z0-9_]*)", _main_code()))

    covered = set(_protocol_methods()) | set(GRID_SPECIFIC_MEMBERS) | {
        "active", "state_corrupted", "total_fills", "total_pnl",
        "total_fees", "total_completed_cycles", "stop_loss_pct", "peak_price",
    }
    uncovered = used - covered
    assert uncovered == set(), (
        f"main.py uses GridEngine members not classified by strategy.py: "
        f"{sorted(uncovered)}. Add each to the Strategy protocol or to "
        f"GRID_SPECIFIC_MEMBERS."
    )


def test_main_never_reaches_into_a_private_member():
    """AUDIT #31. `grid` in main.py is a StrategyRouter whenever STRATEGY_MODE=router,
    and the router refuses to forward underscore names -- forwarding them would let an
    internal typo silently read a strategy's unrelated attribute.

    So a single `grid._something` in main.py raises AttributeError on the iteration it
    runs, every iteration, forever. That is not hypothetical: `grid._last_orderbook` in
    the status log took down the 19:22 demo run, and because the handler logged only
    str(e) the whole 27-minute log said `Loop error (consecutive=1): _last_orderbook`
    with no type and no traceback.

    The previous version of the test above filtered private names out as "not
    contract". They are exactly the contract that breaks.
    """
    import pathlib
    import re

    private = sorted(set(re.findall(r"\bgrid\.(_[a-zA-Z0-9_]*)", _main_code())))
    assert private == [], (
        f"main.py reaches into private strategy members {private}, which the router "
        f"cannot forward. Add a public accessor to the Strategy protocol instead."
    )


def test_protocol_state_attributes_exist_on_an_instance():
    """The protocol declares plain attributes too, and those are also set in __init__."""
    engine = _engine()
    for attr in ("active", "state_corrupted", "total_fills", "total_pnl",
                 "total_fees", "total_completed_cycles"):
        assert hasattr(engine, attr), f"GridEngine instance lacks '{attr}'"


def test_grid_specific_members_are_not_also_in_the_protocol():
    """The two lists must be disjoint, or the boundary means nothing."""
    overlap = set(GRID_SPECIFIC_MEMBERS) & set(_protocol_methods())
    assert overlap == set(), f"members classified both ways: {sorted(overlap)}"


def test_runtime_isinstance_check_works_on_a_real_engine():
    """runtime_checkable Protocols only verify attribute presence, but that is enough
    for the router in step 4 to reject an object that is not a strategy at all."""
    class NotAStrategy:
        pass

    assert not isinstance(NotAStrategy(), Strategy)


# --- #31: what main.py touches must actually resolve on the router ---------

def _router_over_real_strategies():
    """A StrategyRouter over real GridEngine + TrendFollower instances, as main.py's
    _install_strategy builds it."""
    from router import StrategyRouter

    return StrategyRouter(
        strategies={"grid": _engine(), "trend": _trend_follower()},
        default="grid", exchange=None, symbol="DOGEUSDT",
    )


def _members_main_touches() -> list[str]:
    import pathlib
    import re

    return sorted(set(re.findall(r"\bgrid\.([a-zA-Z_][a-zA-Z0-9_]*)", _main_code())))


@pytest.mark.parametrize("active", ["grid", "trend"])
def test_every_member_main_touches_resolves_on_the_router(active):
    """AUDIT #31, and the test that would have caught it.

    In router mode `grid` in main.py is a StrategyRouter, not a GridEngine. The
    classification test above only checks that each name is *listed* somewhere in
    strategy.py -- it never asks whether the name actually resolves through the router
    at runtime. `_last_orderbook` was listed nowhere and filtered out as private, and
    the router refuses to forward private names, so the live loop raised AttributeError
    on every single iteration for 27 minutes.

    This resolves each name against a real router with each strategy live, which is the
    thing production actually does.
    """
    router = _router_over_real_strategies()
    router.active_name = active

    unresolved = []
    for name in _members_main_touches():
        try:
            getattr(router, name)
        except AttributeError:
            unresolved.append(name)

    assert unresolved == [], (
        f"main.py touches {unresolved} on the strategy, but they do not resolve on a "
        f"StrategyRouter with '{active}' live -- that is an AttributeError every "
        f"iteration in router mode."
    )


def test_stats_written_through_the_router_reach_the_live_strategy():
    """main.py assigns to these when it rebuilds the engine after recovery. Without
    __setattr__ forwarding they land in the router's own __dict__, where they shadow
    the real strategy's values for the rest of the process."""
    router = _router_over_real_strategies()

    router.total_fills = 17
    router.total_pnl = 4.25
    router.peak_price = 0.0812

    grid = router.strategies["grid"]
    assert grid.total_fills == 17
    assert grid.total_pnl == 4.25
    assert grid.peak_price == 0.0812
    assert "total_fills" not in router.__dict__, "write shadowed on the router"
    assert router.total_fills == 17


def test_the_router_keeps_its_own_state_out_of_the_strategies():
    """Forwarding writes must not send the router's own bookkeeping downstream."""
    router = _router_over_real_strategies()
    router.switches = 5
    router.active_name = "trend"

    assert router.switches == 5
    assert router.active_name == "trend"
    assert not hasattr(router.strategies["grid"], "switches")


@pytest.mark.parametrize("name", [m for m in GRID_SPECIFIC_MEMBERS])
def test_grid_specific_members_are_callable_the_same_way(name):
    """hasattr is not enough for the forwarded surface.

    main.py calls these on whatever the router has live, with the arguments GridEngine
    expects. `TrendFollower.log_sl_status(self)` satisfied hasattr while
    `grid.log_sl_status(position_side)` raised TypeError on every stop-status log with
    the follower active (AUDIT #31). Either accept the same parameters or take
    *args/**kwargs and ignore them.
    """
    grid_member = getattr(GridEngine, name, None)
    trend_member = getattr(TrendFollower, name, None)
    if not callable(grid_member) or not callable(trend_member):
        pytest.skip(f"{name} is not a method on both")

    trend_params = inspect.signature(trend_member).parameters
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in trend_params.values()):
        return  # accepts anything

    grid_params = list(inspect.signature(grid_member).parameters)
    missing = [p for p in grid_params if p not in trend_params]
    assert missing == [], (
        f"TrendFollower.{name}{tuple(trend_params)} cannot take the call main.py makes "
        f"via GridEngine.{name}{tuple(grid_params)} -- missing {missing}"
    )


def _grid_calls_in_main() -> list[tuple[str, int, tuple]]:
    """Every `grid.<name>(...)` call site in main.py as (name, n_positional, keywords)."""
    import ast
    import pathlib

    # Parses the real syntax tree, so comments are already excluded and this one never
    # needed _main_code(). Left reading the file directly on purpose.
    source = pathlib.Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "grid":
            keywords = tuple(sorted(k.arg for k in node.keywords if k.arg))
            calls.append((func.attr, len(node.args), keywords))
    return sorted(set(calls))


@pytest.mark.parametrize("impl", IMPLEMENTATIONS, ids=lambda c: c.__name__)
def test_every_call_main_makes_would_bind(impl):
    """Resolving the attribute is only half of it -- the call has to bind too.

    This reads main.py's actual call sites and checks each against the signature on
    each strategy, so an argument-count drift on the forwarded surface fails here
    instead of raising TypeError against a live account (AUDIT #31).
    """
    failures = []
    for name, n_args, keywords in _grid_calls_in_main():
        member = getattr(impl, name, None)
        if not callable(member):
            continue
        try:
            inspect.signature(member).bind(
                None, *[object()] * n_args, **{k: object() for k in keywords}
            )
        except TypeError as e:
            failures.append(f"{impl.__name__}.{name}({n_args} args, {list(keywords)}): {e}")

    assert failures == [], "main.py makes calls these strategies cannot accept:\n" + "\n".join(failures)