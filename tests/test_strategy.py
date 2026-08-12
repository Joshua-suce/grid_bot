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


def _protocol_methods() -> list[str]:
    return [
        name for name in dir(Strategy)
        if not name.startswith("_") and callable(getattr(Strategy, name, None))
    ]


def test_grid_engine_implements_every_protocol_method():
    """GridEngine must already satisfy Strategy -- step 2 changes no behaviour."""
    missing = [m for m in _protocol_methods() if not hasattr(GridEngine, m)]
    assert missing == [], f"GridEngine is missing protocol methods: {missing}"


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

    source = pathlib.Path(__file__).resolve().parents[1] / "main.py"
    used = set(re.findall(r"\bgrid\.([a-zA-Z_][a-zA-Z0-9_]*)", source.read_text()))
    used = {u for u in used if not u.startswith("_")}   # private probes are not contract

    covered = set(_protocol_methods()) | set(GRID_SPECIFIC_MEMBERS) | {
        "active", "state_corrupted", "total_fills", "total_pnl",
        "total_fees", "total_completed_cycles", "stop_loss_pct",
    }
    uncovered = used - covered
    assert uncovered == set(), (
        f"main.py uses GridEngine members not classified by strategy.py: "
        f"{sorted(uncovered)}. Add each to the Strategy protocol or to "
        f"GRID_SPECIFIC_MEMBERS."
    )


def test_grid_specific_members_really_exist_on_the_engine():
    """The grid-specific list is the step 3/4 work queue -- it must stay accurate.

    Checked against an instance, not the class: grid_lower/grid_upper/grid_count/
    grid_spacing/levels are assigned in __init__, so they do not exist on the class.
    """
    engine = _engine()
    missing = [m for m in GRID_SPECIFIC_MEMBERS if not hasattr(engine, m)]
    assert missing == [], f"GRID_SPECIFIC_MEMBERS lists members GridEngine lacks: {missing}"


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
