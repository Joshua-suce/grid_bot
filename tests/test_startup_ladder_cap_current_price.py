"""AUDIT #169. The ladder/cap coherence check (AUDIT #66/#120) references
`current_price` -- but on a restart WITH saved state, the common path since every
supervise.py restart hits it, `current_price` was only ever passed as a keyword
argument into grid.load_from_dict(current_price=exchange.get_price(...)), never
assigned to a local. Python scopes a name as local to the whole function the
moment it is assigned ANYWHERE in that function (it IS assigned, in the sibling
`else:` branch used only when no saved state exists), so referencing it on the
saved-state path raises UnboundLocalError every single time.

That exception is swallowed by the broad `except Exception` right below the check,
logged only as an unlabelled DEBUG line:

    2026-09-03 03:49:52 | DEBUG | Ladder/cap coherence check skipped: cannot
    access local variable 'current_price' where it is not associated with a value

Net effect: the exact safety check added to catch AUDIT #120's failure mode (a
grid wider than its cap goes one-sided, gets capped, sits stuck for hours -- 89
recenters in one session) has never actually run on a normal restart. It only
ever fired on a cold start with no saved state, which is the rare path.

Structural, not a full run_bot() integration test -- run_bot is not
unit-testable, per this file's own established precedent (see
tests/test_startup_keeps_stops.py::test_startup_cleanup_asks_to_keep_stops and
tests/test_ladder_fits_cap.py::test_main_warns_rather_than_aborting).
"""

import pathlib


def _main_source():
    path = pathlib.Path(__file__).resolve().parent.parent / "main.py"
    return path.read_text(encoding="utf-8")


def test_current_price_is_bound_before_the_ladder_cap_check_on_the_saved_state_path():
    source = _main_source()

    branch_start = source.index('if saved_state and "grid" in saved_state:')
    # The sibling top-level `else:` (8-space indent, matching the `if` above) marks
    # the end of the saved-state branch. Nested else clauses inside it (e.g. the
    # is_in_recovery() branch) sit at deeper indentation and won't match this.
    branch_end = source.index("\n        else:\n", branch_start)
    saved_state_branch = source[branch_start:branch_end]

    ladder_check = source.index("LADDER OUTGROWS THE CAP")
    assert branch_end < ladder_check, (
        "test's own assumptions about the source layout are stale -- "
        "the saved-state branch no longer precedes the ladder/cap check"
    )

    assert "current_price =" in saved_state_branch, (
        "current_price is never assigned to a local variable on the saved-state "
        "startup path (the common restart path) before the ladder/cap coherence "
        "check references it -- referencing an unassigned local raises "
        "UnboundLocalError, silently swallowed by the broad `except Exception` "
        "right after, so the AUDIT #120 safety check never actually runs on a "
        "normal restart"
    )


def test_the_ladder_cap_check_itself_still_only_warns():
    """Companion to test_ladder_fits_cap.py's own version of this assertion --
    kept here too so a future edit to the saved-state branch can't accidentally
    make the fix turn a warning into a hard abort."""
    source = _main_source()

    guard = source.index("LADDER OUTGROWS THE CAP")
    window = source[guard - 400:guard + 400]
    assert "logger.warning" in window
