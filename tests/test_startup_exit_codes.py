"""The process exit code is a message to supervise.py, so it is behaviour.

RestartPolicy.should_restart treats a clean exit as a deliberate stop and stays down
(supervise.py:63-70). Every startup guard used to end in a bare `return`, which exits
0 -- so "the exchange write path is unreachable" and "your config is wrong" said the
same thing to the supervisor, and the temporary one kept the bot down as if a human
had chosen it. AUDIT #126.
"""
from __future__ import annotations

import inspect
import re

import ccxt
import pytest
from loguru import logger

import main as main_module
from supervise import RestartPolicy


# --------------------------------------------------------------- the primitive
def test_a_transient_abort_asks_the_supervisor_to_try_again():
    with pytest.raises(SystemExit) as e:
        main_module.abort_startup(transient=True)
    assert e.value.code == 1


def test_a_permanent_abort_tells_the_supervisor_to_stay_down():
    with pytest.raises(SystemExit) as e:
        main_module.abort_startup(transient=False)
    assert e.value.code == 0


def test_the_two_codes_are_not_the_same():
    """The whole fix is that these differ. A mutant returning one code for both
    would satisfy every 'it exits' assertion ever written about this path."""
    codes = []
    for transient in (True, False):
        with pytest.raises(SystemExit) as e:
            main_module.abort_startup(transient=transient)
        codes.append(e.value.code)
    assert codes[0] != codes[1], "transient and permanent aborts must be distinguishable"


# ------------------------------------------------- the supervisor's half of it
@pytest.mark.parametrize("code,expected", [(0, False), (1, True)])
def test_supervise_honours_the_codes_abort_startup_emits(code, expected):
    """Pins the contract from the OTHER side. If supervise ever stops distinguishing
    them, abort_startup's care becomes decorative and this test says so."""
    policy = RestartPolicy(max_restarts=5, window_seconds=600)
    assert policy.should_restart(code) is expected


def test_a_transient_abort_actually_earns_a_restart():
    """End to end across the two modules, with no hardcoded literals: whatever
    abort_startup(transient=True) raises must be a code supervise will restart on."""
    with pytest.raises(SystemExit) as e:
        main_module.abort_startup(transient=True)
    assert RestartPolicy(max_restarts=5, window_seconds=600).should_restart(e.value.code)


# ------------------------------------------------------- auth vs unreachable
@pytest.mark.parametrize("exc", [
    ccxt.AuthenticationError("bad key"),
    Exception("binance {'code':-2015,'msg':'Invalid API-key, IP, or permissions'}"),
    Exception("binance {'code':-2014,'msg':'API-key format invalid'}"),
    Exception("Signature for this request is not valid"),
    Exception("401 Unauthorized"),
])
def test_credential_failures_are_not_transient(exc):
    assert main_module._is_auth_failure(exc) is True


@pytest.mark.parametrize("exc", [
    ccxt.NetworkError("Connection reset by peer"),
    ccxt.RequestTimeout("timed out"),
    Exception("HTTP 408 Request Timeout"),
    Exception("Max retries exceeded with url"),
    Exception("binance {'code':-1007,'msg':'Timeout waiting for response'}"),
])
def test_reachability_failures_are_transient(exc):
    assert main_module._is_auth_failure(exc) is False


# ------------------------------------------------------------ wired up for real
def _run_with_failing_exchange(monkeypatch, exc, tmp_path):
    class Boom:
        def __init__(self, *a, **k):
            raise exc

    monkeypatch.setattr(main_module, "Exchange", Boom)
    monkeypatch.setattr(main_module.settings, "telegram_enabled", False)
    monkeypatch.setattr(main_module.settings, "state_dir", str(tmp_path))
    monkeypatch.setattr(main_module, "setup_logging", lambda *a, **k: None)

    sink = []
    handle = logger.add(lambda m: sink.append(str(m)), level="INFO")
    try:
        main_module.run_bot()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    finally:
        logger.remove(handle)
    pytest.fail("run_bot returned instead of aborting")


def test_an_unreachable_venue_at_startup_asks_for_a_restart(monkeypatch, tmp_path):
    """2026-08-18 02:04: every signed Binance endpoint answered HTTP 408. Exiting 0
    there hands the supervisor a decision nobody made."""
    code = _run_with_failing_exchange(
        monkeypatch, ccxt.NetworkError("HTTP 408 Request Timeout"), tmp_path)
    assert code == 1


def test_bad_credentials_at_startup_stay_down(monkeypatch, tmp_path):
    """Rotated keys fail identically forever. Restarting into them is a busy loop."""
    code = _run_with_failing_exchange(
        monkeypatch, ccxt.AuthenticationError("Invalid API-key"), tmp_path)
    assert code == 0


# ------------------------------------------- the guards all speak this language
def _guard_region() -> str:
    """run_bot's startup half -- everything before the trading loop opens."""
    src = inspect.getsource(main_module.run_bot)
    cut = src.index("while True:")
    return src[:cut]


def _bare_returns_in_run_bots_own_body() -> list[int]:
    """Bare returns that belong to run_bot itself, not to a helper defined inside it.

    run_bot nests several small functions (_refresh_sl_stops, _sl_needs_update,
    _detect_trail_fill ...) and an early `return` inside one of those is ordinary
    control flow, not a process exit. Only returns at run_bot's own body depth end
    the process.
    """
    lines = _guard_region().splitlines()
    found, nested_at = [], None
    for i, ln in enumerate(lines):
        stripped, indent = ln.strip(), len(ln) - len(ln.lstrip())
        if re.match(r"def \w+", stripped):
            nested_at = indent
        elif stripped and nested_at is not None and indent <= nested_at \
                and not stripped.startswith(("#", '"', "'")):
            nested_at = None
        if re.fullmatch(r"return\s*", stripped) and nested_at in (None, 0):
            found.append(i + 1)
    return found


def test_no_startup_guard_still_exits_by_falling_out_of_the_function():
    """A bare `return` in this region is an exit code of 0 chosen by accident.

    This is the regression that matters: the fix is easy to make and easy to undo
    by adding one new guard in the old style. It already caught one the first pass
    missed -- the grid-spacing validation abort.
    """
    strays = _bare_returns_in_run_bots_own_body()
    assert not strays, (
        f"bare return(s) at run_bot-relative line(s) {strays} -- each exits 0 and "
        "tells supervise.py a human chose to stop (AUDIT #126)"
    )


def test_the_guard_region_scan_is_not_looking_at_an_empty_string():
    """Guard on the guard. If `while True:` ever moves or is reformatted, the slice
    above could silently become empty and the test above would pass vacuously --
    the same way the seed-call-site scan once matched its own definition."""
    region = _guard_region()
    assert len(region) > 2000, "guard region looks truncated; the scan proves nothing"
    assert "abort_startup" in region


def test_the_nested_def_filter_does_not_swallow_run_bots_own_returns():
    """The filter above is the kind of cleverness that silently makes a test vacuous.
    Prove it still SEES a bare return at run_bot's own depth by feeding it one."""
    lines = _guard_region().splitlines()
    body_indent = len(lines[1]) - len(lines[1].lstrip()) if len(lines) > 1 else 4
    probe = _guard_region() + "\n" + " " * body_indent + "return\n"

    import unittest.mock as _m
    with _m.patch.object(main_module, "run_bot") as fake:
        fake.__doc__ = None
        # exercise the parser directly rather than through inspect
        found, nested_at = [], None
        for i, ln in enumerate(probe.splitlines()):
            s, ind = ln.strip(), len(ln) - len(ln.lstrip())
            if re.match(r"def \w+", s):
                nested_at = ind
            elif s and nested_at is not None and ind <= nested_at \
                    and not s.startswith(("#", '"', "'")):
                nested_at = None
            if re.fullmatch(r"return\s*", s) and nested_at in (None, 0):
                found.append(i + 1)
    assert found, "the scan cannot see a bare return at run_bot's own depth"


def test_every_startup_guard_routes_through_abort_startup():
    """Counts real call sites, excluding the definition."""
    src = inspect.getsource(main_module)
    calls = re.findall(r"^[ \t]+abort_startup\(transient=", src, re.M)
    assert len(calls) >= 8, f"only {len(calls)} abort_startup call sites found"


def test_both_transient_verdicts_are_actually_used():
    """A fix that marked every site transient=True would restart-loop on a bad
    config; one that marked them all False is the original bug. Both must appear."""
    src = inspect.getsource(main_module)
    assert re.search(r"^[ \t]+abort_startup\(transient=True\)", src, re.M)
    assert re.search(r"^[ \t]+abort_startup\(transient=False\)", src, re.M)
