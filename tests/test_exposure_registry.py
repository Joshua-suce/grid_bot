"""One exposure budget for the whole account, not one per process. AUDIT #97.

max_exposure_pct reads like an account limit and is enforced like a per-symbol one.
RiskManager holds the number, GridEngine.get_exposure_pct measures ITS symbol, and
risk.check_all compares them -- all inside one process. Point a second instance at the
same Binance account on another pair and each independently permits the full 50%: the
account reaches 100% with both halves reporting healthy, and neither can see the other.

That matters because running more symbols is the honest answer to trading more. On
2026-08-17 the DOGE ladder used 2.1% of its 50% cap, so the capital for a second pair is
already sitting there -- and this is the guard that has to exist before it is used.
"""

import json
import time

import pytest

from exposure_registry import STALE_AFTER_SECONDS, ExposureRegistry


@pytest.fixture
def registries(tmp_path):
    def make(symbol, stale_after=STALE_AFTER_SECONDS):
        return ExposureRegistry(tmp_path, demo=True, symbol=symbol, stale_after=stale_after)
    return make


# --- the hazard ----------------------------------------------------------------------

def test_two_instances_see_one_anothers_exposure(registries):
    """The whole point. Alone, each of these is a healthy 30% against a 50% cap."""
    doge, sol = registries("DOGEUSDT"), registries("SOLUSDT")

    doge.publish(0.30)
    sol.publish(0.30)

    assert doge.account_exposure_pct(0.30) == pytest.approx(0.60)
    assert sol.account_exposure_pct(0.30) == pytest.approx(0.60)


def test_a_lone_instance_is_unaffected(registries):
    """Today's configuration. The registry must not invent exposure that is not there."""
    doge = registries("DOGEUSDT")
    doge.publish(0.021)

    assert doge.account_exposure_pct(0.021) == pytest.approx(0.021)
    assert doge.describe_others() == ""


def test_the_risk_cap_is_evaluated_against_the_account(registries):
    """Stated as the decision it drives rather than as arithmetic."""
    from risk import RiskManager

    risk = RiskManager(max_exposure_pct=0.50)
    doge, sol = registries("DOGEUSDT"), registries("SOLUSDT")
    doge.publish(0.30)
    sol.publish(0.30)

    assert risk._check_exposure(0.30) is True, "one ladder alone is inside the cap"
    assert risk._check_exposure(doge.account_exposure_pct(0.30)) is False, (
        "the account is at 60% of a 50% cap and the check passed anyway"
    )


# --- a process's own figure is never taken from the file ------------------------------

def test_own_exposure_comes_from_the_caller_not_the_file(registries):
    """Read-modify-write between processes can drop an update. Dropping your OWN would
    under-report the account, which is the unsafe direction, so it is never read back."""
    doge = registries("DOGEUSDT")
    doge.publish(0.10)

    # The file still says 10%; the live figure has moved to 40% this instant.
    assert doge.account_exposure_pct(0.40) == pytest.approx(0.40)


def test_a_lost_write_by_another_process_self_heals(registries, tmp_path):
    """Others' entries can be lost to a race. Every process republishes every poll, so
    the exposure is back within one interval -- the bound this design accepts."""
    doge, sol = registries("DOGEUSDT"), registries("SOLUSDT")
    doge.publish(0.20)
    sol.publish(0.25)

    (tmp_path / "exposure_demo.json").write_text(json.dumps({"DOGEUSDT": {
        "pct": 0.20, "at": time.time()}}), encoding="utf-8")
    assert doge.account_exposure_pct(0.20) == pytest.approx(0.20), "sanity: SOL is gone"

    sol.publish(0.25)                                    # its very next poll
    assert doge.account_exposure_pct(0.20) == pytest.approx(0.45)


# --- dead processes must not hold budget forever --------------------------------------

def test_a_stale_entry_is_ignored(registries):
    """A process that died holds no position. Counting its last reading forever would
    wedge the survivors under a cap they cannot get below by any action they can take."""
    doge, sol = registries("DOGEUSDT"), registries("SOLUSDT")
    doge.publish(0.20)
    sol.publish(0.25, now=time.time() - STALE_AFTER_SECONDS - 1)

    assert doge.account_exposure_pct(0.20) == pytest.approx(0.20)
    assert doge.others() == {}


def test_a_fresh_entry_just_inside_the_window_still_counts(registries):
    doge, sol = registries("DOGEUSDT"), registries("SOLUSDT")
    doge.publish(0.20)
    sol.publish(0.25, now=time.time() - (STALE_AFTER_SECONDS - 5))

    assert doge.account_exposure_pct(0.20) == pytest.approx(0.45)


def test_stale_entries_are_swept_on_publish(registries, tmp_path):
    """Otherwise a long-lived deployment accumulates every symbol it ever ran."""
    doge, sol = registries("DOGEUSDT"), registries("SOLUSDT")
    sol.publish(0.25, now=time.time() - STALE_AFTER_SECONDS - 1)

    doge.publish(0.20)

    assert set(json.loads((tmp_path / "exposure_demo.json").read_text())) == {"DOGEUSDT"}


# --- it must never be the thing that stops the bot ------------------------------------

def test_a_corrupt_file_is_treated_as_empty(registries, tmp_path):
    """A torn write from a concurrent process must cost one poll of visibility, not the
    trading loop. The file is rebuilt every interval anyway."""
    doge = registries("DOGEUSDT")
    (tmp_path / "exposure_demo.json").write_text("{not json", encoding="utf-8")

    assert doge.account_exposure_pct(0.20) == pytest.approx(0.20)
    doge.publish(0.20)
    assert json.loads((tmp_path / "exposure_demo.json").read_text())["DOGEUSDT"]["pct"] == 0.20


def test_an_unwritable_directory_does_not_raise(registries, tmp_path, monkeypatch):
    """publish() is on the hot path. It degrades to this process's own view -- which is
    exactly the behaviour that existed before the registry -- and says so once."""
    doge = registries("DOGEUSDT")

    def boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("exposure_registry.tempfile.mkstemp", boom)

    doge.publish(0.20)                       # must not raise
    assert doge.account_exposure_pct(0.20) == pytest.approx(0.20)


def test_demo_and_live_never_share_a_budget(tmp_path):
    """Different accounts with different balances. A paper position throttling a real one
    is the AUDIT #67 class of bug, and the file name is the only thing preventing it."""
    demo = ExposureRegistry(tmp_path, demo=True, symbol="DOGEUSDT")
    live = ExposureRegistry(tmp_path, demo=False, symbol="DOGEUSDT")

    demo.publish(0.45)

    assert live.filepath != demo.filepath
    assert live.account_exposure_pct(0.10) == pytest.approx(0.10)


def test_the_file_is_swapped_atomically_not_written_in_place(registries, tmp_path):
    """A reader in another process must never see a half-written budget. Assert no
    leftover temp files and valid JSON after a burst of publishes."""
    doge = registries("DOGEUSDT")
    for pct in (0.1, 0.2, 0.3):
        doge.publish(pct)

    assert json.loads((tmp_path / "exposure_demo.json").read_text())["DOGEUSDT"]["pct"] == 0.3
    assert [p.name for p in tmp_path.glob(".exposure*")] == [], "temp file left behind"


# --- wiring -------------------------------------------------------------------------

def test_the_trading_loop_checks_the_account_figure_not_its_own():
    """The registry is inert unless risk.check_all is given its answer. Both call sites
    matter: a PAUSED strategy still holds its position, so it still spends budget."""
    from pathlib import Path

    import main

    src = Path(main.__file__).read_text(encoding="utf-8")
    checks = [src[m:m + 400] for m in
              [i for i in range(len(src)) if src.startswith("risk.check_all(", i)]]

    assert len(checks) >= 2, f"expected both risk.check_all sites, found {len(checks)}"
    for call in checks:
        assert "account_exposure" in call or "account_exposure_pct" in call, (
            "a risk.check_all still measures one symbol against an account-wide cap:\n"
            + call.splitlines()[0]
        )
