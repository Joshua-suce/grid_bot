"""Demo state and live state must never occupy the same file. AUDIT #67.

The filename was `grid_{symbol}.json` with no account distinction, so flipping
DEMO_MODE and restarting loaded one account's state against the other. What that
carries across is not cosmetic:

  - order ids from the other exchange, which check_fills reads as vanished orders
  - hard-stop ratchets anchored to the other account's prices
  - a PnL reconciler holding the other account's totals, and a last_income_time_ms far
    ahead of the new account's income, so real income before it is skipped forever
  - has_saved_grid True, which tells startup NOT to flatten a pre-existing position it
    knows nothing about (AUDIT #37)

Caught before the demo -> live switch rather than after it.
"""

import json
import sys

from loguru import logger

from state import StateManager


def _capture_logs(build):
    """Run `build` with a log sink attached and return (result, captured text)."""
    sink = []
    logger.remove()
    handle = logger.add(lambda m: sink.append(str(m)), level="INFO")
    try:
        result = build()
    finally:
        logger.remove(handle)
        logger.add(sys.stderr, level="INFO")
    return result, "".join(sink)


def test_demo_and_live_use_different_files(tmp_path):
    demo = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=True)
    live = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=False)

    assert demo.filepath != live.filepath
    assert demo.mode == "demo" and live.mode == "live"


def test_live_cannot_read_demo_state(tmp_path):
    """The whole point: a demo run must be invisible to a live start."""
    demo = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=True)
    demo.save({"grid": {"levels": [1, 2, 3]}, "pnl_reconciler": {"realized_pnl": 999.0}})

    live = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=False)

    assert live.load() is None, "live start inherited demo state"
    assert demo.load() is not None, "demo state was destroyed instead of isolated"


def test_each_account_keeps_its_own_state(tmp_path):
    demo = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=True)
    live = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=False)

    demo.save({"who": "demo"})
    live.save({"who": "live"})

    assert demo.load() == {"who": "demo"}
    assert live.load() == {"who": "live"}


def test_symbols_stay_separate_too(tmp_path):
    doge = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=True)
    sol = StateManager(state_dir=str(tmp_path), symbol="SOLUSDT", demo=True)

    doge.save({"who": "doge"})
    sol.save({"who": "sol"})

    assert doge.load() == {"who": "doge"}
    assert sol.load() == {"who": "sol"}


def test_a_legacy_file_is_reported_not_silently_adopted(tmp_path):
    """An unattributable file must not be loaded -- and must not vanish quietly either,
    or a fresh start looks clean while real state sits on disk."""
    legacy = tmp_path / "grid_dogeusdt.json"
    legacy.write_text(json.dumps({"grid": {"levels": [1]}}))

    sm, logs = _capture_logs(
        lambda: StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=True)
    )

    assert sm.load() is None, "the unattributable file was adopted"
    assert legacy.exists(), "the unattributable file was deleted"
    assert "LEGACY STATE IGNORED" in logs


def test_no_warning_once_the_account_has_its_own_state(tmp_path):
    """The legacy notice is for the migration, not a permanent nag."""
    (tmp_path / "grid_dogeusdt.json").write_text(json.dumps({"grid": {}}))
    StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=True).save({"a": 1})

    _, logs = _capture_logs(
        lambda: StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=True)
    )

    assert "LEGACY STATE IGNORED" not in logs
