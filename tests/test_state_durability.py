"""State has to survive the crash it exists for. AUDIT #57.

os.replace is atomic with respect to the rename, but without an fsync the rename can
land while the bytes behind it are still in the page cache. A power loss or hard kill
then leaves a present-but-EMPTY state file. load() handles that -- renames it aside and
starts fresh -- but starting fresh means losing the stop ratchets, which is #50b's
failure mode arriving by a different road.

And every caller discards save()'s result, so a disk that stopped accepting writes
produced one identical log line per iteration and nothing else, while every restart
quietly reloaded older state.
"""

import json

import pytest

from state import StateManager


def test_state_round_trips(tmp_path):
    sm = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT")
    assert sm.save({"hard_sl": 0.0673, "peak": 0.0710}) is True
    assert sm.load() == {"hard_sl": 0.0673, "peak": 0.0710}


def test_the_payload_is_fsynced_before_the_rename(tmp_path, monkeypatch):
    """Pin the ordering: fsync must happen while the temp file is still the temp file.
    An fsync after the rename does not protect the rename."""
    events = []

    real_fsync = __import__("os").fsync
    real_replace = __import__("os").replace

    def spy_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def spy_replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr("state.os.fsync", spy_fsync)
    monkeypatch.setattr("state.os.replace", spy_replace)

    StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT").save({"a": 1})

    assert events == ["fsync", "replace"], (
        f"durability ordering is wrong: {events} -- the rename can outrun the data"
    )


def test_a_failed_save_reports_failure(tmp_path, monkeypatch):
    sm = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT")
    monkeypatch.setattr("state.os.replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))

    assert sm.save({"a": 1}) is False
    assert sm.consecutive_save_failures == 1


def test_repeated_failures_accumulate_and_reset_on_success(tmp_path, monkeypatch):
    """The counter is what makes a persistently unwritable disk visible at all."""
    sm = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT")
    boom = lambda *a: (_ for _ in ()).throw(OSError("disk full"))
    monkeypatch.setattr("state.os.replace", boom)

    for _ in range(7):
        sm.save({"a": 1})
    assert sm.consecutive_save_failures == 7

    monkeypatch.undo()
    assert sm.save({"a": 1}) is True
    assert sm.consecutive_save_failures == 0


def test_a_failed_save_leaves_no_temp_files_behind(tmp_path, monkeypatch):
    """A save that fails every iteration must not fill the disk with .state_* files
    while it does so."""
    sm = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT")
    monkeypatch.setattr("state.os.replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))

    for _ in range(20):
        sm.save({"a": 1})

    assert list(tmp_path.glob(".state_*")) == []


def test_a_failed_save_does_not_destroy_the_previous_good_state(tmp_path, monkeypatch):
    sm = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT")
    sm.save({"hard_sl": 0.0673})

    monkeypatch.setattr("state.os.replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    sm.save({"hard_sl": 0.0999})

    monkeypatch.undo()
    assert sm.load() == {"hard_sl": 0.0673}, "a failed write corrupted good state"


def test_an_empty_state_file_is_moved_aside_rather_than_parsed(tmp_path):
    sm = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT")
    sm.filepath.write_text("")

    assert sm.load() is None
    assert sm.filepath.with_suffix(".empty").exists()


def test_a_corrupt_state_file_is_backed_up_not_deleted(tmp_path):
    sm = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT")
    sm.filepath.write_text("{not json")

    assert sm.load() is None
    # derived from filepath, not hardcoded -- the filename is account-scoped (#67)
    backups = list(tmp_path.glob(f"{sm.filepath.stem}.corrupt*"))
    assert backups, "corrupt state was discarded instead of preserved for diagnosis"
    assert backups[0].read_text() == "{not json"
