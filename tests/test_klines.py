"""Market-data sources for replay. AUDIT #86.

replay.py was reading the bot's own PRICE= lines as if they were the price path. They
are samples of it, ~15s apart, and everything between two samples is invisible.
Measured on sessions that actually traded:

    2026-08-15 22:38   live 16 fills / 5 cycles
                       snapshots  6 / 2      1m candles  7 / 3      trades  8 / 4

Only the network-free logic is covered here; fetching is exercised by running
replay.py --klines against a real session.
"""

from pathlib import Path

import pytest

from klines import path_from_klines, session_window, verify_against_log


def candle(o, h, l, c, ts=0):
    return [ts, o, h, l, c, 100.0]


# --- expanding candles into a path -----------------------------------------------

def test_an_up_bar_is_assumed_to_dip_first():
    assert path_from_klines([candle(1.0, 1.2, 0.9, 1.1)]) == [1.0, 0.9, 1.2, 1.1]


def test_a_down_bar_is_assumed_to_pop_first():
    assert path_from_klines([candle(1.0, 1.2, 0.9, 0.95)]) == [1.0, 1.2, 0.9, 0.95]


def test_a_flat_bar_still_reaches_both_extremes():
    """The sequence is a guess; the extremes are not. Every resting order the market
    actually reached has to be reached here too."""
    p = path_from_klines([candle(1.0, 1.5, 0.5, 1.0)])
    assert min(p) == 0.5 and max(p) == 1.5


def test_every_candle_contributes_four_points():
    assert len(path_from_klines([candle(1, 2, 0.5, 1.5)] * 7)) == 28


def test_no_candles_is_an_empty_path():
    assert path_from_klines([]) == []


def test_extra_columns_are_tolerated():
    """ccxt appends fields on some venues; unpacking must not break on them."""
    assert path_from_klines([[0, 1.0, 1.2, 0.9, 1.1, 100.0, "extra"]])


# --- refusing to replay the wrong hours -------------------------------------------

def test_a_window_that_contains_the_session_is_accepted():
    assert verify_against_log([0.069, 0.071], [0.0695, 0.0705]) is None


def test_a_window_that_misses_the_session_is_rejected():
    """A wrong timezone or run index yields a plausible path for the WRONG hours, and
    every conclusion off it is quietly false."""
    problem = verify_against_log([0.050, 0.052], [0.0695, 0.0705])
    assert problem and "wrong window" in problem


def test_a_session_poking_slightly_outside_is_tolerated():
    """Candle boundaries do not align to the poll clock, so demand containment, not
    equality."""
    assert verify_against_log([0.0690, 0.0710], [0.06899, 0.07101]) is None


def test_missing_data_on_either_side_is_reported():
    assert verify_against_log([], [0.07]) is not None
    assert verify_against_log([0.07], []) is not None


# --- reading a session's time window ----------------------------------------------

def test_the_window_spans_the_first_and_last_stamp(tmp_path):
    log = tmp_path / "grid.log"
    log.write_text(
        "2026-08-15 22:38:13 | INFO | GRID BOT STARTING\n"
        "2026-08-15 22:39:00 | INFO | PRICE=0.07 |\n"
        "2026-08-16 01:00:00 | INFO | PRICE=0.07 |\n", encoding="utf-8")

    start, end = session_window(log, 0)
    assert end > start
    assert (end - start) / 3600000 == pytest.approx(2.36, abs=0.05)


def test_the_newest_run_is_the_default(tmp_path):
    log = tmp_path / "grid.log"
    log.write_text(
        "2026-08-15 01:00:00 | GRID BOT STARTING\n"
        "2026-08-15 01:30:00 | x\n2026-08-15 02:00:00 | x\n"
        "2026-08-15 10:00:00 | GRID BOT STARTING\n"
        "2026-08-15 10:30:00 | x\n2026-08-15 11:00:00 | x\n",
        encoding="utf-8")

    newest, older = session_window(log, 0), session_window(log, 1)
    assert newest[0] > older[0]


def test_a_log_with_no_runs_is_an_error(tmp_path):
    log = tmp_path / "grid.log"
    log.write_text("nothing here\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        session_window(log, 0)


def test_a_run_index_past_the_end_is_an_error(tmp_path):
    log = tmp_path / "grid.log"
    log.write_text("2026-08-15 01:00:00 | GRID BOT STARTING\n"
                   "2026-08-15 02:00:00 | x\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        session_window(log, 5)


def test_a_run_without_two_timestamps_is_an_error(tmp_path):
    log = tmp_path / "grid.log"
    log.write_text("2026-08-15 01:00:00 | GRID BOT STARTING\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        session_window(log, 0)
