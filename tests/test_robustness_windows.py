"""--robustness did not test robustness.

It shifted the START by 6 candles per run via the warmup and never moved the end:
12 runs over 4,320 candles moved the start 66 candles -- 2.75 days of 180 -- and
shared about 98.5% of their data. Twelve views of one window, reported as twelve
trials, under a flag whose own help says "Do this before believing any single
result."

The direction of the error is the dangerous one. A spread measured across
98.5%-identical data UNDERSTATES the real variance, so weak edges look robust.
Measured on the same file and config the moment disjoint windows were introduced:

    old, 12 start offsets   mean -87.99  stdev 34.65  ratio -2.54  "exceeds noise"
    new, 4 disjoint windows mean -23.64  stdev 77.83  ratio -0.30  "chance"

Same data, same config. The standard deviation more than doubles once the runs stop
sharing their candles, and a verdict of "signal" becomes a verdict of "noise".
AUDIT #141.
"""
from __future__ import annotations

import inspect

import pytest

import run_backtest as rb


# ------------------------------------------------------------- the windows
def test_windows_do_not_overlap():
    """The whole defect in one assertion."""
    windows = rb.disjoint_windows(4320, 4)
    for (a1, b1), (a2, b2) in zip(windows, windows[1:]):
        assert b1 <= a2, f"window {(a1, b1)} overlaps {(a2, b2)}"


def test_windows_move_the_end_not_just_the_start():
    """The old mode kept the same end candle for every run, which is why they shared
    98.5% of their data."""
    ends = [b for _, b in rb.disjoint_windows(4320, 4)]
    assert len(set(ends)) == len(ends), "every window ends at the same candle"


def test_the_windows_cover_the_data():
    windows = rb.disjoint_windows(4320, 4)
    assert windows[0][0] == 0
    covered = sum(b - a for a, b in windows)
    assert covered >= 4320 - 4, f"only {covered} of 4320 candles measured"


def test_each_window_is_long_enough_to_warm_up_and_still_trade():
    """A window barely longer than the warmup measures the ATR seed, not a strategy."""
    for a, b in rb.disjoint_windows(4320, 4):
        assert (b - a) > rb.ROBUSTNESS_WARMUP * 2


def test_too_many_windows_is_refused_rather_than_silently_useless():
    """Splitting 4,320 candles 100 ways gives 43-candle windows that cannot absorb a
    50-candle warmup. Returning them anyway would report a confident mean of nothing."""
    with pytest.raises(ValueError, match="warmup"):
        rb.disjoint_windows(4320, 100)


def test_a_single_window_is_refused():
    """One trial is not a robustness check, and reporting a stdev of 0.0 for it would
    read as perfect consistency."""
    with pytest.raises(ValueError):
        rb.disjoint_windows(4320, 1)


@pytest.mark.parametrize("count", [2, 3, 4, 6, 8])
def test_it_holds_for_every_sensible_split(count):
    windows = rb.disjoint_windows(4320, count)
    assert len(windows) == count
    for (a1, b1), (a2, b2) in zip(windows, windows[1:]):
        assert b1 <= a2


def test_the_overlap_check_would_actually_catch_an_overlap():
    """Guard on the guard: prove the assertion above can fail, or it is decorative."""
    bad = [(0, 100), (50, 150)]
    overlapping = [b1 > a2 for (a1, b1), (a2, b2) in zip(bad, bad[1:])]
    assert any(overlapping)


# ------------------------------------------------------ wired into the command
def test_robustness_runs_the_windows_rather_than_shifting_the_warmup():
    """The old code passed warmup=offset to move the start. Anything still doing that
    is measuring the same window again."""
    src = inspect.getsource(rb)
    assert "disjoint_windows(" in src
    assert "50 + i * 6" not in src, "the start-offset mode is still present"


def test_each_run_gets_its_own_slice_of_the_dataframe():
    src = inspect.getsource(rb)
    assert "df.iloc[a:b]" in src, "every run still sees the whole frame"


def test_the_report_says_the_windows_share_nothing():
    """The old output said "12 start offsets" and read as 12 trials. A reader has to
    be able to tell what was actually measured."""
    src = inspect.getsource(rb)
    assert "NON-OVERLAPPING" in src
    assert "shared data" in src


def test_the_verdict_admits_how_few_independent_trials_there_are():
    """Four genuinely independent windows is weak evidence, and saying "exceeds the
    noise floor" without that caveat is how the old flag misled."""
    src = inspect.getsource(rb)
    idx = src.index("exceeds the noise floor")
    assert "weak evidence" in src[idx:idx + 400]


def test_the_deprecated_flag_still_works_but_says_so():
    """--offsets is in every script and note anyone has written. Breaking it silently
    would be worse than accepting it with a warning."""
    src = inspect.getsource(rb)
    assert '"--offsets"' in src
    assert "Deprecated alias" in src
