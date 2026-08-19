"""How many timeframes must agree before the follower gets the symbol. AUDIT #117.

_merge_timeframes required a hardcoded `len(trending) >= 2`. The threshold itself is
defensible -- a trend verdict PAUSES the grid, so one timeframe's opinion should not stop
grid trading -- but being a constant made it both invisible and unadjustable.

On DOGEUSDT the 1h ADX sits inside the dead band for hours, so the only timeframe with an
opinion is often a single one. Every REGIME line of the 2026-08-19 00:19-03:00 run read:

    1h=ranging(adx=17.3) 30m=downtrend(adx=26.3) 1d=uncertain(adx=20.3)
    bands: range<=20 trend>=25 | 2 of 3 timeframe(s) voting -> ranging

30m was trending and alone. len(trending) was 1, the threshold was 2, and the follower was
never once eligible across 2h41m -- the AUDIT #33 "dormant bot" shape, reached through the
vote rather than the dead band.

WHAT THIS IS NOT: a free win. Setting it to 1 also pauses the grid on one timeframe's
say-so, and the follower's measured edge over 70 trades was -0.14%/trade, 95% interval
[-0.50%, +0.21%] -- consistent with zero. This makes the trade-off adjustable and visible,
not favourable. The default stays 2.
"""

import pytest

from trend_filter import MarketRegime, TrendFilter


def filt(votes, min_votes=2):
    f = TrendFilter(trend_min_votes=min_votes)
    f._timeframes = dict(votes)
    f._adx_by_timeframe = {tf: 26.0 for tf in votes}
    return f


UP, DOWN = MarketRegime.UPTREND, MarketRegime.DOWNTREND
RANGE, UNSURE = MarketRegime.RANGING, MarketRegime.UNCERTAIN

# The exact 2026-08-19 shape.
LIVE = {"1h": RANGE, "30m": DOWN, "1d": UNSURE}


# --- the live failure -------------------------------------------------------------------

def test_a_lone_trending_timeframe_is_ignored_by_default():
    """The premise. One trending timeframe against the default threshold of 2."""
    assert filt(LIVE)._merge_timeframes() != DOWN


def test_a_lone_trending_timeframe_wins_at_one_vote():
    """The knob. 30m=downtrend alone now carries the verdict."""
    assert filt(LIVE, min_votes=1)._merge_timeframes() == DOWN


def test_the_default_is_still_two():
    from config import Settings

    assert Settings.model_fields["regime_trend_min_votes"].default == 2


def test_the_constructor_default_is_two_as_well():
    """config and the class must not disagree; a caller that omits the argument should
    get the conservative behaviour, not whatever the class happens to prefer."""
    assert TrendFilter().trend_min_votes == 2


# --- the threshold still means something -------------------------------------------------

def test_two_agreeing_timeframes_trend_at_either_setting():
    for n in (1, 2):
        assert filt({"1h": UP, "30m": UP, "1d": UNSURE}, min_votes=n)._merge_timeframes() == UP


def test_disagreeing_directions_are_still_uncertain():
    """Two trending timeframes pointing opposite ways is not a trend at any threshold."""
    assert filt({"1h": UP, "30m": DOWN, "1d": UNSURE})._merge_timeframes() == UNSURE


def test_one_vote_does_not_resurrect_an_all_abstaining_board():
    """UNCERTAIN is an abstention (AUDIT #78). Lowering the bar must not turn silence
    into a verdict."""
    assert filt({"1h": UNSURE, "30m": UNSURE, "1d": UNSURE},
                min_votes=1)._merge_timeframes() == UNSURE


def test_ranging_is_unaffected_by_the_trend_threshold():
    """The knob gates TRENDING only. A ranging board must read the same either way."""
    board = {"1h": RANGE, "30m": RANGE, "1d": UNSURE}

    assert filt(board, min_votes=1)._merge_timeframes() == filt(board, min_votes=2)._merge_timeframes()


# --- guards -------------------------------------------------------------------------------

def test_zero_is_clamped_to_one():
    """0 would make an empty trending list satisfy `>= 0` and trend on nothing at all."""
    assert TrendFilter(trend_min_votes=0).trend_min_votes == 1
    assert filt({"1h": RANGE, "30m": RANGE, "1d": RANGE},
                min_votes=0)._merge_timeframes() != UP


def test_config_refuses_a_threshold_no_board_can_reach():
    """Above 3 nothing could ever trend, which is the dormancy this fixes, inverted."""
    from config import Settings

    field = Settings.model_fields["regime_trend_min_votes"]
    bounds = {type(m).__name__: getattr(m, "ge", getattr(m, "le", None))
              for m in field.metadata}

    assert bounds.get("Ge") == 1
    assert bounds.get("Le") == 3


# --- the log has to say why ----------------------------------------------------------------

def test_explain_reports_the_threshold():
    """The old line said '1 of 3 timeframe(s) voting' and stopped, so a board with one
    trending timeframe looked like an undecided market rather than a blocked vote."""
    line = filt(LIVE).explain()

    assert "trend needs 2" in line


def test_explain_names_the_blocked_handover():
    line = filt(LIVE).explain()

    assert "1 trending timeframe(s), 2 needed" in line
    assert "follower stays ineligible" in line


def test_explain_stays_quiet_when_nothing_is_trending():
    """Only report a blocked handover when one was actually blocked."""
    line = filt({"1h": RANGE, "30m": RANGE, "1d": UNSURE}).explain()

    assert "follower stays ineligible" not in line


def test_explain_stays_quiet_once_the_threshold_is_met():
    line = filt({"1h": UP, "30m": UP, "1d": UNSURE}).explain()

    assert "follower stays ineligible" not in line


# --- and it is wired ------------------------------------------------------------------------

def test_main_passes_the_knob():
    from pathlib import Path

    import main

    src = Path(main.__file__).read_text(encoding="utf-8")
    at = src.index("trend = TrendFilter(")
    block = src[at:at + 600]

    assert "trend_min_votes=settings.regime_trend_min_votes," in block
