"""Ranking spacings, or refusing to. AUDIT #99.

replay.py --sweep printed a spacing table and its own docstring told you not to believe
it. Three sweeps named three winners, and the cause was never sample size: every spacing
in a row was replayed against the SAME price path, so a row is a block, and the number
carrying information is the DIFFERENCE within the row rather than the level. Pooling the
levels leaves the session's path in the answer, and the path is bigger than the effect.

The live table from that docstring is the fixture below. Read as levels it disagrees with
itself; read as within-session differences it becomes a well-posed question with -- as it
turns out -- the honest answer "not on this evidence".
"""

import math

import pytest

from spacing_study import (Comparison, SessionResult, compare_spacings, format_report,
                           level_floor, rank, resolution_bias, split_by_floor,
                           t_critical_95, theoretical_optimum)

# The three sessions from replay.py's docstring, verbatim.
LIVE = [
    SessionResult("2026-08-15 -0", 2.3,
                  {0.0008: -0.00, 0.0010: 1.66, 0.0012: -0.25, 0.0015: 0.89,
                   0.0020: 1.81, 0.0025: 2.27, 0.0030: 2.64}),
    SessionResult("2026-08-15 -2", 8.2,
                  {0.0008: 3.58, 0.0010: 2.74, 0.0012: 3.41, 0.0015: 2.67,
                   0.0020: 1.15, 0.0025: 0.65, 0.0030: -0.15}),
    SessionResult("2026-08-16 -2", 2.6,
                  {0.0008: 8.49, 0.0010: 3.95, 0.0012: 6.49, 0.0015: 2.84,
                   0.0020: 4.91, 0.0025: 7.36, 0.0030: 8.35}),
]


# --- normalisation --------------------------------------------------------------------

def test_sessions_are_compared_per_day_not_per_session():
    """Sessions run from 20 minutes to 8 hours. Raw USDT calls a long session's result a
    spacing effect: +3.58 over 8.2h is a WORSE rate than +2.64 over 2.3h."""
    long_run, short_run = LIVE[1], LIVE[0]

    assert long_run.net[0.0008] > short_run.net[0.0030], "fixture no longer shows the trap"
    assert long_run.per_day(0.0008) < short_run.per_day(0.0030)


def test_a_zero_length_session_is_refused_rather_than_divided_by():
    with pytest.raises(ValueError):
        SessionResult("empty", 0.0, {0.002: 1.0}).per_day(0.002)


# --- the pairing, which is the whole point --------------------------------------------

def test_differences_are_taken_within_a_session():
    """Each row was replayed against one path, so differencing inside the row divides
    that path out. This is the operation the pooled table never performed."""
    c = next(c for c in compare_spacings(LIVE, baseline=0.0020) if c.spacing == 0.0008)

    expected = [r.per_day(0.0008) - r.per_day(0.0020) for r in LIVE]
    assert c.diffs == pytest.approx(expected)
    assert c.n == 3


def test_pairing_removes_between_session_scale():
    """The property that makes the design work. Scale one session's whole row by 10 --
    a wilder path, every spacing moved together -- and the LEVELS explode while the
    differences move only by that session's own contribution."""
    louder = [SessionResult(r.name, r.hours, {s: v * 10 for s, v in r.net.items()})
              if i == 2 else r
              for i, r in enumerate(LIVE)]

    pooled_before = sum(r.per_day(0.0008) for r in LIVE) / 3
    pooled_after = sum(r.per_day(0.0008) for r in louder) / 3

    assert abs(pooled_after - pooled_before) > 100, "fixture does not shift the levels"


def test_the_baseline_is_not_compared_with_itself():
    assert all(c.spacing != 0.0020 for c in compare_spacings(LIVE, baseline=0.0020))


def test_a_session_missing_a_spacing_is_skipped_not_zero_filled():
    """A missing replay is absent evidence. Scoring it as 0.0 invents a data point that
    happens to say 'no difference', which drags every interval toward a false verdict."""
    partial = LIVE + [SessionResult("partial", 3.0, {0.0020: 1.0})]

    c = next(c for c in compare_spacings(partial, baseline=0.0020) if c.spacing == 0.0008)
    assert c.n == 3


# --- the interval and the refusal -----------------------------------------------------

def test_the_live_table_yields_no_winner():
    """The result that matters, and the one three sweeps got wrong. On these three
    sessions every interval spans zero: there is no winner to report."""
    winner, comparisons = rank(LIVE, baseline=0.0020)

    assert winner is None, f"named {winner.spacing:.2%} a winner off three sessions"
    assert all(not c.decisive for c in comparisons)


def test_the_largest_mean_is_not_promoted_to_a_winner():
    """Largest mean of a set of indistinguishable numbers is not a winner. Here that is
    0.30%, on a spread of +8.7 / -3.8 / +31.8 USDT/day -- one session doing nearly all
    the talking."""
    _winner, comparisons = rank(LIVE, baseline=0.0020)
    best = max(comparisons, key=lambda c: c.mean)

    assert best.spacing == 0.0030
    assert best.mean > 0
    assert not best.decisive, "the largest mean was treated as a result"


def test_normalising_by_duration_changes_which_spacing_leads():
    """Worth pinning, because it is half of why the old answer was wrong. Pooling raw
    session totals makes 0.08% the leader -- but that pools an 8.2h session against a
    2.3h one, so it is mostly a statement about which session ran longest. Per day, the
    lead moves to 0.30%. Neither is a winner; the point is that the old ranking was not
    even measuring a rate."""
    totals = {s: sum(r.net[s] for r in LIVE) for s in (0.0008, 0.0030)}
    assert totals[0.0008] > totals[0.0030], "raw totals no longer favour 0.08%"

    means = {c.spacing: c.mean for c in compare_spacings(LIVE, baseline=0.0020)}
    assert means[0.0030] > means[0.0008], "per-day differencing no longer flips the lead"


def test_a_genuine_effect_is_detected():
    """The refusal must not be unconditional -- a real, consistent edge has to surface,
    or the module is just a way of never answering."""
    sessions = [
        SessionResult(f"s{i}", 4.0, {0.0020: 1.0 + i * 0.5, 0.0008: 3.0 + i * 0.5})
        for i in range(8)
    ]

    winner, _ = rank(sessions, baseline=0.0020)

    assert winner is not None and winner.spacing == 0.0008
    lo, hi = winner.ci
    assert lo > 0, "a perfectly consistent +2/day edge was still called indistinct"


def test_a_consistently_worse_spacing_is_marked_worse_not_promoted():
    sessions = [SessionResult(f"s{i}", 4.0, {0.0020: 2.0, 0.0030: 0.5}) for i in range(8)]

    winner, comparisons = rank(sessions, baseline=0.0020)
    worse = next(c for c in comparisons if c.spacing == 0.0030)

    assert winner is None
    assert worse.decisive and worse.mean < 0


def test_noise_of_the_same_size_as_the_effect_stays_indistinct():
    """Consistency, not magnitude, is what an interval responds to."""
    swings = [+6.0, -5.0, +7.0, -6.0, +5.0, -4.0]
    sessions = [SessionResult(f"s{i}", 4.0, {0.0020: 0.0, 0.0008: d})
                for i, d in enumerate(swings)]

    winner, _ = rank(sessions, baseline=0.0020)
    assert winner is None


def test_a_spacing_that_changed_nothing_is_not_a_discovery():
    """Every session identical to the baseline: mean 0, spread 0, interval exactly
    [0, 0]. That is the strongest evidence of NO effect available, and an interval which
    merely TOUCHES zero must not be read as excluding it -- the boundary is the whole
    difference between 'no difference' and 'a difference of zero, confidently'."""
    sessions = [SessionResult(f"s{i}", 4.0, {0.0020: 2.0, 0.0025: 2.0}) for i in range(6)]

    winner, comparisons = rank(sessions, baseline=0.0020)
    c = next(c for c in comparisons if c.spacing == 0.0025)

    assert c.mean == 0.0 and c.ci == (0.0, 0.0)
    assert not c.decisive, "an interval of exactly [0, 0] was reported as a result"
    assert winner is None


def test_one_session_can_never_be_decisive():
    """n=1 has no spread to estimate. It must widen to infinity, not divide by zero."""
    c = Comparison(0.0008, 0.0020, [5.0])

    assert c.n == 1
    assert c.ci == (float("-inf"), float("inf"))
    assert not c.decisive
    assert math.isnan(c.sd)


# --- how far away an answer is --------------------------------------------------------

def test_it_reports_how_many_sessions_an_answer_would_take():
    """The useful half of a refusal. Noisier evidence must demand more sessions."""
    tight = Comparison(0.0008, 0.0020, [2.0, 2.2, 1.8, 2.1, 1.9])
    loose = Comparison(0.0008, 0.0020, [2.0, 8.0, -4.0, 6.0, -2.0])

    assert tight.sessions_needed() < loose.sessions_needed()
    assert tight.sessions_needed() >= 2


def test_a_zero_effect_needs_an_unbounded_sample():
    """No sample size resolves a difference that is not there, and claiming a finite one
    would promise an answer that never arrives."""
    import sys

    assert Comparison(0.0008, 0.0020, [1.0, -1.0, 1.0, -1.0]).sessions_needed() == sys.maxsize


def test_the_report_says_no_winner_and_what_it_would_take():
    text = format_report(LIVE, baseline=0.0020)

    assert "NO WINNER" in text
    assert "sessions" in text
    assert "WINNER 0" not in text


# --- Student's t ----------------------------------------------------------------------

def test_t_values_match_the_table():
    for df, expected in ((1, 12.706), (5, 2.571), (10, 2.228), (30, 2.042)):
        assert t_critical_95(df) == pytest.approx(expected)


def test_t_converges_to_the_normal_limit():
    assert t_critical_95(100_000) == pytest.approx(1.960)
    assert t_critical_95(35) < t_critical_95(30)
    assert t_critical_95(35) > t_critical_95(40)


def test_t_is_wider_than_the_normal_for_small_samples():
    """Using 1.96 at n=3 would report intervals ~2x too narrow and manufacture winners."""
    assert t_critical_95(2) > 2 * 1.96


# --- the resolution guard -------------------------------------------------------------

def test_even_fill_loss_is_survivable():
    """A coarse path that costs every spacing the same fraction still ranks: the
    differences are unaffected by a common factor."""
    fine = {0.0008: 400, 0.0015: 200, 0.0030: 100}
    coarse = {0.0008: 200, 0.0015: 100, 0.0030: 50}

    assert resolution_bias(fine, coarse) is None


def test_spacing_dependent_fill_loss_is_refused():
    """The fatal case: 1m candles hide intra-minute oscillation, which is where a tight
    ladder lives. Reading that as a parameter effect is the exact error being avoided."""
    fine = {0.0008: 400, 0.0015: 200, 0.0030: 100}
    coarse = {0.0008: 80, 0.0015: 150, 0.0030: 95}

    problem = resolution_bias(fine, coarse)

    assert problem is not None
    assert "20%" in problem and "95%" in problem


def test_resolution_needs_something_to_compare():
    assert resolution_bias({0.0008: 10}, {0.0008: 5}) is not None


def test_a_ratio_from_a_handful_of_fills_is_not_evidence():
    """3 fills against 2 is not a recovery rate, and a few of those look reassuringly
    flat. Spacings too thin to estimate are dropped, not counted as agreement."""
    fine = {0.0020: 3, 0.0025: 2, 0.0030: 2}
    coarse = {0.0020: 3, 0.0025: 2, 0.0030: 2}

    problem = resolution_bias(fine, coarse)

    assert problem is not None and "too few" in problem


def test_thin_spacings_are_dropped_but_the_rest_still_judged():
    fine = {0.0015: 200, 0.0020: 150, 0.0040: 3}
    coarse = {0.0015: 40, 0.0020: 140, 0.0040: 3}

    problem = resolution_bias(fine, coarse)

    assert problem is not None and "20%" in problem and "93%" in problem


# --- spacings the engine will not even place -------------------------------------------

def test_the_level_floor_matches_the_engines_own_gate():
    """_is_level_profitable needs spacing >= round trip x MIN_PROFIT_MULTIPLIER."""
    floor = level_floor(maker=0.0002, taker=0.0004, taker_share=0.05,
                        min_profit_multiplier=3.0)

    assert floor == pytest.approx(0.00126)


def test_the_old_sweeps_winner_was_below_the_floor():
    """The finding that invalidates the previous answer. 0.08% -- named best by two of
    the three earlier sweeps -- cannot place a single level at these settings, and
    measured on 14,777 real ticks on 2026-08-17 it produced exactly zero fills while
    0.15% produced 14. Its column was an untraded position drifting, not a spacing."""
    floor = level_floor(0.0002, 0.0004, 0.05, 3.0)
    rankable, inert = split_by_floor((0.0008, 0.0010, 0.0012, 0.0015, 0.0020, 0.0030),
                                     floor)

    assert inert == [0.0008, 0.0010, 0.0012]
    assert rankable == [0.0015, 0.0020, 0.0030]


def test_the_swept_range_is_entirely_above_the_floor():
    """The module's own spacing set must not reintroduce inert columns."""
    from spacing_study import SPACINGS

    _rankable, inert = split_by_floor(SPACINGS, level_floor(0.0002, 0.0004, 0.05, 3.0))
    assert inert == []


def test_a_lower_multiplier_unlocks_tighter_spacings():
    """The floor is a config consequence, not a constant -- at 2.0 the 0.10% column
    becomes a real measurement rather than an empty one."""
    assert split_by_floor((0.0010,), level_floor(0.0002, 0.0004, 0.05, 3.0))[0] == []
    assert split_by_floor((0.0010,), level_floor(0.0002, 0.0004, 0.05, 2.0))[0] == [0.0010]


def test_the_study_refuses_to_rank_on_data_its_own_check_rejected():
    """A table produced off data this module has already shown to be spacing-biased is
    exactly the failure it exists to prevent -- and the previous harness's table was
    believed three times. The gate is the whole design, so it is pinned here."""
    from pathlib import Path

    import spacing_study

    src = Path(spacing_study.__file__).read_text(encoding="utf-8")
    body = src[src.index("def main("):]

    assert "if not a.force:" in body, "the study no longer gates on the resolution check"
    gate = body.index("if not a.force:")
    loop = body.index("for log, run in discover_sessions")
    assert gate < loop, "the sessions are swept before the data is validated"
    assert "validate_resolution(" in body[gate:loop]


# --- the reference point ---------------------------------------------------------------

def test_the_diffusion_optimum_is_twice_the_round_trip():
    """Crossings scale as quadratic variation / s^2, so profit goes as (1/s^2)(s - f),
    maximised at s = 2f. Offered as a reference, never as a measurement."""
    s = theoretical_optimum(maker=0.0002, taker=0.0004, taker_share=0.05)

    assert s == pytest.approx(0.00084)
    assert s == pytest.approx(2 * 2 * (0.0002 * 0.95 + 0.0004 * 0.05))
