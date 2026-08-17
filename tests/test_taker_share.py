"""The fee input has to be measured, and measured on the right fills. AUDIT #100.

taker_fill_share_pct sets round_trip_fee_pct, which sets every break-even price and the
level-profitability floor. AUDIT #51 is what happens when it is too low: exits clamped to
"break-even" book a real loss, and the floor passes levels that do not clear their own
fees. So the question "is 5% still right?" is not bookkeeping, it decides whether the
bot's idea of profit matches the exchange's.

Two ways to get the answer wrong, and both have happened here:

  * measure the wrong fills. Account-wide taker was 41.6% on 2026-08-17 while grid CYCLES
    were 0.0% -- the difference is entirely stops, reconcile closes and crossed unwinds.
    Feeding 41.6% into the model would raise the floor 35% and reject levels that are
    genuinely profitable.

  * measure too few. A handful of fills produces a share that looks like a number and is
    an anecdote, and a zero-taker sample under a normal-approximation interval reads as
    certainty rather than as "no taker fill yet".
"""

import math

import pytest

from taker_share import (DEFAULT_MIN_FILLS, GRID_PURPOSES, TakerSample, fee_model,
                         tally, verdict, wilson_interval)

CONFIGURED = 0.05
MAKER, TAKER, MULT = 0.0002, 0.0004, 3.0


def sample(fills, taker_fills, notional=None, taker_notional=None, commission=0.0):
    notional = fills * 25.0 if notional is None else notional
    taker_notional = (taker_fills * 25.0 if taker_notional is None else taker_notional)
    return TakerSample(fills=fills, taker_fills=taker_fills, notional=notional,
                       taker_notional=taker_notional, commission=commission)


# --- what gets measured ---------------------------------------------------------------

def test_only_cycle_legs_count_as_cycle_fees():
    """round_trip_fee_pct describes a grid cycle. Stops, reconcile closes and unwinds are
    forced exits -- real costs, but not the cost a LEVEL pays, and charging levels for
    them is what config.py explicitly warns against."""
    assert set(GRID_PURPOSES) == {"grid_entry", "grid_exit"}
    for forced in ("stop_trail", "stop_hard", "reconcile", "unwind", "emergency"):
        assert forced not in GRID_PURPOSES


def test_the_share_is_weighted_by_notional_not_by_fill_count():
    """Fees are charged on notional, so one 10,000u taker fill outweighs ninety-nine 1u
    maker fills. Counting fills would report 1% where the money says 99%."""
    s = TakerSample(fills=100, taker_fills=1, notional=10_099.0, taker_notional=10_000.0)

    assert s.share_by_count == pytest.approx(0.01)
    assert s.share_by_notional == pytest.approx(0.99, abs=0.01)


def test_the_realised_rate_is_commission_over_notional():
    s = sample(100, 0, notional=10_000.0, commission=2.0)
    assert s.realised_rate == pytest.approx(0.0002)


def test_an_empty_sample_does_not_divide_by_zero():
    s = TakerSample()
    assert s.share_by_notional == 0.0 and s.share_by_count == 0.0
    assert s.realised_rate == 0.0


def test_forced_exits_are_kept_out_of_the_cycle_bucket():
    """The measurement that matters, on the shape of the real data: a stop firing at the
    taker rate must not raise the share a LEVEL is charged. On 2026-08-17 this was the
    difference between 41.6% and 0.0%."""
    grid, by_purpose = tally([
        ("grid_entry", 100.0, 0.02, False),
        ("grid_exit", 100.0, 0.02, False),
        ("stop_hard", 5_000.0, 2.0, True),
        ("reconcile", 400.0, 0.16, True),
        ("unwind", 300.0, 0.12, True),
    ])

    assert grid.fills == 2
    assert grid.share_by_notional == 0.0, "a stop-out was charged to the grid levels"
    assert by_purpose["stop_hard"].share_by_notional == 1.0, "forced exits still reported"


def test_cycle_legs_land_in_both_buckets():
    """Per-purpose is for reading; the cycle bucket is what feeds the model."""
    grid, by_purpose = tally([("grid_entry", 100.0, 0.02, True)])

    assert grid.fills == 1 and grid.taker_fills == 1
    assert by_purpose["grid_entry"].fills == 1


def test_a_zero_notional_fill_is_skipped_not_counted_as_maker():
    """It would otherwise dilute the share toward zero — the unsafe direction — with
    fills that carry no fee at all."""
    grid, _ = tally([("grid_entry", 0.0, 0.0, True), ("grid_entry", 100.0, 0.04, True)])

    assert grid.fills == 1
    assert grid.share_by_notional == 1.0


def test_untagged_fills_never_reach_the_cycle_bucket():
    """Orders from before the AUDIT #56 tagging, and any whose order record aged out of
    the window, are of unknown purpose. Guessing they are cycles is how the account-wide
    number gets in by the back door."""
    grid, by_purpose = tally([("untagged", 1_000.0, 0.4, True),
                              ("UNMAPPED", 1_000.0, 0.4, True)])

    assert grid.fills == 0
    assert set(by_purpose) == {"untagged", "UNMAPPED"}


# --- the uncertainty --------------------------------------------------------------------

def test_a_zero_taker_sample_is_not_reported_as_certainty():
    """The failure mode that makes a small sample dangerous. Under the normal
    approximation, 0 of 40 gives an interval of exactly zero width -- 'the share is 0%,
    certainly' -- from a sample that has merely not seen a taker fill yet. Wilson keeps
    an upper bound."""
    lo, hi = wilson_interval(0, 40)

    assert lo == 0.0
    assert hi > 0.05, f"0 of 40 claimed the share is under 5% (upper bound {hi:.1%})"


def test_the_interval_never_goes_negative():
    for k, n in ((0, 10), (1, 500), (2, 30)):
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_more_fills_narrow_the_interval():
    narrow = wilson_interval(50, 1000)
    wide = wilson_interval(5, 100)

    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_no_sample_at_all_admits_the_whole_range():
    assert wilson_interval(0, 0) == (0.0, 1.0)


# --- the fee model ----------------------------------------------------------------------

def test_the_configured_share_reproduces_the_live_floor():
    """The numbers the bot actually runs on: 5% taker gives a 0.042% round trip and a
    0.126% floor, which is what makes 0.08%/0.10%/0.12% spacings inert."""
    m = fee_model(MAKER, TAKER, CONFIGURED, MULT)

    assert m["round_trip"] == pytest.approx(0.00042)
    assert m["floor"] == pytest.approx(0.00126)


def test_a_higher_share_raises_the_floor():
    low = fee_model(MAKER, TAKER, 0.05, MULT)["floor"]
    high = fee_model(MAKER, TAKER, 0.40, MULT)["floor"]

    assert high > low
    assert 0.002 / high < 1.2, "a 40% share leaves the live spacing barely above its floor"


def test_pure_maker_and_pure_taker_bound_the_model():
    assert fee_model(MAKER, TAKER, 0.0, MULT)["round_trip"] == pytest.approx(2 * MAKER)
    assert fee_model(MAKER, TAKER, 1.0, MULT)["round_trip"] == pytest.approx(2 * TAKER)


# --- the verdict, which is asymmetric on purpose -----------------------------------------

def test_a_thin_sample_concludes_nothing():
    """Neither confirmation nor alarm. The 2026-08-17 testnet window had 244 cycle fills
    across 35 days; a live account starting from zero will sit here for a while."""
    code, why = verdict(sample(40, 0), CONFIGURED)

    assert code == "INSUFFICIENT"
    assert "40" in why


def test_a_share_confidently_above_config_is_an_alarm():
    """The AUDIT #51 direction. Break-even prices computed from too low a fee are not
    break-even, and the bot books small real losses as flat."""
    code, why = verdict(sample(1000, 300), CONFIGURED)

    assert code == "UNDERSTATED"
    assert "before trading on it" in why


def test_a_share_above_config_but_inside_the_noise_is_not_an_alarm():
    """Firing on every point estimate above 5% would cry wolf on ordinary variation."""
    code, _ = verdict(sample(250, 18), CONFIGURED)      # 7.2%, CI still covers 5%

    assert code == "ELEVATED"


def test_a_share_below_config_is_the_safe_side():
    """Overstating the share costs a little trading; understating it costs money. Config
    at or above the truth is the outcome to report as fine."""
    code, why = verdict(sample(1000, 5), CONFIGURED)

    assert code == "OK"
    assert "safe side" in why


def test_the_testnet_result_would_pass_and_that_is_the_trap():
    """244 cycle fills, zero taker -- the actual 2026-08-17 testnet measurement. It reads
    OK, and it is worthless for a live decision: a synthetic book rests post-only orders
    that a real one would queue behind real flow. The tool says OK; the operator still
    has to not carry it across."""
    code, _ = verdict(sample(244, 0), CONFIGURED, min_fills=200)

    assert code == "OK"


def test_the_minimum_sample_is_enforced_at_the_boundary():
    assert verdict(sample(DEFAULT_MIN_FILLS - 1, 0), CONFIGURED)[0] == "INSUFFICIENT"
    assert verdict(sample(DEFAULT_MIN_FILLS, 0), CONFIGURED)[0] != "INSUFFICIENT"


def test_the_caller_can_demand_a_larger_sample():
    assert verdict(sample(300, 0), CONFIGURED, min_fills=1000)[0] == "INSUFFICIENT"


# --- the report must name the account it measured -----------------------------------------

def test_the_runner_distinguishes_demo_from_live():
    """A testnet share presented as a live one is the whole hazard this tool exists for."""
    from pathlib import Path

    import taker_share

    src = Path(taker_share.__file__).read_text(encoding="utf-8")
    body = src[src.index("def main("):]

    assert "settings.demo_mode" in body, "the report does not check which account it read"
    assert "DEMO (testnet)" in body and "LIVE" in body
    assert "Do NOT carry this share to live" in src
