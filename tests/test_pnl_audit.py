"""The exchange's own ledger, read offline. AUDIT #87.

The point of this module is to be the one number that does NOT come from the bot's
bookkeeping, so the tests here care most about the two ways it could quietly lie:
double-counting a paginated row, and averaging a catastrophe into invisibility.
"""

import time

import pytest

import pnl_audit
from pnl_audit import daily, excluding, fetch_income, outlier_days, summarize

DAY = 86400000


def row(t, kind, amount, tran=None):
    return {"time": t, "incomeType": kind, "income": str(amount),
            "tranId": tran if tran is not None else f"{t}-{kind}-{amount}"}


# --- summing the ledger -------------------------------------------------------------

def test_net_is_realized_plus_commission_plus_funding():
    s = summarize([row(0, "REALIZED_PNL", 10.0), row(1, "COMMISSION", -3.0),
                   row(2, "FUNDING_FEE", 0.5)])
    assert s["net"] == 7.5


def test_fills_are_counted_from_commission_entries():
    """Every fill is charged, so commission rows are the honest fill count -- realized
    rows only appear when something closes."""
    s = summarize([row(0, "COMMISSION", -0.01), row(1, "COMMISSION", -0.01),
                   row(2, "REALIZED_PNL", 0.5)])
    assert s["fills"] == 2


def test_per_day_divides_by_the_span_actually_covered():
    s = summarize([row(0, "REALIZED_PNL", 10.0), row(4 * DAY, "REALIZED_PNL", 10.0)])
    assert s["span_days"] == 4.0
    assert s["per_day"] == 5.0


def test_a_single_entry_does_not_divide_by_zero():
    s = summarize([row(0, "REALIZED_PNL", 10.0)])
    assert s["span_days"] == 0.0 and s["per_day"] == 0.0


def test_an_empty_ledger_is_not_an_error():
    assert summarize([])["net"] == 0.0


def test_unknown_income_types_still_reach_the_net():
    """Binance adds types (INSURANCE_CLEAR, TRANSFER...). Silently dropping one would
    make the audit disagree with the balance for no visible reason."""
    assert summarize([row(0, "INSURANCE_CLEAR", -2.0)])["net"] == -2.0


# --- pagination must not double-count -----------------------------------------------

NOW = int(time.time() * 1000)


class _Ledger:
    """Binance's income endpoint: rows at or after startTime, capped at `limit`.

    Faithful on the one detail that matters -- it pages by TIME, not by cursor -- so
    entries sharing a millisecond straddle a page boundary exactly as they do live.
    """

    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: int(r["time"]))
        self.calls = []

    def get_income_history(self, symbol, since_ms=None, limit=1000):
        self.calls.append(since_ms)
        return [r for r in self.rows if int(r["time"]) >= (since_ms or 0)][:limit]


def recent(offset_ms, kind, amount, tran=None):
    return row(NOW - 5 * DAY + offset_ms, kind, amount, tran)


def test_entries_sharing_a_millisecond_across_a_page_boundary_survive():
    """A full page ending mid-millisecond is the normal case -- a fill stamps its
    COMMISSION and REALIZED_PNL identically. Resuming at newest+1 steps over whichever
    did not fit, and a dropped COMMISSION understates the fee bill.
    """
    rows = [recent(i, "COMMISSION", -0.01, tran=f"c{i}") for i in range(1000)]
    rows.append(recent(999, "REALIZED_PNL", 5.0, tran="straddler"))
    got = fetch_income(_Ledger(rows), "DOGEUSDT", days=30)
    assert any(r["tranId"] == "straddler" for r in got), "boundary row was skipped"
    assert len(got) == 1001


def test_a_row_returned_twice_is_counted_once():
    """Re-reading the boundary is the cost of not skipping it; dedup is what makes that
    safe. Double-counting a COMMISSION would inflate the very number being measured."""
    rows = [recent(i, "COMMISSION", -0.01, tran=f"c{i}") for i in range(1000)]
    rows.append(recent(999, "COMMISSION", -0.01, tran="straddler"))
    got = fetch_income(_Ledger(rows), "DOGEUSDT", days=30)
    assert len(got) == len({r["tranId"] for r in got})
    assert summarize(got)["totals"]["COMMISSION"] == pytest.approx(-10.01)


def test_paging_stops_instead_of_spinning_when_a_full_page_repeats():
    """Resuming AT newest means a page of identically-stamped rows returns itself
    forever unless no-new-rows ends the walk."""
    rows = [recent(0, "COMMISSION", -0.01, tran=f"c{i}") for i in range(1000)]
    ledger = _Ledger(rows)
    got = fetch_income(ledger, "DOGEUSDT", days=30)
    assert len(ledger.calls) < 100
    assert len(got) == 1000


def test_a_short_page_ends_the_walk():
    """Fewer than `limit` rows means the ledger is exhausted; asking again is a wasted
    round trip against a rate limit."""
    ledger = _Ledger([recent(0, "COMMISSION", -1.0)])
    fetch_income(ledger, "DOGEUSDT", days=30)
    assert len(ledger.calls) == 1


def test_the_request_cap_is_honoured():
    rows = [recent(i, "COMMISSION", -0.01, tran=f"c{i}") for i in range(50000)]
    ledger = _Ledger(rows)
    fetch_income(ledger, "DOGEUSDT", days=30, max_requests=3)
    assert len(ledger.calls) == 3


# --- the day breakdown ---------------------------------------------------------------

def test_entries_group_into_utc_days():
    d = daily([row(0, "REALIZED_PNL", 1.0), row(DAY, "REALIZED_PNL", 2.0)])
    assert sorted(d) == ["1970-01-01", "1970-01-02"]


def test_each_day_carries_its_own_net():
    d = daily([row(0, "REALIZED_PNL", 5.0), row(1, "COMMISSION", -2.0)])
    assert d["1970-01-01"]["net"] == 3.0


# --- refusing to let one catastrophe hide in an average ------------------------------

def flat_days_plus(disaster):
    """Twenty ordinary days and one that is not."""
    rows = []
    for i in range(20):
        rows.append(row(i * DAY, "REALIZED_PNL", 1.0))
    rows.append(row(20 * DAY, "REALIZED_PNL", disaster))
    return rows


def test_a_catastrophic_day_is_flagged():
    flagged = outlier_days(daily(flat_days_plus(-50.0)))
    assert [d for d, _, _ in flagged] == ["1970-01-21"]


def test_a_second_bad_day_is_not_masked_by_the_worst_one():
    """Masking is the failure mode that matters. One -50 inflates the standard
    deviation so much that a -15 day sits well inside 3.5 sd of it and a mean/stdev
    screen reports the account as having had exactly one bad day. Median/MAD is not
    moved by the outlier, so it sees both -- which is the difference between "one
    freak event" and "this keeps happening".
    """
    rows = flat_days_plus(-50.0) + [row(21 * DAY, "REALIZED_PNL", -15.0)]
    nets = [v["net"] for v in daily(rows).values()]
    mean = sum(nets) / len(nets)
    sd = (sum((n - mean) ** 2 for n in nets) / len(nets)) ** 0.5
    assert abs(-15.0 - mean) < 3.5 * sd, "mean/stdev would have caught it after all"

    flagged = [d for d, _, _ in outlier_days(daily(rows))]
    assert "1970-01-22" in flagged and "1970-01-21" in flagged


def test_an_ordinary_run_of_days_flags_nothing():
    rows = [row(i * DAY, "REALIZED_PNL", 1.0 + (i % 3) * 0.1) for i in range(20)]
    assert outlier_days(daily(rows)) == []


def test_identical_days_do_not_divide_by_zero_mad():
    rows = [row(i * DAY, "REALIZED_PNL", 1.0) for i in range(20)]
    assert outlier_days(daily(rows)) == []


def test_too_few_days_to_judge_returns_nothing():
    assert outlier_days(daily([row(0, "REALIZED_PNL", -50.0)])) == []


def test_a_windfall_is_flagged_too():
    """An outsized WIN is equally not trading, and treating it as recurring income is
    the more expensive mistake."""
    assert outlier_days(daily(flat_days_plus(+50.0)))


def test_removing_the_outlier_shows_the_underlying_run():
    rows = flat_days_plus(-50.0)
    rest = summarize(excluding(rows, {"1970-01-21"}))
    assert rest["net"] == 20.0
    assert rest["fills"] == 0


def test_excluding_nothing_changes_nothing():
    rows = flat_days_plus(-50.0)
    assert summarize(excluding(rows, set()))["net"] == summarize(rows)["net"]


def test_the_module_states_what_it_found():
    """The event days and the fee/edge ratio are the reason this exists; a docstring
    that loses them turns the module back into an unexplained script."""
    assert "2026-08-08" in pnl_audit.__doc__
    assert "158%" in pnl_audit.__doc__


def test_the_docstring_does_not_carry_the_retracted_figures():
    """-53.07/-2.20 per day came from a paginator that skipped boundary rows. It was
    wrong by a sign and it was nearly reported. If it ever reappears here, it is
    because someone restored an old draft."""
    for retracted in ("-53.07", "-2.20 USDT/day", "4212"):
        assert retracted not in pnl_audit.__doc__
