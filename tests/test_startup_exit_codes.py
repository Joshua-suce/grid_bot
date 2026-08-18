"""A startup abort has to say WHY, in the exit code. AUDIT #104.

supervise.py restarts a crash and honours a clean exit -- "a clean exit is a decision
someone made". That rule is correct, so main.py has to mean it. Every startup abort
returned normally, which exits 0, so an unreachable exchange looked exactly like a
deliberate shutdown and the bot stayed down until a human noticed.

2026-08-18 02:04, with every signed Binance endpoint answering HTTP 408:

    ERROR | ACCOUNT NOT SAFE TO TRADE | 1 problem(s) found
    ERROR |   - account configuration could not be read ... unverified
    [supervise] exited 0 after 83s
    [supervise] clean exit -- not restarting

The distinction that has to survive: "the exchange says 5x and your config says 25x" is
a real misconfiguration -- restarting changes nothing, a human must fix it, and stopping
is right. "I could not reach the endpoint to ask" is weather. Same list, opposite
handling, and they were indistinguishable.
"""

import re
from pathlib import Path

import pytest

import main
from main import ACCOUNT_UNREADABLE, verify_account_config


def cfg():
    """The real settings object. A hand-rolled stub only has the attributes the test
    author remembered, so it drifts from what verify_account_config actually reads."""
    from config import settings

    return settings


class FakeExchange:
    """Only what verify_account_config actually reaches for: the account config, the
    maintenance-margin rate (isolated only), the fee schedule and the order floor."""

    def __init__(self, config):
        self._config = config

    def get_account_config(self, symbol):
        return self._config

    def get_maint_margin_ratio(self, symbol, notional):
        return 0.004

    def get_commission_rates(self, symbol):
        from config import settings

        return {"maker_pct": settings.maker_fee_pct, "taker_pct": settings.taker_fee_pct}

    def get_min_notional(self, symbol):
        # verify_account_config checks the order floor against the
        # exchange now, same as it checks leverage and fees (AUDIT #107).
        from grid import MIN_NOTIONAL_USDT

        return MIN_NOTIONAL_USDT


def test_an_unreadable_account_reports_the_named_problem():
    """The caller keys off this exact value, so it cannot drift into a bare string."""
    problems = verify_account_config(FakeExchange(None), cfg(), 4938.0)

    assert problems == [ACCOUNT_UNREADABLE]


def test_the_unreadable_case_is_distinguishable_from_a_real_mismatch():
    """A leverage mismatch is a different animal and must not carry the same marker."""
    settings = cfg()
    # get_account_config's real shape -- "isolated" included, which the liquidation
    # check reads. A stub missing it raises KeyError instead of testing anything.
    wrong = {"leverage": settings.leverage + 20, "margin_mode": "cross",
             "dual_side": False, "isolated": False, "max_notional": 600000}
    problems = verify_account_config(FakeExchange(wrong), settings, 4938.0)

    assert problems, "an exchange on the wrong leverage is a problem"
    assert ACCOUNT_UNREADABLE not in problems


def test_an_unreachable_exchange_exits_nonzero():
    """The live failure. Retries are already exhausted by the time this is reached, so
    the exchange is genuinely unreachable -- which is what the supervisor's backoff is
    for. Exiting 0 tells it the opposite."""
    src = Path(main.__file__).read_text(encoding="utf-8")
    at = src.index("ACCOUNT NOT SAFE TO TRADE")
    block = src[at:at + 1200]

    assert "if ACCOUNT_UNREADABLE in account_problems:" in block
    guard = block.index("if ACCOUNT_UNREADABLE in account_problems:")
    assert "raise SystemExit(1)" in block[guard:guard + 200]


def test_a_real_misconfiguration_still_stops_for_good():
    """The other half, and the reason this is not just 'always exit 1'. Restarting into
    a leverage mismatch forever would churn the book and never fix anything."""
    src = Path(main.__file__).read_text(encoding="utf-8")
    at = src.index("ACCOUNT NOT SAFE TO TRADE")
    block = src[at:at + 1200]
    guard = block.index("if ACCOUNT_UNREADABLE in account_problems:")
    after = block[guard:]

    assert re.search(r"raise SystemExit\(1\)\s*\n\s*return", after), (
        "the non-transient path no longer exits cleanly")


def test_the_startup_balance_read_also_exits_nonzero():
    """The same defect, one call earlier, fixed in AUDIT #102. Pinned together so a
    future edit cannot quietly reintroduce either."""
    src = Path(main.__file__).read_text(encoding="utf-8")
    at = src.index("Could not read balance to verify account configuration")

    assert "raise SystemExit(1)" in src[at:at + 700]


def test_supervise_keeps_honouring_a_genuine_clean_exit():
    """The supervisor's rule is right; the fix belongs on the side that was lying to it."""
    import supervise

    src = Path(supervise.__file__).read_text(encoding="utf-8")

    assert "CLEAN_EXIT = 0" in src
    assert "exit_code == CLEAN_EXIT" in src
