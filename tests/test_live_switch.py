"""The account has to be configured the way the money math assumes. AUDIT #69/#71/#72.

Every figure the bot computes about money -- notional per order, margin reserved, how
far liquidation sits from the stop -- comes from config values the exchange is free to
disagree with, and nothing checked that they matched. This was not hypothetical: while
writing the check, .env said LEVERAGE=25 and the exchange reported 5 on the same
account. On demo that is a number in a log. On a live account it is a 5x error in every
margin figure the bot believes, entered silently.
"""

from types import SimpleNamespace

import pytest

from main import verify_account_config
from state import StateManager


def cfg(**over):
    base = dict(
        symbol="DOGEUSDT", leverage=25, capital_per_grid_usdt=5.0, grid_count=8,
        stop_loss_pct=0.03, maker_fee_pct=0.02, taker_fee_pct=0.04,
        min_profit_multiplier=3.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


_DEFAULT = object()  # so account=None can mean "the read failed", not "use defaults"


class FakeExchange:
    def __init__(self, account=_DEFAULT, mmr=0.006, fees=_DEFAULT):
        self._account = {
            "leverage": 25, "margin_mode": "cross", "isolated": False,
            "dual_side": False, "max_notional": 600000.0,
        } if account is _DEFAULT else account
        self._mmr = mmr
        # Verbatim from the demo account, 2026-08-14.
        self._fees = {"maker_pct": 0.02, "taker_pct": 0.04} if fees is _DEFAULT else fees

    def get_account_config(self, symbol):
        return self._account

    def get_maint_margin_ratio(self, symbol, notional):
        return self._mmr

    def get_commission_rates(self, symbol):
        return self._fees


# --- leverage ---------------------------------------------------------------------

def test_a_matching_account_is_clean():
    assert verify_account_config(FakeExchange(), cfg(), 4931.09) == []


def test_leverage_mismatch_blocks_the_start():
    """The measured case: config 25x, exchange 5x."""
    ex = FakeExchange({"leverage": 5, "margin_mode": "cross", "isolated": False,
                       "dual_side": False, "max_notional": 4800000.0})

    problems = verify_account_config(ex, cfg(leverage=25), 4931.09)

    assert len(problems) == 1
    assert "leverage mismatch" in problems[0]
    assert "25x" in problems[0] and "5x" in problems[0]


def test_leverage_mismatch_the_other_way_also_blocks():
    ex = FakeExchange({"leverage": 50, "margin_mode": "cross", "isolated": False,
                       "dual_side": False, "max_notional": 5000.0})
    assert any("leverage mismatch" in p for p in verify_account_config(ex, cfg(), 4931.09))


def test_unreadable_account_config_blocks_rather_than_assuming():
    """Silence is not agreement. If nothing can be read, nothing is verified."""
    problems = verify_account_config(FakeExchange(account=None), cfg(), 4931.09)

    assert len(problems) == 1
    assert "could not be read" in problems[0]


# --- position mode ----------------------------------------------------------------

def test_hedge_mode_blocks_the_start():
    """The bot sends no positionSide, which Binance rejects outright in hedge mode."""
    ex = FakeExchange({"leverage": 25, "margin_mode": "cross", "isolated": False,
                       "dual_side": True, "max_notional": 600000.0})

    problems = verify_account_config(ex, cfg(), 4931.09)

    assert any("HEDGE" in p and "One-way" in p for p in problems)


# --- margin mode ------------------------------------------------------------------

def _isolated(leverage=25):
    return {"leverage": leverage, "margin_mode": "isolated", "isolated": True,
            "dual_side": False, "max_notional": 600000.0}


def test_isolated_margin_blocks_when_the_stop_does_not_clear_liquidation():
    """At 25x with a 0.6% maintenance rate, liquidation is ~3.4% from entry and the
    stop is at 3.0%. Four tenths of a percent, against a mark price that wanders."""
    problems = verify_account_config(
        FakeExchange(_isolated(25), mmr=0.006), cfg(leverage=25, stop_loss_pct=0.03), 4931.09
    )

    assert any("ISOLATED" in p and "liquidation" in p for p in problems)


def test_isolated_margin_passes_when_the_stop_has_real_daylight():
    """Same account on 5x: liquidation ~19.4% away, a 3% stop is nowhere near it."""
    assert verify_account_config(
        FakeExchange(_isolated(5), mmr=0.006), cfg(leverage=5, stop_loss_pct=0.03), 4931.09
    ) == []


def test_cross_margin_at_the_same_leverage_is_fine():
    """The identical geometry that fails on isolated passes on cross -- the whole
    wallet backs the position, so liquidation is not 3.4% away."""
    assert verify_account_config(FakeExchange(), cfg(leverage=25, stop_loss_pct=0.03),
                                 4931.09) == []


def test_isolated_margin_blocks_when_the_maintenance_rate_is_unknown():
    problems = verify_account_config(
        FakeExchange(_isolated(25), mmr=None), cfg(), 4931.09
    )
    assert any("maintenance-margin rate could not be read" in p for p in problems)


def test_the_maintenance_rate_used_is_the_one_passed_in():
    """A higher maintenance rate moves liquidation NEARER, so a stop that cleared at
    0.6% must not silently still clear at 2%."""
    ok = verify_account_config(
        FakeExchange(_isolated(10), mmr=0.006), cfg(leverage=10, stop_loss_pct=0.05), 4931.09
    )
    tight = verify_account_config(
        FakeExchange(_isolated(10), mmr=0.05), cfg(leverage=10, stop_loss_pct=0.05), 4931.09
    )

    assert ok == []
    assert any("ISOLATED" in p for p in tight)


# --- funding ----------------------------------------------------------------------

def test_a_balance_that_cannot_fund_the_ladder_blocks():
    """8 resting orders reserve 5 USDT of margin each whether they fill or not."""
    problems = verify_account_config(FakeExchange(), cfg(), balance=30.0)

    assert any("cannot fund the ladder" in p for p in problems)
    assert any("40.00" in p for p in problems)


def test_exactly_enough_balance_is_allowed():
    assert verify_account_config(FakeExchange(), cfg(), balance=40.0) == []


def test_percent_sizing_skips_the_funding_check():
    """With CAPITAL_PER_GRID_USDT=0 the size is a fraction of balance, so it cannot
    outgrow the balance by construction."""
    assert verify_account_config(
        FakeExchange(), cfg(capital_per_grid_usdt=0.0), balance=30.0
    ) == []


def test_a_ladder_above_the_symbol_notional_cap_blocks():
    ex = FakeExchange({"leverage": 25, "margin_mode": "cross", "isolated": False,
                       "dual_side": False, "max_notional": 100.0})

    assert any("only allowed up to" in p for p in verify_account_config(ex, cfg(), 4931.09))


def test_several_problems_are_all_reported_not_just_the_first():
    """Fixing one and restarting into the next is a bad way to find out."""
    ex = FakeExchange({"leverage": 5, "margin_mode": "cross", "isolated": False,
                       "dual_side": True, "max_notional": 600000.0})

    problems = verify_account_config(ex, cfg(), balance=10.0)

    assert len(problems) >= 3


# --- fee schedule -------------------------------------------------------------------

def test_a_higher_maker_fee_than_configured_blocks():
    """The minimum profitable spacing is built from MAKER_FEE_PCT. If the account
    actually pays more, rungs go closer together than a cycle can pay for and every
    completed cycle loses the difference -- silently, because the arithmetic is
    self-consistent."""
    ex = FakeExchange(fees={"maker_pct": 0.05, "taker_pct": 0.08})

    problems = verify_account_config(ex, cfg(maker_fee_pct=0.02), 4931.09)

    assert any("MAKER_FEE_PCT" in p for p in problems)
    assert any("MAKER_FEE_PCT=0.0500" in p for p in problems), "no corrected value given"


def test_matching_fees_are_clean():
    assert verify_account_config(
        FakeExchange(fees={"maker_pct": 0.02, "taker_pct": 0.04}), cfg(), 4931.09
    ) == []


def test_a_rounding_difference_in_fees_does_not_block():
    """A fee tier is a defect; a hair of floating point is not."""
    ex = FakeExchange(fees={"maker_pct": 0.0200001, "taker_pct": 0.04})
    assert verify_account_config(ex, cfg(maker_fee_pct=0.02), 4931.09) == []


def test_a_higher_taker_fee_alone_does_not_block():
    """Taker prices forced exits, and no spacing decision rides on it. It is the
    dominant cost in the measured history, so it is logged loudly -- but a stale figure
    makes estimates optimistic, it does not make cycles unprofitable."""
    ex = FakeExchange(fees={"maker_pct": 0.02, "taker_pct": 0.05})

    assert verify_account_config(ex, cfg(taker_fee_pct=0.04), 4931.09) == []


def test_lower_fees_than_configured_do_not_block():
    """Overstating fees only makes spacing wider than it needs to be."""
    ex = FakeExchange(fees={"maker_pct": 0.01, "taker_pct": 0.02})
    assert verify_account_config(ex, cfg(), 4931.09) == []


def test_unreadable_fees_do_not_block_but_are_not_silent():
    """Unlike leverage, an unknown fee rate cannot mis-size anything -- it can only make
    the spacing floor stale, and the floor is still enforced against the config."""
    assert verify_account_config(FakeExchange(fees=None), cfg(), 4931.09) == []


# --- StateManager: demo is not a default (#71) --------------------------------------

def test_state_manager_will_not_guess_the_account():
    """A default would let #67 be undone by omission: a live caller that forgot the
    argument would silently get the demo file."""
    with pytest.raises(TypeError):
        StateManager(state_dir="state", symbol="DOGEUSDT")


def test_state_manager_rejects_demo_passed_positionally():
    """Keyword-only, so `StateManager(dir, symbol, True)` cannot mean the wrong thing."""
    with pytest.raises(TypeError):
        StateManager("state", "DOGEUSDT", True)


# --- first-run position guard (#72) -------------------------------------------------

def test_a_fresh_account_has_no_history(tmp_path):
    sm = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=False)
    assert sm.has_history() is False


def test_a_saved_state_counts_as_history(tmp_path):
    sm = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=True)
    sm.save({"a": 1})
    assert sm.has_history() is True


@pytest.mark.parametrize("suffix", [".bak.1786498297", ".empty", ".corrupt.123.456"])
def test_unrestorable_leftovers_still_count_as_history(tmp_path, suffix):
    """A backup or a corrupt file cannot be loaded, but it proves the bot ran here --
    so a position IS an orphan of a dead session and closing it is right."""
    sm = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=True)
    (tmp_path / f"{sm.filepath.stem}{suffix}").write_text("{}")

    assert sm.has_history() is True


def test_the_other_account_does_not_count_as_history(tmp_path):
    """The whole point: a demo run must not make a live start believe it has been here
    before, or the first live start market-closes a position it did not open."""
    demo = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=True)
    demo.save({"a": 1})

    live = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=False)

    assert live.has_history() is False


def test_another_symbol_does_not_count_as_history(tmp_path):
    StateManager(state_dir=str(tmp_path), symbol="SOLUSDT", demo=True).save({"a": 1})

    sm = StateManager(state_dir=str(tmp_path), symbol="DOGEUSDT", demo=True)

    assert sm.has_history() is False
