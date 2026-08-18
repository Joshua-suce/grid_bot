"""The script that puts account settings back after a testnet reset. AUDIT #106.

A Binance testnet reset returned this account as 10x ISOLATED against the 25x CROSS the
previous session ran on, and startup refused to trade -- correctly, since order notional
is CAPITAL_PER_GRID_USDT x LEVERAGE and every margin figure would have been out by 2.5x.
fix_account.py is the other half of that refusal.

What has to hold, because this one writes to the account:

  * a dry run writes nothing, and it is the default
  * an open position or a resting order stops it before anything is touched
  * "already set" is success, not failure -- it has to be safe to run twice
  * the result is verified by RE-READING the exchange, never from the call's own reply
"""

import pytest

import fix_account
from fix_account import describe, open_exposure, plan, set_margin_mode, set_position_mode


@pytest.fixture(autouse=True)
def restore_exchange():
    """Each test swaps fix_account.Exchange for a stub. Put the real one back afterwards
    so a later test cannot inherit a fake that happens to still be installed."""
    real = fix_account.Exchange
    yield
    fix_account.Exchange = real


class FakeInner:
    def __init__(self, fail_margin=None, fail_dual=None):
        self.calls = []
        self._fail_margin = fail_margin
        self._fail_dual = fail_dual

    def fapiPrivatePostMarginType(self, params):
        self.calls.append(("marginType", params))
        if self._fail_margin:
            raise Exception(self._fail_margin)
        return {"code": 200}

    def fapiPrivatePostPositionSideDual(self, params):
        self.calls.append(("positionSide", params))
        if self._fail_dual:
            raise Exception(self._fail_dual)
        return {"code": 200}


class FakeExchange:
    def __init__(self, cfg, positions=None, orders=None, inner=None):
        self._cfg = cfg
        self._positions = positions or []
        self._orders = orders or []
        self.exchange = inner or FakeInner()
        self.leverage_calls = []

    def get_account_config(self, symbol):
        return self._cfg

    def get_positions(self, symbol):
        return self._positions

    def get_open_orders(self, symbol):
        return self._orders

    def set_leverage(self, symbol, leverage):
        self.leverage_calls.append(leverage)
        return True


def cfg(leverage=25, margin="cross", dual=False, max_notional=600000):
    return {"leverage": leverage, "margin_mode": margin, "dual_side": dual,
            "isolated": margin == "isolated", "max_notional": max_notional}


# --- what needs changing --------------------------------------------------------------

def test_a_matching_account_needs_nothing():
    from config import settings

    assert plan(cfg(leverage=settings.leverage)) == []


def test_the_real_reset_is_detected():
    """The state the account actually came back in."""
    from config import settings

    todo = plan(cfg(leverage=10, margin="isolated", max_notional=1_800_000))
    what = [row[0] for row in todo]

    assert "leverage" in what
    assert "margin mode" in what
    assert ("10x", f"{settings.leverage}x") == todo[what.index("leverage")][1:]


def test_hedge_mode_is_treated_as_wrong():
    """The bot sends no positionSide, which hedge mode rejects outright (-4061)."""
    todo = plan(cfg(dual=True))

    assert [row[0] for row in todo] == ["position mode"]


def test_describe_reads_an_unreadable_account_as_such():
    assert describe(None) == "unreadable"
    assert "10x" in describe(cfg(leverage=10))


# --- the guards -----------------------------------------------------------------------

def test_an_open_position_blocks_everything(capsys):
    ex = FakeExchange(cfg(leverage=10, margin="isolated"),
                      positions=[{"contracts": 1778.0}])
    fix_account.Exchange = lambda *a, **k: ex

    rc = fix_account.main(["--apply"])

    assert rc == 2
    assert ex.leverage_calls == [], "leverage was changed with a position open"
    assert ex.exchange.calls == [], "margin type was touched with a position open"
    assert "REFUSING" in capsys.readouterr().out


def test_a_resting_order_blocks_everything():
    ex = FakeExchange(cfg(leverage=10, margin="isolated"),
                      orders=[{"id": "1"}])
    fix_account.Exchange = lambda *a, **k: ex

    assert fix_account.main(["--apply"]) == 2
    assert ex.leverage_calls == []
    assert ex.exchange.calls == []


def test_a_short_position_counts_as_open():
    """positionAmt is negative for a short; abs() or it reads as flat."""
    ex = FakeExchange(cfg(), positions=[{"positionAmt": "-1771"}])

    qty, orders = open_exposure(ex, "DOGEUSDT")

    assert qty == pytest.approx(1771)


def test_an_unreadable_position_field_does_not_crash():
    ex = FakeExchange(cfg(), positions=[{"contracts": None}, {}])

    assert open_exposure(ex, "DOGEUSDT") == (0.0, 0)


# --- dry run is the default -----------------------------------------------------------

def test_a_dry_run_writes_nothing(capsys):
    ex = FakeExchange(cfg(leverage=10, margin="isolated"))
    fix_account.Exchange = lambda *a, **k: ex

    rc = fix_account.main([])

    assert rc == 0
    assert ex.leverage_calls == []
    assert ex.exchange.calls == []
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "--apply" in out


def test_apply_says_so_before_acting(capsys):
    ex = FakeExchange(cfg(leverage=10))
    fix_account.Exchange = lambda *a, **k: ex
    fix_account.main(["--apply"])

    assert "APPLY" in capsys.readouterr().out


# --- idempotence ----------------------------------------------------------------------

def test_already_set_is_success_not_failure():
    """Binance answers "you asked for what is already set" with an error code. A script
    meant to be safe to run twice has to read that as done."""
    ex = FakeExchange(cfg(), inner=FakeInner(
        fail_margin='binanceusdm {"code":-4046,"msg":"No need to change margin type."}'))

    assert set_margin_mode(ex, "DOGEUSDT", "cross") is True


def test_position_mode_already_one_way_is_success():
    ex = FakeExchange(cfg(), inner=FakeInner(
        fail_dual='binanceusdm {"code":-4059,"msg":"No need to change position side."}'))

    assert set_position_mode(ex) is True


def test_a_position_rejection_is_reported_as_failure():
    ex = FakeExchange(cfg(), inner=FakeInner(
        fail_margin='binanceusdm {"code":-4047,"msg":"Margin type cannot be changed if '
                    'there exists position."}'))

    assert set_margin_mode(ex, "DOGEUSDT", "cross") is False


def test_the_wire_value_for_cross_is_crossed():
    """Binance's API spells it CROSSED; positionRisk reports it back as 'cross'."""
    ex = FakeExchange(cfg())
    set_margin_mode(ex, "DOGEUSDT", "cross")

    assert ex.exchange.calls[0][1]["marginType"] == "CROSSED"


def test_the_symbol_is_sent_bare():
    ex = FakeExchange(cfg())
    set_margin_mode(ex, "DOGE/USDT:USDT", "cross")

    assert ex.exchange.calls[0][1]["symbol"] == "DOGEUSDT"


# --- the result is verified, not assumed ----------------------------------------------

class LyingExchange(FakeExchange):
    """set_leverage returns True and the account stays where it was. Observed live on
    2026-08-18: "Leverage set to 25x" logged, account still on 10x."""

    def set_leverage(self, symbol, leverage):
        self.leverage_calls.append(leverage)
        return True


def test_a_change_that_did_not_take_is_reported_as_failure(capsys):
    ex = LyingExchange(cfg(leverage=10))
    fix_account.Exchange = lambda *a, **k: ex

    rc = fix_account.main(["--apply"])

    assert rc == 1, "trusted the call's own reply instead of re-reading the account"
    assert "STILL WRONG" in capsys.readouterr().out


def test_a_change_that_took_reports_success(capsys):
    from config import settings

    class Fixed(FakeExchange):
        def set_leverage(self, symbol, leverage):
            self.leverage_calls.append(leverage)
            self._cfg = cfg(leverage=leverage)
            return True

    ex = Fixed(cfg(leverage=10))
    fix_account.Exchange = lambda *a, **k: ex

    rc = fix_account.main(["--apply"])

    assert rc == 0
    assert ex.leverage_calls == [settings.leverage]
    assert "matches config" in capsys.readouterr().out


def test_an_unreadable_account_stops_before_writing(capsys):
    ex = FakeExchange(None)
    fix_account.Exchange = lambda *a, **k: ex

    assert fix_account.main(["--apply"]) == 1
    assert ex.leverage_calls == []


# --- it must never place a trade ------------------------------------------------------

def test_the_script_places_no_orders():
    """It changes account settings. A stray order-placing call in here would be a
    trade nobody asked for."""
    from pathlib import Path

    src = Path(fix_account.__file__).read_text(encoding="utf-8")
    for forbidden in ("place_limit_order", "place_stop_market", "create_order",
                      "close_all_positions", "market_close"):
        assert forbidden not in src, f"{forbidden} has no business in this script"
