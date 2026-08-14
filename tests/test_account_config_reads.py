"""Reading and setting account configuration. AUDIT #69.

set_leverage swallowed every failure behind one warning. That is survivable on an
account already at the right leverage and quietly catastrophic otherwise: with
CAPITAL_PER_GRID_USDT sizing, notional per order is margin x LEVERAGE, so a rejected
call leaves the bot spending the exchange's leverage while planning with the config's.
"""

import ccxt
import pytest

from exchange import CircuitBreaker, Exchange

# Verbatim from the demo account, 2026-08-14.
POSITION_RISK = [{
    "symbol": "DOGEUSDT", "positionAmt": "0", "entryPrice": "0.0",
    "markPrice": "0.06977000", "liquidationPrice": "0", "leverage": "5",
    "maxNotionalValue": "4800000", "marginType": "cross", "isolatedMargin": "0.00000000",
    "positionSide": "BOTH", "notional": "0", "isolated": False, "adlQuantile": 0,
}]

LEVERAGE_BRACKET = [{
    "symbol": "DOGEUSDT",
    "brackets": [
        {"bracket": 1, "initialLeverage": 50, "notionalCap": 5000,
         "notionalFloor": 0, "maintMarginRatio": 0.006, "cum": 0.0},
        {"bracket": 2, "initialLeverage": 40, "notionalCap": 25000,
         "notionalFloor": 5000, "maintMarginRatio": 0.007, "cum": 5.0},
        {"bracket": 3, "initialLeverage": 25, "notionalCap": 600000,
         "notionalFloor": 25000, "maintMarginRatio": 0.01, "cum": 80.0},
    ],
}]


COMMISSION_RATE = {
    "symbol": "DOGEUSDT", "makerCommissionRate": "0.000200",
    "takerCommissionRate": "0.000400", "rpiCommissionRate": "0",
}


def make_exchange(fake):
    ex = Exchange.__new__(Exchange)
    ex.exchange = fake
    ex.demo = False
    ex.has_credentials = True
    ex.max_retries = 1
    ex.retry_delay = 0.0
    ex._circuit_breaker = CircuitBreaker(failure_threshold=1000, recovery_time=0)
    return ex


class Backend:
    def __init__(self, applied=None, raises=None, risk=None, dual=False, bracket=None,
                 commission=None):
        self.applied = applied
        self.raises = raises
        self.risk = POSITION_RISK if risk is None else risk
        self.dual = dual
        self.bracket = LEVERAGE_BRACKET if bracket is None else bracket
        self.commission = COMMISSION_RATE if commission is None else commission
        self.calls = []

    def set_leverage(self, leverage, symbol):
        self.calls.append((leverage, symbol))
        if self.raises:
            raise self.raises
        return {"symbol": symbol, "leverage": self.applied if self.applied is not None else leverage,
                "maxNotionalValue": "600000"}

    def fapiPrivateV2GetPositionRisk(self, params=None):
        if isinstance(self.risk, Exception):
            raise self.risk
        return self.risk

    def fapiPrivateGetPositionSideDual(self):
        if isinstance(self.dual, Exception):
            raise self.dual
        return {"dualSidePosition": self.dual}

    def fapiPrivateGetLeverageBracket(self, params=None):
        if isinstance(self.bracket, Exception):
            raise self.bracket
        return self.bracket

    def fapiPrivateGetCommissionRate(self, params=None):
        if isinstance(self.commission, Exception):
            raise self.commission
        return self.commission


# --- set_leverage ------------------------------------------------------------------

def test_a_successful_set_reports_true():
    ex = make_exchange(Backend())
    assert ex.set_leverage("DOGEUSDT", 25) is True


def test_a_rejected_set_reports_false():
    """An open position, a bracket cap, or a key without futures permission."""
    ex = make_exchange(Backend(raises=ccxt.ExchangeError("no need to change leverage")))
    assert ex.set_leverage("DOGEUSDT", 25) is False


def test_a_200_that_applied_something_else_reports_false():
    """Trust the exchange's echo over the request: this is the case worth catching."""
    ex = make_exchange(Backend(applied=20))
    assert ex.set_leverage("DOGEUSDT", 25) is False


def test_the_echo_is_read_from_info_when_ccxt_nests_it():
    class Nested(Backend):
        def set_leverage(self, leverage, symbol):
            return {"info": {"leverage": "20"}}

    assert make_exchange(Nested()).set_leverage("DOGEUSDT", 25) is False


def test_a_reply_without_a_leverage_field_is_accepted():
    """Not every ccxt version echoes it. Absence is not evidence of a mismatch --
    verify_account_config reads the account back regardless."""
    class Silent(Backend):
        def set_leverage(self, leverage, symbol):
            return {}

    assert make_exchange(Silent()).set_leverage("DOGEUSDT", 25) is True


# --- get_account_config ------------------------------------------------------------

def test_account_config_is_readable_while_flat():
    """fetch_positions returns nothing at all when flat and the v3 account endpoint
    omits the symbol, so positionRisk is the one that answers."""
    acct = make_exchange(Backend()).get_account_config("DOGEUSDT")

    assert acct["leverage"] == 5
    assert acct["margin_mode"] == "cross"
    assert acct["isolated"] is False
    assert acct["dual_side"] is False
    assert acct["max_notional"] == 4800000.0


def test_isolated_margin_is_reported_as_isolated():
    risk = [dict(POSITION_RISK[0], marginType="isolated", isolated=True)]
    acct = make_exchange(Backend(risk=risk)).get_account_config("DOGEUSDT")

    assert acct["isolated"] is True
    assert acct["margin_mode"] == "isolated"


def test_hedge_mode_is_reported():
    acct = make_exchange(Backend(dual=True)).get_account_config("DOGEUSDT")
    assert acct["dual_side"] is True


@pytest.mark.parametrize("backend", [
    Backend(risk=ccxt.RequestTimeout("down")),
    Backend(dual=ccxt.RequestTimeout("down")),
    Backend(risk=[]),
    Backend(risk=[{"symbol": "DOGEUSDT", "marginType": "cross"}]),  # no leverage field
])
def test_an_unreadable_account_returns_none_rather_than_a_guess(backend):
    assert make_exchange(backend).get_account_config("DOGEUSDT") is None


# --- maintenance margin ------------------------------------------------------------

@pytest.mark.parametrize("notional,expected", [
    (500.0, 0.006),      # one side of the current ladder — bracket 1
    (4999.0, 0.006),
    (5000.0, 0.006),     # boundary belongs to the bracket whose cap it is
    (5001.0, 0.007),
    (30000.0, 0.01),
])
def test_the_bracket_matching_the_notional_is_used(notional, expected):
    """The rate is tiered, and it is what sets the distance to liquidation. A constant
    would misstate exactly the number the stop has to clear."""
    ex = make_exchange(Backend())
    assert ex.get_maint_margin_ratio("DOGEUSDT", notional) == expected


def test_beyond_every_bracket_the_widest_one_applies():
    """Erring toward the highest rate puts liquidation NEARER, never further."""
    ex = make_exchange(Backend())
    assert ex.get_maint_margin_ratio("DOGEUSDT", 10_000_000.0) == 0.01


def test_an_unreadable_bracket_returns_none():
    ex = make_exchange(Backend(bracket=ccxt.RequestTimeout("down")))
    assert ex.get_maint_margin_ratio("DOGEUSDT", 500.0) is None


def test_malformed_brackets_do_not_crash_the_check():
    ex = make_exchange(Backend(bracket=[{"symbol": "DOGEUSDT", "brackets": [{"junk": 1}]}]))
    assert ex.get_maint_margin_ratio("DOGEUSDT", 500.0) is None


# --- commission rates ---------------------------------------------------------------

def test_commission_rates_come_back_as_percentages():
    """Binance returns fractions; MAKER_FEE_PCT/TAKER_FEE_PCT are percentages. Getting
    that conversion wrong by 100x would silently disable the fee floor."""
    fees = make_exchange(Backend()).get_commission_rates("DOGEUSDT")

    assert fees == {"maker_pct": 0.02, "taker_pct": 0.04}


def test_a_higher_fee_tier_is_reported_as_such():
    ex = make_exchange(Backend(commission={
        "symbol": "DOGEUSDT", "makerCommissionRate": "0.000500",
        "takerCommissionRate": "0.000800",
    }))
    assert ex.get_commission_rates("DOGEUSDT") == {"maker_pct": 0.05, "taker_pct": 0.08}


@pytest.mark.parametrize("commission", [
    ccxt.RequestTimeout("down"),
    {"symbol": "DOGEUSDT"},
    {"symbol": "DOGEUSDT", "makerCommissionRate": "not-a-number",
     "takerCommissionRate": "0.000400"},
])
def test_unreadable_commission_rates_return_none(commission):
    assert make_exchange(Backend(commission=commission)).get_commission_rates("DOGEUSDT") is None
