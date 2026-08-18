"""The order floor is an assumption about the exchange, so ask it. AUDIT #107.

MIN_NOTIONAL_USDT is 5.0 -- DOGE's number -- and every order the grid or the trend
follower declines to place below the floor is declined against it. Nothing ever asked
the exchange whether it was true, and exchange.py had no way to: it exposed leverage,
margin mode, position mode, max notional and fee rates for verification, and not this.

The failure it prevents is the quiet kind. On a symbol whose real minimum is higher,
orders sized at the floor come back -4164 and the ladder never fills, while the bot's
sizing arithmetic agrees with itself the whole way down -- the same shape as the fee
mismatch that the fee check exists to catch.

It was also defined THREE times: grid.py, trend_follower.py and backtest.py, as
independent copies. trend_follower now imports grid's, so the production path has one.
"""

from pathlib import Path

import pytest

import main
from grid import MIN_NOTIONAL_USDT


# --- one definition in the production path ------------------------------------------

def test_the_follower_shares_the_grid_constant():
    """Two copies meant a symbol change had two places to remember and no way to notice
    missing one."""
    import trend_follower

    assert trend_follower.MIN_NOTIONAL_USDT is MIN_NOTIONAL_USDT


def test_the_follower_does_not_redefine_it():
    src = Path(__import__("trend_follower").__file__).read_text(encoding="utf-8")

    assert "MIN_NOTIONAL_USDT = " not in src, "the constant was redefined, not imported"


# --- reading it from the exchange -----------------------------------------------------

class FakeCcxt:
    def __init__(self, market):
        self._market = market

    def market(self, symbol):
        if self._market is None:
            raise KeyError(symbol)
        return self._market


def reader(market):
    from exchange import Exchange

    ex = Exchange.__new__(Exchange)
    ex.exchange = FakeCcxt(market)
    return ex


def test_it_reads_the_unified_limit():
    ex = reader({"limits": {"cost": {"min": 5.0}}})

    assert ex.get_min_notional("DOGEUSDT") == pytest.approx(5.0)


def test_it_falls_back_to_the_raw_filter():
    """Some ccxt builds carry the value only in Binance's own filter list."""
    ex = reader({"limits": {"cost": {}},
                 "info": {"filters": [{"filterType": "PRICE_FILTER"},
                                      {"filterType": "MIN_NOTIONAL", "notional": "20"}]}})

    assert ex.get_min_notional("DOGEUSDT") == pytest.approx(20.0)


def test_an_unknown_market_is_none_not_a_crash():
    assert reader(None).get_min_notional("NOPEUSDT") is None


def test_an_absent_limit_is_none_not_zero():
    """Zero would read as "no minimum" and disable the check silently."""
    ex = reader({"limits": {"cost": {}}, "info": {}})

    assert ex.get_min_notional("DOGEUSDT") is None


def test_an_unparseable_limit_is_none():
    ex = reader({"limits": {"cost": {"min": "not-a-number"}}})

    assert ex.get_min_notional("DOGEUSDT") is None


# --- what startup does with it --------------------------------------------------------

def _verify(exchange_min):
    """verify_account_config with everything else healthy, so only the floor is in play."""
    from config import settings

    class Ex:
        def get_account_config(self, symbol):
            return {"leverage": settings.leverage, "margin_mode": "cross",
                    "dual_side": False, "isolated": False, "max_notional": 600000}

        def get_maint_margin_ratio(self, symbol, notional):
            return 0.004

        def get_commission_rates(self, symbol):
            return {"maker_pct": settings.maker_fee_pct,
                    "taker_pct": settings.taker_fee_pct}

        def get_min_notional(self, symbol):
            return exchange_min

    return main.verify_account_config(Ex(), settings, 4938.0)


def test_a_higher_exchange_minimum_blocks_startup():
    """The case that matters. Orders sized between the two are rejected -4164 and the
    rungs never appear, with nothing in the bot's own arithmetic to reveal it."""
    problems = _verify(MIN_NOTIONAL_USDT + 15)

    assert any("-4164" in p for p in problems), problems


def test_a_matching_minimum_is_no_problem():
    assert _verify(MIN_NOTIONAL_USDT) == []


def test_a_lower_exchange_minimum_is_no_problem():
    """The bot being stricter than the exchange rejects nothing -- it only means some
    orders it could have placed, it will not."""
    assert _verify(MIN_NOTIONAL_USDT / 2) == []


def test_an_unreadable_minimum_does_not_block():
    """It is public market data, so failing to read it means something odd -- but
    refusing to trade over an unverified floor that has been right all along would be
    a worse trade than warning."""
    assert _verify(None) == []


def test_the_check_names_both_numbers():
    """A problem that does not say what to change costs a debugging session."""
    problems = _verify(20.0)
    text = " ".join(problems)

    assert "20.00" in text and f"{MIN_NOTIONAL_USDT:.2f}" in text
