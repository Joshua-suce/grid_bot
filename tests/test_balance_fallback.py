"""A second route to the balance when the heavy endpoint is the broken one. AUDIT #103.

ccxt's fetch_balance calls fapi/v3/account -- the heaviest account endpoint there is,
returning full state including every position, for one number. On 2026-08-18 Binance's
demo backend answered -1007 on the v3 account family while v2 was healthy:

    FAILED  fapi/v3/account   -1007
    FAILED  fapi/v3/balance   -1007
    OK      fapi/v2/balance   balance=5000.00  avail=5000.00
    OK      fapi/v2/account   wallet=5000.00

set_leverage and the clock sync had both just succeeded, so this was never "the exchange
is down". Retries cannot help: they ask the same broken endpoint three more times.

The bug this module mostly exists for is the one the first version walked straight into.
Routing the fallback through _retry gated it on the circuit breaker -- which the primary
had just opened, three failures a call against a threshold of five. So the breaker tripped
by the broken endpoint refused the healthy one, and the fallback worked exactly once:
get_balance() returned 5000.0, then get_total_equity() raised "Circuit breaker open".
"""

import ccxt
import pytest

import exchange as exchange_mod
from exchange import Exchange

V2_ROWS = [
    {"asset": "USDT", "balance": "5000.00000000", "crossWalletBalance": "5000.00000000",
     "crossUnPnl": "12.50000000", "availableBalance": "4900.00000000"},
    {"asset": "BNB", "balance": "1.0", "crossUnPnl": "0.0", "availableBalance": "1.0"},
]
TIMEOUT = ccxt.RequestTimeout('binanceusdm {"code":-1007,"msg":"Timeout waiting..."}')


def _ex(v3_error=TIMEOUT, v2_rows=V2_ROWS, v2_error=None):
    ex = Exchange.__new__(Exchange)
    ex.max_retries = 2
    ex.retry_delay = 0
    ex._circuit_breaker = exchange_mod.CircuitBreaker()
    ex._balance_cache = {}
    ex._balance_cache_at = {}
    ex._balance_cache_time = 0.0
    ex._balance_cache_ttl = 5.0
    ex._is_timestamp_error = lambda e: False
    calls = {"v3": 0, "v2": 0}

    class Inner:
        def fetch_balance(self):
            calls["v3"] += 1
            if v3_error:
                raise v3_error
            return {"USDT": {"free": 1.0, "total": 2.0, "used": 1.0}}

        def fapiPrivateV2GetBalance(self):
            calls["v2"] += 1
            if v2_error:
                raise v2_error
            return v2_rows

    ex.exchange = Inner()
    return ex, calls


# --- the fallback itself -------------------------------------------------------------

def test_a_dead_v3_falls_back_to_v2():
    ex, calls = _ex()

    assert ex.get_balance() == 4900.0
    assert calls["v3"] > 0 and calls["v2"] == 1


def test_total_includes_unrealised_pnl():
    """v3 reports total as margin balance -- wallet plus open PnL -- and risk.check_all
    measures drawdown against it. A `total` that quietly dropped open PnL would read a
    losing position as no drawdown at all, disarming the kill switch."""
    ex, _ = _ex()

    assert ex.get_total_equity() == pytest.approx(5012.50)


def test_used_is_what_the_position_is_holding():
    ex, _ = _ex()

    info = ex.get_balance_info()
    assert info == {"free": 4900.0, "total": pytest.approx(5012.50),
                    "used": pytest.approx(112.50)}


def test_every_asset_is_mapped_not_just_usdt():
    ex, _ = _ex()
    ex.get_balance()

    assert ex.get_balance("BNB") == 1.0


# --- the breaker interaction, which is the whole point ---------------------------------

def test_the_fallback_is_not_gated_by_the_breaker_the_primary_opened():
    """The bug the first version had. The primary's failures open the breaker; if the
    fallback goes through the same gate it is refused, and the second call onward gets
    nothing. Live symptom: get_balance() returned 5000.0, get_total_equity() raised."""
    ex, calls = _ex()
    ex._circuit_breaker.open = True
    ex._circuit_breaker.failures = 99

    assert ex.get_balance() == 4900.0
    assert calls["v2"] == 1, "the fallback was refused by the breaker"


def test_repeated_reads_keep_working():
    """One success is not enough -- the loop reads the balance every iteration."""
    ex, calls = _ex()

    for _ in range(4):
        assert ex.get_balance() == 4900.0
    assert calls["v2"] == 4


def test_a_working_fallback_closes_the_breaker():
    """It is proof the venue is reachable, which is exactly what the breaker wants to
    know. Leaving it open would block tickers and orders that are perfectly healthy."""
    ex, _ = _ex()
    ex._circuit_breaker.record_failure()
    ex._circuit_breaker.record_failure()

    ex.get_balance()

    assert ex._circuit_breaker.failures == 0
    assert ex._circuit_breaker.open is False


# --- and it must not paper over anything else -------------------------------------------

def test_both_routes_down_raises_the_original_error():
    """The caller needs the real reason, not "fallback failed"."""
    ex, _ = _ex(v2_error=TIMEOUT)

    with pytest.raises(ccxt.RequestTimeout):
        ex.get_balance()


def test_an_empty_v2_response_is_not_treated_as_a_zero_balance():
    """Reporting free=0 would have risk.check_all see the account wiped out."""
    ex, _ = _ex(v2_rows=[])

    with pytest.raises(ccxt.RequestTimeout):
        ex.get_balance()


def test_a_non_transient_primary_error_does_not_fall_back():
    """Bad credentials or a rejected request are real answers. Retrying them on another
    endpoint hides the problem and burns rate limit."""
    ex, calls = _ex(v3_error=ccxt.AuthenticationError("bad key"))

    with pytest.raises(ccxt.AuthenticationError):
        ex.get_balance()
    assert calls["v2"] == 0


def test_a_healthy_v3_never_touches_the_fallback():
    ex, calls = _ex(v3_error=None)

    assert ex.get_balance() == 1.0
    assert calls["v2"] == 0
