"""Balance reads must be retried like every other read. AUDIT #102.

Every read on Exchange goes through _retry -- ticker, order book, ohlcv, income, open
orders, positions, fetch_order. The three balance readers were the only ones calling ccxt
directly, each with a hand-rolled timestamp retry and nothing else. So the one endpoint
with no protection against a transient failure was the one the startup sequence must read
before it can do anything at all.

2026-08-18 01:24:20, twenty seconds into a fresh start:

    ERROR | Could not read balance to verify account configuration:
            binanceusdm {"code":-1007,"msg":"Timeout waiting for response from backend
            server. Send status unknown; execution status unknown."}
    [supervise] exited 0 after 20s
    [supervise] clean exit -- not restarting

Two independent defects put the bot down for the night:

  * -1007 is a backend timeout, raised by ccxt as RequestTimeout, which _retry already
    handles. The call never reached it.

  * the failure path returned normally, so the process exited 0, and supervise.py read
    that as a deliberate shutdown to be honoured rather than a crash to be restarted.
"""

import re
from pathlib import Path

import ccxt
import pytest

import exchange as exchange_mod
from exchange import Exchange


def _ex(responses):
    """An Exchange whose fetch_balance yields `responses` in order (raise or return)."""
    ex = Exchange.__new__(Exchange)
    ex.max_retries = 3
    ex.retry_delay = 0
    ex._circuit_breaker = exchange_mod.CircuitBreaker()
    ex._balance_cache = {}
    ex._balance_cache_at = {}
    ex._balance_cache_time = 0.0
    ex._balance_cache_ttl = 5.0
    ex._is_timestamp_error = lambda e: False
    calls = {"n": 0}

    class Inner:
        def fetch_balance(self):
            r = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            if isinstance(r, Exception):
                raise r
            return r

    ex.exchange = Inner()
    return ex, calls


PAYLOAD = {"USDT": {"free": 4898.0, "total": 4938.0, "used": 40.0}}
TIMEOUT = ccxt.RequestTimeout(
    'binanceusdm {"code":-1007,"msg":"Timeout waiting for response from backend server."}')


# --- the failure that took the bot down --------------------------------------------------

def test_a_backend_timeout_is_retried_not_raised():
    """The live failure, exactly: one -1007 then a good response."""
    ex, calls = _ex([TIMEOUT, PAYLOAD])

    assert ex.get_balance() == 4898.0
    assert calls["n"] == 2, "gave up on the first timeout"


def test_all_three_readers_are_retried():
    """get_balance was not special. Equity and the info dict read the same endpoint and
    were equally unprotected."""
    for reader, expected in (
        (lambda e: e.get_balance(), 4898.0),
        (lambda e: e.get_total_equity(), 4938.0),
        (lambda e: e.get_balance_info()["used"], 40.0),
    ):
        ex, calls = _ex([TIMEOUT, PAYLOAD])
        assert reader(ex) == expected
        assert calls["n"] == 2


def test_a_persistently_unreachable_exchange_still_raises():
    """Retrying is not swallowing. If it never comes back the caller must hear about it."""
    ex, _ = _ex([TIMEOUT])

    with pytest.raises(ccxt.RequestTimeout):
        ex.get_balance()


def test_a_network_error_is_retried_too():
    ex, calls = _ex([ccxt.NetworkError("connection reset"), PAYLOAD])

    assert ex.get_balance() == 4898.0
    assert calls["n"] == 2


def test_a_bad_request_is_not_retried():
    """_retry only widens the net for transient classes. Hammering a rejected request
    three times just wastes the rate limit."""
    ex, calls = _ex([ccxt.InvalidOrder("nonsense")])

    with pytest.raises(ccxt.InvalidOrder):
        ex.get_balance()
    assert calls["n"] == 1


# --- no reader may go around the wrapper again -------------------------------------------

def test_no_balance_reader_calls_ccxt_directly():
    """The property, pinned at source. A new getter that calls self.exchange.fetch_balance
    directly reintroduces exactly this outage, and would pass every test above."""
    src = Path(exchange_mod.__file__).read_text(encoding="utf-8")
    direct = [m.start() for m in re.finditer(r"self\.exchange\.fetch_balance", src)]

    assert len(direct) == 1, f"{len(direct)} direct fetch_balance calls; expected only _fetch_balance's"
    wrapper = src.index("def _fetch_balance")
    nxt = src.index("\n    def ", wrapper + 1)
    assert wrapper < direct[0] < nxt, "the direct call is outside _fetch_balance"


def test_the_wrapper_labels_the_call_for_the_breaker():
    """_retry counts failures per label into the circuit breaker; an unlabelled call is
    invisible in the logs when diagnosing an outage."""
    src = Path(exchange_mod.__file__).read_text(encoding="utf-8")
    body = src[src.index("def _fetch_balance"):]

    assert 'label="fetch_balance"' in body[:body.index("\n    def ", 1)]


# --- and a hiccup must not look like a decision -------------------------------------------

def test_the_startup_balance_failure_exits_nonzero():
    """supervise.py honours exit 0 as a deliberate shutdown and stays down. A backend
    timeout is not a decision, and after AUDIT #102's retries it means the exchange is
    genuinely unreachable -- which is what the supervisor's backoff exists for."""
    import main

    src = Path(main.__file__).read_text(encoding="utf-8")
    at = src.index("Could not read balance to verify account configuration")
    after = src[at:at + 400]

    assert "raise SystemExit(1)" in after, (
        "startup balance failure still returns cleanly; supervise will not restart it")


def test_supervise_still_honours_a_real_clean_exit():
    """The supervisor's rule is correct and must stay: the fix belongs on the side that
    was misreporting, not here."""
    import supervise

    src = Path(supervise.__file__).read_text(encoding="utf-8")

    assert "CLEAN_EXIT = 0" in src
    assert "exit_code == CLEAN_EXIT" in src


# --- a clock repair must not spend the retry budget ---------------------------------------

def test_a_timestamp_resync_earns_its_attempt_back():
    """AUDIT #102. Resyncing the clock is a repair, not a retry-and-hope. Spending an
    attempt on it means the clock gets fixed and nothing is left to use the fix -- at
    max_retries=1 the call resynced and raised anyway, which is what the balance getters'
    own unconditional resync used to prevent before they were folded into _retry."""
    ex, calls = _ex([ccxt.InvalidNonce("-1021 timestamp drift"), PAYLOAD])
    ex.max_retries = 1
    ex._is_timestamp_error = lambda e: isinstance(e, ccxt.InvalidNonce)
    ex._sync_time = lambda: None

    assert ex.get_balance() == 4898.0
    assert calls["n"] == 2


def test_an_endless_nonce_error_still_terminates():
    """The budget is a loan, not a licence. A backend answering -1021 forever must not
    spin the loop."""
    ex, calls = _ex([ccxt.InvalidNonce("-1021 timestamp drift")])
    ex.max_retries = 2
    ex._is_timestamp_error = lambda e: isinstance(e, ccxt.InvalidNonce)
    ex._sync_time = lambda: None

    with pytest.raises(ccxt.InvalidNonce):
        ex.get_balance()
    assert calls["n"] <= 5, f"{calls['n']} attempts on a permanent nonce error"
