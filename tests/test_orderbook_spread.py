"""The spread reading was dead for the entire life of the bot. AUDIT #55.

`get_orderbook_depth` derived the spread from `fetch_ticker`, but binanceusdm does not
populate bid/ask on the ticker -- both come back None. The guard `if bid and ask` never
fired, `_last_spread` never moved off its initial 0.0, and every status line ever written
reported `spread=0.0000%`.

Measured live against the DEMO account while fixing this:

    ticker bid=None  ask=None  last=0.07023
    book   top bid=[0.07023, ...]  top ask=[0.07024, ...]   -> real spread 0.0142%

The order book fetched one line later always had the answer, so the ticker call was both
wrong and a wasted round trip on every single iteration.
"""

from exchange import CircuitBreaker, Exchange


def _exchange(fake):
    ex = Exchange.__new__(Exchange)
    ex.exchange = fake
    ex.demo = False
    ex.has_credentials = True
    ex.max_retries = 1
    ex.retry_delay = 0.0
    ex._circuit_breaker = CircuitBreaker(failure_threshold=1000, recovery_time=0)
    ex._last_spread = 0.0
    return ex


class _Book:
    """Reproduces the real binanceusdm shape: ticker has no bid/ask, book does."""

    def __init__(self, bids=None, asks=None, fail=False):
        self._bids = bids if bids is not None else [[0.07023, 50008596.0], [0.07022, 10.0]]
        self._asks = asks if asks is not None else [[0.07024, 4252688.0], [0.07025, 10.0]]
        self._fail = fail
        self.ticker_calls = 0

    def fetch_ticker(self, symbol):
        self.ticker_calls += 1
        return {"bid": None, "ask": None, "last": 0.07023}

    def fetch_order_book(self, symbol, limit=10):
        if self._fail:
            raise ConnectionError("book unavailable")
        return {"bids": self._bids, "asks": self._asks}


def test_the_spread_is_read_from_the_book_not_the_empty_ticker():
    ex = _exchange(_Book())

    depth = ex.get_orderbook_depth("DOGEUSDT")

    expected = (0.07024 - 0.07023) / 0.07024
    assert depth["spread_pct"] > 0, (
        "spread is still 0 with a real two-sided book -- the reading is dead"
    )
    assert abs(depth["spread_pct"] - expected) < 1e-12
    assert abs(depth["spread_pct"] * 100 - 0.0142) < 0.001


def test_the_wasted_ticker_call_is_gone():
    """One fetch per iteration, not two. The ticker added a round trip and contributed
    nothing but the None that broke the calculation."""
    fake = _Book()

    ex = _exchange(fake)
    ex.get_orderbook_depth("DOGEUSDT")

    assert fake.ticker_calls == 0, "still paying for a ticker fetch that has no bid/ask"


def test_a_failed_fetch_serves_the_last_known_spread_not_a_fabricated_zero():
    """A transient failure must not make the book look infinitely tight. Zero is a
    meaningful spread value, so inventing it on error is the same class of lie as
    reporting an unreadable order book as clean (#50c, #54)."""
    fake = _Book()
    ex = _exchange(fake)
    ex.get_orderbook_depth("DOGEUSDT")
    good = ex._last_spread
    assert good > 0

    fake._fail = True
    depth = ex.get_orderbook_depth("DOGEUSDT")

    assert depth["spread_pct"] == good, "a failed fetch fabricated a zero spread"
    assert depth["stale"] is True, "stale data is not flagged as stale"


def test_a_one_sided_book_does_not_crash_or_invent_a_spread():
    ex = _exchange(_Book(bids=[], asks=[[0.07024, 1.0]]))

    depth = ex.get_orderbook_depth("DOGEUSDT")

    assert depth["spread_pct"] == 0.0      # never observed one, not "tight"
    assert depth["stale"] is False


def test_depth_imbalance_reflects_the_book():
    ex = _exchange(_Book(bids=[[0.07023, 300.0]], asks=[[0.07024, 100.0]]))

    depth = ex.get_orderbook_depth("DOGEUSDT")

    assert depth["bid_vol"] == 300.0 and depth["ask_vol"] == 100.0
    assert depth["imbalance"] > 0, "more bids than asks should read as positive imbalance"
