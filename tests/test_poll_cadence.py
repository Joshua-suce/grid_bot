"""The loop must poll at the interval it was configured with. AUDIT #96.

Two independent reasons a 10-second poll ran at 13-17 seconds in the 2026-08-17 12:55
session, both measured rather than inferred.

`time.sleep(settings.poll_interval)` sat at the END of the iteration, so the real cadence
was `work + interval`. The configured number was a floor nobody ever reached, and the
drift grows silently with every call added to the loop.

And the work itself was half redundant. Timed against the live venue, seven REST round
trips at ~400ms each:

    fetch_ticker      (get_price)                  399ms
    fetch_balance     (get_balance_cached)         403ms   <-- same payload
    fetch_balance     (get_total_equity_cached)    397ms   <-- as the line above
    fetch_open_orders (enforce_order_limit)        399ms   <-- same book
    fetch_open_orders (check_fills)                384ms   <-- as the line above
    fetch_positions   (get_position_breakdown)     385ms
    fetch_open_orders (get_stop_orders)            399ms

Four of those seven read something another call in the same iteration had already read.

This does NOT buy fills, and the module is not a throughput claim: over that session the
price path was 2.21% against 0.2016% spacing, allowing ~11 crossings, and the bot booked
11 fills. What a true cadence buys is reaction time on the things that run once per
iteration -- the stop-loss refresh, the risk kill switches, the awaiting-counter release.
"""

import re
from pathlib import Path

import main
from exchange import Exchange
from grid import GridEngine
from main import sleep_until_next_poll


# --- sleep the remainder, not the whole -----------------------------------------------

def test_the_sleep_is_shortened_by_the_work_already_done():
    import time

    started = time.monotonic() - 0.30
    slept = sleep_until_next_poll(started, 0.40)

    assert 0.05 <= slept <= 0.15, f"slept {slept:.3f}s after 0.30s of work in a 0.40s poll"


def test_an_iteration_that_overran_does_not_sleep_at_all():
    """Negative remainders must not become a negative sleep, and must not become a full
    one either -- an iteration slower than the interval is already late."""
    import time

    started = time.monotonic() - 5.0

    assert sleep_until_next_poll(started, 0.20) == 0.0


def test_an_instant_iteration_still_sleeps_the_whole_interval():
    import time

    started = time.monotonic()
    slept = sleep_until_next_poll(started, 0.20)

    assert 0.15 <= slept <= 0.20


def test_the_loop_measures_from_the_top_of_the_iteration():
    """The clock has to start before the work, or the deadline means nothing. Both halves
    are load-bearing and live ~500 lines apart."""
    src = Path(main.__file__).read_text(encoding="utf-8")
    body = src[src.index("        while True:"):]

    start = body.index("iteration_started = time.monotonic()")
    use = body.index("sleep_until_next_poll(iteration_started, settings.poll_interval)")

    assert start < use, "the iteration clock is started after it is read"
    assert re.search(r"consecutive_errors = 0\s*\n\s*sleep_until_next_poll\(", body), (
        "the end-of-iteration sleep is still a fixed one"
    )


def test_a_raising_iteration_still_backs_off_the_full_interval():
    """The deadline sleep shortens itself by the work already done, so on the error path
    it would retry fastest after the failures that take longest to reach -- the loop
    would spin hardest on the worst defects. That one path stays a full sleep."""
    src = Path(main.__file__).read_text(encoding="utf-8")
    handler = src[src.index("Loud and running beats silent and flat."):]
    handler = handler[:handler.index("continue")]

    assert "time.sleep(settings.poll_interval)" in handler, (
        "the loop-level exception handler now backs off by less the later it fails"
    )


# --- one balance read per iteration ---------------------------------------------------

def _cache_only_exchange():
    ex = Exchange.__new__(Exchange)
    ex._balance_cache = {}
    ex._balance_cache_at = {}
    ex._balance_cache_time = 0.0
    ex._balance_cache_ttl = 5.0
    return ex


def test_free_and_equity_come_from_a_single_round_trip():
    import types

    ex = _cache_only_exchange()
    calls = {"n": 0}

    def info(self, asset="USDT"):
        calls["n"] += 1
        return {"free": 4800.0, "total": 4900.0, "used": 30.0}

    ex.get_balance_info = types.MethodType(info, ex)

    ex.get_balance_cached()
    ex.get_total_equity_cached()

    assert calls["n"] == 1, f"the free+equity pair cost {calls['n']} fetch_balance calls"


# --- one open-order read per iteration ------------------------------------------------

class CountingExchange:
    def __init__(self, orders=()):
        self.orders = list(orders)
        self.fetches = 0

    def get_open_orders(self, symbol):
        self.fetches += 1
        return list(self.orders)


def _quiet_engine(ex):
    """A GridEngine reduced to check_fills' own bookkeeping -- no levels, so the sweep
    runs end to end without needing the rest of the ladder wired up."""
    g = GridEngine.__new__(GridEngine)
    g.symbol = "DOGEUSDT"
    g.exchange = ex
    g.levels = []
    g._current_price_or_none = lambda: None
    g._release_awaiting_levels = lambda price: None
    g._repair_ladder = lambda price: None
    g.reconcile_position_entry = lambda: None
    return g


def test_check_fills_uses_the_book_it_is_given():
    ex = CountingExchange()

    _quiet_engine(ex).check_fills(0.0, open_orders=[])

    assert ex.fetches == 0, "re-read a book the caller had already read"


def test_check_fills_still_reads_its_own_book_when_not_given_one():
    """Every other caller, and every existing test, passes nothing."""
    ex = CountingExchange()

    _quiet_engine(ex).check_fills(0.0)

    assert ex.fetches == 1


def test_enforce_order_limit_uses_the_book_it_is_given():
    ex = Exchange.__new__(Exchange)
    ex._open_order_count = 0
    ex._last_order_count_time = 0.0
    counter = CountingExchange()
    ex.get_open_orders = counter.get_open_orders

    assert ex.enforce_order_limit("DOGEUSDT", keep_count=10, orders=[]) == 0
    assert counter.fetches == 0


def test_enforce_order_limit_still_reads_its_own_book_when_not_given_one():
    ex = Exchange.__new__(Exchange)
    ex._open_order_count = 0
    ex._last_order_count_time = 0.0
    counter = CountingExchange()
    ex.get_open_orders = counter.get_open_orders

    ex.enforce_order_limit("DOGEUSDT", keep_count=10)

    assert counter.fetches == 1


def test_a_cancel_forces_a_fresh_book_before_the_fill_sweep():
    """check_fills decides what filled by ABSENCE from the list. Handing it a snapshot
    taken before enforce_order_limit cancelled orders would read those cancellations as
    fills -- inventing trades, entries and P&L that never happened."""
    src = Path(main.__file__).read_text(encoding="utf-8")
    window = src[src.index("open_orders = exchange.get_open_orders(settings.symbol)"):
                 src.index("fills = grid.check_fills(")]

    assert re.search(r"if exchange\.enforce_order_limit\(", window), (
        "the limit check no longer gates the re-read"
    )
    assert window.count("exchange.get_open_orders(settings.symbol)") == 2, (
        "no re-read after a cancel: the fill sweep would see a stale book"
    )


def test_every_strategy_accepts_the_shared_book():
    """main.py calls check_fills through the Strategy protocol, so a signature that only
    GridEngine understands is a TypeError the moment the router hands over."""
    import inspect

    from grid import GridEngine as G
    from router import StrategyRouter
    from trend_follower import TrendFollower

    for impl in (G, StrategyRouter, TrendFollower):
        params = inspect.signature(impl.check_fills).parameters
        assert "open_orders" in params, f"{impl.__name__}.check_fills cannot take the book"
        assert params["open_orders"].default is None, (
            f"{impl.__name__}.check_fills makes the shared book mandatory"
        )
