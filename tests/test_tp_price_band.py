"""A target the venue will never accept must not be re-offered every minute.

2026-08-20 05:35:28. A 3R target of 0.2445 on a wide ATR stop, against a Binance
PERCENT_PRICE ceiling of 0.22887 with spot at 0.2180:

    -4016 Limit price can't be higher than 0.22887.

The old handler treated that like any placement failure: ERROR, back off 60s, try
again. Retrying cannot fix it -- only price moving toward the target makes it
placeable -- so it produced a recurring red line for a condition that is not a
fault. check_fills' software comparison is what actually closes the position; the
resting order is an optimisation. AUDIT #135.
"""
from __future__ import annotations

import time

import pytest

from trend_follower import (TP_BAND_RETRY_SECONDS, TrendFollower,
                            is_price_band_rejection)


# ------------------------------------------------------- telling the two apart
@pytest.mark.parametrize("msg", [
    'binanceusdm {"code":-4016,"msg":"Limit price can\'t be higher than 0.22887."}',
    'binanceusdm {"code":-4016,"msg":"Limit price can\'t be lower than 0.19."}',
    'binanceusdm {"code":-4015,"msg":"Client order id length should be less..."}',
    'binanceusdm {"code":-1013,"msg":"Filter failure: PERCENT_PRICE"}',
])
def test_band_rejections_are_recognised(msg):
    assert is_price_band_rejection(Exception(msg)) is True


@pytest.mark.parametrize("msg", [
    'binanceusdm {"code":-2019,"msg":"Margin is insufficient."}',
    'binanceusdm {"code":-1021,"msg":"Timestamp for this request is outside..."}',
    "Connection reset by peer",
    "HTTP 408 Request Timeout",
])
def test_ordinary_failures_are_not_band_rejections(msg):
    """These DO want a short retry. Conflating them would bury a real fault for ten
    minutes at a time."""
    assert is_price_band_rejection(Exception(msg)) is False


def test_the_two_verdicts_are_actually_distinguishable():
    """Guard on the guard: a predicate stuck on one answer would satisfy exactly half
    the assertions above and look fine in isolation."""
    band = is_price_band_rejection(Exception('{"code":-4016,"msg":"x"}'))
    other = is_price_band_rejection(Exception("Connection reset by peer"))
    assert band != other


# ------------------------------------------------------------ the arming path
BAND_ERROR = Exception(
    'binanceusdm {"code":-4016,"msg":"Limit price can\'t be higher than 0.22887."}')


class _Ex:
    def __init__(self, error=None):
        self.error = error
        self.attempts = 0

    def place_limit_order(self, symbol, side, price, qty, **kw):
        self.attempts += 1
        if self.error:
            raise self.error
        return {"id": "tp-1"}


def _follower(exchange, target=0.2445):
    t = TrendFollower.__new__(TrendFollower)
    t.exchange = exchange
    t.symbol = "ADAUSDT"
    t._side = "long"
    t._take_profit_price = target
    t._tp_order_id = None
    t._tp_retry_after = 0.0
    t._event_journal = None
    t._notifier = None
    return t


def test_a_band_rejection_backs_off_far_longer_than_a_normal_failure():
    """Ten minutes, not one. The condition resolves on price, not on time."""
    tf = _follower(_Ex(BAND_ERROR))
    before = time.time()

    tf._arm_take_profit(111.0)

    assert tf._tp_retry_after >= before + TP_BAND_RETRY_SECONDS - 1
    assert TP_BAND_RETRY_SECONDS > 60.0, "no longer than the ordinary-failure backoff"


def test_an_ordinary_failure_still_retries_soon():
    """The band path must not swallow real faults into a ten-minute silence."""
    tf = _follower(_Ex(Exception("Connection reset by peer")))
    before = time.time()

    tf._arm_take_profit(111.0)

    assert before + 30 < tf._tp_retry_after < before + 120


def test_the_band_rejection_is_announced_once_not_every_attempt():
    """The whole complaint was a recurring red line. Say it once."""
    ex = _Ex(BAND_ERROR)
    tf = _follower(ex)

    tf._arm_take_profit(111.0)
    assert tf._tp_band_deferred is True

    tf._tp_retry_after = 0.0          # let it try again
    tf._arm_take_profit(111.0)

    assert ex.attempts == 2, "it stopped trying altogether"
    assert tf._tp_band_deferred is True


def test_a_successful_placement_clears_the_deferral():
    """Otherwise the next band rejection on a new target goes unannounced."""
    tf = _follower(_Ex(BAND_ERROR))
    tf._arm_take_profit(111.0)
    assert tf._tp_band_deferred is True

    tf.exchange = _Ex()               # venue now accepts it
    tf._tp_retry_after = 0.0
    tf._tp_order_id = None
    tf._arm_take_profit(111.0)

    assert tf._tp_order_id == "tp-1"
    assert tf._tp_band_deferred is False


def test_the_deferral_flag_survives_construction_by_new():
    """_tp_band_deferred must be a CLASS attribute. An __init__-only attribute has
    broken this suite twice: tests build strategies with __new__, and the first read
    then raises AttributeError."""
    assert TrendFollower._tp_band_deferred is False
    assert TrendFollower.__new__(TrendFollower)._tp_band_deferred is False


def test_the_target_is_kept_so_the_software_exit_still_works():
    """Deferring the resting order must not abandon the target itself -- check_fills
    compares against _take_profit_price and that is what actually closes."""
    tf = _follower(_Ex(BAND_ERROR))

    tf._arm_take_profit(111.0)

    assert tf._take_profit_price == 0.2445
    assert tf._tp_order_id is None
