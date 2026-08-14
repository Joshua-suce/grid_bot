"""Every order says why it exists. AUDIT #56.

Thirty days of ledger, split by the maker flag:

    maker fills  106,398 notional   +120.02 realized  -21.28 comm  ->  +98.74
    TAKER fills   62,948 notional   -106.86 realized  -25.18 comm  -> -132.04

The grid is profitable. Forced exits take it back and more -- the worst ten taker closes
alone are -112.98. But "taker" lumps stop-outs, reconcile closes, emergency closes and
crossed unwinds together, so it does not say which one to fix.

Orders now carry a two-character purpose tag in their clientOrderId, so realized PnL can
be attributed to the mechanism responsible, retroactively, from exchange data alone.
"""

import uuid

import pytest

from exchange import (
    PURPOSE_SEP,
    PURPOSE_TAGS,
    _purpose_tag,
    purpose_of_client_order_id,
)


def _id(purpose):
    return f"{_purpose_tag(purpose)}{uuid.uuid4().hex[:29]}"


@pytest.mark.parametrize("purpose", sorted(PURPOSE_TAGS))
def test_every_purpose_round_trips(purpose):
    assert purpose_of_client_order_id(_id(purpose)) == purpose


@pytest.mark.parametrize("purpose", sorted(PURPOSE_TAGS))
def test_ids_fit_binance_limits(purpose):
    """Binance caps clientOrderId at 36 chars and restricts the alphabet."""
    cid = _id(purpose)
    assert len(cid) <= 36
    assert all(c.isalnum() or c in "._:/-" for c in cid), cid


def test_a_legacy_id_is_not_mistaken_for_a_tagged_one():
    """The bug this separator exists to prevent. Ids used to be "g" + 31 hex chars, so
    any legacy id whose second character was an 'e' decoded as the "ge" (grid_entry)
    tag -- 1 in 16 of them. The first attribution report mis-labelled 80 of 1480 legacy
    executions exactly this way."""
    legacy_collision = "ge" + "a" * 30            # old format, second char happens to be 'e'
    assert purpose_of_client_order_id(legacy_collision) == "untagged"

    for hex_char in "0123456789abcdef":
        assert purpose_of_client_order_id(f"g{hex_char}{'0' * 30}") == "untagged"


def test_separator_is_not_a_hex_digit():
    """The whole scheme rests on this: the random part is hex, so a non-hex separator
    can never be produced by accident."""
    assert PURPOSE_SEP not in "0123456789abcdefABCDEF"


def test_unknown_and_empty_ids_are_untagged_not_guessed():
    for value in ("", None, "x", "zz_abc", "??_abc"):
        assert purpose_of_client_order_id(value) == "untagged"


def test_tags_are_unique():
    tags = list(PURPOSE_TAGS.values())
    assert len(tags) == len(set(tags)), f"duplicate purpose tags: {tags}"


def test_an_unknown_purpose_name_falls_back_rather_than_raising():
    """A mis-typed purpose at a call site must not take the order path down with it."""
    assert purpose_of_client_order_id(_id("not_a_real_purpose")) == "other"


# --- the tags actually reach the exchange ----------------------------------

class _Spy:
    def __init__(self):
        self.params = []

    def create_limit_order(self, symbol, side, amount, price, params):
        self.params.append(params)
        return {"id": "1"}

    def create_market_order(self, symbol, side, amount, params):
        self.params.append(params)
        return {"id": "2"}

    def create_order(self, symbol, type_, side, amount, price, params):
        self.params.append(params)
        return {"id": "3"}


def _exchange(spy):
    from exchange import CircuitBreaker, Exchange

    ex = Exchange.__new__(Exchange)
    ex.exchange = spy
    ex.demo = False
    ex.has_credentials = True
    ex.max_retries = 1
    ex.retry_delay = 0.0
    ex._circuit_breaker = CircuitBreaker(failure_threshold=1000, recovery_time=0)
    return ex


def test_limit_orders_carry_the_tag_and_purpose_never_reaches_ccxt():
    spy = _Spy()
    ex = _exchange(spy)

    ex.place_limit_order("DOGEUSDT", "buy", 0.07, 100,
                         params={"reduceOnly": True, "purpose": "unwind"}, post_only=False)

    sent = spy.params[0]
    assert purpose_of_client_order_id(sent["newClientOrderId"]) == "unwind"
    assert "purpose" not in sent, "the purpose key leaked through to ccxt"
    assert sent["reduceOnly"] is True, "carrying purpose in params clobbered a real param"


def test_stop_orders_carry_the_tag():
    spy = _Spy()
    ex = _exchange(spy)

    ex.place_stop_market("DOGEUSDT", "sell", 632, 0.0673, purpose="stop_trail")

    assert purpose_of_client_order_id(spy.params[0]["newClientOrderId"]) == "stop_trail"


def test_market_closes_carry_the_tag():
    """The path behind the single worst loss on record: BUY 31,761 @ 0.07120, -45.68."""
    spy = _Spy()
    ex = _exchange(spy)

    ex.close_position("DOGEUSDT", "short", 31761, purpose="emergency")

    assert purpose_of_client_order_id(spy.params[0]["newClientOrderId"]) == "emergency"


def test_retries_keep_one_client_order_id_so_tagging_did_not_break_idempotency():
    """A retried timeout must reuse the id, or Binance creates a second order at the
    same level instead of rejecting the duplicate."""
    class Flaky(_Spy):
        def __init__(self):
            super().__init__()
            self.n = 0

        def create_limit_order(self, symbol, side, amount, price, params):
            self.params.append(params)
            self.n += 1
            if self.n == 1:
                import ccxt
                raise ccxt.RequestTimeout("timeout")
            return {"id": "1"}

    spy = Flaky()
    ex = _exchange(spy)
    ex.retry_delay = 0.0

    ex.place_limit_order("DOGEUSDT", "buy", 0.07, 100, max_attempts=2,
                         params={"purpose": "grid_entry"})

    ids = [p["newClientOrderId"] for p in spy.params]
    assert len(set(ids)) == 1, f"retry used a different clientOrderId: {ids}"
    assert purpose_of_client_order_id(ids[0]) == "grid_entry"
