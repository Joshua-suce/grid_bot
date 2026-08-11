import time

import ccxt
import pytest

from exchange import CircuitBreaker, Exchange


def make_exchange(fake):
    ex = Exchange.__new__(Exchange)
    ex.exchange = fake
    ex.demo = False
    ex.has_credentials = True
    ex.max_retries = 1
    ex.retry_delay = 0.0
    ex._circuit_breaker = CircuitBreaker(failure_threshold=1000, recovery_time=0)
    ex._mock_orders = {}
    ex._mock_filled = {}
    return ex


class FakeBackend:
    def __init__(self, orders, cancel_failures=0, batch_failures=0):
        self._orders = {o["id"]: o for o in orders}
        self._cancel_failures = cancel_failures
        self._batch_failures = batch_failures
        self._calls = {"cancel": 0, "batch": 0}

    def fetch_open_orders(self, symbol):
        return list(self._orders.values())

    def cancel_all_orders(self, symbol):
        self._calls["batch"] += 1
        if self._calls["batch"] <= self._batch_failures:
            raise ccxt.RequestTimeout("batch timeout")
        self._orders.clear()

    def cancel_order(self, order_id, symbol):
        self._calls["cancel"] += 1
        if self._calls["cancel"] <= self._cancel_failures:
            raise ccxt.RequestTimeout("cancel timeout")
        self._orders.pop(order_id, None)

    def fapiPrivateDeleteAlgoOrder(self, params):
        pass

    def amount_to_precision(self, symbol, amount):
        return str(amount)


def test_exchange_requires_credentials_in_demo_mode():
    """No mock/simulated trading fallback: DEMO mode without real API keys must
    fail loudly at construction time rather than silently trading against a
    fabricated $10k mock balance (see AUDIT.md)."""
    with pytest.raises(ValueError, match="credentials"):
        Exchange({"apiKey": "", "secret": ""}, demo=True)


def test_exchange_requires_credentials_in_live_mode():
    with pytest.raises(ValueError, match="credentials"):
        Exchange({"apiKey": "", "secret": ""}, demo=False)


def test_cancel_order_network_error_returns_false():
    fake = FakeBackend([{"id": "1"}], cancel_failures=9999)
    ex = make_exchange(fake)
    assert ex.cancel_order("1", "DOGEUSDT") is False


def test_cancel_order_already_gone_returns_true():
    fake = FakeBackend([{"id": "1"}], cancel_failures=9999)

    class NotFoundBackend(FakeBackend):
        def cancel_order(self, order_id, symbol):
            raise ccxt.OrderNotFound("not found")

    ex = make_exchange(NotFoundBackend([{"id": "1"}]))
    assert ex.cancel_order("1", "DOGEUSDT") is True


def test_cancel_everything_retries_until_book_clean():
    fake = FakeBackend(
        [{"id": f"{i}"} for i in range(3)],
        cancel_failures=2,
        batch_failures=1,
    )
    ex = make_exchange(fake)
    cancelled = ex.cancel_everything("DOGEUSDT", timeout_seconds=10)
    assert fake._orders == {}
    assert cancelled >= 3


def test_cancel_everything_reports_when_backend_stays_down():
    fake = FakeBackend(
        [{"id": f"{i}"} for i in range(3)],
        cancel_failures=9999,
        batch_failures=9999,
    )
    ex = make_exchange(fake)
    cancelled = ex.cancel_everything("DOGEUSDT", timeout_seconds=1)
    assert cancelled == 0
    assert len(fake._orders) == 3


class PostOnlyBackend:
    def __init__(self):
        self._placed_without_postonly = 0
        self._tried_postonly = 0

    def create_limit_order(self, symbol, side, amount, price, params):
        if params.get("postOnly") is True:
            self._tried_postonly += 1
            raise ccxt.InvalidOrder("-2019 Post only order would be immediately matched")
        self._placed_without_postonly += 1
        return {"id": "placed-ok", "side": side}

    def fetch_open_orders(self, symbol):
        return []

    def fetch_positions(self, symbols):
        return []

    def amount_to_precision(self, symbol, amount):
        return str(amount)


def test_place_limit_order_survives_postonly_reject_on_last_attempt():
    fake = PostOnlyBackend()
    ex = make_exchange(fake)
    order = ex.place_limit_order("DOGEUSDT", "buy", 0.069, 100, max_attempts=1, post_only=True)
    assert order["id"] == "placed-ok"
    assert fake._tried_postonly == 1
    assert fake._placed_without_postonly == 1


class PostOnlyFallbackFailsOnceBackend(PostOnlyBackend):
    def __init__(self):
        super().__init__()
        self._fallback_calls = 0

    def create_limit_order(self, symbol, side, amount, price, params):
        if params.get("postOnly") is True:
            self._tried_postonly += 1
            raise ccxt.InvalidOrder("-2019 Post only order would be immediately matched")
        self._fallback_calls += 1
        if self._fallback_calls == 1:
            # e.g. an unmapped exchange error code on the retried clientOrderId.
            raise ccxt.ExchangeError("unexpected rejection on fallback")
        self._placed_without_postonly += 1
        return {"id": "placed-ok-2nd-try", "side": side}


def test_place_limit_order_retries_when_postonly_fallback_also_fails():
    """Regression: the postOnly-rejected fallback call used to have no error
    handling of its own, so any failure there propagated out of place_limit_order
    entirely instead of participating in the normal retry loop. Observed live as
    57 unhandled 'exceptions must derive from BaseException' failures from ccxt
    (an unmapped error code) when the fallback hit a transient rejection.
    """
    fake = PostOnlyFallbackFailsOnceBackend()
    ex = make_exchange(fake)
    order = ex.place_limit_order("DOGEUSDT", "buy", 0.069, 100, max_attempts=2, post_only=True)
    assert order["id"] == "placed-ok-2nd-try"
    assert fake._fallback_calls == 2


def test_place_limit_order_raises_real_error_when_fallback_exhausts_retries():
    class AlwaysFailFallbackBackend(PostOnlyBackend):
        def create_limit_order(self, symbol, side, amount, price, params):
            if params.get("postOnly") is True:
                raise ccxt.InvalidOrder("-2019 Post only order would be immediately matched")
            raise ccxt.ExchangeError("persistent failure")

    fake = AlwaysFailFallbackBackend()
    ex = make_exchange(fake)
    with pytest.raises(ccxt.ExchangeError):
        ex.place_limit_order("DOGEUSDT", "buy", 0.069, 100, max_attempts=2, post_only=True)


class DriftBackend:
    def __init__(self, value):
        self.calls = 0
        self.synced = 0
        self._value = value

    def fetch_ticker(self, symbol):
        self.calls += 1
        if self.calls == 1:
            raise ccxt.InvalidNonce("-1021 Timestamp for this request is outside of the recvWindow")
        return {"last": self._value}

    def amount_to_precision(self, symbol, amount):
        return str(amount)


def test_retry_resyncs_time_on_invalid_nonce():
    fake = DriftBackend(0.070)
    ex = make_exchange(fake)
    ex.max_retries = 3
    ex._sync_time = lambda: setattr(fake, "synced", fake.synced + 1)
    result = ex.get_ticker("DOGEUSDT")
    assert result["last"] == 0.070
    assert fake.calls == 2
    assert fake.synced == 1
    assert ex._circuit_breaker.failures == 0


def test_retry_resyncs_on_raw_minus_1021_exchange_error():
    class RawDrift(DriftBackend):
        def fetch_ticker(self, symbol):
            self.calls += 1
            if self.calls == 1:
                raise ccxt.ExchangeError("-1021 your time is ahead of server")
            return {"last": 0.070}

    fake = RawDrift(0.070)
    ex = make_exchange(fake)
    ex.max_retries = 3
    ex._sync_time = lambda: setattr(fake, "synced", fake.synced + 1)
    result = ex.get_ticker("DOGEUSDT")
    assert result["last"] == 0.070
    assert fake.synced == 1


class BalanceDriftBackend:
    def __init__(self, free):
        self.calls = 0
        self._free = free

    def fetch_balance(self):
        self.calls += 1
        if self.calls == 1:
            raise ccxt.InvalidNonce("-1021 timestamp drift")
        return {"USDT": {"free": self._free, "total": self._free, "used": 0.0}}

    def amount_to_precision(self, symbol, amount):
        return str(amount)


def test_get_balance_resyncs_on_invalid_nonce():
    fake = BalanceDriftBackend(123.0)
    ex = make_exchange(fake)
    ex._sync_time = lambda: None
    assert ex.get_balance("USDT") == 123.0
    assert fake.calls == 2


def test_get_total_equity_resyncs_on_invalid_nonce():
    fake = BalanceDriftBackend(456.0)
    ex = make_exchange(fake)
    ex._sync_time = lambda: None
    assert ex.get_total_equity("USDT") == 456.0
    assert fake.calls == 2


def test_cancel_order_resyncs_on_invalid_nonce():
    class CancelDriftBackend(FakeBackend):
        def __init__(self, orders):
            super().__init__(orders)
            self.synced = 0

        def cancel_order(self, order_id, symbol):
            self._calls["cancel"] += 1
            if self._calls["cancel"] == 1:
                raise ccxt.InvalidNonce("-1021 timestamp drift")
            self._orders.pop(order_id, None)

    fake = CancelDriftBackend([{"id": "1"}])
    ex = make_exchange(fake)
    ex._sync_time = lambda: setattr(fake, "synced", fake.synced + 1)
    assert ex.cancel_order("1", "DOGEUSDT") is True
    assert fake.synced == 1
    assert fake._orders == {}


def test_place_limit_order_retries_with_same_client_order_id():
    seen = []

    class RetryBackend:
        def __init__(self):
            self.calls = 0

        def create_limit_order(self, symbol, side, amount, price, params):
            self.calls += 1
            seen.append(params.get("newClientOrderId"))
            if self.calls == 1:
                raise ccxt.RequestTimeout("request timed out")
            return {"id": "abc", "side": side}

        def amount_to_precision(self, symbol, amount):
            return str(amount)

    fake = RetryBackend()
    ex = make_exchange(fake)
    order = ex.place_limit_order("DOGEUSDT", "buy", 0.069, 100, max_attempts=2, post_only=True)
    assert order["id"] == "abc"
    assert fake.calls == 2
    assert seen[0] == seen[1]
    assert seen[0].startswith("g")


def test_place_limit_order_uses_fresh_client_order_id_per_invocation():
    ids = []

    class CaptureBackend:
        def create_limit_order(self, symbol, side, amount, price, params):
            ids.append(params.get("newClientOrderId"))
            return {"id": f"o{len(ids)}", "side": side}

        def amount_to_precision(self, symbol, amount):
            return str(amount)

    fake = CaptureBackend()
    ex = make_exchange(fake)
    ex.place_limit_order("DOGEUSDT", "buy", 0.069, 100, post_only=True)
    ex.place_limit_order("DOGEUSDT", "sell", 0.071, 100, post_only=True)
    assert len(ids) == 2
    assert ids[0] != ids[1]
