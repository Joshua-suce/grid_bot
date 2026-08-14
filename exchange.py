from __future__ import annotations

import time
import uuid
from typing import Any

import ccxt
import pandas as pd
from loguru import logger


class PostOnlyWouldCross(Exception):
    """A post-only order was rejected (-2019) because it would cross the spread.

    Distinct from a real placement failure: nothing is wrong with the account or the
    order, the price is simply on the wrong side of the book at this instant. Callers
    should skip the level quietly and retry later rather than log an error, alert, or
    resubmit it as a taker order.
    """


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_time: int = 120):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failures = 0
        self.last_failure_time = 0.0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.open = True
            logger.error(
                "CIRCUIT BREAKER OPEN | {} consecutive failures — pausing for {}s",
                self.failures, self.recovery_time,
            )

    def allow_request(self) -> bool:
        if not self.open:
            return True
        if (time.time() - self.last_failure_time) >= self.recovery_time:
            logger.info("CIRCUIT BREAKER HALF-OPEN | attempting request")
            self.open = False
            self.failures = 0
            return True
        return False


MAX_OPEN_ORDERS = 250


class Exchange:
    def __init__(self, config: dict, demo: bool = False, max_retries: int = 3, retry_delay: float = 5.0) -> None:
        self.demo = demo
        self.config = config
        self.has_credentials = bool(config.get("apiKey") and config.get("secret"))
        if not self.has_credentials:
            # No mock/simulated trading fallback: both DEMO and LIVE mode trade against a
            # real Binance account. Config.validate() already enforces this before an
            # Exchange is ever constructed in main.py; this is a defense-in-depth check
            # for any other caller, so a missing-credentials bug fails loudly instead of
            # quietly falling back to fabricated balances/orders (see AUDIT.md).
            raise ValueError(
                "Binance API credentials are required (both DEMO and LIVE mode use a real "
                "account -- there is no mock trading fallback). Set apiKey/secret."
            )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._circuit_breaker = CircuitBreaker()
        self._last_spread: float = 0.0
        self._last_time_sync: float = 0.0
        self._time_sync_interval: float = 20.0
        self._open_order_count: int = 0
        self._last_order_count_time: float = 0.0
        self._balance_cache: dict[str, float] = {}
        # Per-KEY timestamps. A single shared timestamp meant whichever getter ran
        # first refreshed it for all of them, so the next getter saw "fresh" and
        # returned its own stale value. get_balance_cached() runs at main.py:990 and
        # get_total_equity_cached() at :991, back to back -- so equity was fetched once
        # and then served from cache for the rest of the run, while free balance updated
        # every iteration. That equity feeds risk.check_all's drawdown check, which is
        # the kill switch (AUDIT #39).
        self._balance_cache_at: dict[str, float] = {}
        self._balance_cache_time: float = 0.0
        self._balance_cache_ttl: float = 5.0

        self.exchange = ccxt.binanceusdm(config)
        if demo:
            self.exchange.enable_demo_trading(True)

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                self.exchange.load_markets()
                self._sync_time()
                logger.info("Connected to Binance Futures {} mode", "DEMO" if demo else "LIVE")
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning("Connection attempt {}/{} failed: {} — retrying in {}s", attempt, max_retries, e, retry_delay)
                    time.sleep(retry_delay)
                else:
                    logger.error("Connection failed after {} attempts", max_retries)
        raise last_error

    def _sync_time(self) -> None:
        try:
            local_before = int(self.exchange.milliseconds())
            diff = self.exchange.load_time_difference()
            self._last_time_sync = time.time()
            if abs(diff) > 500:
                logger.info("TIME SYNC | diff={}ms (local_before={})", diff, local_before)
            elif abs(diff) > 100:
                logger.debug("TIME SYNC | diff={}ms (local_before={})", diff, local_before)
        except Exception as e:
            self._last_time_sync = time.time()
            logger.debug("Time sync failed: {}", e)

    def maybe_resync_time(self) -> None:
        if time.time() - self._last_time_sync >= self._time_sync_interval:
            self._sync_time()

    def reconnect(self) -> bool:
        """Re-establish exchange connection after a network outage."""
        logger.info("RECONNECTING | re-establishing exchange connection...")
        try:
            self.exchange = ccxt.binanceusdm(self.config)
            if self.demo:
                self.exchange.enable_demo_trading(True)
            self.exchange.load_markets()
            self._sync_time()
            self._circuit_breaker.record_success()
            logger.info("RECONNECTED | exchange connection restored")
            return True
        except Exception as e:
            logger.error("RECONNECT FAILED | {}", e)
            return False

    def _invalidate_balance_cache(self) -> None:
        """Clear the balance cache to force a fresh fetch."""
        self._balance_cache = {}
        self._balance_cache_at = {}
        self._balance_cache_time = 0.0

    def _cache_fresh(self, key: str) -> bool:
        return (time.time() - self._balance_cache_at.get(key, 0.0)) < self._balance_cache_ttl

    def _cache_put(self, key: str, value: float) -> None:
        self._balance_cache[key] = value
        self._balance_cache_at[key] = time.time()
        self._balance_cache_time = self._balance_cache_at[key]

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info("Leverage set to {}x for {}", leverage, symbol)
        except Exception as e:
            logger.warning("Could not set leverage: {}", e)

    @staticmethod
    def _is_timestamp_error(e: Exception) -> bool:
        """True if the error is a timestamp/drift rejection (Binance -1021).

        In ccxt >= 4 the -1021 mapping resolves to InvalidNonce, which is a
        NetworkError, not an ExchangeError, so plain ExchangeError handlers
        never fire for it. Match on both to stay robust across ccxt versions.
        """
        return isinstance(e, ccxt.InvalidNonce) or "-1021" in str(e)

    def _retry(self, fn, *args, label: str = "API call", max_attempts: int | None = None, **kwargs) -> Any:
        if not self._circuit_breaker.allow_request():
            logger.warning("{} skipped — circuit breaker open", label)
            raise ccxt.NetworkError("Circuit breaker open")
        last_err = None
        attempts = max_attempts or self.max_retries
        for attempt in range(1, attempts + 1):
            try:
                result = fn(*args, **kwargs)
                self._circuit_breaker.record_success()
                return result
            except Exception as e:
                last_err = e
                if self._is_timestamp_error(e):
                    self._sync_time()
                    delay = self.retry_delay * attempt
                    logger.warning(
                        "{} timestamp drift (attempt {}/{}): synced time, retrying in {}s",
                        label, attempt, attempts, delay,
                    )
                    time.sleep(delay)
                    continue
                if isinstance(e, (ccxt.RequestTimeout, ccxt.NetworkError)):
                    if not isinstance(e, ccxt.RateLimitExceeded):
                        self._circuit_breaker.record_failure()
                    delay = self.retry_delay * attempt
                    logger.warning(
                        "{} failed (attempt {}/{}): {} — retrying in {}s",
                        label, attempt, attempts, e, delay,
                    )
                    time.sleep(delay)
                elif isinstance(e, (ccxt.OrderNotFound, ccxt.InvalidOrder)):
                    # "That order does not exist" is a legitimate ANSWER to a probe, not
                    # a failure. fetch_order already catches these, logs at debug and
                    # returns None -- but this logged ERROR first, so every routine
                    # check of an order we had just cancelled printed a red line. Three
                    # of them appeared right after the 22:04 recenter, which cancels
                    # everything and then checks what it cancelled (AUDIT #35).
                    # Deliberately NOT a circuit-breaker failure. The exchange answered;
                    # it just said the order is gone. Counting these tripped the breaker
                    # on routine probing -- a recenter cancels every order and then checks
                    # what it cancelled, which is five -2013 replies in a row against a
                    # threshold of five, and an open breaker refuses EVERY request for
                    # 120s including stop-loss placement (AUDIT #39).
                    logger.debug("{}: {}", label, e)
                    raise
                else:
                    logger.error("{} failed: {}", label, e)
                    self._circuit_breaker.record_failure()
                    raise
        logger.error("{} failed after {} retries", label, attempts)
        raise last_err

    def get_ticker(self, symbol: str) -> dict:
        return self._retry(self.exchange.fetch_ticker, symbol, label="fetch_ticker")

    def get_price(self, symbol: str) -> float:
        ticker = self.get_ticker(symbol)
        return ticker["last"]

    def get_spread(self, symbol: str) -> float:
        ticker = self.get_ticker(symbol)
        bid = ticker.get("bid", 0)
        ask = ticker.get("ask", 0)
        if bid and ask and ask > 0:
            spread_pct = (ask - bid) / ask
            self._last_spread = spread_pct
            return spread_pct
        return 0.0

    def get_orderbook_depth(self, symbol: str, limit: int = 10) -> dict:
        try:
            ticker = self._retry(self.exchange.fetch_ticker, symbol, label="fetch_ticker(book)")
            bid = ticker.get("bid", 0)
            ask = ticker.get("ask", 0)
            if bid and ask and ask > 0:
                self._last_spread = (ask - bid) / ask

            book = self._retry(self.exchange.fetch_order_book, symbol, limit=limit, label="fetch_order_book")
            bid_vol = sum(b[1] for b in book.get("bids", [])[:5])
            ask_vol = sum(a[1] for a in book.get("asks", [])[:5])
            imbalance = (bid_vol - ask_vol) / max(1, bid_vol + ask_vol)
            return {
                "bid_vol": bid_vol,
                "ask_vol": ask_vol,
                "imbalance": imbalance,
                "spread_pct": self._last_spread,
            }
        except Exception as e:
            logger.debug("Orderbook fetch failed: {}", e)
            return {"bid_vol": 0, "ask_vol": 0, "imbalance": 0, "spread_pct": 0}

    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        raw = self._retry(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit, label="fetch_ohlcv")
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def get_balance(self, asset: str = "USDT") -> float:
        """Return the available free asset balance."""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get(asset, {}).get("free", 0))
        except Exception as e:
            if self._is_timestamp_error(e):
                self._sync_time()
                balance = self.exchange.fetch_balance()
                return float(balance.get(asset, {}).get("free", 0))
            raise

    def get_balance_cached(self, asset: str = "USDT") -> float:
        """Return cached free balance, fetching only if stale (>5s old)."""
        cache_key = f"free_{asset}"
        if cache_key in self._balance_cache and self._cache_fresh(cache_key):
            return self._balance_cache[cache_key]
        value = self.get_balance(asset)
        self._cache_put(cache_key, value)
        return value

    def get_total_equity(self, asset: str = "USDT") -> float:
        """Return the total asset equity, including used margin or reserved funds."""
        try:
            balance = self.exchange.fetch_balance()
            total = float(balance.get(asset, {}).get("total", 0))
            if total > 0:
                return total
            return float(balance.get(asset, {}).get("free", 0))
        except Exception as e:
            if self._is_timestamp_error(e):
                self._sync_time()
                balance = self.exchange.fetch_balance()
                total = float(balance.get(asset, {}).get("total", 0))
                if total > 0:
                    return total
                return float(balance.get(asset, {}).get("free", 0))
            raise

    def get_total_equity_cached(self, asset: str = "USDT") -> float:
        """Return cached total equity, fetching only if stale."""
        cache_key = f"total_{asset}"
        if cache_key in self._balance_cache and self._cache_fresh(cache_key):
            return self._balance_cache[cache_key]
        value = self.get_total_equity(asset)
        self._cache_put(cache_key, value)
        return value

    def get_balance_info(self, asset: str = "USDT") -> dict[str, float]:
        """Return both free and total balance values for clearer reconciliation."""
        try:
            balance = self.exchange.fetch_balance()
            asset_bal = balance.get(asset, {})
            result = {
                "free": float(asset_bal.get("free", 0)),
                "total": float(asset_bal.get("total", 0)),
                "used": float(asset_bal.get("used", 0)),
            }
            self._cache_put(f"free_{asset}", result["free"])
            self._cache_put(f"total_{asset}", result["total"])
            self._cache_put(f"used_{asset}", result["used"])
            return result
        except Exception as e:
            if self._is_timestamp_error(e):
                self._sync_time()
                balance = self.exchange.fetch_balance()
                asset_bal = balance.get(asset, {})
                return {
                    "free": float(asset_bal.get("free", 0)),
                    "total": float(asset_bal.get("total", 0)),
                    "used": float(asset_bal.get("used", 0)),
                }
            raise

    def get_balance_info_cached(self, asset: str = "USDT") -> dict[str, float]:
        """Return cached balance info, fetching only if stale."""
        now = time.time()
        if self._cache_fresh(f"free_{asset}") and self._cache_fresh(f"total_{asset}"):
            if f"free_{asset}" in self._balance_cache:
                return {
                    "free": self._balance_cache[f"free_{asset}"],
                    "total": self._balance_cache.get(f"total_{asset}", 0.0),
                    "used": self._balance_cache.get(f"used_{asset}", 0.0),
                }
        return self.get_balance_info(asset)

    def get_income_history(
        self, symbol: str, since_ms: int | None = None, income_type: str | None = None, limit: int = 1000,
    ) -> list[dict]:
        """Fetch Binance's own income ledger (REALIZED_PNL, COMMISSION, FUNDING_FEE, ...).

        This is the exchange's ground-truth accounting for a symbol, independent of any
        bookkeeping the bot does locally. Returns raw entries as given by Binance, e.g.
        {"symbol": "DOGEUSDT", "incomeType": "REALIZED_PNL", "income": "0.12345678",
         "time": 1710000000000, "tranId": ..., "asset": "USDT", ...}.
        """
        params: dict[str, Any] = {"limit": limit}
        try:
            params["symbol"] = self.exchange.market(symbol)["id"]
        except Exception:
            params["symbol"] = symbol.replace("/", "").split(":")[0]
        if since_ms is not None:
            params["startTime"] = int(since_ms)
        if income_type is not None:
            params["incomeType"] = income_type
        return self._retry(self.exchange.fapiPrivateGetIncome, params, label="fetch_income")

    def place_limit_order(
        self, symbol: str, side: str, price: float, amount: float, max_attempts: int = 3, params: dict | None = None,
        post_only: bool = True, allow_taker_fallback: bool = False,
    ) -> dict[str, Any]:
        """Place a limit order.

        Normalizes common parameter names and precedence:
        - If 'postOnly' or 'post_only' is present inside params, that value overrides the post_only argument.
        - 'params' is passed through to the underlying exchange API (ccxt) so use keys like 'reduceOnly' and 'stopPrice' there.

        allow_taker_fallback controls what happens when a post-only order is rejected
        with -2019 because it would cross the spread. Defaults to False, which raises
        PostOnlyWouldCross instead of resubmitting as a taker order.

        That default matters for grid economics. A grid level earns the spacing between
        levels and pays a fee to do it; re-sending a crossing order without postOnly
        fills it immediately at the *taker* rate, so the level pays double the fee and
        captures none of the spread it existed to collect. Silently doing that turned
        grid levels into guaranteed small losses. Exit paths (reduce-only unwinds,
        hedges) already pass postOnly=False explicitly and never reach this branch --
        for them, getting out is worth the taker fee.

        Returns the exchange order dict.
        """
        last_err = None
        client_order_id = f"g{uuid.uuid4().hex[:31]}"
        for attempt in range(1, max_attempts + 1):
            try:
                # Merge and normalize params: allow callers to pass post-only via either the post_only
                # kwarg or inside params as 'postOnly' (or 'post_only'). The params dict is what
                # will be forwarded to the exchange library (ccxt) which expects flags in 'params'.
                order_params = dict(params or {})
                # Reuse a single clientOrderId across retries so an ambiguous timeout retry is
                # idempotent (Binance rejects the duplicate while the original order is open,
                # instead of creating a second order at the same grid level).
                if "newClientOrderId" not in order_params:
                    order_params["newClientOrderId"] = client_order_id
                # If params contains a postOnly/post_only key, let it override the post_only arg
                if "postOnly" in order_params:
                    post_only = bool(order_params.pop("postOnly"))
                elif "post_only" in order_params:
                    post_only = bool(order_params.pop("post_only"))
                # Ensure explicit postOnly flag is present in params when required/desired
                if post_only:
                    order_params["postOnly"] = True
                else:
                    # If caller explicitly set post_only False, set postOnly=False so the exchange can act accordingly
                    order_params["postOnly"] = False

                order = self.exchange.create_limit_order(symbol, side, amount, price, order_params)
                order_id = order.get("id", "unknown")
                logger.info(
                    "ORDER PLACED | {} {} {} @ {} (id={}) | attempt={} | postOnly={}",
                    side.upper(), amount, symbol, price, order_id, attempt, post_only,
                )
                return order
            except ccxt.InsufficientFunds as e:
                logger.error("Insufficient funds for {} {} @ {}: {}", side, amount, price, e)
                raise
            except ccxt.InvalidOrder as e:
                err_str = str(e)
                if post_only and ("-2019" in err_str or "would trigger immediate match" in err_str.lower() or "post only" in err_str.lower()):
                    if not allow_taker_fallback:
                        # The level is on the wrong side of the book right now. Resting
                        # it is impossible and crossing it is a guaranteed loss, so
                        # decline to place at all -- the caller re-places it once price
                        # moves back or the grid recenters.
                        logger.debug(
                            "POST-ONLY WOULD CROSS | {} {} @ {} — skipping rather than paying taker",
                            side.upper(), amount, price,
                        )
                        raise PostOnlyWouldCross(
                            f"{side} @ {price} would cross the spread; not placing as taker"
                        ) from e
                    logger.debug("Post-only order rejected (would cross spread) @ {} — placing without postOnly", price)
                    order_params = dict(params or {})
                    order_params["postOnly"] = False
                    # Use a fresh clientOrderId for the fallback: the postOnly attempt above
                    # was REJECTED by the exchange (no order was ever created), but Binance's
                    # testnet has been observed to reject a resubmission that reuses the same
                    # clientOrderId as a phantom duplicate, and that failure was propagating
                    # out of this function entirely (unhandled -- see except block below),
                    # surfacing as a bare, unhelpful "exceptions must derive from BaseException"
                    # from ccxt for unmapped error codes. Give the fallback its own id and its
                    # own error handling so it participates in the normal retry loop instead.
                    order_params["newClientOrderId"] = f"{client_order_id}f{attempt}"
                    try:
                        order = self.exchange.create_limit_order(symbol, side, amount, price, order_params)
                    except Exception as fallback_err:
                        last_err = fallback_err
                        logger.warning(
                            "Post-only fallback also failed @ {} (attempt {}/{}): {}",
                            price, attempt, max_attempts, fallback_err,
                        )
                        if attempt < max_attempts:
                            time.sleep(self.retry_delay * attempt)
                            continue
                        logger.error(
                            "Order placement failed after {} attempts (post-only fallback): {}",
                            max_attempts, fallback_err,
                        )
                        raise
                    order_id = order.get("id", "unknown")
                    logger.info(
                        "ORDER PLACED | {} {} {} @ {} (id={}) | attempt={} | postOnly=False",
                        side.upper(), amount, symbol, price, order_id, attempt,
                    )
                    return order
                logger.error("Invalid order: {} {} {} @ {}: {}", side, amount, symbol, price, e)
                raise
            except Exception as e:
                last_err = e
                if attempt < max_attempts:
                    delay = self.retry_delay * attempt
                    if "-1008" in str(e):
                        delay = max(delay, 10.0)
                        logger.warning("Order placement throttled by exchange protection (-1008) — backing off {}s", delay)
                    else:
                        logger.warning("Order placement failed (attempt {}/{}): {} — retrying in {}s", attempt, max_attempts, e, delay)
                    time.sleep(delay)
                else:
                    logger.error("Order placement failed after {} attempts: {}", max_attempts, e)
        raise last_err

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info("ORDER CANCELLED | id={}", order_id)
            return True
        except ccxt.OrderNotFound:
            logger.debug("Cancel order {} — already gone", order_id)
            return True
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            if self._is_timestamp_error(e):
                self._sync_time()
                try:
                    self.exchange.cancel_order(order_id, symbol)
                    logger.info("ORDER CANCELLED | id={}", order_id)
                    return True
                except ccxt.NetworkError as e2:
                    logger.warning("Cancel order {} status UNKNOWN after time sync: {}", order_id, e2)
                    return False
                except Exception:
                    pass
            elif isinstance(e, ccxt.NetworkError):
                logger.warning(
                    "Cancel order {} status UNKNOWN (network): {} — order may still be open, will re-verify",
                    order_id, e,
                )
                return False
        # Regular cancel failed — try algo/conditional order cancel
        try:
            self.exchange.fapiPrivateDeleteAlgoOrder({
                'symbol': symbol,
                'algoId': order_id,
            })
            logger.info("ALGO ORDER CANCELLED | id={}", order_id)
            return True
        except Exception:
            pass
        logger.warning("Failed to cancel order {} (may already be gone)", order_id)
        return False

    def get_open_orders(self, symbol: str) -> list[dict]:
        return self._retry(self.exchange.fetch_open_orders, symbol, label="fetch_open_orders")

    def get_positions(self, symbol: str) -> list[dict]:
        return self._retry(self.exchange.fetch_positions, [symbol], label="fetch_positions")

    def fetch_order(self, order_id: str, symbol: str, max_attempts: int = 3, delay: float = 1.0) -> dict | None:
        for attempt in range(1, max_attempts + 1):
            try:
                return self._retry(self.exchange.fetch_order, order_id, symbol, label="fetch_order")
            except (ccxt.OrderNotFound, ccxt.InvalidOrder) as e:
                if attempt < max_attempts:
                    logger.debug(
                        "fetch_order {} attempt {}/{} not found, retrying in {}s",
                        order_id, attempt, max_attempts, delay,
                    )
                    time.sleep(delay)
                    continue
                logger.debug("fetch_order {} failed after retries: {}", order_id, e)
                return None
            except Exception as e:
                logger.debug("fetch_order {} failed: {}", order_id, e)
                return None

    def close_position(self, symbol: str, side: str, amount: float, max_attempts: int | None = None) -> dict:
        close_side = "sell" if side == "long" else "buy"
        order = self._retry(
            self.exchange.create_market_order, symbol, close_side, amount,
            label="close_position", max_attempts=max_attempts,
        )
        logger.info("POSITION CLOSED | {} {} {}", close_side.upper(), amount, symbol)
        return order

    def place_stop_market(self, symbol: str, side: str, amount: float, stop_price: float) -> dict:
        """Place a stop-market order (e.g. stop-loss to close a long position)."""
        params = {"stopPrice": stop_price, "reduceOnly": True}
        order = self._retry(
            self.exchange.create_order, symbol, "stop_market", side, amount, None, params,
            label="stop_market",
        )
        logger.info(
            "STOP-MARKET PLACED | {} {} {} @ stop={} (id={})",
            side.upper(), amount, symbol, stop_price, order["id"],
        )
        return order

    def close_all_positions(self, symbol: str) -> int:
        """Close any existing open positions. Returns number of positions closed."""
        positions = self.get_positions(symbol)
        closed = 0
        for pos in positions:
            amt = float(pos.get("contracts", 0) or 0)
            if amt == 0:
                continue
            side = pos.get("side", "")
            if not side:
                continue
            try:
                self.close_position(symbol, side, abs(amt), max_attempts=1)
                closed += 1
                logger.warning("Closed existing {} position: {} {}", side, abs(amt), symbol)
            except Exception as e:
                logger.error("Failed to close {} position: {}", side, e)
        return closed

    def get_open_order_ids(self, symbol: str) -> set[str]:
        """Return set of open order IDs on the exchange."""
        orders = self.get_open_orders(symbol)
        return {o["id"] for o in orders}

    def get_open_order_count(self, symbol: str, max_age: float = 5.0) -> int:
        """Return count of open limit orders (excludes stop/conditional orders)."""
        now = time.time()
        if now - self._last_order_count_time < max_age:
            return self._open_order_count
        orders = self.get_open_orders(symbol)
        self._open_order_count = len(orders)
        self._last_order_count_time = now
        return self._open_order_count

    def can_place_order(self, symbol: str) -> bool:
        """Check if we have room to place more orders."""
        count = self.get_open_order_count(symbol)
        if count >= MAX_OPEN_ORDERS:
            logger.warning("OPEN ORDER LIMIT | {} orders open (max {})", count, MAX_OPEN_ORDERS)
            return False
        return True

    def enforce_order_limit(self, symbol: str, keep_count: int = 10, tracked_ids: set[str] | None = None) -> int:
        """Cancel oldest untracked open orders if near the limit. Returns number cancelled."""
        tracked = tracked_ids or set()
        orders = self.get_open_orders(symbol)
        count = len(orders)
        if count < MAX_OPEN_ORDERS - keep_count:
            self._open_order_count = count
            self._last_order_count_time = time.time()
            return 0
        excess = count - (MAX_OPEN_ORDERS - keep_count)
        logger.warning("ENFORCE ORDER LIMIT | {} open, cancelling {} untracked", count, excess)
        cancelled = 0
        for order in orders:
            if cancelled >= excess:
                break
            order_id = order.get("id")
            if not order_id or order_id in tracked:
                continue
            if self.cancel_order(order_id, symbol):
                cancelled += 1
        self._open_order_count = max(0, count - cancelled)
        self._last_order_count_time = time.time()
        return cancelled

    def cancel_all_open_orders(self, symbol: str) -> int:
        """Cancel every open order for the symbol and return the number cancelled."""
        orders = self.get_open_orders(symbol)
        if orders:
            ids = [o.get("id") for o in orders if o.get("id")]
            logger.info("Open orders before cancellation for {}: {}", symbol, ids)
        cancelled = 0
        for order in orders:
            order_id = order.get("id")
            if not order_id:
                continue
            if self.cancel_order(order_id, symbol):
                cancelled += 1
        logger.info("Cancelled {} open orders for {}", cancelled, symbol)
        return cancelled

    def get_stop_orders(self, symbol: str) -> list[dict] | None:
        """Fetch all open stop/conditional orders (stop-market, take-profit, etc).

        Returns None when the book could not be READ. That is an unknown state, and it
        is not the same as an empty one: returning [] on failure told every caller "this
        position has no stops", which is the single most dangerous lie this class can
        tell. Reconciling stops against a fabricated empty book either tears down live
        protection or double-places it (AUDIT #54, same family as #50c).
        """
        try:
            return self._retry(
                self.exchange.fetch_open_orders, symbol,
                params={"stop": True},
                label="fetch_stop_orders",
            )
        except Exception as e:
            logger.warning("Failed to fetch stop orders: {} — status UNKNOWN, not empty", e)
            return None

    def cancel_all_stop_orders(self, symbol: str) -> int | None:
        """Cancel every stop/conditional (algo) order for the symbol.

        Returns the number cancelled, or None if the stop book could not be read -- in
        which case nothing is known to have been cancelled and the caller must not treat
        the result as a clean sweep (AUDIT #54).
        """
        stop_orders = self.get_stop_orders(symbol)
        if stop_orders is None:
            logger.warning(
                "STOP CANCEL UNVERIFIED | could not read stop orders for {} — no cancel "
                "was attempted and the book state is UNKNOWN", symbol,
            )
            return None
        if not stop_orders:
            return 0
        cancelled = 0
        for order in stop_orders:
            algo_id = order.get("id")
            if not algo_id:
                continue
            try:
                self.exchange.fapiPrivateDeleteAlgoOrder({
                    'symbol': symbol,
                    'algoId': algo_id,
                })
                cancelled += 1
            except Exception as e:
                logger.debug("Algo cancel failed for {}: {}", algo_id, e)
                try:
                    self.exchange.cancel_order(algo_id, symbol)
                    cancelled += 1
                except Exception:
                    pass
        logger.info("Cancelled {} stop/conditional orders for {}", cancelled, symbol)
        return cancelled

    def cancel_everything(self, symbol: str, timeout_seconds: float = 300.0) -> int:
        """Cancel ALL open orders: limit, stop, conditional — everything.

        Blocks up to ``timeout_seconds`` re-verifying against the exchange and retrying,
        so a flaky write path can never report success while orders are still open
        (which is how duplicate grid levels historically piled up). Returns the number
        of orders confirmed cancelled.
        """
        deadline = time.time() + timeout_seconds

        def _fetch_regular() -> dict[str, dict] | None:
            try:
                return {o["id"]: o for o in self.get_open_orders(symbol) if o.get("id")}
            except Exception as e:
                logger.warning("Cleanup: failed to fetch open orders ({}), status unknown", e)
                return None

        pending = _fetch_regular()

        cancelled = 0
        seen_ids: set[str] = set()

        # 1) Try one batch cancel for the regular (limit) orders.
        if pending:
            try:
                self.exchange.cancel_all_orders(symbol)
                cancelled += len(pending)
                logger.info("BATCH CANCELLED {} orders for {}", len(pending), symbol)
                pending = {}
            except (ccxt.RequestTimeout, ccxt.NetworkError) as e:
                logger.warning("Batch cancel timed out ({}); will retry individually", e)
            except Exception as e:
                logger.warning("Batch cancel failed ({}); will retry individually", e)

        # 2) Stop/conditional (algo) orders — attempt once each.
        stop_book = self.get_stop_orders(symbol)
        if stop_book is None:
            # Unreadable, not empty. The regular-order verification below still runs;
            # this just must not be mistaken for "there were no stops" (AUDIT #54).
            logger.warning(
                "CLEANUP | stop/conditional book for {} could not be read — any algo "
                "orders there are NOT known to be cancelled", symbol,
            )
            stop_book = []
        for order in stop_book:
            algo_id = order.get("id")
            if not algo_id or algo_id in seen_ids:
                continue
            seen_ids.add(algo_id)
            try:
                self.exchange.fapiPrivateDeleteAlgoOrder({
                    'symbol': symbol,
                    'algoId': algo_id,
                })
                cancelled += 1
            except Exception:
                if self.cancel_order(algo_id, symbol):
                    cancelled += 1

        # 3) Verify + retry loop: keep cancelling until the book is confirmed clean.
        while time.time() < deadline:
            still_open = _fetch_regular()
            if still_open is None:
                time.sleep(5.0)
                continue
            if not still_open:
                break
            ids = list(still_open)
            for i, oid in enumerate(ids):
                if time.time() >= deadline:
                    break
                if self.cancel_order(oid, symbol):
                    cancelled += 1
                if i < len(ids) - 1:
                    time.sleep(1.0)
            time.sleep(2.0)

        remaining = _fetch_regular()
        if remaining:
            logger.warning(
                "CLEANUP INCOMPLETE | {} orders still open for {} after {}s of retries "
                "(backend unreachable for writes)", len(remaining), symbol, timeout_seconds,
            )
        elif remaining is None:
            # AUDIT #50. _fetch_regular returns None when the READ failed, and `if
            # remaining:` treated that identically to an empty book -- so a cleanup that
            # could not verify anything logged "CLEANUP VERIFIED | book clean". Startup
            # runs this before building a fresh ladder; believing a false all-clear means
            # laying a new grid on top of orders that were never cancelled, i.e. double
            # exposure with no record of it. Unknown is not clean.
            logger.warning(
                "CLEANUP UNVERIFIED | could not read the order book for {} — cancels "
                "were sent but the result is UNKNOWN, not confirmed clean", symbol,
            )
        else:
            logger.info("CLEANUP VERIFIED | book clean for {}", symbol)
        logger.info("CANCEL EVERYTHING | {} total orders confirmed cancelled for {}", cancelled, symbol)
        return cancelled

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exchange.close()
