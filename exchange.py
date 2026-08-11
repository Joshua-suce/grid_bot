from __future__ import annotations

import time
import uuid
from typing import Any

import ccxt
import pandas as pd
from loguru import logger


DEMO_MOCK_BALANCE = 10000.0


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
        self._mock_orders: dict[str, dict] = {}
        self._mock_filled: dict[str, dict] = {}
        self._mock_positions: dict[str, float] = {}
        self._mock_balance: float = DEMO_MOCK_BALANCE
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._circuit_breaker = CircuitBreaker()
        self._last_spread: float = 0.0
        self._last_time_sync: float = 0.0
        self._time_sync_interval: float = 20.0
        self._open_order_count: int = 0
        self._last_order_count_time: float = 0.0
        self._balance_cache: dict[str, float] = {}
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
                if demo and self.has_credentials:
                    logger.info("Connected to Binance Futures DEMO mode")
                elif demo:
                    logger.warning(
                        "Demo mode WITHOUT API keys — using mock balance of {} USDT. "
                        "Get demo keys from demo.binance.com",
                        DEMO_MOCK_BALANCE,
                    )
                else:
                    logger.info("Connected to Binance Futures LIVE mode")
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
        self._balance_cache_time = 0.0

    def set_leverage(self, symbol: str, leverage: int) -> None:
        if self.demo and not self.has_credentials:
            logger.info("Leverage set to {}x for {} (mock)", leverage, symbol)
            return
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
        if self.demo and not self.has_credentials:
            return self._mock_balance
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
        now = time.time()
        cache_key = f"free_{asset}"
        if (now - self._balance_cache_time) < self._balance_cache_ttl and cache_key in self._balance_cache:
            return self._balance_cache[cache_key]
        value = self.get_balance(asset)
        self._balance_cache[cache_key] = value
        self._balance_cache_time = now
        return value

    def get_total_equity(self, asset: str = "USDT") -> float:
        """Return the total asset equity, including used margin or reserved funds."""
        if self.demo and not self.has_credentials:
            return self._mock_balance
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
        now = time.time()
        cache_key = f"total_{asset}"
        if (now - self._balance_cache_time) < self._balance_cache_ttl and cache_key in self._balance_cache:
            return self._balance_cache[cache_key]
        value = self.get_total_equity(asset)
        self._balance_cache[cache_key] = value
        self._balance_cache_time = now
        return value

    def get_balance_info(self, asset: str = "USDT") -> dict[str, float]:
        """Return both free and total balance values for clearer reconciliation."""
        if self.demo and not self.has_credentials:
            return {"free": self._mock_balance, "total": self._mock_balance, "used": 0.0}
        try:
            balance = self.exchange.fetch_balance()
            asset_bal = balance.get(asset, {})
            result = {
                "free": float(asset_bal.get("free", 0)),
                "total": float(asset_bal.get("total", 0)),
                "used": float(asset_bal.get("used", 0)),
            }
            self._balance_cache[f"free_{asset}"] = result["free"]
            self._balance_cache[f"total_{asset}"] = result["total"]
            self._balance_cache[f"used_{asset}"] = result["used"]
            self._balance_cache_time = time.time()
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
        if (now - self._balance_cache_time) < self._balance_cache_ttl:
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
        if self.demo and not self.has_credentials:
            return []
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
        post_only: bool = True,
    ) -> dict[str, Any]:
        """Place a limit order.

        Normalizes common parameter names and precedence:
        - If 'postOnly' or 'post_only' is present inside params, that value overrides the post_only argument.
        - 'params' is passed through to the underlying exchange API (ccxt) so use keys like 'reduceOnly' and 'stopPrice' there.

        Returns the exchange order dict.
        """
        if self.demo and not self.has_credentials:
            fake_id = f"MOCK-{side[:1].upper()}-{int(time.time()*1000)}"
            self._mock_orders[fake_id] = {"side": side, "price": price}
            logger.info(
                "MOCK ORDER | {} {} {} @ {} (id={})",
                side.upper(), amount, symbol, price, fake_id,
            )
            return {"id": fake_id}

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
        if self.demo and not self.has_credentials:
            self._mock_orders.pop(order_id, None)
            logger.info("MOCK CANCEL | id={}", order_id)
            return True
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
        if self.demo and not self.has_credentials:
            current = self._retry(self.exchange.fetch_ticker, symbol, label="fetch_ticker(mock)")
            current = current["last"]
            open_list = []
            filled = []
            for oid, order in self._mock_orders.items():
                if order["side"] == "buy" and current <= order["price"]:
                    filled.append(oid)
                    continue
                if order["side"] == "sell" and current >= order["price"]:
                    filled.append(oid)
                    continue
                open_list.append({"id": oid})
            for oid in filled:
                self._mock_filled[oid] = self._mock_orders.pop(oid)
            return open_list
        return self._retry(self.exchange.fetch_open_orders, symbol, label="fetch_open_orders")

    def get_positions(self, symbol: str) -> list[dict]:
        if self.demo and not self.has_credentials:
            return []
        return self._retry(self.exchange.fetch_positions, [symbol], label="fetch_positions")

    def fetch_order(self, order_id: str, symbol: str, max_attempts: int = 3, delay: float = 1.0) -> dict | None:
        if self.demo and not self.has_credentials:
            order = self._mock_orders.get(order_id)
            if order is not None:
                return {"id": order_id, "status": "open", "side": order["side"], "price": order["price"]}
            filled = self._mock_filled.get(order_id)
            if filled is not None:
                return {"id": order_id, "status": "closed", "side": filled["side"], "price": filled["price"]}
            return None

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
        if self.demo and not self.has_credentials:
            key = f"MOCK-CLOSE-{side[:1].upper()}-{int(time.time()*1000)}"
            logger.info("MOCK CLOSE POSITION | {} {} {} (id={})", close_side.upper(), amount, symbol, key)
            return {"id": key}
        order = self._retry(
            self.exchange.create_market_order, symbol, close_side, amount,
            label="close_position", max_attempts=max_attempts,
        )
        logger.info("POSITION CLOSED | {} {} {}", close_side.upper(), amount, symbol)
        return order

    def place_stop_market(self, symbol: str, side: str, amount: float, stop_price: float) -> dict:
        """Place a stop-market order (e.g. stop-loss to close a long position)."""
        if self.demo and not self.has_credentials:
            fake_id = f"MOCK-SL-{int(time.time()*1000)}"
            self._mock_orders[fake_id] = {"side": side, "price": stop_price, "type": "stop_market"}
            logger.info(
                "MOCK STOP-MARKET | {} {} {} @ stop={} (id={})",
                side.upper(), amount, symbol, stop_price, fake_id,
            )
            return {"id": fake_id}
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
        if self.demo and not self.has_credentials:
            return 0
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

    def get_stop_orders(self, symbol: str) -> list[dict]:
        """Fetch all open stop/conditional orders (stop-market, take-profit, etc)."""
        if self.demo and not self.has_credentials:
            return []
        try:
            return self._retry(
                self.exchange.fetch_open_orders, symbol,
                params={"stop": True},
                label="fetch_stop_orders",
            )
        except Exception as e:
            logger.warning("Failed to fetch stop orders: {}", e)
            return []

    def cancel_all_stop_orders(self, symbol: str) -> int:
        """Cancel every stop/conditional (algo) order for the symbol."""
        if self.demo and not self.has_credentials:
            return 0
        stop_orders = self.get_stop_orders(symbol)
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
        for order in self.get_stop_orders(symbol):
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
        else:
            logger.info("CLEANUP VERIFIED | book clean for {}", symbol)
        logger.info("CANCEL EVERYTHING | {} total orders confirmed cancelled for {}", cancelled, symbol)
        return cancelled

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exchange.close()
