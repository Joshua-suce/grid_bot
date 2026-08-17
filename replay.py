"""Replay recorded prices through the real GridEngine. No network, no waiting.

Every behavioural defect found live so far -- #75 phantom fills, #77 one-way
starvation, #79 the vacated ladder line, #80 the overstated cycle P&L -- was fully
visible in data the bot had already written to its own log. Finding them cost 8-hour
live runs anyway, because there was no way to re-run a recorded session.

This is that way. It drives the actual GridEngine against a paper exchange that fills
resting limit orders when the recorded price crosses them, and asserts the invariants
that live runs kept violating:

  * the ladder keeps one level per grid line and never loses one       (#79)
  * no post-only order is ever placed on the wrong side of the market  (#77)
  * the grid's own P&L claim matches the fills that actually happened  (#80)

What it CANNOT do: rank grid spacings. --sweep runs, and prints a confident-looking
table, but the table has no resolving power. Measured on 13.1h / 101,013 real trade
ticks across three sessions, sweeping 0.08%-0.30%:

    session        hours   0.08   0.10   0.12   0.15   0.20   0.25   0.30
    2026-08-15 -0    2.3  -0.00  +1.66  -0.25  +0.89  +1.81  +2.27  +2.64
    2026-08-15 -2    8.2  +3.58  +2.74  +3.41  +2.67  +1.15  +0.65  -0.15
    2026-08-16 -2    2.6  +8.49  +3.95  +6.49  +2.84  +4.91  +7.36  +8.35

Adjacent spacings on the SAME session differ by more than the best-to-worst spread of
the pooled table (3.65 vs 1.90 USDT/day). The curves are not merely noisy, they are
shaped differently per session and disagree on the winner. That is what a measurement
looks like when it is reading the price path rather than the parameter.

Three separate sweeps have now named three different "best" spacings -- 0.20% from a
187h live survey, 0.08% from poll snapshots, 0.08% again from the tape but on a curve
whose own neighbours swing wider than the result. Pooling more sessions will not fix
this; the variance is between sessions, not within them. Do not report a winner off
this harness.

Two further reasons, found while building spacing_study.py (AUDIT #99):

  * The three tightest columns place NO ORDERS. _is_level_profitable requires spacing
    to clear the round trip times MIN_PROFIT_MULTIPLIER -- 0.126% at current settings --
    so 0.08%, 0.10% and 0.12% are inert, and 0.08% is the column two of those sweeps
    named the winner. Measured on 14,777 real ticks (2026-08-17): zero fills at all
    three, against 14 at 0.15%. That column is an untraded position drifting.

  * The table pools raw session totals, so an 8.2h session outvotes a 2.3h one on
    length alone. Normalised per day the ranking changes hands.

Use spacing_study.py instead: it differences within a session, puts an interval on the
answer, and refuses when the interval spans zero. --sweep remains useful for what it was
always good at -- driving the engine hard enough to expose ladder defects.

Usage:
    py replay.py                       # replay the newest log
    py replay.py logs/grid_2026-08-15.log
    py replay.py --run 1               # an earlier run in the same log (0 = newest)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import grid as _grid
from config import settings
from exchange import PostOnlyWouldCross
from grid import GridEngine


class VirtualClock:
    """Deterministic time for the engine under replay.

    GridEngine caches the position and the open-order map for 2.0 wall-clock seconds
    (grid.py break-even and _open_orders_map). A replay pushes ~2000 ticks through in
    about ten seconds of real time, so those caches straddle real seconds arbitrarily
    and the SAME code gives different answers run to run -- measured: 6 fills/14 lines
    on one run, 5 fills/13 lines on the next.

    That makes a replay useless as a regression test, and it is also unfaithful: live
    polls are ~15s apart, so those caches have always expired by the next tick. The
    clock advances one poll interval per tick, which is both deterministic and closer
    to what the bot really sees.
    """

    def __init__(self, step: float, start: float = 1_700_000_000.0):
        self._now = float(start)
        self._step = float(step)

    def advance(self) -> None:
        self._now += self._step

    def time(self) -> float:
        return self._now

    def sleep(self, seconds):        # never actually wait during a replay
        self._now += float(seconds or 0)

    def __getattr__(self, name):     # anything else falls through to the real module
        import time as _real
        return getattr(_real, name)


# --------------------------------------------------------------------------------
# a paper exchange: same surface GridEngine calls, none of the network
# --------------------------------------------------------------------------------

class _Ccxt:
    """The precision helpers GridEngine reaches for via exchange.exchange."""

    def __init__(self, price_dp: int, amount_dp: int):
        self._p, self._a = price_dp, amount_dp

    def price_to_precision(self, symbol, price):
        return f"{float(price):.{self._p}f}"

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.{self._a}f}"


class PaperExchange:
    """Fills a resting limit order the moment the recorded price reaches it.

    Position is NETTED, one-way, exactly as Binance USDM runs it -- a buy against a
    short reduces it and realises P&L rather than opening a second leg. Getting that
    wrong is what makes the grid's own (exit-entry)*qty arithmetic disagree with the
    account, so the paper book models it properly and reports the real number.
    """

    def __init__(self, symbol: str, balance: float = 5000.0,
                 maker_fee: float = 0.0002, price_dp: int = 5, amount_dp: int = 0):
        self.symbol = symbol
        self.exchange = _Ccxt(price_dp, amount_dp)
        self.free = balance
        self.maker_fee = maker_fee
        self.price = 0.0
        self.orders: dict[str, dict] = {}
        self.qty = 0.0          # signed: + long, - short
        self.entry = 0.0
        self.realized = 0.0     # gross, before fees
        self.fees = 0.0
        self.trades: list[dict] = []
        self.crossing_orders: list[dict] = []
        self._seq = 0

    # --- driving ---------------------------------------------------------------

    def tick(self, price: float) -> list[str]:
        self.price = price
        hit = [
            oid for oid, o in self.orders.items()
            if o["status"] == "open"
            and ((o["side"] == "buy" and price <= o["price"])
                 or (o["side"] == "sell" and price >= o["price"]))
        ]
        for oid in hit:
            self._execute(oid)
        return hit

    def _execute(self, oid: str) -> None:
        o = self.orders[oid]
        o["status"] = "closed"
        o["filled"] = o["amount"]
        o["info"]["executedQty"] = str(o["amount"])
        signed = o["amount"] if o["side"] == "buy" else -o["amount"]
        px = o["price"]
        fee = abs(signed) * px * self.maker_fee
        self.fees += fee

        if self.qty == 0 or (self.qty > 0) == (signed > 0):
            # opening or adding: blend the entry
            total = self.qty + signed
            self.entry = (self.entry * self.qty + px * signed) / total if total else 0.0
            self.qty = total
        else:
            closing = min(abs(signed), abs(self.qty))
            direction = 1 if self.qty > 0 else -1
            self.realized += (px - self.entry) * closing * direction
            self.qty += signed
            if abs(self.qty) < 1e-9:
                self.qty, self.entry = 0.0, 0.0
            elif (self.qty > 0) != (direction > 0):
                self.entry = px          # flipped through zero
        self.trades.append({"id": oid, "side": o["side"], "price": px,
                            "amount": o["amount"], "fee": fee})

    @property
    def net(self) -> float:
        """What the account actually made: realised P&L less every fee paid."""
        return self.realized - self.fees

    # --- the surface GridEngine calls ------------------------------------------

    def get_price(self, symbol):
        return self.price

    def get_balance(self, asset="USDT"):
        return self.free

    def can_place_order(self, symbol):
        return True

    def get_orderbook_depth(self, symbol, limit=10):
        return {"bids": [[self.price, 1e6]], "asks": [[self.price, 1e6]],
                "bid_volume": 1e6, "ask_volume": 1e6, "imbalance": 1.0}

    def place_limit_order(self, symbol, side, price, amount, max_attempts=3,
                          params=None, post_only=True, allow_taker_fallback=False,
                          purpose="gg"):
        # Strictly through the market. Resting AT the touch is a legitimate maker
        # order; only a bid above it or an offer below it would take liquidity.
        crosses = (side == "buy" and price > self.price) or \
                  (side == "sell" and price < self.price)
        if post_only and crosses:
            # The real client raises PostOnlyWouldCross on -2019, and grid.py catches it
            # as a deliberate no-op: leave the level unplaced and retry next pass, never
            # take liquidity. Returning None instead sent the engine down its generic
            # error branch -- error log, journal entry, Telegram alert -- which is a
            # path production never takes, and made a normal retry look like 287
            # failures (AUDIT #85).
            self.crossing_orders.append({"side": side, "price": price,
                                         "market": self.price})
            raise PostOnlyWouldCross(
                f"post-only {side} {price} would cross market {self.price}")
        self._seq += 1
        oid = str(self._seq)
        self.orders[oid] = {"id": oid, "side": side, "price": float(price),
                            "amount": float(amount), "status": "open",
                            "filled": 0.0, "info": {"executedQty": "0"}}
        return dict(self.orders[oid])

    def fetch_order(self, order_id, symbol, **kw):
        o = self.orders.get(str(order_id))
        return dict(o) if o else None

    def cancel_order(self, order_id, symbol):
        o = self.orders.get(str(order_id))
        if o and o["status"] == "open":
            o["status"] = "canceled"
            return True
        return False

    def get_open_orders(self, symbol):
        return [dict(o) for o in self.orders.values() if o["status"] == "open"]

    def get_open_order_ids(self, symbol):
        return {o["id"] for o in self.orders.values() if o["status"] == "open"}

    def get_positions(self, symbol):
        if self.qty == 0:
            return []
        return [{"contracts": abs(self.qty), "entryPrice": self.entry,
                 "side": "long" if self.qty > 0 else "short"}]

    def close_position(self, symbol, *a, **kw):
        if self.qty:
            self.realized += (self.price - self.entry) * self.qty
            self.qty, self.entry = 0.0, 0.0
        return True

    def cancel_all_open_orders(self, symbol):
        n = 0
        for o in self.orders.values():
            if o["status"] == "open":
                o["status"] = "canceled"
                n += 1
        return n

    def cancel_everything(self, symbol):
        return self.cancel_all_open_orders(symbol)

    def enforce_order_limit(self, *a, **kw):
        return 0


# --------------------------------------------------------------------------------
# reading a recorded run
# --------------------------------------------------------------------------------

def load_prices(path: Path, run: int = 0) -> list[float]:
    """Prices from one run in a log. run=0 is the newest run in the file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = [m.start() for m in re.finditer(r"GRID BOT STARTING", text)]
    if not starts:
        raise SystemExit(f"{path} has no run in it")
    if run >= len(starts):
        raise SystemExit(f"{path} has {len(starts)} run(s); asked for index {run}")
    chunk = text[starts[-1 - run]:]
    end = chunk.find("GRID BOT STARTING", 1)
    if end != -1:
        chunk = chunk[:end]
    return [float(p) for p in re.findall(r"PRICE=([\d.]+)", chunk)]


# --------------------------------------------------------------------------------
# the replay
# --------------------------------------------------------------------------------

def replay(prices: list[float], *, grid_count: int | None = None,
           spacing_pct: float | None = None, quiet: bool = False,
           min_profit_multiplier: float | None = None) -> dict:
    start = prices[0]
    count = grid_count or settings.grid_count
    span = (spacing_pct if spacing_pct is not None
            else settings.range_min_spacing_pct) * count / 2
    lower, upper = start * (1 - span), start * (1 + span)

    paper = PaperExchange(settings.symbol, maker_fee=settings.maker_fee_pct / 100)
    paper.price = start

    grid = GridEngine(
        paper, settings.symbol,
        grid_lower=lower, grid_upper=upper, grid_count=count,
        capital_per_grid_pct=settings.capital_per_grid_pct,
        stop_loss_pct=settings.stop_loss_pct,
        maker_fee_pct=settings.maker_fee_pct / 100,
        taker_fee_pct=settings.taker_fee_pct / 100,
        taker_fill_share=settings.taker_fill_share_pct / 100,
        recenter_cooldown=0, replacement_cooldown=0, order_pacing_seconds=0,
        capital_per_grid_usdt=settings.capital_per_grid_usdt,
        leverage=settings.leverage,
        trailing_sl_trigger_pct=settings.trailing_sl_trigger_pct,
        max_exposure_pct=settings.max_exposure_pct,
        # Overridable so a sweep can ask what the MARKET pays at a spacing, separately
        # from whether the configured fee floor currently allows quoting there.
        min_profit_multiplier=(settings.min_profit_multiplier
                               if min_profit_multiplier is None
                               else min_profit_multiplier),
    )
    clock = VirtualClock(step=max(settings.poll_interval, 1))
    real_time = _grid.time
    _grid.time = clock
    try:
        grid.initialize(start, paper.free)
        grid.activate(paper.free)

        lines_at_start = len({l.price for l in grid.levels})
        worst_lines, worst_at = lines_at_start, None

        for i, p in enumerate(prices):
            clock.advance()
            paper.tick(p)
            grid.check_fills(paper.free)
            n = len({l.price for l in grid.levels})
            if n < worst_lines:
                worst_lines, worst_at = n, i
    finally:
        _grid.time = real_time

    return {
        "ticks": len(prices),
        "fills": grid.total_fills,
        "cycles": grid.total_completed_cycles,
        "grid_net": grid.total_pnl - grid.total_fees,
        "paper_net": paper.net,
        "paper_fees": paper.fees,
        "open_qty": paper.qty,
        "levels": len(grid.levels),
        "lines_start": lines_at_start,
        "lines_worst": worst_lines,
        "lines_worst_tick": worst_at,
        "lines_end": len({l.price for l in grid.levels}),
        "crossing": paper.crossing_orders,
        "spacing": grid.grid_spacing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", nargs="?", help="log file (default: newest in logs/)")
    ap.add_argument("--run", type=int, default=0, help="0 = newest run in the file")
    ap.add_argument("--sweep", action="store_true",
                    help="replay the same path at a range of spacings")
    ap.add_argument("--klines", action="store_true",
                    help="drive from exchange candles instead of poll snapshots; "
                         "snapshots are ~15s apart and hide most fills (#86)")
    ap.add_argument("--timeframe", default="1m",
                    help="candle size for --klines, or 'trades' for the raw tape (default 1m)")
    a = ap.parse_args()

    path = Path(a.log) if a.log else max(Path("logs").glob("grid_*.log"),
                                         key=lambda p: p.stat().st_mtime)
    snapshots = load_prices(path, a.run)
    prices, source = snapshots, "poll snapshots"

    if a.klines:
        # Snapshots are ~15s apart and everything between them is invisible: measured,
        # that hid ~60% of real fills. Candle high/low are the extremes that reach
        # resting orders, so they are what a grid replay actually needs (AUDIT #86).
        from config import settings as _s
        from exchange import Exchange
        from klines import (fetch_klines, fetch_trade_path, path_from_klines,
                            session_window, verify_against_log)
        start, end = session_window(path, a.run)
        ex = Exchange(_s.exchange_config, demo=_s.demo_mode)
        if a.timeframe == "trades":
            prices = fetch_trade_path(ex, _s.symbol, start, end)
            if not prices:
                raise SystemExit("no trades returned for that window")
            source_n = f"{len(prices):,} trades"
        else:
            candles = fetch_klines(ex, _s.symbol, start, end, timeframe=a.timeframe)
            if not candles:
                raise SystemExit("no candles returned for that window")
            prices = path_from_klines(candles)
            source_n = f"{len(candles)} {a.timeframe} candles"
        problem = verify_against_log(prices, snapshots)
        if problem:
            # A wrong window gives a plausible path for the wrong hours, and every
            # conclusion off it is quietly false. Refuse rather than mislead.
            raise SystemExit(f"REFUSING TO REPLAY: {problem}")
        source = source_n

    print(f"replaying {path.name} run -{a.run}: {len(prices)} ticks from {source}, "
          f"{min(prices)}-{max(prices)} ({(max(prices)-min(prices))/min(prices):.3%})\n")

    r = replay(prices)
    print(f"  fills                {r['fills']}")
    print(f"  completed cycles     {r['cycles']}")
    print(f"  grid claims          {r['grid_net']:+.4f} USDT")
    print(f"  actually made        {r['paper_net']:+.4f} USDT   "
          f"(fees {r['paper_fees']:.4f}, open {r['open_qty']:+.0f})")
    gap = r["grid_net"] - r["paper_net"]
    if abs(gap) > 0.005:
        print(f"  MISMATCH             {gap:+.4f} USDT — the grid's cycle arithmetic "
              f"does not describe the fills that happened (#80)")

    print(f"\n  ladder lines         {r['lines_start']} at start, "
          f"{r['lines_worst']} worst, {r['lines_end']} at end")
    if r["lines_worst"] < r["lines_start"]:
        print(f"  LADDER LOST A LINE   first at tick {r['lines_worst_tick']} (#79)")
    if r["crossing"]:
        print(f"  CROSSING ORDERS      {len(r['crossing'])} post-only orders placed on "
              f"the wrong side of the market (#77)")
        for c in r["crossing"][:3]:
            print(f"      {c['side']} {c['price']} with market at {c['market']}")

    if a.sweep:
        print(f"\n  {'spacing':>9} {'fills':>6} {'cycles':>7} {'actually made':>14}")
        for pct in (0.0008, 0.0010, 0.0012, 0.0015, 0.0020, 0.0025, 0.0030):
            s = replay(prices, spacing_pct=pct, quiet=True)
            tag = "  <- config" if abs(pct - settings.range_min_spacing_pct) < 1e-9 else ""
            print(f"  {pct:>8.2%} {s['fills']:>6} {s['cycles']:>7} "
                  f"{s['paper_net']:>+14.4f}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
