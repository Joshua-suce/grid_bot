"""Read-only web dashboard for the grid bot.

    py dashboard.py            # then open http://localhost:8787
    py dashboard.py --port 9000 --host 0.0.0.0

It reads the bot's OWN FILES and never calls the exchange:

    state/grid_{symbol}_{mode}.json   the ladder, stops, risk state, verified PnL
    logs/events_{mode}.jsonl          balance/exposure/position snapshots, orders, fills
    logs/trades_{mode}.csv            per-fill history
    logs/grid_YYYY-MM-DD.log          the PRICE= status line, and health signals
    logs/attribution_{mode}.json      optional: maker/taker split, written by attribute_pnl

That constraint is the design. The bot shares a rate limit with anything else using the
same key, and its order placement is the latency-sensitive part; a dashboard polling
Binance next to a live trading process buys nothing and adds a failure mode. Everything
above is rewritten every iteration anyway. Nothing here writes, so it cannot corrupt
state, and stopping it can never affect a running bot.

Stdlib only -- no new dependencies.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REFRESH_SECONDS = 5

# The bot's per-iteration status line. Parsed rather than recomputed so the dashboard
# reports the same numbers the bot does, from the same instant.
#
# f4db690 relabelled the line ("account(...)=-71.00" read as a negative BALANCE --
# AUDIT #144) but never touched this regex or its own fixture in
# tests/test_dashboard.py, which hand-copies a real log line rather than deriving it
# from main.py's format string. Both silently went stale together: every log line
# main.py wrote from that commit forward stopped matching, load_status() returned
# None for the whole file, and the dashboard's status card went blank with no error
# anywhere -- the tests still passed because they were testing the regex against
# themselves, not against what main.py emits (AUDIT #152). Group names are
# unchanged -- render() below reads them by name -- only the literal labels moved.
STATUS = re.compile(
    r"PRICE=(?P<price>[\d.]+) \| fills=(?P<fills>\d+) \| "
    r"gross=(?P<gross>-?[\d.]+) fees=(?P<fees>-?[\d.]+) net=(?P<net>-?[\d.]+) \| "
    r"session_pnl=(?P<session>[+-][\d.]+) pnl_since\((?P<window>[^)]*)\)=(?P<account>[+-][\d.]+) "
    r"today=(?P<today>[+-][\d.]+) \| balance_free=(?P<free>[\d.]+) "
    r"total_equity=(?P<equity>[\d.]+) \| grid=(?P<grid>\w+) \| "
    r"regime=(?P<regime>[\w.]+)\(adx=(?P<adx>[\d.]+)\) \| spread=(?P<spread>[\d.]+)%"
)


# --------------------------------------------------------------------------- loading


def _mode(settings) -> str:
    return "demo" if settings.demo_mode else "live"


def load_state(settings) -> dict | None:
    path = Path(settings.state_dir) / f"grid_{settings.symbol.lower()}_{_mode(settings)}.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_status(settings) -> dict | None:
    """The most recent PRICE= line, plus how long ago it was written."""
    logs = sorted(Path(settings.log_dir).glob("grid_*.log"))
    if not logs:
        return None
    try:
        lines = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        m = STATUS.search(line)
        if m:
            out = {k: v for k, v in m.groupdict().items()}
            try:
                stamp = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                out["age_seconds"] = (datetime.now() - stamp).total_seconds()
                out["at"] = line[:19]
            except ValueError:
                out["age_seconds"] = None
            return out
    return None


def load_events(settings, limit: int = 4000) -> list[dict]:
    path = Path(settings.log_dir) / f"events_{_mode(settings)}.jsonl"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_trades(settings) -> list[dict]:
    path = Path(settings.log_dir) / f"trades_{_mode(settings)}.csv"
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def load_attribution(settings) -> dict | None:
    """Maker/taker and per-purpose split. The one thing that needs the exchange, so it
    is a cache written by `py attribute_pnl.py --json`, shown with its age."""
    path = Path(settings.log_dir) / f"attribution_{_mode(settings)}.json"
    try:
        data = json.loads(path.read_text())
        data["_age_hours"] = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(data["generated_at"])
        ).total_seconds() / 3600
        return data
    except Exception:
        return None


def current_run_log(settings) -> tuple[str, bool]:
    """Today's log from the LAST startup banner onward, and whether it was found.

    Scoping matters more than it looks. A day's log holds every run that day, so the
    unscoped version showed three recenters and four rejected stops from a session that
    ended ten hours earlier, under a heading that reads as "right now". Stale problems
    presented as current are worse than no health panel.
    """
    logs = sorted(Path(settings.log_dir).glob("grid_*.log"))
    if not logs:
        return "", False
    try:
        text = logs[-1].read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "", False
    marker = "GRID BOT STARTING"
    idx = text.rfind(marker)
    if idx == -1:
        # The run began before today's file. Whole-file is the honest fallback, and the
        # caller says so rather than implying this run produced all of it.
        return text, False
    return text[idx:], True


def load_signals(settings) -> tuple[list[tuple[str, int, str]], bool]:
    """Health signals, using check_run's patterns so there is one definition of them."""
    try:
        from check_run import SIGNALS
    except Exception:
        return [], False
    text, scoped = current_run_log(settings)
    if not text:
        return [], scoped
    found = []
    for label, pattern, meaning in SIGNALS:
        n = len(re.findall(pattern, text))
        if n:
            found.append((label, n, meaning))
    return found, scoped


# ----------------------------------------------------------------------- derivations


def latest(events: list[dict], name: str) -> dict | None:
    for e in reversed(events):
        if e.get("event") == name:
            return e
    return None


def equity_series(events: list[dict], cap: int = 240) -> list[tuple[str, float]]:
    points = [
        (e["ts"], float(e["total_equity"]))
        for e in events
        if e.get("event") == "balance_snapshot" and e.get("total_equity") is not None
    ]
    if len(points) <= cap:
        return points
    step = len(points) // cap
    return points[::step][-cap:]


def fill_stats(trades: list[dict]) -> dict:
    """What the bot's own journal says. Exchange-verified totals come from the state
    file's reconciler; this is the per-fill texture -- how many, how big, how they land."""
    out = {"n": 0, "cycles": 0, "gross": 0.0, "fees": 0.0, "wins": 0, "losses": 0,
           "worst": 0.0, "best": 0.0, "notional": 0.0}
    for row in trades:
        try:
            pnl = float(row.get("cycle_pnl") or 0)
            fee = float(row.get("fee") or 0)
            qty = float(row.get("quantity") or 0)
            price = float(row.get("price") or 0)
        except (TypeError, ValueError):
            continue
        out["n"] += 1
        out["fees"] += fee
        out["notional"] += qty * price
        if str(row.get("completed_cycle")).lower() in ("true", "1"):
            out["cycles"] += 1
            out["gross"] += pnl
            if pnl > 0:
                out["wins"] += 1
                out["best"] = max(out["best"], pnl)
            elif pnl < 0:
                out["losses"] += 1
                out["worst"] = min(out["worst"], pnl)
    return out


# --------------------------------------------------------------------------- render


def esc(v) -> str:
    return html.escape(str(v))


def fmt(v, places=2, sign=False) -> str:
    try:
        s = f"{float(v):+.{places}f}" if sign else f"{float(v):.{places}f}"
        return s
    except (TypeError, ValueError):
        return "—"


def render_ladder(state: dict, price: float | None) -> str:
    grid = (state or {}).get("grid") or {}
    levels = grid.get("levels") or []
    if not levels:
        return "<p class='muted'>No ladder in state yet.</p>"

    rows = sorted(levels, key=lambda l: float(l.get("price") or 0), reverse=True)
    price_shown = price is None
    out = ["<table class='ladder'>"]
    out.append("<tr><th>price</th><th>side</th><th>status</th><th>qty</th>"
               "<th>fills</th><th>pnl</th></tr>")
    for lvl in rows:
        p = float(lvl.get("price") or 0)
        if not price_shown and price is not None and p < price:
            gap_up = None
            out.append(
                f"<tr class='now'><td>{fmt(price, 5)}</td><td colspan='5'>"
                f"&#9664; price is here</td></tr>"
            )
            price_shown = True
        side = str(lvl.get("side") or "")
        status = str(lvl.get("status") or "")
        # Whitelist, never interpolate. Escaping the CELL is not enough when the same
        # value also lands in a class attribute: a '>' inside class='...' closes the tag
        # and everything after it becomes markup. Values here come from the bot's own
        # state file, which is exactly the sort of "trusted" input that stops being so.
        klass = {"buy": "buy", "sell": "sell"}.get(side, "other")
        if status == "awaiting_counter":
            klass += " awaiting"
        elif not lvl.get("order_id"):
            klass += " unarmed"
        pnl = float(lvl.get("total_pnl") or 0)
        out.append(
            f"<tr class='{klass}'>"
            f"<td>{fmt(p, 5)}</td>"
            f"<td class='side'>{esc(side)}</td>"
            f"<td>{esc(status)}{'' if lvl.get('order_id') else ' <span class=muted>(no order)</span>'}</td>"
            f"<td>{fmt(lvl.get('quantity'), 0)}</td>"
            f"<td>{esc(lvl.get('fill_count') or 0)}</td>"
            f"<td class='{'pos' if pnl > 0 else 'neg' if pnl < 0 else ''}'>"
            f"{fmt(pnl, 4, sign=True) if pnl else '—'}</td>"
            f"</tr>"
        )
    if not price_shown and price is not None:
        out.append(f"<tr class='now'><td>{fmt(price, 5)}</td>"
                   f"<td colspan='5'>&#9664; price is here (below the ladder)</td></tr>")
    out.append("</table>")
    return "".join(out)


def render_sparkline(points: list[tuple[str, float]]) -> str:
    if len(points) < 2:
        return "<p class='muted'>Not enough equity samples yet.</p>"
    values = [v for _, v in points]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    w, h = 720, 120
    step = w / (len(values) - 1)
    coords = " ".join(
        f"{i * step:.1f},{h - (v - lo) / span * (h - 8) - 4:.1f}"
        for i, v in enumerate(values)
    )
    change = values[-1] - values[0]
    klass = "pos" if change > 0 else "neg" if change < 0 else ""
    return (
        f"<svg viewBox='0 0 {w} {h}' preserveAspectRatio='none' class='spark {klass}'>"
        f"<polyline points='{coords}' fill='none' stroke='currentColor' stroke-width='2'/>"
        f"</svg>"
        f"<div class='sparkfoot'><span>{fmt(lo, 2)}</span>"
        f"<span class='{klass}'>{fmt(change, 2, sign=True)} over {len(values)} samples</span>"
        f"<span>{fmt(hi, 2)}</span></div>"
    )


def render_attribution(attr: dict | None) -> str:
    if not attr:
        return (
            "<p class='muted'>No attribution cache yet. The maker/taker split is the one "
            "figure that needs the exchange, so it is not fetched here — run "
            "<code>py attribute_pnl.py 7 --json</code> to refresh it.</p>"
        )
    age = attr.get("_age_hours", 0)
    rows = ["<table class='kv'><tr><th>bucket</th><th>execs</th><th>notional</th>"
            "<th>realized</th><th>fees</th><th>NET</th></tr>"]
    for name, b in sorted(attr.get("buckets", {}).items(),
                          key=lambda kv: kv[1].get("net", 0)):
        net = b.get("net", 0)
        rows.append(
            f"<tr><td>{esc(name)}</td><td>{esc(b.get('n', 0))}</td>"
            f"<td>{fmt(b.get('notional'), 0)}</td><td>{fmt(b.get('pnl'), 2, sign=True)}</td>"
            f"<td>{fmt(b.get('comm'), 2, sign=True)}</td>"
            f"<td class='{'pos' if net > 0 else 'neg'}'>{fmt(net, 2, sign=True)}</td></tr>"
        )
    rows.append("</table>")
    stale = " stale" if age > 24 else ""
    return (
        f"<p class='muted{stale}'>as of {esc(attr.get('generated_at', '')[:19])} "
        f"({age:.1f}h ago, {esc(attr.get('days', '?'))}d window)</p>" + "".join(rows)
    )


def render(settings) -> str:
    mode = _mode(settings)
    state = load_state(settings)
    status = load_status(settings)
    events = load_events(settings)
    trades = load_trades(settings)
    attr = load_attribution(settings)
    signals, scoped = load_signals(settings)

    grid = (state or {}).get("grid") or {}
    risk = (state or {}).get("risk") or {}
    rec = (state or {}).get("pnl_reconciler") or {}
    bal = latest(events, "balance_snapshot") or {}
    pos = latest(events, "position_snapshot")

    price = float(status["price"]) if status else None
    age = status.get("age_seconds") if status else None
    if age is None:
        alive, alive_text = "unknown", "no status line found"
    elif age < 60:
        alive, alive_text = "ok", f"last tick {age:.0f}s ago"
    elif age < 600:
        alive, alive_text = "warn", f"last tick {age / 60:.1f} min ago"
    else:
        alive, alive_text = "bad", f"SILENT for {age / 60:.0f} min"

    stats = fill_stats(trades)
    used = float(bal.get("used") or 0)
    equity = float(bal.get("total_equity") or 0)

    def card(title, body, cls=""):
        return f"<section class='card {cls}'><h2>{title}</h2>{body}</section>"

    # --- header ---------------------------------------------------------------
    head = (
        f"<div class='head'>"
        f"<span class='badge {mode}'>{mode.upper()}</span>"
        f"<span class='sym'>{esc(settings.symbol)}</span>"
        f"<span class='price'>{fmt(price, 5)}</span>"
        f"<span class='dot {alive}'></span><span class='muted'>{esc(alive_text)}</span>"
        f"<span class='grow'></span>"
        f"<span class='muted'>grid {esc(status['grid']) if status else '—'} · "
        f"{esc(status['regime']) if status else '—'} "
        f"adx {esc(status['adx']) if status else '—'} · "
        f"spread {esc(status['spread']) if status else '—'}%</span>"
        f"</div>"
    )

    # --- now ------------------------------------------------------------------
    money = (
        "<table class='kv'>"
        f"<tr><td>session PnL</td><td class='{_cls(status and status['session'])}'>"
        f"{esc(status['session']) if status else '—'}</td></tr>"
        f"<tr><td>account ({esc(status['window']) if status else '—'})</td>"
        f"<td class='{_cls(status and status['account'])}'>"
        f"{esc(status['account']) if status else '—'}</td></tr>"
        f"<tr><td>today</td><td class='{_cls(status and status['today'])}'>"
        f"{esc(status['today']) if status else '—'}</td></tr>"
        f"<tr><td>verified realized</td><td>{fmt(rec.get('realized_pnl'), 4, sign=True)}</td></tr>"
        f"<tr><td>commission</td><td>{fmt(rec.get('commission'), 4, sign=True)}</td></tr>"
        f"<tr><td>funding</td><td>{fmt(rec.get('funding_fee'), 4, sign=True)}</td></tr>"
        "</table>"
    )

    exposure = (
        "<table class='kv'>"
        f"<tr><td>free</td><td>{fmt(bal.get('free'))}</td></tr>"
        f"<tr><td>margin used</td><td>{fmt(used)}</td></tr>"
        f"<tr><td>equity</td><td>{fmt(equity)}</td></tr>"
        f"<tr><td>exposure</td><td>{fmt(float(bal.get('exposure_pct') or 0) * 100, 1)}% "
        f"<span class='muted'>of {fmt(settings.max_exposure_pct * 100, 0)}% cap</span></td></tr>"
        f"<tr><td>peak balance</td><td>{fmt(risk.get('peak_balance'))}</td></tr>"
        f"<tr><td>drawdown</td><td class='{_cls(-1 if _dd(risk, equity) else 0)}'>"
        f"{fmt(_dd(risk, equity) * 100, 2)}%</td></tr>"
        f"<tr><td>consecutive losses</td><td>{esc(risk.get('consecutive_losses', '—'))}</td></tr>"
        f"<tr><td>in recovery</td><td>{'YES' if risk.get('in_recovery') else 'no'}</td></tr>"
        "</table>"
    )

    if pos:
        position = (
            "<table class='kv'>"
            f"<tr><td>side</td><td class='side'>{esc(pos.get('side'))}</td></tr>"
            f"<tr><td>qty</td><td>{fmt(pos.get('qty'), 0)}</td></tr>"
            f"<tr><td>entry</td><td>{fmt(pos.get('entry_price'), 5)}</td></tr>"
            f"<tr><td>unrealized</td><td class='{_cls(pos.get('unrealized_pnl'))}'>"
            f"{fmt(pos.get('unrealized_pnl'), 4, sign=True)}</td></tr>"
            "</table>"
        )
    else:
        position = "<p class='muted'>Flat — no position snapshot recorded.</p>"

    stops = (
        "<table class='kv'>"
        f"<tr><td>hard (long)</td><td>{fmt(grid.get('_hard_sl_price'), 5)}</td></tr>"
        f"<tr><td>trail (long)</td><td>{fmt(grid.get('_trailing_sl_price'), 5)}</td></tr>"
        f"<tr><td>hard (short)</td><td>{fmt(grid.get('_hard_sl_price_short'), 5)}</td></tr>"
        f"<tr><td>trail (short)</td><td>{fmt(grid.get('_trailing_sl_price_short'), 5)}</td></tr>"
        f"<tr><td>peak / trough</td><td>{fmt(grid.get('_peak_price'), 5)} / "
        f"{fmt(grid.get('_trough_price'), 5)}</td></tr>"
        f"<tr><td>buys blocked</td><td>{'YES' if grid.get('_block_buys') else 'no'}</td></tr>"
        f"<tr><td>sells blocked</td><td>{'YES' if grid.get('_block_sells') else 'no'}</td></tr>"
        f"<tr><td>volatility mult</td><td>{fmt(grid.get('_volatility_mult'))}</td></tr>"
        "</table>"
    )

    # --- performance ----------------------------------------------------------
    perf = (
        "<table class='kv'>"
        f"<tr><td>fills</td><td>{stats['n']}</td></tr>"
        f"<tr><td>completed cycles</td><td>{stats['cycles']}</td></tr>"
        f"<tr><td>win / loss</td><td>{stats['wins']} / {stats['losses']}</td></tr>"
        f"<tr><td>gross (journal)</td><td class='{_cls(stats['gross'])}'>"
        f"{fmt(stats['gross'], 4, sign=True)}</td></tr>"
        f"<tr><td>fees (journal)</td><td class='neg'>{fmt(-stats['fees'], 4, sign=True)}</td></tr>"
        f"<tr><td>net (journal)</td><td class='{_cls(stats['gross'] - stats['fees'])}'>"
        f"{fmt(stats['gross'] - stats['fees'], 4, sign=True)}</td></tr>"
        f"<tr><td>best / worst cycle</td><td>{fmt(stats['best'], 4, sign=True)} / "
        f"<span class='neg'>{fmt(stats['worst'], 4, sign=True)}</span></td></tr>"
        f"<tr><td>traded notional</td><td>{fmt(stats['notional'], 0)}</td></tr>"
        "</table>"
    )

    where = "since this run started" if scoped else "in today's log (run start not found)"
    if signals:
        sig_rows = "".join(
            f"<tr><td class='n'>{n}</td><td>{esc(label)}</td>"
            f"<td class='muted'>{esc(meaning)}</td></tr>"
            for label, n, meaning in signals
        )
        health = (f"<p class='muted'>{esc(where)}</p>"
                  f"<table class='kv signals'>{sig_rows}</table>")
    else:
        health = f"<p class='muted ok'>Nothing flagged {esc(where)}.</p>"

    body = (
        head
        + "<div class='grid2'>"
        + card("Ladder", render_ladder(state, price), "tall")
        + "<div class='col'>"
        + card("PnL", money)
        + card("Position", position)
        + card("Stops &amp; gates", stops)
        + "</div>"
        + "</div>"
        + "<div class='grid2'>"
        + card("Equity", render_sparkline(equity_series(events)))
        + card("Balance &amp; risk", exposure)
        + "</div>"
        + "<div class='grid2'>"
        + card("Fills (bot journal)", perf)
        + card("Attribution (exchange-verified)", render_attribution(attr))
        + "</div>"
        + card("Health (this run)", health)
        + f"<p class='foot'>read-only · files only, no exchange calls · "
        f"refreshed {esc(datetime.now().strftime('%H:%M:%S'))}</p>"
    )
    return body


def _cls(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "pos" if f > 0 else "neg" if f < 0 else ""


def _dd(risk: dict, equity: float) -> float:
    try:
        peak = float(risk.get("peak_balance") or 0)
        return (peak - equity) / peak if peak > 0 and equity else 0.0
    except (TypeError, ValueError):
        return 0.0


CSS = """
*{box-sizing:border-box}
body{margin:0;padding:18px;background:#0e1116;color:#d7dde5;
     font:14px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
h2{margin:0 0 10px;font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:#7d8899}
.head{display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.badge{padding:3px 9px;border-radius:4px;font-weight:700;letter-spacing:.06em}
.badge.demo{background:#1d3a5c;color:#8fc4ff}
.badge.live{background:#5c1d1d;color:#ff9b9b}
.sym{font-weight:700}
.price{font-size:24px;font-weight:700;color:#fff}
.grow{flex:1}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.dot.ok{background:#3fb950}.dot.warn{background:#d29922}
.dot.bad{background:#f85149}.dot.unknown{background:#6e7681}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.col{display:flex;flex-direction:column;gap:14px}
.card{background:#161b22;border:1px solid #232a34;border-radius:8px;padding:14px;
      overflow-x:auto}
.card.tall{max-height:640px;overflow-y:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:4px 8px;text-align:left;white-space:nowrap}
th{color:#7d8899;font-weight:500;border-bottom:1px solid #232a34;font-size:11px;
   text-transform:uppercase;letter-spacing:.05em}
.kv td:first-child{color:#8b95a3}
.kv td:last-child{text-align:right}
.signals td:last-child{text-align:left;white-space:normal}
.signals .n{color:#d29922;font-weight:700;text-align:right;width:1%}
.ladder tr.buy .side{color:#3fb950}
.ladder tr.sell .side{color:#f85149}
.ladder tr.unarmed{opacity:.45}
.ladder tr.awaiting{background:#1f2410}
.ladder tr.now{background:#243044;color:#8fc4ff;font-weight:700}
.ladder tr.now td{border-top:1px solid #4a6a9a;border-bottom:1px solid #4a6a9a}
.pos{color:#3fb950}.neg{color:#f85149}
.muted{color:#6e7681}
.muted.ok{color:#3fb950}
.muted.stale{color:#d29922}
.spark{width:100%;height:120px;color:#58a6ff}
.spark.pos{color:#3fb950}.spark.neg{color:#f85149}
.sparkfoot{display:flex;justify-content:space-between;color:#6e7681;font-size:12px;
           margin-top:4px}
code{background:#0e1116;padding:1px 5px;border-radius:3px;color:#8fc4ff}
.foot{color:#4d545e;font-size:12px;text-align:center;margin:18px 0 0}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
"""

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>grid bot — {symbol} {mode}</title>
<style>{css}</style></head>
<body><div id="root">{body}</div>
<script>
// Poll the fragment rather than reloading: keeps scroll position in the ladder.
setInterval(async () => {{
  try {{
    const r = await fetch('/fragment', {{cache: 'no-store'}});
    if (r.ok) document.getElementById('root').innerHTML = await r.text();
  }} catch (e) {{ /* bot or dashboard restarting; next tick will catch up */ }}
}}, {refresh}000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    settings = None

    def _send(self, body: str, ctype: str = "text/html; charset=utf-8") -> None:
        raw = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        try:
            if self.path.startswith("/fragment"):
                self._send(render(self.settings))
            elif self.path in ("/", "/index.html"):
                self._send(PAGE.format(
                    css=CSS, body=render(self.settings), refresh=REFRESH_SECONDS,
                    symbol=esc(self.settings.symbol), mode=_mode(self.settings),
                ))
            else:
                self.send_error(404)
        except Exception as e:  # a dashboard must never be the thing that breaks
            self._send(f"<pre style='color:#f85149'>dashboard error: {esc(e)}</pre>")

    def log_message(self, *args) -> None:
        pass  # the bot owns the terminal; a request log per 5s would bury it


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to reach it from another device on your network")
    args = parser.parse_args()

    from config import settings

    Handler.settings = settings
    server = HTTPServer((args.host, args.port), Handler)
    shown = "localhost" if args.host == "127.0.0.1" else args.host
    print(f"grid bot dashboard  ->  http://{shown}:{args.port}")
    print(f"  mode    {_mode(settings).upper()}  {settings.symbol}")
    print("  reads files only — never calls the exchange, never writes")
    print("  Ctrl+C to stop (does not affect the bot)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
