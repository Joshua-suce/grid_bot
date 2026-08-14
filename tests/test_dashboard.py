"""The dashboard reads the bot's files and nothing else. AUDIT #73.

Two things are being pinned here.

The first is the architectural guarantee: no exchange calls, ever. The bot shares a rate
limit with anything using the same key and its order placement is the latency-sensitive
part, so a dashboard polling Binance next to a live trading process buys nothing and adds
a failure mode. Everything it shows is rewritten to disk every iteration anyway.

The second is that what it displays is TRUE OF NOW. The first version counted health
signals across the whole day's log, so a card headed "Health" showed three recenters and
four rejected stops from a session that had ended ten hours earlier. Stale problems
presented as current are worse than no health panel.
"""

import json
import re
from types import SimpleNamespace

import pytest

import dashboard

# Verbatim from logs/grid_2026-08-14.log.
STATUS_LINE = (
    "2026-08-14 22:34:11 | INFO    | __main__:run_bot:1720 | PRICE=0.06971 | fills=0 | "
    "gross=0.00 fees=0.00 net=0.00 | session=+0.00 account(since 2026-08-14)=+1.98 "
    "today=+1.98 | balance_free=4911.10 total_equity=4931.09 | grid=ON | "
    "regime=uncertain(adx=14.8) | spread=0.0143%"
)
START_LINE = (
    "2026-08-14 22:25:21 | INFO    | __main__:run_bot:592 | "
    "GRID BOT STARTING | mode=DEMO (testnet) | symbol=DOGEUSDT"
)
RECENTER_LINE = (
    "2026-08-14 06:43:15 | INFO    | grid:recenter:1 | "
    "RECENTERING GRID | price 0.06972 outside [0.06936-0.07116] (margin 2.0%)"
)


@pytest.fixture
def env(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "state").mkdir()
    return SimpleNamespace(
        symbol="DOGEUSDT", demo_mode=True,
        state_dir=str(tmp_path / "state"), log_dir=str(tmp_path / "logs"),
        max_exposure_pct=0.50, _root=tmp_path,
    )


def write_log(env, *lines):
    (env._root / "logs" / "grid_2026-08-14.log").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_state(env, state):
    (env._root / "state" / "grid_dogeusdt_demo.json").write_text(json.dumps(state))


# --- the architectural guarantee ----------------------------------------------------

def test_the_dashboard_cannot_call_the_exchange():
    """Not a style rule. A dashboard that polls Binance competes with order placement
    for the same rate limit, next to real money, for data already on disk."""
    assert not hasattr(dashboard, "Exchange")
    assert not hasattr(dashboard, "ccxt")

    src = open(dashboard.__file__, encoding="utf-8").read()
    body = src.split('"""', 2)[-1]  # skip the module docstring, which discusses them
    for forbidden in ("import ccxt", "from exchange import", "Exchange(",
                      "fetch_balance", "fetch_positions", "fapiPrivate"):
        assert forbidden not in body, f"dashboard reaches the exchange via {forbidden}"


def test_the_dashboard_never_writes():
    src = open(dashboard.__file__, encoding="utf-8").read()
    body = src.split('"""', 2)[-1]
    for forbidden in ("write_text(", "os.replace", "\"w\"", "'w'", "unlink("):
        assert forbidden not in body, f"dashboard mutates state via {forbidden}"


# --- status line --------------------------------------------------------------------

def test_the_status_line_parses(env):
    write_log(env, STATUS_LINE)
    s = dashboard.load_status(env)

    assert s["price"] == "0.06971"
    assert s["session"] == "+0.00"
    assert s["account"] == "+1.98"
    assert s["window"] == "since 2026-08-14"
    assert s["free"] == "4911.10"
    assert s["equity"] == "4931.09"
    assert s["grid"] == "ON"
    assert s["regime"] == "uncertain"
    assert s["adx"] == "14.8"
    assert s["spread"] == "0.0143"


def test_the_latest_status_line_wins(env):
    older = STATUS_LINE.replace("PRICE=0.06971", "PRICE=0.06900")
    write_log(env, older, STATUS_LINE)

    assert dashboard.load_status(env)["price"] == "0.06971"


def test_a_log_with_no_status_line_is_not_an_error(env):
    write_log(env, START_LINE)
    assert dashboard.load_status(env) is None


# --- health scoping (the bug) -------------------------------------------------------

def test_health_is_scoped_to_the_current_run(env):
    """The measured case: recenters at 06:43 from a session that ended at 12:05, under
    a card headed 'Health', while the run that started at 22:25 was clean."""
    write_log(env, RECENTER_LINE, RECENTER_LINE, START_LINE, STATUS_LINE)

    signals, scoped = dashboard.load_signals(env)

    assert scoped is True
    assert not any(label == "recenter" for label, _, _ in signals), (
        "signals from a previous run were reported as current"
    )


def test_problems_in_the_current_run_are_still_reported(env):
    write_log(env, START_LINE, RECENTER_LINE, STATUS_LINE)

    signals, scoped = dashboard.load_signals(env)

    assert scoped is True
    assert any(label == "recenter" and n == 1 for label, n, _ in signals)


def test_only_the_most_recent_start_scopes_it(env):
    write_log(env, START_LINE, RECENTER_LINE, START_LINE, STATUS_LINE)

    signals, _ = dashboard.load_signals(env)

    assert not any(label == "recenter" for label, _, _ in signals)


def test_a_run_that_began_before_todays_log_says_so(env):
    """Falling back to the whole file is fine; implying this run produced all of it
    is not."""
    write_log(env, RECENTER_LINE, STATUS_LINE)

    signals, scoped = dashboard.load_signals(env)

    assert scoped is False
    assert any(label == "recenter" for label, _, _ in signals)


def test_the_page_says_which_window_health_covers(env):
    write_log(env, START_LINE, STATUS_LINE)
    write_state(env, {"grid": {"levels": []}})

    page = dashboard.render(env)

    assert "since this run started" in page


# --- ladder -------------------------------------------------------------------------

LEVELS = [
    {"price": 0.06982, "side": "sell", "status": "pending", "quantity": 1790,
     "fill_count": 0, "total_pnl": 0.0, "order_id": "1"},
    {"price": 0.06958, "side": "buy", "status": "pending", "quantity": 1796,
     "fill_count": 0, "total_pnl": 0.0, "order_id": "2"},
]


def test_the_ladder_marks_where_price_sits():
    out = dashboard.render_ladder({"grid": {"levels": LEVELS}}, price=0.06970)

    rows = re.findall(r"<tr[^>]*>", out)
    marker = [i for i, r in enumerate(rows) if "now" in r]
    assert len(marker) == 1, "price marker missing or duplicated"
    # header, sell, MARKER, buy
    assert marker[0] == 2


def test_price_above_the_whole_ladder_lands_at_the_top():
    out = dashboard.render_ladder({"grid": {"levels": LEVELS}}, price=0.07500)
    rows = re.findall(r"<tr[^>]*>", out)
    assert "now" in rows[1], "price above every rung was not shown at the top"


def test_price_below_the_whole_ladder_lands_at_the_bottom():
    out = dashboard.render_ladder({"grid": {"levels": LEVELS}}, price=0.05000)
    rows = re.findall(r"<tr[^>]*>", out)
    assert "now" in rows[-1]
    assert "below the ladder" in out


def test_a_rung_with_no_live_order_is_marked():
    levels = [dict(LEVELS[0], order_id="")]
    out = dashboard.render_ladder({"grid": {"levels": levels}}, price=0.0697)

    assert "unarmed" in out and "no order" in out


def test_an_awaiting_counter_rung_is_marked():
    levels = [dict(LEVELS[0], status="awaiting_counter")]
    out = dashboard.render_ladder({"grid": {"levels": levels}}, price=0.0697)

    assert "awaiting" in out


def test_an_empty_ladder_does_not_crash():
    assert "No ladder" in dashboard.render_ladder({"grid": {"levels": []}}, 0.0697)
    assert "No ladder" in dashboard.render_ladder(None, None)


# --- fills --------------------------------------------------------------------------

def test_fill_stats_count_only_completed_cycles_as_pnl():
    """An entry fill has no cycle_pnl yet. Counting it as a zero-PnL cycle would halve
    the win rate."""
    rows = [
        {"cycle_pnl": "0", "fee": "0.02", "quantity": "1790", "price": "0.0697",
         "completed_cycle": "False"},
        {"cycle_pnl": "0.15", "fee": "0.02", "quantity": "1790", "price": "0.0697",
         "completed_cycle": "True"},
        {"cycle_pnl": "-0.05", "fee": "0.02", "quantity": "1790", "price": "0.0697",
         "completed_cycle": "True"},
    ]
    s = dashboard.fill_stats(rows)

    assert s["n"] == 3
    assert s["cycles"] == 2
    assert s["wins"] == 1 and s["losses"] == 1
    assert s["gross"] == pytest.approx(0.10)
    assert s["fees"] == pytest.approx(0.06)


def test_fill_stats_survive_a_malformed_row():
    rows = [{"cycle_pnl": "oops", "fee": "x", "quantity": "", "price": "",
             "completed_cycle": "True"}]
    assert dashboard.fill_stats(rows)["n"] == 0


# --- attribution --------------------------------------------------------------------

def test_missing_attribution_says_how_to_get_it():
    out = dashboard.render_attribution(None)
    assert "attribute_pnl.py" in out


def test_attribution_shows_its_age():
    out = dashboard.render_attribution({
        "generated_at": "2026-08-14T12:00:00+00:00", "days": 7, "_age_hours": 30.0,
        "buckets": {"gx": {"n": 4, "notional": 500.0, "pnl": -1.2, "comm": -0.2,
                           "net": -1.4}},
    })
    assert "30.0h ago" in out
    assert "stale" in out, "a day-old snapshot was not flagged as stale"


# --- safety -------------------------------------------------------------------------

def test_log_content_cannot_inject_html(env):
    """Log lines are data. An exchange error message containing markup must render as
    text, not as part of the page."""
    evil = START_LINE.replace("DOGEUSDT", "<script>alert(1)</script>")
    write_log(env, evil, STATUS_LINE)
    write_state(env, {"grid": {"levels": [
        dict(LEVELS[0], side="<img src=x onerror=alert(1)>")]}})

    page = dashboard.render(env)

    assert "<script>alert(1)</script>" not in page
    assert "<img src=x onerror" not in page


def test_render_survives_every_file_being_absent(env):
    """A dashboard must never be the thing that breaks."""
    page = dashboard.render(env)
    assert "DOGEUSDT" in page


def test_render_survives_corrupt_state(env):
    (env._root / "state" / "grid_dogeusdt_demo.json").write_text("{not json")
    write_log(env, START_LINE, STATUS_LINE)

    assert "DOGEUSDT" in dashboard.render(env)


def test_the_live_badge_is_distinguishable(env):
    """Mistaking a live dashboard for a demo one is the expensive direction."""
    env.demo_mode = False
    write_log(env, START_LINE, STATUS_LINE)

    assert "badge live" in dashboard.render(env)
    env.demo_mode = True
    assert "badge demo" in dashboard.render(env)


@pytest.mark.parametrize("age,expected", [(5, "dot ok"), (300, "dot warn"), (5000, "dot bad")])
def test_a_silent_bot_is_visible(env, age, expected, monkeypatch):
    write_log(env, START_LINE, STATUS_LINE)
    monkeypatch.setattr(dashboard, "load_status",
                        lambda s: {**dashboard.STATUS.search(STATUS_LINE).groupdict(),
                                   "age_seconds": age, "at": "x"})

    assert expected in dashboard.render(env)
