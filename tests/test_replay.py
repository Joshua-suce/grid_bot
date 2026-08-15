"""The offline replay harness. AUDIT #81.

Every behavioural defect found live -- #75, #77, #79, #80 -- was already sitting in a
recorded log. Each one still cost an 8-hour live run to find, because nothing could
re-run a recorded session. These tests cover the harness that ends that, and the last
one is a genuine regression guard: it replays the exact price path that broke the
ladder on 2026-08-15 and asserts the ladder survives it.
"""

from pathlib import Path

import pytest

from replay import PaperExchange, load_prices, replay

SYM = "DOGEUSDT"


# --- the paper book has to net like Binance, or it proves nothing ---------------

def book(price=0.07):
    p = PaperExchange(SYM, maker_fee=0.0)
    p.price = price
    return p


def rest(p, side, price, amount):
    """Place an order away from the market, then move price onto it."""
    o = p.place_limit_order(SYM, side, price, amount)
    assert o is not None, "the paper book rejected an order that does not cross"
    return o


def test_a_buy_then_a_sell_realises_the_spread():
    p = book(0.0700)
    rest(p, "buy", 0.0699, 1000)
    p.tick(0.0699)
    rest(p, "sell", 0.0701, 1000)
    p.tick(0.0701)

    assert p.qty == 0
    assert round(p.realized, 6) == round((0.0701 - 0.0699) * 1000, 6)


def test_a_buy_against_a_short_reduces_it_rather_than_opening_a_second_leg():
    """One-way netting. Getting this wrong is precisely why the grid's own
    (exit-entry)*qty arithmetic disagrees with the account (#80)."""
    p = book(0.0700)
    rest(p, "sell", 0.0701, 1000)
    p.tick(0.0701)
    assert p.qty == -1000

    rest(p, "buy", 0.0699, 600)
    p.tick(0.0699)

    assert p.qty == -400, "the buy opened a long instead of covering the short"
    assert round(p.realized, 6) == round((0.0701 - 0.0699) * 600, 6)


def test_closing_more_than_the_position_flips_it():
    p = book(0.0700)
    rest(p, "sell", 0.0701, 1000)
    p.tick(0.0701)
    rest(p, "buy", 0.0699, 1500)
    p.tick(0.0699)

    assert p.qty == 500
    assert p.entry == 0.0699, "the flipped leg kept the old short's entry"


def test_fees_are_charged_on_every_fill_and_subtracted_from_net():
    p = PaperExchange(SYM, maker_fee=0.001)
    p.price = 0.0700
    rest(p, "buy", 0.0699, 1000)
    p.tick(0.0699)

    assert round(p.fees, 8) == round(0.0699 * 1000 * 0.001, 8)
    assert round(p.net, 8) == round(p.realized - p.fees, 8)


def test_a_post_only_order_through_the_market_is_rejected():
    p = book(0.0700)
    assert p.place_limit_order(SYM, "buy", 0.0705, 100) is None
    assert p.place_limit_order(SYM, "sell", 0.0695, 100) is None
    assert p.crossing_orders, "a crossing order was accepted silently"


def test_resting_at_the_touch_is_not_crossing():
    p = book(0.0700)
    assert p.place_limit_order(SYM, "buy", 0.0700, 100) is not None
    assert p.place_limit_order(SYM, "sell", 0.0700, 100) is not None
    assert not p.crossing_orders


def test_an_unfilled_order_still_reads_open():
    p = book(0.0700)
    o = rest(p, "buy", 0.0690, 100)
    p.tick(0.0699)
    assert p.fetch_order(o["id"], SYM)["status"] == "open"


def test_a_filled_order_reports_its_quantity_both_ways():
    """check_fills reads status, and order_was_filled falls back to executedQty --
    a paper fill has to satisfy both or #75's guard never gets exercised."""
    p = book(0.0700)
    o = rest(p, "buy", 0.0690, 100)
    p.tick(0.0689)

    got = p.fetch_order(o["id"], SYM)
    assert got["status"] == "closed"
    assert got["filled"] == 100
    assert got["info"]["executedQty"] == "100.0"


def test_positions_report_the_shape_the_engine_reads():
    p = book(0.0700)
    rest(p, "sell", 0.0701, 1000)
    p.tick(0.0701)

    pos = p.get_positions(SYM)[0]
    assert pos["contracts"] == 1000
    assert pos["side"] == "short"
    assert pos["entryPrice"] == 0.0701


def test_a_flat_book_reports_no_position():
    assert book().get_positions(SYM) == []


# --- reading runs out of a log ---------------------------------------------------

def test_the_newest_run_is_the_default(tmp_path):
    log = tmp_path / "grid.log"
    log.write_text(
        "GRID BOT STARTING\nPRICE=0.01 |\nPRICE=0.02 |\n"
        "GRID BOT STARTING\nPRICE=0.03 |\nPRICE=0.04 |\n", encoding="utf-8")

    assert load_prices(log, 0) == [0.03, 0.04]
    assert load_prices(log, 1) == [0.01, 0.02]


def test_asking_for_a_run_that_is_not_there_is_an_error(tmp_path):
    log = tmp_path / "grid.log"
    log.write_text("GRID BOT STARTING\nPRICE=0.01 |\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_prices(log, 5)


# --- end to end ------------------------------------------------------------------

def sawtooth(low, high, rungs):
    out = []
    for _ in range(rungs):
        out += [low + (high - low) * i / 20 for i in range(21)]
        out += [high - (high - low) * i / 20 for i in range(21)]
    return out


def test_a_market_that_actually_swings_produces_cycles():
    r = replay(sawtooth(0.0690, 0.0710, 6))
    assert r["fills"] > 0, "a 2.9% saw-tooth produced no fills at all"
    assert r["cycles"] > 0


@pytest.mark.xfail(strict=True, reason=(
    "OPEN, AUDIT #82. #79 keeps the ladder whole on the recorded 2026-08-15 path, "
    "but a market that genuinely swings still drives it 14 -> 13. The recorded run "
    "was nearly flat (0.444%), so it never exercised this. Deterministic replay of a "
    "2.9% saw-tooth does, and the line is lost for the rest of the run."))
def test_the_ladder_keeps_every_line_through_a_swinging_market():
    r = replay(sawtooth(0.0690, 0.0710, 6))
    assert r["lines_worst"] == r["lines_start"], (
        f"ladder fell from {r['lines_start']} lines to {r['lines_worst']}")


@pytest.mark.xfail(strict=True, reason=(
    "OPEN, AUDIT #83. In a rising market the engine retries one rung as a SELL below "
    "the market over and over -- 136 attempts at 0.06931 while price ran 0.0694-0.0699. "
    "Binance rejects every one of those post-only (-2019), so live this is a rung that "
    "silently never re-arms. #77 re-sides a rung on RELEASE; this path does not."))
def test_no_crossing_orders_are_ever_placed():
    r = replay(sawtooth(0.0690, 0.0710, 6))
    assert not r["crossing"], f"engine placed {len(r['crossing'])} crossing orders"


def test_the_grids_own_pnl_matches_what_the_fills_actually_made():
    """#80 end to end. The engine's total_pnl/total_fees and the paper book's netted
    accounting are independent implementations; if the engine still priced level
    round trips instead of position reductions they would diverge."""
    r = replay(sawtooth(0.0690, 0.0710, 6))

    assert abs(r["grid_net"] - r["paper_net"]) < 1e-6, (
        f"grid claims {r['grid_net']:+.4f}, fills actually made {r['paper_net']:+.4f}")


def test_a_one_way_market_does_not_manufacture_profit():
    """A ladder eaten by a trend completes 'cycles' while the account is just
    accumulating inventory. The money reported must not follow the cycle count."""
    climb = [0.0690 + 0.0020 * i / 400 for i in range(401)]
    r = replay(climb)

    assert abs(r["grid_net"] - r["paper_net"]) < 1e-6
    if r["cycles"] > 0:
        assert r["grid_net"] <= r["paper_net"] + 1e-6


@pytest.mark.skipif(not Path("logs/grid_2026-08-15.log").exists(),
                    reason="the recorded run is not in this checkout")
def test_the_recorded_run_that_broke_the_ladder_no_longer_breaks_it():
    """2026-08-15 06:33-14:53. Live, this exact path drove the book from 14 grid
    lines to 13 and left 0.07001 empty for 5h13m while price crossed it 62 times."""
    r = replay(load_prices(Path("logs/grid_2026-08-15.log"), 0))

    assert r["lines_worst"] == 14, (
        f"ladder dropped to {r['lines_worst']} lines at tick {r['lines_worst_tick']}")
    assert r["lines_end"] == 14
