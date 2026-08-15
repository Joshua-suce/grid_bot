"""P&L must follow the netted position, not the ladder. AUDIT #80.

On 2026-08-15 the bot reported +0.41 for a run in which Binance paid +0.24 -- 69%
high. Two faults compounded:

  * profit was booked when a LEVEL completed its own round trip. In one-way netting
    that is not when money moves. Fill #2 covered the short and realised +0.25 while
    the grid called it 'open' and booked nothing; fill #3 then ADDED to a long while
    the grid called it 'complete' and booked a +0.25 that never happened.

  * fees were charged only on 'completed' fills, so opening fills traded free in the
    bot's books -- 0.10 booked against 0.15 actually paid.

The level cannot be patched into agreement by choosing better prices or quantities:
in a netted account the level is not the thing that holds the position.
"""

import pytest

from grid import GridEngine


def ledger(qty=0.0, entry=0.0):
    g = GridEngine.__new__(GridEngine)
    g._pos_qty, g._pos_entry = qty, entry
    return g


# --- opening and adding realise nothing ------------------------------------------

def test_opening_a_long_realises_nothing():
    g = ledger()
    assert g._apply_to_position("buy", 1000, 0.07) == 0.0
    assert (g._pos_qty, g._pos_entry) == (1000, 0.07)


def test_opening_a_short_realises_nothing():
    g = ledger()
    assert g._apply_to_position("sell", 1000, 0.07) == 0.0
    assert g._pos_qty == -1000
    assert g._pos_entry == 0.07


def test_adding_to_a_long_reaverages_the_entry():
    g = ledger(1000, 0.0700)
    assert g._apply_to_position("buy", 1000, 0.0690) == 0.0
    assert g._pos_qty == 2000
    assert round(g._pos_entry, 8) == 0.0695


def test_adding_to_a_short_reaverages_the_entry():
    g = ledger(-1000, 0.0700)
    assert g._apply_to_position("sell", 3000, 0.0708) == 0.0
    assert g._pos_qty == -4000
    assert round(g._pos_entry, 8) == 0.0706


# --- reducing is where money happens ---------------------------------------------

def test_reducing_a_long_realises_the_gain():
    g = ledger(1000, 0.0690)
    got = g._apply_to_position("sell", 400, 0.0700)
    assert round(got, 8) == round((0.0700 - 0.0690) * 400, 8)
    assert g._pos_qty == 600
    assert g._pos_entry == 0.0690, "the remaining position lost its entry"


def test_reducing_a_short_realises_the_gain():
    g = ledger(-1000, 0.0700)
    got = g._apply_to_position("buy", 400, 0.0690)
    assert round(got, 8) == round((0.0700 - 0.0690) * 400, 8)
    assert g._pos_qty == -600


def test_a_losing_reduction_realises_a_loss():
    g = ledger(1000, 0.0700)
    assert g._apply_to_position("sell", 1000, 0.0690) < 0


def test_closing_flat_clears_the_entry():
    g = ledger(1000, 0.0690)
    g._apply_to_position("sell", 1000, 0.0700)
    assert g._pos_qty == 0
    assert g._pos_entry == 0.0


def test_flipping_through_zero_prices_only_what_closed():
    """The remainder is a NEW position opened at this fill, not a continuation."""
    g = ledger(-1000, 0.0700)
    got = g._apply_to_position("buy", 1500, 0.0690)

    assert round(got, 8) == round((0.0700 - 0.0690) * 1000, 8), "priced the wrong size"
    assert g._pos_qty == 500
    assert g._pos_entry == 0.0690


# --- guards ----------------------------------------------------------------------

@pytest.mark.parametrize("qty,price", [(0, 0.07), (-5, 0.07), (100, 0.0), (100, -1)])
def test_nonsense_fills_are_ignored(qty, price):
    g = ledger(1000, 0.07)
    assert g._apply_to_position("buy", qty, price) == 0.0
    assert g._pos_qty == 1000, "a rejected fill still moved the position"


def test_seeding_adopts_a_position_the_exchange_already_holds():
    """Without this a restart books the close of a pre-existing position as pure
    profit, because the ledger believes it opened flat."""
    g = ledger()
    g.seed_position(-5348, 0.07011)
    assert (g._pos_qty, g._pos_entry) == (-5348, 0.07011)

    got = g._apply_to_position("buy", 5348, 0.07011)
    assert abs(got) < 1e-9, "closing at the entry price should realise nothing"


def test_seeding_flat_clears_the_entry():
    g = ledger(1000, 0.07)
    g.seed_position(0, 0.07)
    assert (g._pos_qty, g._pos_entry) == (0.0, 0.0)


# --- the run that exposed it -----------------------------------------------------

MAKER = 0.0002

# side, qty, price -- exactly the five fills of 2026-08-15 06:33-14:53
RUN = [("sell", 1781, 0.07015), ("buy", 1785, 0.07001), ("buy", 1785, 0.07001),
       ("buy", 1789, 0.06987), ("sell", 1785, 0.07001)]

EXCHANGE_SAID = 0.2394      # reconciler: 2.1802 - 1.9408
OLD_CODE_SAID = 0.4053      # what the log printed: gross 0.5053 - fees 0.1000


def run_the_ledger():
    g = ledger()
    realized = fees = 0.0
    for side, qty, price in RUN:
        realized += g._apply_to_position(side, qty, price)
        fees += qty * price * MAKER          # every fill, not just 'completed' ones
    return realized - fees, g


def test_the_run_now_reports_close_to_what_binance_paid():
    net, _ = run_the_ledger()
    assert abs(net - EXCHANGE_SAID) < 0.02, (
        f"ledger says {net:+.4f}, exchange said {EXCHANGE_SAID:+.4f}")


def test_the_run_no_longer_reports_the_inflated_figure():
    net, _ = run_the_ledger()
    assert abs(net - OLD_CODE_SAID) > 0.10, (
        f"still reporting the old overstated {OLD_CODE_SAID:+.4f}")


def test_the_residual_long_is_carried_not_booked():
    """Fills bought 5359 and sold 3566: 1793 DOGE is still open and its gain is
    unrealised. Booking it would be the same class of error in reverse."""
    _, g = run_the_ledger()
    assert g._pos_qty == 1793
    assert round(g._pos_entry, 5) == 0.06994


def test_the_old_books_undercharged_this_run_by_half():
    """Fees were taken only on the two fills the ladder called 'complete'. The other
    three traded free. (The wiring itself is covered end to end by test_replay's
    grid-vs-paper agreement; this pins the measured size of the gap.)"""
    all_five = sum(q * p * MAKER for _, q, p in RUN)
    completed_only = sum(q * p * MAKER for _, q, p in (RUN[2], RUN[4]))

    assert round(all_five, 6) == 0.124968
    assert round(completed_only, 6) == 0.049987
    assert all_five - completed_only > 0.07


# --- adopting what the account already holds -------------------------------------

class _Positions:
    """Minimal stand-in for the exchange wrapper's get_positions."""

    def __init__(self, rows):
        self.rows = rows

    def get_positions(self, symbol):
        if isinstance(self.rows, Exception):
            raise self.rows
        return self.rows


def seeded(rows, start_qty=0.0, start_entry=0.0):
    g = ledger(start_qty, start_entry)
    g.exchange = _Positions(rows)
    g.symbol = "DOGEUSDT"
    g._seed_position_from_exchange()
    return g


def test_an_inherited_short_is_adopted_before_trading():
    """The state this bot actually woke up in on 2026-08-15."""
    g = seeded([{"contracts": 5348, "side": "short", "entryPrice": 0.07011}])
    assert g._pos_qty == -5348
    assert g._pos_entry == 0.07011


def test_an_inherited_long_is_adopted():
    g = seeded([{"contracts": 1793, "side": "long", "entryPrice": 0.06994}])
    assert g._pos_qty == 1793


def test_a_flat_account_leaves_the_saved_ledger_alone():
    g = seeded([], start_qty=100, start_entry=0.07)
    assert g._pos_qty == 100, "a flat exchange reply wiped restored state"


def test_a_zero_size_row_is_not_adopted():
    g = seeded([{"contracts": 0, "side": "long", "entryPrice": 0.07}], start_qty=50, start_entry=0.07)
    assert g._pos_qty == 50


def test_a_row_without_an_entry_price_is_not_adopted():
    g = seeded([{"contracts": 900, "side": "long", "entryPrice": 0}], start_qty=50, start_entry=0.07)
    assert g._pos_qty == 50


def test_an_exchange_error_leaves_the_ledger_on_saved_state():
    """A network blip must not silently reset the books to flat."""
    g = seeded(RuntimeError("connection reset"), start_qty=-5348, start_entry=0.07011)
    assert g._pos_qty == -5348


def test_adopting_stops_the_inheritance_being_booked_as_profit():
    g = seeded([{"contracts": 5348, "side": "short", "entryPrice": 0.07011}])
    realised = g._apply_to_position("buy", 5348, 0.07008)

    assert 0 < realised < 0.20, f"realised {realised:+.4f} on a 0.00003 move"
    assert realised < 1.0, "booked the whole inherited position as session profit"
