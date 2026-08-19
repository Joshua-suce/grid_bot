"""Moving the bot to another symbol without stranding money. AUDIT #112.

Three things must be true, in order:

  1. the CURRENT symbol is flat. Switching away from an open position abandons it --
     the stop stays on the exchange but nothing manages, unwinds or reports it. The
     script refuses, and it does not close positions itself: that is a trade.
  2. the NEW symbol fits. A rung must clear the exchange minimum, and the price tick
     must be fine enough that rungs do not round onto each other at the fee floor.
     ADAUSDT's tick is 0.057% of price against DOGE's 0.0142% -- four times coarser,
     and worth checking before the ladder is built rather than after.
  3. the saved state goes. It holds the OLD symbol's bounds and levels; restoring it
     against a new symbol is meaningless. Archived, not deleted.
"""

import pytest

import switch_symbol
from switch_symbol import MIN_TICKS_PER_RUNG, open_exposure, state_path


class FakeExchange:
    def __init__(self, positions=None, orders=None):
        self._positions = positions or []
        self._orders = orders or []

    def get_positions(self, symbol):
        return self._positions

    def get_open_orders(self, symbol):
        return self._orders


@pytest.fixture(autouse=True)
def restore_exchange():
    real = switch_symbol.Exchange
    yield
    switch_symbol.Exchange = real


def other_symbol():
    """A symbol that is definitely not the configured one.

    main() short-circuits with "Already on that symbol" before it reaches the flat check,
    so a hardcoded target quietly stops exercising the guard the moment .env moves to it.
    That happened on 2026-08-19 when SYMBOL became ADAUSDT: these two tests went from
    asserting rc == 2 to getting rc == 0, and the guard they exist to protect was no
    longer covered by anything.
    """
    from config import settings

    for candidate in ("ADAUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"):
        if candidate != settings.symbol:
            return candidate
    raise AssertionError("no distinct symbol available to switch to")


# --- guard 1: never abandon a position --------------------------------------------------

def test_an_open_position_blocks_the_switch(capsys):
    """The live case: SHORT 5350 DOGE open on 2026-08-18."""
    ex = FakeExchange(positions=[{"contracts": 5350.0}])
    switch_symbol.Exchange = lambda *a, **k: ex

    rc = switch_symbol.main([other_symbol()])

    assert rc == 2
    out = capsys.readouterr().out
    assert "abandon" in out
    assert "will not place that trade" in out


def test_resting_orders_block_it_too():
    ex = FakeExchange(orders=[{"id": "1"}])
    switch_symbol.Exchange = lambda *a, **k: ex

    assert switch_symbol.main([other_symbol()]) == 2


def test_a_short_counts_as_open():
    """positionAmt is negative for a short; without abs() it reads as flat."""
    assert open_exposure(FakeExchange(positions=[{"positionAmt": "-5350"}]),
                         "DOGEUSDT")[0] == pytest.approx(5350)


def test_unreadable_position_fields_do_not_crash():
    assert open_exposure(FakeExchange(positions=[{"contracts": None}, {}]),
                         "DOGEUSDT") == (0.0, 0)


def test_switching_to_the_same_symbol_is_a_no_op(capsys):
    from config import settings

    rc = switch_symbol.main([settings.symbol])

    assert rc == 0
    assert "Already on that symbol" in capsys.readouterr().out


# --- guard 2: the new symbol must fit ---------------------------------------------------

def test_the_tick_floor_is_a_real_constraint():
    """5 ticks a rung is the line. ADA sits at ~9, DOGE at ~17; a symbol quoted in
    whole cents against a 0.5% spacing would sit at 1 and round its rungs together."""
    assert MIN_TICKS_PER_RUNG >= 3, "fewer than 3 ticks cannot express a ladder"


def test_state_path_matches_what_main_writes():
    """If these disagree the script archives nothing and the bot restores the old
    symbol's ladder against the new one."""
    from config import settings

    p = state_path(settings.symbol)

    assert p.name.startswith(f"grid_{settings.symbol.lower()}")
    assert p.parent.name == settings.state_dir
    assert ("_demo" in p.name) == settings.demo_mode


# --- and it must never trade -------------------------------------------------------------

def test_the_script_places_no_orders():
    """It moves a file and edits .env. An order-placing call in here would be a trade
    nobody asked for -- and the one trade this script deliberately refuses to make."""
    from pathlib import Path

    src = Path(switch_symbol.__file__).read_text(encoding="utf-8")
    for forbidden in ("place_limit_order", "place_stop_market", "create_order",
                      "close_all_positions", "cancel_all"):
        assert forbidden not in src, f"{forbidden} has no business in this script"


def test_a_dry_run_is_the_default():
    from pathlib import Path

    src = Path(switch_symbol.__file__).read_text(encoding="utf-8")

    assert '"--apply", action="store_true"' in src
    assert "if not args.apply:" in src
