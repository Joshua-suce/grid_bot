"""The ladder must not lose a grid line. AUDIT #79.

Measured 2026-08-15 06:33-14:53. A filled SELL at 0.07001 migrated its level down to
0.06987 -- a line a held rung was already parked on. _handle_fill's occupancy test only
looks for queued replacements, not levels in 'awaiting_counter', so it saw the slot as
free. The held rung was then stranded (own line taken by the arrival, counter line taken
by a live sell) and NOTHING was left owning 0.07001.

Price then spent 5h13m -- 1226 consecutive polls, 100% of the remaining run -- strictly
inside the resulting 0.06987/0.07015 gap, crossing the vacated 0.07001 line 62 times
with no order on it. The saved state confirms it: 14 levels on 13 distinct lines, two
stacked on 0.06987, and 0.07001 gone.
"""

from grid import GridEngine, GridLevel


class FakeExchange:
    """Five decimals, matching the real DOGEUSDT tick."""

    class exchange:
        @staticmethod
        def price_to_precision(symbol, price):
            return f"{price:.5f}"

        @staticmethod
        def amount_to_precision(symbol, amount):
            return f"{amount:.6f}"


SPACING = 0.00014154


def engine(levels, spacing=SPACING, count=14):
    g = GridEngine.__new__(GridEngine)
    g.grid_spacing = spacing
    g.grid_count = count
    g.symbol = "DOGEUSDT"
    g.exchange = FakeExchange()
    g.levels = levels
    return g


def live(price, side):
    l = GridLevel(price=price, side=side)
    l.order_id = "live"
    l.status = "pending"
    return l


def held(price, side, counter_price, counter_side):
    l = GridLevel(price=price, side=side)
    l.status = "awaiting_counter"
    l.awaiting_price = counter_price
    l.awaiting_side = counter_side
    l.order_id = None
    return l


def measured_ladder():
    """The book exactly as state\\grid_dogeusdt_demo.json recorded it at shutdown."""
    stranded = held(0.06987, "buy", 0.07015, "sell")
    levels = [
        live(0.06916, "buy"), live(0.06930, "buy"), live(0.06944, "buy"),
        live(0.06958, "buy"), live(0.06973, "buy"),
        stranded,
        live(0.06987, "buy"),          # migrated here after fill #5, stranding the above
        live(0.07015, "sell"), live(0.07029, "sell"), live(0.07043, "sell"),
        live(0.07058, "sell"), live(0.07072, "sell"), live(0.07086, "sell"),
        live(0.07100, "sell"),
    ]
    return levels, stranded


# --- finding the hole ----------------------------------------------------------------

def test_the_vacated_line_is_detected():
    levels, _ = measured_ladder()
    assert engine(levels)._ladder_holes() == [0.07001]


def test_a_complete_ladder_has_no_holes():
    levels = [live(0.06987 + k * SPACING, "sell") for k in range(6)]
    assert engine(levels)._ladder_holes() == []


def test_a_ragged_gap_is_not_treated_as_a_hole():
    """Levels off the lattice must not have lines invented inside them, or the bot
    posts orders where the ladder never had any."""
    levels = [live(0.06987, "buy"), live(0.07040, "sell")]   # 3.7 spacings, not clean
    assert engine(levels)._ladder_holes() == []


def test_a_gap_wider_than_the_ladder_is_ignored():
    levels = [live(0.01, "buy"), live(0.09, "sell")]
    assert engine(levels)._ladder_holes() == []


def test_zero_spacing_cannot_divide_by_zero():
    levels, _ = measured_ladder()
    assert engine(levels, spacing=0.0)._ladder_holes() == []


# --- filling it ----------------------------------------------------------------------

def test_the_stranded_rung_takes_the_vacated_line():
    """The whole point: 0.07001 was where price was trading."""
    levels, stranded = measured_ladder()

    engine(levels)._release_awaiting_levels(0.07002)

    assert stranded.status == "pending", "the rung stayed stranded"
    assert stranded.price == 0.07001, f"went to {stranded.price}, not the empty line"
    assert stranded.awaiting_price is None


def test_the_ladder_is_whole_again_afterwards():
    levels, _ = measured_ladder()

    engine(levels)._release_awaiting_levels(0.07002)

    assert len({l.price for l in levels}) == 14


def test_a_hole_below_price_comes_back_as_a_bid():
    levels, stranded = measured_ladder()

    engine(levels)._release_awaiting_levels(0.07002)   # 0.07001 sits just below

    # asserting the side alone proves nothing -- the rung was already a buy, so a
    # version that never moves it passes
    assert (stranded.price, stranded.side) == (0.07001, "buy")


def test_a_hole_above_price_comes_back_as_an_offer():
    """Re-arming it as a bid above the market would post a crossing order."""
    levels, stranded = measured_ladder()

    engine(levels)._release_awaiting_levels(0.06990)   # 0.07001 sits above

    assert stranded.side == "sell"


def test_two_stranded_rungs_do_not_both_take_one_hole():
    levels, first = measured_ladder()
    second = held(0.06987, "buy", 0.07015, "sell")
    levels.insert(0, second)

    engine(levels)._release_awaiting_levels(0.07002)

    took = [l for l in (first, second) if l.status == "pending"]
    assert len(took) == 1, "both rungs released onto the same line"
    assert len({l.price for l in levels if l.status == "pending"}) == \
        len([l for l in levels if l.status == "pending"])


def test_a_rung_stays_held_when_the_ladder_has_no_hole():
    """Option 3 must not relocate rungs for its own sake."""
    stranded = held(0.06987, "buy", 0.07001, "sell")
    levels = [stranded, live(0.06987, "buy"), live(0.07001, "sell")]

    engine(levels)._release_awaiting_levels(0.06994)

    assert stranded.status == "awaiting_counter"


def test_no_price_means_no_relocation():
    levels, stranded = measured_ladder()

    engine(levels)._release_awaiting_levels(None)

    assert stranded.status == "awaiting_counter"


# --- the earlier routes still win ----------------------------------------------------

def test_the_free_counter_slot_still_takes_precedence():
    """Option 1 is what #61 intended and a hole must not pre-empt it."""
    stranded = held(0.06987, "buy", 0.07015, "sell")
    levels = [stranded, live(0.06987, "buy"), live(0.07043, "sell")]
    #                              0.07015 and 0.07029 are both empty lines

    engine(levels)._release_awaiting_levels(0.07002)

    assert (stranded.price, stranded.side) == (0.07015, "sell")


def test_the_rungs_own_line_still_takes_precedence_over_a_hole():
    """Option 2: if its own line is free and price has cleared it, it belongs there."""
    stranded = held(0.06987, "buy", 0.07015, "sell")
    levels = [stranded, live(0.07015, "sell"), live(0.07043, "sell")]
    #                   0.07029 is an empty line, but 0.06987 is the rung's own

    engine(levels)._release_awaiting_levels(0.07002)

    assert stranded.price == 0.06987
    assert stranded.side == "buy"
