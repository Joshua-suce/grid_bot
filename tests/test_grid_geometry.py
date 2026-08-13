"""The price must not sit in the widest hole in the ladder. AUDIT #44.

The dynamic ladder was built as

    buys  = linspace(lower, price, n, endpoint=False)   -> step (price-lower)/n
    sells = linspace(price, upper, m+1)[1:]             -> step (upper-price)/m

which places the last buy one FULL step below the price and the first sell one full
step above it. The gap straddling the price is therefore always exactly twice the
spacing everywhere else -- the widest hole in the ladder, parked permanently where the
price actually is. A grid earns when price crosses levels, so this doubled the movement
required before the bot could trade at all.

Measured on the 2026-08-13 17:00 run: ... 0.06954 | 0.07006 ... a 0.748% centre gap
against 0.37-0.39% elsewhere. The price spent 77 minutes inside it, ranging
0.06973-0.06994, and the bot recorded zero fills.
"""

import pytest

from grid import GridEngine

LOWER, UPPER = 0.06849107, 0.07110893      # the real grid from that run
PRICE = 0.0698
LIVE_LOW, LIVE_HIGH = 0.06973, 0.06994     # the range it actually traded


class _Ex:
    class exchange:
        @staticmethod
        def amount_to_precision(s, a): return f"{float(a):.0f}"
        @staticmethod
        def price_to_precision(s, p): return f"{float(p):.5f}"

    def get_price(self, s): return PRICE
    def get_balance(self, s="USDT"): return 4762.0
    def get_positions(self, s): return []
    def get_open_orders(self, s): return []
    def get_open_order_ids(self, s): return set()
    def can_place_order(self, s): return True
    def place_limit_order(self, *a, **k): return {"id": "x"}


def _engine(count=10, lower=LOWER, upper=UPPER):
    return GridEngine(
        exchange=_Ex(), symbol="DOGEUSDT",
        grid_lower=lower, grid_upper=upper, grid_count=count,
        capital_per_grid_pct=0.018, stop_loss_pct=0.03,
        min_profit_multiplier=3.0, max_exposure_pct=0.5, leverage=5,
    )


def _gaps(levels, price):
    """(gap straddling the price, mean of the others), as percentages."""
    p = sorted(l.price for l in levels)
    centre, others = None, []
    for a, b in zip(p, p[1:]):
        pct = (b / a - 1) * 100
        if a < price < b:
            centre = pct
        else:
            others.append(pct)
    return centre, sum(others) / len(others)


@pytest.mark.parametrize("count", [6, 9, 10, 11, 14, 20])
def test_the_gap_at_the_price_is_not_wider_than_the_rest(count):
    """The regression. It was exactly 2.0x for every grid_count."""
    g = _engine(count)
    g._initialize_dynamic(PRICE)

    centre, others = _gaps(g.levels, PRICE)

    assert centre is not None, "no gap straddles the price"
    assert centre <= others * 1.35, (
        f"grid_count={count}: the gap at the price is {centre:.3f}% against {others:.3f}% "
        f"elsewhere ({centre/others:.2f}x) -- the price is sitting in the widest hole"
    )


def test_the_nearest_level_is_about_half_a_spacing_away():
    """The consequence of the fix: the bot needs half a spacing of movement to trade,
    not a full one."""
    g = _engine(10)
    g._initialize_dynamic(PRICE)

    nearest = min(abs(l.price - PRICE) for l in g.levels) / PRICE * 100
    _, others = _gaps(g.levels, PRICE)

    assert nearest < others * 0.75, (
        f"nearest level is {nearest:.3f}% away with {others:.3f}% spacing -- "
        f"that is most of a full step, not half"
    )


def test_the_silent_77_minutes_would_have_traded():
    """The specific window: 16:59-18:15 on 2026-08-13, price 0.06973-0.06994, zero fills.
    The shipped ladder had no level inside that range at all."""
    g = _engine(10)
    g._initialize_dynamic(PRICE)

    inside = [l.price for l in g.levels if LIVE_LOW <= l.price <= LIVE_HIGH]

    assert inside, (
        "still no level inside the range the price actually traded -- the bot would "
        "have sat idle through those 77 minutes again"
    )


@pytest.mark.parametrize("count", [6, 9, 10, 11, 14, 20])
def test_every_level_stays_inside_the_grid(count):
    """Half-step offsets must not push a level outside the configured range."""
    g = _engine(count)
    g._initialize_dynamic(PRICE)

    for l in g.levels:
        assert LOWER <= l.price <= UPPER, f"level {l.price} escaped [{LOWER}, {UPPER}]"


@pytest.mark.parametrize("count", [6, 9, 10, 11, 14, 20])
def test_sides_are_assigned_correctly(count):
    """Buys below the price, sells above -- a buy above the market crosses and is
    post-only rejected (see #42)."""
    g = _engine(count)
    g._initialize_dynamic(PRICE)

    for l in g.levels:
        if l.side == "buy":
            assert l.price < PRICE, f"buy level {l.price} is above the price {PRICE}"
        else:
            assert l.price > PRICE, f"sell level {l.price} is below the price {PRICE}"


def test_a_degenerate_count_falls_back_instead_of_dividing_by_zero():
    """grid_count can shrink to 1 after tick-rounding dedup; half_count would be 0."""
    g = _engine(1)

    g._initialize_dynamic(PRICE)          # must not raise ZeroDivisionError

    assert g.levels, "no levels built at all"
