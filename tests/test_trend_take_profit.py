"""A defined reward, stated in units of the risk taken. AUDIT #101.

The trend follower had no target at all: it exited on the trailing stop or on the regime
turning, and the reward was whatever the trend happened to give. That is the classical
design and it is defensible -- trend following is usually paid by the rare very large
winner, and a fixed target caps exactly that.

What it cannot do is answer "what is my reward:risk on this trade", which is the only
question some operators care about. take_profit_r answers it: 1R is the distance from
entry to the stop the trade OPENED with, so take_profit_r=3 puts the target three times
that far away and one winner pays for three losers.

Two things this must get right, and both are easy to get wrong:

  * R is fixed at entry. The stop ratchets as price runs, so a target recomputed from the
    live stop would creep toward entry and quietly shrink the reward it exists to
    guarantee -- the trade would be sold as 3:1 and settle nearer 1:1.

  * the target must never win a tie against the stop. They sit on opposite sides of
    entry, so only a gap can put price past both, and on a gap the market never offered
    the target.
"""

import time

import pytest

from trend_follower import TrendFollower


class FakeInner:
    @staticmethod
    def price_to_precision(symbol, price):
        return f"{float(price):.5f}"

    @staticmethod
    def amount_to_precision(symbol, qty):
        return f"{float(qty):.0f}"


class FakeExchange:
    def __init__(self, price=0.0700):
        self.exchange = FakeInner()
        self.price = price
        self.closed = []
        # What the ACCOUNT holds. This used to be absent and get_positions returned []
        # unconditionally -- the fake reported flat while the test had just driven a
        # real entry through _record_entry. A close-guard that verifies the position
        # against the exchange then passes against a lie (AUDIT #132).
        self.position_qty = 0.0

    def get_price(self, symbol):
        return self.price

    def close_position(self, symbol, side, amount):
        self.closed.append((side, amount))
        self.position_qty = max(0.0, self.position_qty - abs(amount))
        return {"id": "c1"}

    def get_open_order_ids(self, symbol):
        return set()

    def get_positions(self, symbol):
        if self.position_qty <= 0:
            return []
        return [{"contracts": self.position_qty,
                 "info": {"positionAmt": str(self.position_qty)}}]


def follower(take_profit_r=0.0, atr_pct=0.01, price=0.0700):
    ex = FakeExchange(price)
    t = TrendFollower(ex, "DOGEUSDT", capital_pct=0.10, stop_loss_pct=0.005,
                      atr_stop_multiplier=2.0, take_profit_r=take_profit_r,
                      leverage=1, min_hold_seconds=0)
    t._atr_pct = atr_pct
    t.active = True
    return t, ex


def enter(t, side="buy", price=0.0700, qty=1000.0):
    """Drive a filled entry through the real _record_entry."""
    t._record_entry({"side": side, "average": price, "filled": qty, "status": "closed"})
    t._entry_time = 0.0                      # min_hold satisfied
    t.exchange.position_qty = qty            # the account really holds it now
    return t


# --- where the target lands -------------------------------------------------------------

def test_the_target_is_the_stop_distance_times_r():
    """1R is entry-to-stop. At 1% ATR and a 2x multiplier the stop is 2% away, so a 3R
    target sits 6% above entry."""
    t, _ = follower(take_profit_r=3.0, atr_pct=0.01)
    enter(t)

    assert t._initial_risk == pytest.approx(0.0700 * 0.02, rel=1e-3)
    assert t._take_profit_price == pytest.approx(0.0700 * 1.06, rel=1e-3)


def test_a_short_targets_the_other_way():
    t, _ = follower(take_profit_r=3.0, atr_pct=0.01)
    enter(t, side="sell")

    assert t._take_profit_price == pytest.approx(0.0700 * 0.94, rel=1e-3)


def test_zero_disables_it_and_keeps_the_original_behaviour():
    """The classical design has to remain reachable: no target, ride the trail."""
    t, _ = follower(take_profit_r=0.0)
    enter(t)

    assert t._take_profit_price is None


def test_a_wider_stop_pushes_the_target_out_with_it():
    """R is a unit of risk, not a fixed percentage. Doubling the ATR doubles both."""
    near, _ = follower(take_profit_r=3.0, atr_pct=0.01)
    far, _ = follower(take_profit_r=3.0, atr_pct=0.02)
    enter(near), enter(far)

    assert far._initial_risk == pytest.approx(2 * near._initial_risk, rel=1e-3)
    assert (far._take_profit_price - 0.07) == pytest.approx(
        2 * (near._take_profit_price - 0.07), rel=1e-3)


def test_r_is_frozen_at_entry_not_recomputed_from_the_ratcheting_stop():
    """The defect this is built to avoid. The stop ratchets up as price runs; a target
    re-derived from it would walk toward entry and deliver far less than the R it was
    sold as."""
    t, ex = follower(take_profit_r=3.0, atr_pct=0.01)
    enter(t)
    target_at_entry = t._take_profit_price
    risk_at_entry = t._initial_risk

    for p in (0.0710, 0.0720, 0.0730):       # price runs, trailing stop ratchets up
        ex.price = p
        t.update_trailing_sl(p)

    assert t.get_stop_loss_price() > 0.0700, "fixture no longer ratchets the stop"
    assert t._initial_risk == risk_at_entry
    assert t._take_profit_price == target_at_entry


# --- exiting on it ------------------------------------------------------------------------

def test_reaching_the_target_closes_the_position():
    t, ex = follower(take_profit_r=3.0, atr_pct=0.01)
    enter(t)
    ex.price = t._take_profit_price

    fills = t.check_fills(balance=1000.0)

    assert ex.closed and ex.closed[0][0] == "long"
    assert any(f["completed_cycle"] for f in fills)
    assert t._side is None


def test_short_of_the_target_it_stays_open():
    t, ex = follower(take_profit_r=3.0, atr_pct=0.01)
    enter(t)
    ex.price = t._take_profit_price * 0.999

    t.check_fills(balance=1000.0)

    assert t._side == "long", "closed before the target was reached"
    assert not ex.closed


def test_a_short_closes_when_price_falls_to_its_target():
    t, ex = follower(take_profit_r=3.0, atr_pct=0.01)
    enter(t, side="sell")
    ex.price = t._take_profit_price

    t.check_fills(balance=1000.0)

    assert ex.closed and ex.closed[0][0] == "short"


def test_the_stop_wins_a_tie_with_the_target():
    """Only a gap can put price past both. There the market never traded the target, and
    booking it would invent a win -- the same optimism that makes a backtest lie.

    The price matters. Entry 0.0700, stop 0.0686, target forced to 0.0600: at 0.0500 only
    the stop is live and the test passes whichever branch wins, proving nothing. 0.0650
    is under the stop AND over the target, which is the only arrangement that actually
    asks the question."""
    t, ex = follower(take_profit_r=3.0, atr_pct=0.01)
    enter(t)
    t._take_profit_price = 0.0600
    ex.price = 0.0650

    assert ex.price >= t._take_profit_price, "fixture does not reach the target"
    assert ex.price <= t.get_stop_loss_price(), "fixture does not breach the stop"

    fills = t.check_fills(balance=1000.0)

    # Assert the REASON, not the profit. Profit comes from the exit price and is negative
    # whichever branch wins, so it cannot tell them apart -- a mislabelling mutant passed
    # a profit-only assertion cleanly.
    assert fills and fills[-1]["reason"] == "trailing_stop", (
        f"a gap through the stop was labelled {fills[-1]['reason']!r}")
    assert fills[-1]["profit"] < 0


def test_a_genuine_target_hit_is_labelled_as_one():
    """The other half: the label has to be right when the target really did pay."""
    t, ex = follower(take_profit_r=3.0, atr_pct=0.01)
    enter(t)
    ex.price = t._take_profit_price

    fills = t.check_fills(balance=1000.0)

    assert fills[-1]["reason"] == "take_profit"
    assert fills[-1]["profit"] > 0


def test_the_target_does_not_move_up_with_the_ratcheting_stop():
    """The frozen-R property, asserted through behaviour rather than through the
    attribute. Let price run so the trailing stop ratchets, then offer exactly the
    original target: it must still close. A target re-derived from the live stop would
    have walked away and the trade would stay open past the reward it promised."""
    t, ex = follower(take_profit_r=3.0, atr_pct=0.01)
    enter(t)
    original_target = t._take_profit_price

    for p in (0.0710, 0.0720, 0.0730):
        ex.price = p
        t.check_fills(balance=1000.0)
    assert t._side == "long", "closed early; fixture cannot test the target"
    assert t.get_stop_loss_price() > 0.0700, "fixture no longer ratchets the stop"

    ex.price = original_target
    t.check_fills(balance=1000.0)

    assert t._side is None, "target moved with the stop instead of staying where it was set"


def test_a_negative_r_is_treated_as_disabled():
    """Unclamped, take_profit_r=-1 puts a long's 'target' BELOW entry -- so the first
    tick closes the trade at a loss and calls it a take-profit."""
    t, _ = follower(take_profit_r=-1.0, atr_pct=0.01)
    enter(t)

    assert t.take_profit_r == 0.0
    assert t._take_profit_price is None


def test_the_trailing_stop_still_exits_when_no_target_is_set():
    """take_profit_r=0 must not disable the stop as well."""
    t, ex = follower(take_profit_r=0.0, atr_pct=0.01)
    enter(t)
    ex.price = 0.0650

    t.check_fills(balance=1000.0)

    assert t._side is None and ex.closed


def test_state_is_cleared_so_the_next_trade_sets_its_own_target():
    """A stale target left on the object would exit the NEXT position at a price that
    belonged to the last one."""
    t, ex = follower(take_profit_r=3.0, atr_pct=0.01)
    enter(t)
    ex.price = t._take_profit_price
    t.check_fills(balance=1000.0)

    assert t._take_profit_price is None
    assert t._initial_risk == 0.0


# --- the payoff it is supposed to deliver -------------------------------------------------

def test_a_winner_pays_for_r_losers():
    """The property the whole feature exists for, in money. 3R won against 1R lost."""
    t, ex = follower(take_profit_r=3.0, atr_pct=0.01)
    enter(t, qty=1000.0)
    risk_usdt = t._initial_risk * 1000.0

    ex.price = t._take_profit_price
    fills = t.check_fills(balance=1000.0)
    won = fills[-1]["profit"]

    assert won == pytest.approx(3 * risk_usdt, rel=0.02)


def test_the_configured_default_changes_nothing():
    """Shipping this on by default would silently re-shape every trend trade."""
    from config import Settings

    assert Settings.model_fields["trend_take_profit_r"].default == 0.0


def test_main_passes_the_setting_through():
    """A config nobody reads is a config that does nothing."""
    from pathlib import Path

    import main

    src = Path(main.__file__).read_text(encoding="utf-8")
    ctor = src[src.index("trend = TrendFollower("):]
    ctor = ctor[:ctor.index(")\n")]

    assert "take_profit_r=settings.trend_take_profit_r" in ctor
