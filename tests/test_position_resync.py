"""The ledger's average entry must track the exchange's. AUDIT #88.

_seed_position_from_exchange runs once, in activate(). Everything after that is a local
mirror fed only by fills this ladder observed -- but the scale-out trailing leg, the hard
stop and any reduce-only close all move the blended entry without passing through
_handle_fill. Measured 2026-08-16 22:21: exchange LONG 8516 @ 0.06952357, ladder sold
1796 @ 0.06959, engine booked +0.001293 where the income reconciler verified +0.09.
"""

import pytest

from grid import GridEngine


class FakeExchange:
    """Only the surface reconcile_position_entry touches."""

    def __init__(self, positions=None, raises=False):
        self._positions = positions if positions is not None else []
        self.raises = raises
        self.calls = 0

    def get_positions(self, symbol):
        self.calls += 1
        if self.raises:
            raise RuntimeError("network")
        return self._positions


def pos(side, qty, entry):
    return [{"side": side, "contracts": qty, "entryPrice": entry}]


@pytest.fixture
def engine(monkeypatch):
    e = GridEngine.__new__(GridEngine)          # no __init__: only the ledger matters
    e.symbol = "DOGEUSDT"
    e._pos_qty = 0.0
    e._pos_entry = 0.0
    e._mirror_mismatch_polls = 0
    e.exchange = FakeExchange()
    return e


# --- the drift this exists to catch -------------------------------------------------

def test_a_drifted_average_is_replaced_by_the_exchanges(engine):
    """The live case: same size, entry 0.0000657 high, so a won round trip books ~1%
    of what it actually made."""
    engine._pos_qty, engine._pos_entry = 8516.0, 0.0695893
    engine.exchange = FakeExchange(pos("long", 8516.0, 0.06952357))

    assert engine.reconcile_position_entry() is True
    assert engine._pos_entry == 0.06952357


def test_the_corrected_entry_produces_the_realistic_profit(engine):
    """End to end: the same sell that booked +0.001293 books the real number once the
    average is right. This is the whole point of the fix."""
    engine._pos_qty, engine._pos_entry = 8516.0, 0.0695893
    engine.exchange = FakeExchange(pos("long", 8516.0, 0.06952357))
    before = GridEngine._apply_to_position(engine, "sell", 1796.0, 0.06959)

    engine._pos_qty, engine._pos_entry = 8516.0, 0.0695893
    engine.reconcile_position_entry()
    after = GridEngine._apply_to_position(engine, "sell", 1796.0, 0.06959)

    assert before == pytest.approx(0.00128, abs=5e-4)      # what it booked
    assert after == pytest.approx(0.119, abs=0.01)         # what it made
    assert after > before * 50


def test_a_short_entry_is_adopted_too(engine):
    engine._pos_qty, engine._pos_entry = -5000.0, 0.0700
    engine.exchange = FakeExchange(pos("short", 5000.0, 0.06990))

    assert engine.reconcile_position_entry() is True
    assert engine._pos_entry == 0.06990


def test_an_entry_that_already_agrees_is_left_alone(engine):
    engine._pos_qty, engine._pos_entry = 8516.0, 0.06952357
    engine.exchange = FakeExchange(pos("long", 8516.0, 0.06952357))

    assert engine.reconcile_position_entry() is False
    assert engine._pos_entry == 0.06952357


# --- refusing to double-count an unattributed fill ----------------------------------

def test_a_quantity_disagreement_defers_at_first(engine):
    """Transiently a size mismatch means check_fills has not attributed a fill yet. The
    exchange's position ALREADY contains it, so adopting immediately and then processing
    the fill would apply the same trade twice."""
    engine._pos_qty, engine._pos_entry = 8516.0, 0.0695893
    engine.exchange = FakeExchange(pos("long", 10312.0, 0.06952357))

    assert engine.reconcile_position_entry() is False
    assert engine._pos_entry == 0.0695893, "adopted an entry while a fill was in flight"
    assert engine._pos_qty == 8516.0


def test_it_resyncs_on_the_next_poll_once_the_fill_lands(engine):
    """The deferral must not outlive the race it exists for."""
    engine._pos_qty, engine._pos_entry = 8516.0, 0.0695893
    engine.exchange = FakeExchange(pos("long", 10312.0, 0.06952357))
    assert engine.reconcile_position_entry() is False

    engine._pos_qty = 10312.0                    # the fill is attributed
    assert engine.reconcile_position_entry() is True
    assert engine._pos_entry == 0.06952357


# --- a mismatch that never resolves is a wrong ledger, not a race -------------------

def test_a_persistent_disagreement_rebuilds_the_mirror(engine):
    """The live failure, 2026-08-17 04:33. Ledger LONG 1389 @ 0.06981, exchange SHORT
    5342 @ 0.07015 -- different side, 6731 apart, and no fill was ever going to close
    that. Deferring unconditionally meant the resync never ran for a whole session
    while _apply_to_position priced closes against a long that did not exist.
    """
    engine._pos_qty, engine._pos_entry = 1389.0, 0.06981296
    engine.exchange = FakeExchange(pos("short", 5342.0, 0.07015001))

    for poll in range(engine.MIRROR_MISMATCH_TOLERANCE_POLLS - 1):
        assert engine.reconcile_position_entry() is False, f"adopted on poll {poll}"

    assert engine.reconcile_position_entry() is True
    assert engine._pos_qty == -5342.0, "side not adopted"
    assert engine._pos_entry == 0.07015001


def test_the_rebuilt_mirror_prices_the_next_trade_correctly(engine):
    """End to end, on the exact trade that exposed this.

    The engine sold 1783 @ 0.07008. Against the phantom LONG that is a closing trade
    and books a profit -- the live run recorded +0.74 on it. Against the real SHORT the
    same sell ADDS exposure and realises nothing at all, which is why the income
    reconciler verified -0.02, the fee and nothing else.

    Not a magnitude error. The mirror had the engine booking profit on a trade that
    closed no position.
    """
    engine._pos_qty, engine._pos_entry = 1389.0, 0.06981296
    engine.exchange = FakeExchange(pos("short", 5342.0, 0.07015001))
    phantom = GridEngine._apply_to_position(engine, "sell", 1783.0, 0.07008)

    engine._pos_qty, engine._pos_entry = 1389.0, 0.06981296
    for _ in range(engine.MIRROR_MISMATCH_TOLERANCE_POLLS):
        engine.reconcile_position_entry()
    real = GridEngine._apply_to_position(engine, "sell", 1783.0, 0.07008)

    assert phantom > 0.3, "the phantom long booked a profit on this sell"
    assert real == 0.0, "on the real short this sell adds exposure and realises nothing"
    assert engine._pos_qty == pytest.approx(-7125.0), "short should have grown"


def test_the_counter_resets_once_the_sizes_agree(engine):
    """Two isolated races must not add up to a rebuild -- that would defeat the
    double-count guard the deferral exists for."""
    engine._pos_qty, engine._pos_entry = 8516.0, 0.06952357
    engine.exchange = FakeExchange(pos("long", 10312.0, 0.06952357))
    engine.reconcile_position_entry()
    engine.reconcile_position_entry()

    engine._pos_qty = 10312.0                    # resolved
    engine.reconcile_position_entry()
    assert engine._mirror_mismatch_polls == 0

    engine._pos_qty = 8516.0                     # a fresh, unrelated race
    assert engine.reconcile_position_entry() is False, "old strikes carried over"
    assert engine._pos_qty == 8516.0


def test_a_rebuild_resets_the_counter(engine):
    engine._pos_qty, engine._pos_entry = 1389.0, 0.06981296
    engine.exchange = FakeExchange(pos("short", 5342.0, 0.07015001))
    for _ in range(engine.MIRROR_MISMATCH_TOLERANCE_POLLS):
        engine.reconcile_position_entry()

    assert engine._mirror_mismatch_polls == 0


def test_a_hair_of_float_noise_is_not_a_quantity_disagreement(engine):
    """Sizes come back through float parsing; an exact-equality gate would defer
    forever on a position that actually matches."""
    engine._pos_qty, engine._pos_entry = 8516.0, 0.0695893
    engine.exchange = FakeExchange(pos("long", 8516.0000001, 0.06952357))

    assert engine.reconcile_position_entry() is True


# --- not making things worse ---------------------------------------------------------

def test_a_flat_exchange_does_not_zero_the_ledger(engine):
    """Flat on the exchange with a live ledger is the unattributed-close case. Zeroing
    the entry here would make the next fill's P&L nonsense instead of merely stale."""
    engine._pos_qty, engine._pos_entry = 8516.0, 0.0695893
    engine.exchange = FakeExchange([])

    assert engine.reconcile_position_entry() is False
    assert engine._pos_entry == 0.0695893


def test_an_unreadable_exchange_keeps_the_existing_mirror(engine):
    """Same rule as POSITION SEED SKIPPED: a network blip must not rewrite the ledger."""
    engine._pos_qty, engine._pos_entry = 8516.0, 0.0695893
    engine.exchange = FakeExchange(raises=True)

    assert engine.reconcile_position_entry() is False
    assert engine._pos_entry == 0.0695893


def test_a_zero_entry_price_is_refused(engine):
    """Binance occasionally reports entryPrice 0 on a position mid-update. Adopting it
    would make every subsequent close look like infinite profit."""
    engine._pos_qty, engine._pos_entry = 8516.0, 0.0695893
    engine.exchange = FakeExchange(pos("long", 8516.0, 0.0))

    assert engine.reconcile_position_entry() is False
    assert engine._pos_entry == 0.0695893


def test_the_quantity_is_never_rewritten(engine):
    """Only the average is adopted. Size is check_fills' business, and a resync that
    moved it would silently erase a fill the ladder still has to account for.

    The sizes here differ by the float hair the tolerance forgives -- which is the only
    reachable state where adopting the quantity is observable at all. An equal-size
    fixture would pass against a method that rewrites both, and did.
    """
    engine._pos_qty, engine._pos_entry = 8516.0, 0.0695893
    engine.exchange = FakeExchange(pos("long", 8516.0000001, 0.06952357))

    assert engine.reconcile_position_entry() is True
    assert engine._pos_qty == 8516.0, "resync rewrote the size, not just the average"
