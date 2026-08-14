"""The stop-loss layer told the truth about coverage only by accident. AUDIT #54.

Three defects, all on the path that produced the worst day on record:

  1. Coverage was `bool(sl_orders)` -- True if ANY leg placed. The scale-out splits the
     position across a trail leg and a hard leg, so one successful placement out of two
     left HALF the position with no stop and reported it protected. The #50 fix that
     blocks new exposure while uncovered was therefore reading a flag that could not see
     the failure it existed to catch.

  2. Refresh was cancel-everything-then-place-everything. A trailing leg ratcheting up
     tore down the unchanged hard leg as well, opening two unprotected round trips
     instead of one -- and doing it on every ratchet.

  3. `get_stop_orders` returned [] both when there were no stops and when the READ
     FAILED. Reconciling against a fabricated empty book means cancelling protection you
     cannot see, or double-placing it.
"""

import main


class _Ex:
    """Records what the reconciler actually did to the exchange."""

    def __init__(self, place_fails=(), cancel_ok=True):
        self.placed: list[tuple] = []
        self.purposes: list[str] = []
        self.cancelled: list[str] = []
        self._place_fails = set(place_fails)
        self._cancel_ok = cancel_ok
        self._n = 0

    def place_stop_market(self, symbol, side, amount, stop_price, purpose="stop_hard"):
        if round(float(stop_price), 8) in {round(float(p), 8) for p in self._place_fails}:
            raise RuntimeError(f"exchange rejected stop @ {stop_price}")
        self._n += 1
        self.placed.append((side, float(amount), float(stop_price)))
        self.purposes.append(purpose)
        return {"id": f"new{self._n}"}

    def cancel_order(self, order_id, symbol):
        self.cancelled.append(order_id)
        return self._cancel_ok


def _live(order_id, price, qty):
    return {"id": order_id, "stopPrice": price, "remaining": qty}


DESIRED = [("trail", 632.0, 0.0682), ("hard", 632.0, 0.0673)]


# --- 1. coverage is a quantity, not a boolean -------------------------------

def test_one_leg_placing_out_of_two_is_not_covered():
    """The defect exactly: trail places, hard is rejected, 632 of 1264 DOGE naked.
    The old `bool(sl_orders)` reported this as protected and let the grid keep buying."""
    ex = _Ex(place_fails=[0.0673])

    kept, covered_qty, desired_qty = main.reconcile_stop_orders(
        ex, "DOGEUSDT", "sell", DESIRED, live=[],
    )

    assert set(kept) == {"trail"}
    assert covered_qty == 632.0
    assert desired_qty == 1264.0
    assert covered_qty < desired_qty, (
        "half the position has no stop but coverage does not reflect it"
    )


def test_both_legs_placing_is_covered():
    ex = _Ex()
    kept, covered_qty, desired_qty = main.reconcile_stop_orders(
        ex, "DOGEUSDT", "sell", DESIRED, live=[],
    )
    assert set(kept) == {"trail", "hard"}
    assert covered_qty == desired_qty == 1264.0


def test_every_leg_failing_reports_zero_coverage():
    ex = _Ex(place_fails=[0.0682, 0.0673])
    kept, covered_qty, desired_qty = main.reconcile_stop_orders(
        ex, "DOGEUSDT", "sell", DESIRED, live=[],
    )
    assert kept == {}
    assert covered_qty == 0.0 and desired_qty == 1264.0


def test_a_partially_filled_stop_counts_only_what_remains():
    """`remaining` beats `amount`: a stop half-consumed no longer covers the position."""
    ex = _Ex()
    live = [{"id": "s1", "stopPrice": 0.0682, "amount": 632.0, "remaining": 300.0}]

    kept, covered_qty, _ = main.reconcile_stop_orders(
        ex, "DOGEUSDT", "sell", [("trail", 632.0, 0.0682)], live,
    )

    # 300 does not cover 632, so it is replaced rather than trusted
    assert ex.placed, "a partially-filled stop was accepted as full coverage"
    assert covered_qty == 632.0


# --- 2. do not churn what has not changed -----------------------------------

def test_an_unchanged_leg_is_left_alone_when_the_other_moves():
    """The trail ratchets, the hard stop does not. The hard stop must not be touched:
    every needless cancel/place is a window with the position unprotected."""
    ex = _Ex()
    live = [_live("trail_old", 0.0670, 632.0), _live("hard_live", 0.0673, 632.0)]
    desired = [("trail", 632.0, 0.0682), ("hard", 632.0, 0.0673)]

    kept, covered_qty, desired_qty = main.reconcile_stop_orders(
        ex, "DOGEUSDT", "sell", desired, live,
    )

    assert kept["hard"]["id"] == "hard_live", "the unchanged hard stop was replaced"
    assert ex.cancelled == ["trail_old"], f"cancelled more than the stale leg: {ex.cancelled}"
    assert [p[2] for p in ex.placed] == [0.0682], "placed more than the moved leg"
    assert covered_qty == desired_qty


def test_nothing_is_touched_when_the_book_already_matches():
    ex = _Ex()
    live = [_live("t", 0.0682, 632.0), _live("h", 0.0673, 632.0)]

    kept, covered_qty, desired_qty = main.reconcile_stop_orders(
        ex, "DOGEUSDT", "sell", DESIRED, live,
    )

    assert ex.placed == [] and ex.cancelled == []
    assert covered_qty == desired_qty
    assert {k: v["id"] for k, v in kept.items()} == {"trail": "t", "hard": "h"}


def test_strays_are_cancelled():
    """An orphan stop from an earlier crash is claimed by no desired leg, so it goes."""
    ex = _Ex()
    live = [_live("t", 0.0682, 632.0), _live("h", 0.0673, 632.0), _live("orphan", 0.0601, 999.0)]

    main.reconcile_stop_orders(ex, "DOGEUSDT", "sell", DESIRED, live)

    assert ex.cancelled == ["orphan"]


# --- 3. an unreadable book is not an empty one ------------------------------

def test_get_stop_orders_returns_none_when_unreadable():
    from exchange import Exchange

    ex = Exchange.__new__(Exchange)
    ex._retry = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("backend down"))
    ex.exchange = type("X", (), {"fetch_open_orders": staticmethod(lambda *a, **k: [])})()

    assert ex.get_stop_orders("DOGEUSDT") is None, (
        "a failed read still reports an empty stop book -- callers will conclude the "
        "position has no stops and act on it"
    )


def test_cancel_all_stop_orders_reports_unknown_rather_than_zero():
    from exchange import Exchange

    ex = Exchange.__new__(Exchange)
    ex.get_stop_orders = lambda symbol: None

    assert ex.cancel_all_stop_orders("DOGEUSDT") is None, (
        "an unreadable stop book reported as '0 cancelled', which reads as a clean sweep"
    )


def test_the_caller_gates_on_quantity_not_truthiness():
    """Structural, and the one that matters most: a reconciler that reports quantities
    is useless if the caller goes back to asking whether the dict is non-empty. That is
    the original defect -- one leg out of two placing, and the position reported covered.
    """
    import inspect

    source = inspect.getsource(main.run_bot)
    assert "covered_qty, desired_qty = reconcile_stop_orders(" in source, (
        "_refresh_sl_stops no longer takes quantities from the reconciler"
    )
    assert "covered = covered_qty >= desired_qty" in source, (
        "coverage is no longer decided by comparing covered quantity against desired"
    )
    assert "covered = bool(sl_orders)" not in source, (
        "coverage is back to a truthiness test: one leg of a two-leg scale-out placing "
        "would report a half-naked position as protected"
    )


def test_reconcile_is_never_reached_with_a_fabricated_empty_book():
    """Structural: the refresh must abort on an unreadable book rather than reconcile
    against []. Reconciling against [] cancels nothing and re-places everything, which
    is the double-protection/no-protection case this exists to prevent."""
    import inspect

    source = inspect.getsource(main.run_bot)
    assert "if live is None:" in source, (
        "_refresh_sl_stops no longer distinguishes an unreadable stop book"
    )
    abort = source.index("STOP REFRESH ABORTED")
    reconcile = source.index("reconcile_stop_orders(")
    assert abort < reconcile, "the unreadable-book guard must precede the reconcile"
