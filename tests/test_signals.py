"""Tests for the regime-signal observer.

The first section is the important one. signals.py claims to be read-only *by
construction* -- not by discipline -- and that claim is what makes it safe to run
alongside a live account. These tests pin it: if someone later hands the generator an
exchange handle, or gives it a method that looks like an order path, the first test
fails and names the change rather than letting it reach a real book.

The rest cover the scoring, because the CSV's whole purpose is to be evidence about
whether the regime calls are any good (AUDIT.md #24) -- and evidence scored by a buggy
judge is worse than no evidence, since it reads as authoritative.
"""

import csv
import inspect
from dataclasses import fields

import pytest

import signals
from signals import ACTIONABLE, RANGE_TOLERANCE_PCT, REGIME_BIAS, Signal, SignalGenerator


class _RecordingNotifier:
    def __init__(self, blow_up: bool = False):
        self.messages: list[str] = []
        self.blow_up = blow_up

    def send(self, message: str) -> bool:
        if self.blow_up:
            raise RuntimeError("telegram down")
        self.messages.append(message)
        return True


class _RecordingJournal:
    def __init__(self):
        self.entries: list[tuple] = []

    def trend_change(self, old, new, adx, source):
        self.entries.append((old, new, adx, source))


def _gen(tmp_path, **kw) -> SignalGenerator:
    kw.setdefault("symbol", "DOGEUSDT")
    return SignalGenerator(log_dir=str(tmp_path), **kw)


# --- read-only by construction -------------------------------------------------

def test_constructor_takes_no_exchange():
    """The safety property is structural: there is no parameter to pass a book to."""
    params = set(inspect.signature(SignalGenerator.__init__).parameters)
    assert "exchange" not in params
    assert "client" not in params


def test_generator_holds_no_order_capable_attribute(tmp_path):
    """Nothing reachable on the instance can place, cancel, or amend an order."""
    gen = _gen(tmp_path, notifier=_RecordingNotifier(), event_journal=_RecordingJournal())
    gen.observe("ranging", 0.10)
    gen.observe("uptrend", 0.10)

    forbidden = ("order", "buy", "sell", "position", "cancel", "close_all")
    for name in dir(gen):
        if name.startswith("__"):
            continue
        assert not any(f in name.lower() for f in forbidden), (
            f"SignalGenerator grew an order-shaped member: {name}"
        )
        value = getattr(gen, name, None)
        for attr in dir(value) if not isinstance(value, (str, int, float, bool, type(None))) else []:
            assert not attr.lower().startswith("create_order"), (
                f"{name} exposes an order path: {attr}"
            )


def test_module_imports_no_exchange():
    """signals.py must not reach the exchange layer even indirectly."""
    source = inspect.getsource(signals)
    assert "import exchange" not in source
    assert "from exchange" not in source
    assert "ccxt" not in source


# --- transitions ---------------------------------------------------------------

def test_first_observe_establishes_baseline_without_emitting(tmp_path):
    """Starting up in `ranging` is not a transition into it."""
    gen = _gen(tmp_path)
    assert gen.observe("ranging", 0.10) is None
    assert gen.history == []
    assert gen.pending is None


def test_unchanged_regime_emits_nothing(tmp_path):
    gen = _gen(tmp_path)
    gen.observe("ranging", 0.10)
    for _ in range(5):
        assert gen.observe("ranging", 0.11) is None
    assert gen.history == []


def test_change_emits_signal_with_bias(tmp_path):
    gen = _gen(tmp_path)
    gen.observe("ranging", 0.10)
    sig = gen.observe("uptrend", 0.12, adx=31.5)

    assert sig is not None
    assert (sig.from_regime, sig.to_regime, sig.bias) == ("ranging", "uptrend", "LONG")
    assert sig.price == pytest.approx(0.12)
    assert sig.adx == pytest.approx(31.5)
    assert gen.pending is sig


def test_regime_case_is_normalised(tmp_path):
    gen = _gen(tmp_path)
    gen.observe("RANGING", 0.10)
    assert gen.observe("ranging", 0.10) is None, "case change is not a regime change"
    sig = gen.observe("UpTrend", 0.10)
    assert sig.to_regime == "uptrend" and sig.bias == "LONG"


def test_unknown_regime_falls_back_to_none_bias(tmp_path):
    gen = _gen(tmp_path)
    gen.observe("ranging", 0.10)
    sig = gen.observe("garbage", 0.10)
    assert sig.bias == "NONE"


def test_every_known_regime_has_a_bias():
    assert set(REGIME_BIAS) == {"uptrend", "downtrend", "ranging", "uncertain"}
    assert REGIME_BIAS["uncertain"] not in ACTIONABLE


# --- scoring --------------------------------------------------------------------

@pytest.mark.parametrize("bias, entry, exit_, expected", [
    ("LONG", 0.10, 0.12, "hit"),
    ("LONG", 0.10, 0.08, "miss"),
    ("SHORT", 0.10, 0.08, "hit"),
    ("SHORT", 0.10, 0.12, "miss"),
])
def test_directional_scoring(bias, entry, exit_, expected):
    sig = Signal("t", "S", "a", "b", bias, entry, 20.0)
    outcome, move = signals.score(sig, exit_)
    assert outcome == expected
    assert move == pytest.approx((exit_ - entry) / entry)


def test_range_is_scored_on_staying_put():
    """A RANGE call is right when nothing happened -- the opposite test to a directional one."""
    inside = Signal("t", "S", "a", "b", "RANGE", 1.0, 15.0)
    assert signals.score(inside, 1.0 + RANGE_TOLERANCE_PCT / 2)[0] == "hit"
    assert signals.score(inside, 1.0 - RANGE_TOLERANCE_PCT / 2)[0] == "hit"
    assert signals.score(inside, 1.0 + RANGE_TOLERANCE_PCT * 2)[0] == "miss"
    assert signals.score(inside, 1.0 - RANGE_TOLERANCE_PCT * 2)[0] == "miss"


def test_range_boundary_separates_inside_from_outside():
    """The band is a threshold, not an exact edge -- don't pin float behaviour on it."""
    sig = Signal("t", "S", "a", "b", "RANGE", 1.0, 15.0)
    assert signals.score(sig, 1.0 + RANGE_TOLERANCE_PCT * 0.99)[0] == "hit"
    assert signals.score(sig, 1.0 + RANGE_TOLERANCE_PCT * 1.01)[0] == "miss"


def test_flat_bias_is_never_scored_as_a_hit():
    sig = Signal("t", "S", "a", "b", "NONE", 1.0, 10.0)
    assert signals.score(sig, 2.0)[0] == "flat"


def test_nonpositive_entry_price_scores_nothing():
    """Guards the division; a zero price is bad data, not a flat outcome."""
    assert signals.score(Signal("t", "S", "a", "b", "LONG", 0.0, 10.0), 1.0) == ("", 0.0)


def test_pending_is_resolved_by_the_next_change(tmp_path):
    gen = _gen(tmp_path)
    gen.observe("ranging", 0.10)
    first = gen.observe("uptrend", 0.10)
    gen.observe("downtrend", 0.12)

    assert first.outcome == "hit"
    assert first.move_pct == pytest.approx(0.2)
    assert first.resolved_price == pytest.approx(0.12)
    assert first.resolved_at != ""
    assert gen.pending is not None and gen.pending is not first


def test_only_the_latest_signal_is_pending(tmp_path):
    gen = _gen(tmp_path)
    gen.observe("ranging", 0.10)
    gen.observe("uptrend", 0.10)
    gen.observe("downtrend", 0.12)
    gen.observe("ranging", 0.11)
    assert sum(1 for s in gen.history if s.outcome == "") == 1


# --- csv ------------------------------------------------------------------------

def test_header_written_once_and_matches_fields(tmp_path):
    gen = _gen(tmp_path)
    _gen(tmp_path)  # second generator, same file -- must not re-write the header

    rows = list(csv.reader(open(gen.filepath)))
    assert rows == [SignalGenerator._HEADER]
    assert SignalGenerator._HEADER == [f.name for f in fields(Signal)], (
        "CSV columns drifted from the Signal dataclass"
    )


def test_only_resolved_signals_are_written(tmp_path):
    gen = _gen(tmp_path)
    gen.observe("ranging", 0.10)
    gen.observe("uptrend", 0.10)

    rows = list(csv.reader(open(gen.filepath)))
    assert len(rows) == 1, "an unresolved signal must not be written yet"

    gen.observe("downtrend", 0.12)
    rows = list(csv.reader(open(gen.filepath)))
    assert len(rows) == 2
    record = dict(zip(rows[0], rows[1]))
    assert record["bias"] == "LONG"
    assert record["outcome"] == "hit"
    assert float(record["move_pct"]) == pytest.approx(0.2, abs=1e-6)


def test_write_failure_does_not_break_the_loop(tmp_path, monkeypatch):
    """A journal that cannot be written must never take the trading loop down."""
    gen = _gen(tmp_path)
    gen.observe("ranging", 0.10)
    gen.observe("uptrend", 0.10)

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", boom)
    gen.observe("downtrend", 0.12)  # must not raise


# --- notification gating ---------------------------------------------------------

def test_actionable_signals_are_pushed(tmp_path):
    notifier = _RecordingNotifier()
    gen = _gen(tmp_path, notifier=notifier)
    gen.observe("ranging", 0.10)
    gen.observe("uptrend", 0.10)
    assert len(notifier.messages) == 1
    assert "SIGNAL LONG" in notifier.messages[0]


def test_uncertain_transitions_are_recorded_but_not_pushed(tmp_path):
    notifier = _RecordingNotifier()
    gen = _gen(tmp_path, notifier=notifier)
    gen.observe("ranging", 0.10)
    sig = gen.observe("uncertain", 0.10)
    assert sig is not None and sig in gen.history
    assert notifier.messages == []


def test_notify_false_suppresses_all_pushes(tmp_path):
    notifier = _RecordingNotifier()
    gen = _gen(tmp_path, notifier=notifier, notify=False)
    gen.observe("ranging", 0.10)
    gen.observe("uptrend", 0.10)
    assert notifier.messages == []


def test_notifier_failure_is_swallowed(tmp_path):
    gen = _gen(tmp_path, notifier=_RecordingNotifier(blow_up=True))
    gen.observe("ranging", 0.10)
    sig = gen.observe("uptrend", 0.10)  # must not raise
    assert sig is not None


def test_event_journal_receives_the_transition(tmp_path):
    journal = _RecordingJournal()
    gen = _gen(tmp_path, event_journal=journal)
    gen.observe("ranging", 0.10)
    gen.observe("uptrend", 0.10, adx=30.0)
    assert journal.entries == [("ranging", "uptrend", 30.0, "signal")]


def test_event_journal_failure_is_swallowed(tmp_path):
    class Broken:
        def trend_change(self, *a):
            raise RuntimeError("journal down")

    gen = _gen(tmp_path, event_journal=Broken())
    gen.observe("ranging", 0.10)
    assert gen.observe("uptrend", 0.10) is not None


# --- accuracy --------------------------------------------------------------------

def test_accuracy_counts_only_resolved_signals(tmp_path):
    gen = _gen(tmp_path)
    gen.observe("ranging", 0.10)
    gen.observe("uptrend", 0.10)     # LONG, resolves +20% -> hit
    gen.observe("downtrend", 0.12)   # SHORT, resolves +8.3% -> miss
    gen.observe("ranging", 0.13)     # pending, uncounted

    a = gen.accuracy()
    assert a["resolved"] == 2
    assert a["hits"] == 1
    assert a["hit_rate"] == pytest.approx(0.5)
    assert a["LONG"] == {"n": 1, "hits": 1, "hit_rate": 1.0}
    assert a["SHORT"] == {"n": 1, "hits": 0, "hit_rate": 0.0}
    assert a["RANGE"]["n"] == 0


def test_accuracy_on_empty_history_is_safe(tmp_path):
    gen = _gen(tmp_path)
    a = gen.accuracy()
    assert a["resolved"] == 0 and a["hit_rate"] == 0.0
    gen.log_accuracy()  # must not raise or divide by zero
