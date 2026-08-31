"""Demo fills and live fills must not land in the same journal. AUDIT #70.

Both journals wrote to one file regardless of account. tools/analyze_performance.py
reads trades.csv to report win rate, net PnL and fee drag -- so a demo run followed by a
live run would be averaged together into a number describing neither. Demo fills cost
nothing; live fills cost money.

signals.csv is deliberately NOT split: it records regime calls against price outcomes,
and the market is the same market whichever account is watching it.
"""

import json

import pytest

from event_journal import EventJournal
from trade_journal import TradeJournal


def _record(journal, **over):
    args = dict(
        symbol="DOGEUSDT", side="buy", price=0.0697, quantity=1792.0,
        grid_spacing=0.000287, fill_number=1, completed_cycle=False,
        cycle_pnl=0.0, cumulative_pnl=0.0, daily_pnl=0.0, trades_today=1,
        fills_today=1, regime="uncertain", regime_adx=18.4, fee=0.025, balance=4931.09,
        equity=4931.09, exposure_pct=0.025, unrealized_pnl=0.0,
    )
    args.update(over)
    journal.record(**args)


def test_trade_journals_are_separate_files(tmp_path):
    demo = TradeJournal(str(tmp_path), demo=True)
    live = TradeJournal(str(tmp_path), demo=False)

    assert demo.filepath != live.filepath
    assert demo.filepath.name == "trades_demo.csv"
    assert live.filepath.name == "trades_live.csv"


def test_a_demo_fill_never_appears_in_the_live_journal(tmp_path):
    demo = TradeJournal(str(tmp_path), demo=True)
    live = TradeJournal(str(tmp_path), demo=False)

    _record(demo, quantity=1792.0)

    assert "1792.0" in demo.filepath.read_text()
    live_rows = live.filepath.read_text().strip().splitlines()
    assert len(live_rows) == 1, "the live journal picked up a demo fill"


def test_both_journals_record_independently(tmp_path):
    demo = TradeJournal(str(tmp_path), demo=True)
    live = TradeJournal(str(tmp_path), demo=False)

    _record(demo, price=0.0697)
    _record(live, price=0.0812)

    assert "0.0697" in demo.filepath.read_text()
    assert "0.0697" not in live.filepath.read_text()
    assert "0.0812" in live.filepath.read_text()
    assert "0.0812" not in demo.filepath.read_text()


def test_event_journals_are_separate_files(tmp_path):
    demo = EventJournal(str(tmp_path), demo=True)
    live = EventJournal(str(tmp_path), demo=False)

    assert demo.filepath.name == "events_demo.jsonl"
    assert live.filepath.name == "events_live.jsonl"


def test_a_demo_event_never_appears_in_the_live_event_log(tmp_path):
    demo = EventJournal(str(tmp_path), demo=True)
    live = EventJournal(str(tmp_path), demo=False)

    demo.daily_reset(-3.31, 42, 4931.09)

    assert "4931.09" in demo.filepath.read_text()
    assert not live.filepath.exists() or live.filepath.read_text() == ""


def test_event_records_still_parse_as_jsonl(tmp_path):
    j = EventJournal(str(tmp_path), demo=True)
    j.daily_reset(-3.31, 42, 4931.09)

    lines = [json.loads(ln) for ln in j.filepath.read_text().strip().splitlines()]
    assert lines and lines[0]["event"]


@pytest.mark.parametrize("cls,attr", [(TradeJournal, "filepath"), (EventJournal, "filepath")])
def test_the_mode_is_recorded_on_the_journal(tmp_path, cls, attr):
    assert cls(str(tmp_path), demo=True).mode == "demo"
    assert cls(str(tmp_path), demo=False).mode == "live"
