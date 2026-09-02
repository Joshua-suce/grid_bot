"""_paged() pagination for attribute_pnl.py.

No test file existed for this module before. The one thing worth guarding is the
same class of bug pnl_audit.py's own docstring documents having already been
caught and fixed there once: resuming a page at last-timestamp-plus-one silently
drops any sibling row stamped at the identical millisecond that did not fit in the
prior page. That module fixed it by resuming AT the boundary (inclusive) and
deduping by key; _paged() used to still have the "+1" version.
"""

from attribute_pnl import _paged


class _Endpoint:
    """Stubs a Binance fapiPrivate* paginated endpoint (fapiPrivateGetUserTrades /
    fapiPrivateGetAllOrders): rows within [startTime, endTime], capped at limit,
    returned in time order -- faithful on the one detail that matters, that it
    pages by TIME not by cursor, so same-millisecond rows straddle a page
    boundary exactly as they do live."""

    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: int(r["time"]))
        self.calls: list[dict] = []

    def __call__(self, params):
        self.calls.append(dict(params))
        start, end, limit = params["startTime"], params["endTime"], params["limit"]
        batch = [r for r in self.rows if start <= int(r["time"]) <= end]
        return batch[:limit]


def trade(t, trade_id):
    """userTrades rows carry a unique `id`."""
    return {"time": t, "id": trade_id}


def order(t, order_id):
    """allOrders rows have no `id` -- `orderId` is the unique key for them."""
    return {"time": t, "orderId": order_id}


def test_trades_sharing_a_millisecond_across_a_page_boundary_survive():
    rows = [trade(i, i) for i in range(999)]
    rows.append(trade(999, "boundary-a"))
    rows.append(trade(999, "boundary-b"))
    got = _paged(_Endpoint(rows), "DOGEUSDT", 0, 10_000, window_ms=20_000)
    ids = {r["id"] for r in got}
    assert {"boundary-a", "boundary-b"} <= ids, "a boundary row was skipped"
    assert len(got) == 1001


def test_orders_sharing_a_millisecond_across_a_page_boundary_survive():
    rows = [order(i, i) for i in range(999)]
    rows.append(order(999, "boundary-a"))
    rows.append(order(999, "boundary-b"))
    got = _paged(_Endpoint(rows), "DOGEUSDT", 0, 10_000, window_ms=20_000)
    order_ids = {r["orderId"] for r in got}
    assert {"boundary-a", "boundary-b"} <= order_ids, "a boundary row was skipped"
    assert len(got) == 1001


def test_a_row_returned_twice_across_the_resumed_boundary_is_counted_once():
    """Re-reading the boundary inclusively is the cost of not skipping it; dedup is
    what makes that safe."""
    rows = [trade(i, i) for i in range(999)]
    rows.append(trade(999, "boundary"))
    got = _paged(_Endpoint(rows), "DOGEUSDT", 0, 10_000, window_ms=20_000)
    ids = [r["id"] for r in got]
    assert len(ids) == len(set(ids))
    assert len(got) == 1000


def test_paging_stops_instead_of_spinning_when_a_full_page_repeats():
    """Resuming AT the last timestamp means a page of identically-stamped rows
    returns itself forever unless no-new-rows ends the walk."""
    rows = [trade(0, i) for i in range(1000)]
    ep = _Endpoint(rows)
    got = _paged(ep, "DOGEUSDT", 0, 10_000, window_ms=20_000)
    assert len(ep.calls) < 100
    assert len(got) == 1000


def test_a_short_page_ends_the_walk():
    ep = _Endpoint([trade(0, "only")])
    _paged(ep, "DOGEUSDT", 0, 10_000, window_ms=20_000)
    assert len(ep.calls) == 1


def test_a_row_exactly_on_a_window_boundary_is_not_dropped():
    """The outer per-window loop resumes the NEXT window at window_end + 1, so a
    row exactly at that instant must land in the following window rather than
    falling into the gap between them."""
    window_ms = 1000
    rows = [trade(0, "before"), trade(window_ms - 1, "at-end"), trade(window_ms, "next-window")]
    got = _paged(_Endpoint(rows), "DOGEUSDT", 0, window_ms + 1, window_ms=window_ms)
    ids = sorted(r["id"] for r in got)
    assert ids == ["at-end", "before", "next-window"]
