"""CAPITAL_PER_GRID_USDT actually decides the order size now. AUDIT #63.

It was `max(fixed, percent)`, which made the setting silently inert whenever the percent
path was larger -- and on a 4930 balance it always was. Every start logged:

    CAPITAL_PER_GRID_USDT (25.00) is smaller than percent-based allocation (88.74);
    using the larger value for per-grid sizing.

so orders went out at 88.74 USDT while the config asked for 25. The line reads as a note
rather than "your setting is being ignored", and it survived several audits because the
behaviour was documented in the docstring.

Fixed means: when set, CAPITAL_PER_GRID_USDT x LEVERAGE is the notional per order, still
bounded by MAX_EXPOSURE_PCT.
"""

import pytest

from grid import GridEngine


class _Ex:
    class exchange:
        @staticmethod
        def amount_to_precision(s, a): return f"{float(a):.0f}"
        @staticmethod
        def price_to_precision(s, p): return f"{float(p):.5f}"

    def get_price(self, s): return 0.0694
    def get_balance(self, s="USDT"): return 4930.0
    def get_positions(self, s): return []
    def get_open_orders(self, s): return []
    def get_open_order_ids(self, s): return set()
    def can_place_order(self, s): return True
    def place_limit_order(self, *a, **k): return {"id": "x"}


def _engine(usdt=5.0, leverage=10, pct=0.018, count=10, max_exposure=0.50):
    return GridEngine(
        exchange=_Ex(), symbol="DOGEUSDT", grid_lower=0.0686, grid_upper=0.0703,
        grid_count=count, capital_per_grid_pct=pct, capital_per_grid_usdt=usdt,
        stop_loss_pct=0.03, min_profit_multiplier=3.0,
        max_exposure_pct=max_exposure, leverage=leverage,
    )


def test_the_configured_usdt_size_is_what_gets_used():
    """5 USDT of capital at 10x is 50 USDT of notional -- not the 88.74 the percent
    path would have produced on this balance."""
    g = _engine(usdt=5.0, leverage=10)

    assert g._calc_usdt_per_grid(4930.0) == pytest.approx(50.0)


def test_it_is_used_even_when_the_percent_path_is_larger():
    """The exact live case: fixed 25, percent 88.74, and 88.74 won."""
    g = _engine(usdt=5.0, leverage=5)          # 5 x 5 = 25
    percent = 4930.0 * 0.018                   # 88.74

    size = g._calc_usdt_per_grid(4930.0)

    assert size == pytest.approx(25.0)
    assert size < percent, "the percent path is still overriding the configured size"


def test_a_calm_market_cannot_inflate_the_configured_size():
    """The configured size is a CEILING. Unbounded, the calm multiplier hits 2.5 and
    turned a configured 50 into 102.50 -- ten rungs of which is 1025 against a 587
    position cap, so the cap blocked the ladder in exactly the quiet markets the
    multiplier exists to keep it trading in (AUDIT #64)."""
    g = _engine(usdt=5.0, leverage=10)
    g.update_volatility(0.003)

    assert g._volatility_mult > 1.0, "precondition: calm markets do raise the multiplier"
    assert g._calc_usdt_per_grid(4930.0) == pytest.approx(50.0)


def test_a_violent_market_may_still_shrink_it():
    """Bounded above, not below -- shrinking in wild markets reduces risk."""
    g = _engine(usdt=5.0, leverage=10)
    g.update_volatility(0.05)

    assert g._volatility_mult < 1.0
    assert g._calc_usdt_per_grid(4930.0) == pytest.approx(50.0 * g._volatility_mult)
    assert g._calc_usdt_per_grid(4930.0) < 50.0


def test_percent_sizing_keeps_the_full_multiplier():
    """The ceiling applies to the configured-USDT path only; percent sizing is a
    fraction of equity and scales as it always did."""
    g = _engine(usdt=0.0, leverage=10)
    g.update_volatility(0.003)

    assert g._volatility_mult > 1.0
    assert g._calc_usdt_per_grid(4930.0) == pytest.approx(
        4930.0 * 0.018 * g._volatility_mult
    )


def test_percent_sizing_still_applies_when_no_usdt_is_configured():
    g = _engine(usdt=0.0, leverage=10)

    assert g._calc_usdt_per_grid(4930.0) == pytest.approx(4930.0 * 0.018)


def test_the_exposure_ceiling_still_binds():
    """An absolute size must not be able to commit more than MAX_EXPOSURE_PCT."""
    g = _engine(usdt=500.0, leverage=10, count=10, max_exposure=0.50)

    size = g._calc_usdt_per_grid(4930.0)

    assert size * 10 == pytest.approx(4930.0 * 0.50)
    assert size < 500.0 * 10


def test_the_new_size_is_smaller_than_what_was_running():
    """The whole point: this reduces exposure. 10 rungs at the configured size must
    commit less than the percent path did."""
    configured = _engine(usdt=5.0, leverage=10)._calc_usdt_per_grid(4930.0) * 10
    percent_path = 4930.0 * 0.018 * 10

    assert configured < percent_path
    assert configured == pytest.approx(500.0)
    assert percent_path == pytest.approx(887.4)


# --- config validation ------------------------------------------------------

def test_the_percent_cap_check_is_skipped_when_usdt_sizing_governs():
    """It compares capital_per_grid_pct against max_position_pct. With USDT sizing that
    number decides nothing, and the absolute figure cannot be compared to a fraction of
    equity without a balance -- so the check must not fire on it."""
    from config import Settings

    # 10 x 0.05 = 0.50 total allocation, which clears the earlier ceiling check, but
    # 5 levels per side against 0.12/0.05 = 2.4 affordable trips this one.
    s = Settings(api_key="k", api_secret="s", grid_count=10,
                 capital_per_grid_pct=0.05, max_position_pct=0.12,
                 capital_per_grid_usdt=5.0)
    s.validate()

    strict = Settings(api_key="k", api_secret="s", grid_count=10,
                      capital_per_grid_pct=0.05, max_position_pct=0.12,
                      capital_per_grid_usdt=0.0)
    with pytest.raises(ValueError, match="GRID_COUNT"):
        strict.validate()
