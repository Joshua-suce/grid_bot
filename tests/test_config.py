import pytest

from config import Settings


def test_validate_requires_credentials_in_demo_mode():
    """No mock/simulated trading fallback: DEMO mode must have real API
    credentials configured too, not just LIVE mode (see AUDIT.md)."""
    s = Settings(api_key="", api_secret="", demo_mode=True)
    with pytest.raises(ValueError, match="DEMO mode requires"):
        s.validate()


def test_validate_requires_credentials_in_live_mode():
    s = Settings(api_key="", api_secret="", demo_mode=False)
    with pytest.raises(ValueError, match="LIVE mode requires"):
        s.validate()


def test_validate_requires_both_key_and_secret():
    s = Settings(api_key="only-a-key", api_secret="", demo_mode=True)
    with pytest.raises(ValueError, match="DEMO mode requires"):
        s.validate()


def test_validate_passes_with_credentials_present():
    s = Settings(api_key="key123", api_secret="secret123", demo_mode=True)
    s.validate()  # must not raise


# --- Spacing must clear fees by a real margin ------------------------------
# Every field these tests care about is passed explicitly so the assertions do not
# depend on whatever happens to be in the developer's .env.

def _coherent(**overrides):
    base = dict(
        api_key="key123", api_secret="secret123", demo_mode=True,
        grid_count=10, capital_per_grid_pct=0.018, max_position_pct=0.12,
        maker_fee_pct=0.02, taker_fee_pct=0.04,
        range_min_spacing_pct=0.002, min_profit_multiplier=3.0,
    )
    base.update(overrides)
    return Settings(**base)


def test_spacing_at_the_fee_floor_is_rejected():
    """The live config that lost money: 0.1% spacing against a 0.04% round trip.

    Fees took 39% of every gross cycle before slippage or an adverse move. The
    reconciled result was commission -45.26 against realized +13.21.
    """
    s = _coherent(range_min_spacing_pct=0.001)
    with pytest.raises(ValueError, match="does not clear fees"):
        s.validate()


def test_spacing_comfortably_above_fees_is_accepted():
    _coherent(range_min_spacing_pct=0.002).validate()  # 5x the round trip


def test_spacing_floor_scales_with_the_fee_rate():
    """A higher fee tier must demand wider spacing, not silently shrink the edge."""
    s = _coherent(maker_fee_pct=0.05, range_min_spacing_pct=0.002)
    with pytest.raises(ValueError, match="does not clear fees"):
        s.validate()


def test_multiplier_of_one_still_permits_break_even_spacing():
    """Opting back into the old break-even behaviour stays possible, just explicit."""
    _coherent(range_min_spacing_pct=0.001, min_profit_multiplier=1.0).validate()


# --- The grid must fit inside the position cap -----------------------------

def test_grid_wider_than_the_position_cap_is_rejected():
    """The live config: 20 levels, but max_position_pct only afforded ~6.7.

    Levels 7-20 could never fill. The cap blocked that side partway through, the book
    went permanently one-sided, and the bot sat in the capped state that made
    recentering destructive (89 recenters in one session).
    """
    s = _coherent(grid_count=20)
    with pytest.raises(ValueError, match="can never fill"):
        s.validate()


def test_grid_that_fits_the_position_cap_is_accepted():
    _coherent(grid_count=10).validate()  # 5 per side vs 6.7 affordable


def test_error_names_a_grid_count_that_would_actually_work():
    s = _coherent(grid_count=20)
    with pytest.raises(ValueError) as exc:
        s.validate()
    suggested = int(str(exc.value).split("GRID_COUNT <= ")[1].split(",")[0])
    _coherent(grid_count=suggested).validate()  # the advice must itself validate


def test_raising_the_position_cap_also_resolves_it():
    """Three ways out of the incoherence; the error names all of them."""
    _coherent(grid_count=20, max_position_pct=0.20).validate()
    _coherent(grid_count=20, capital_per_grid_pct=0.011).validate()


# --- strategy mode ---------------------------------------------------------

def test_strategy_mode_rejects_an_unknown_value():
    """A typo must fail at startup, not silently fall back to trading one strategy."""
    s = _coherent(strategy_mode="trendfollower")
    with pytest.raises(ValueError, match="STRATEGY_MODE must be"):
        s.validate()


@pytest.mark.parametrize("mode", ["grid", "router"])
def test_valid_strategy_modes_are_accepted(mode):
    _coherent(strategy_mode=mode).validate()


def test_strategy_mode_defaults_to_grid():
    """The router's switching thresholds are unvalidated (AUDIT.md), so the
    long-standing single-strategy behaviour must remain the default."""
    assert Settings.model_fields["strategy_mode"].default == "grid"
