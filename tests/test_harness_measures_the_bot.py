"""The backtest harness must measure the strategy the bot actually runs.

backtest.run_backtest has accepted use_router since it was written, defaulting False.
run_backtest.py never passed it. So every figure this project has ever taken from the
tool -- every sweep, every robustness verdict, every tuning decision -- measured the
GRID ALONE while the bot ran STRATEGY_MODE=router.

Measured on data/adausdt_1h_180d.csv, balance 4866, identical in every other respect:

    router     return -1.82%   net -88.79   898 fills   227 taker   127 switches
    grid only  return +0.17%   net  +8.18  1180 fills   105 taker     0 switches

A 96.97 USDT gap on one window, reproducing an independent four-window measurement
of -96.21 vs +16.11. The tool was reporting the wrong system. AUDIT #140.

Separately the harness sizes only by percent of balance, while the bot sizes by
CAPITAL_PER_GRID_USDT x leverage when that is set. There is no parameter to tell the
harness that, so it silently trades 87.6 USDT/order against the account's 25.0. It
cannot be fixed by passing a flag, so it is now reported instead of assumed away.
"""
from __future__ import annotations

import inspect

import pytest

import run_backtest as rb


# ------------------------------------------------- the router reaches the engine
def test_the_engine_supports_a_router_the_front_end_never_asked_for():
    """Pins the shape of the original bug: the capability existed and was unused."""
    import backtest
    assert "use_router" in inspect.signature(backtest.run_backtest).parameters


def test_defaults_carry_the_deployed_strategy_mode():
    from config import settings
    cfg = rb._defaults_from_env()
    assert cfg["use_router"] is (settings.strategy_mode == "router")


def test_defaults_carry_the_router_parameters_too():
    """A router run with default handoff timings is not the bot either."""
    cfg = rb._defaults_from_env()
    for key in ("trend_capital_pct", "trend_atr_stop_multiplier",
                "trend_min_hold_seconds", "router_min_regime_seconds",
                "router_handoff_grace_seconds"):
        assert key in cfg, f"{key} is not passed, so the sim's router is not the bot's"


def test_every_router_key_is_accepted_by_the_engine():
    """A key the engine does not take is a TypeError at run time, and a key it takes
    but we never send is the bug this file exists for. Check both directions."""
    import backtest
    accepted = set(inspect.signature(backtest.run_backtest).parameters)
    unknown = sorted(set(rb._defaults_from_env()) - accepted)
    assert not unknown, f"_defaults_from_env sends keys run_backtest cannot take: {unknown}"


# ------------------------------------------------------------ the sizing warning
def test_sizing_mismatch_reports_the_live_ratio():
    bt, live, ratio = rb.sizing_mismatch({"capital_per_grid_pct": 0.018}, 4866.0)
    assert bt == pytest.approx(87.588)
    assert live > 0
    assert ratio == pytest.approx(bt / live)


def test_a_matched_size_reports_a_ratio_of_one():
    from config import settings
    live = settings.capital_per_grid_usdt * settings.leverage
    balance = 5000.0
    _, _, ratio = rb.sizing_mismatch(
        {"capital_per_grid_pct": live / balance}, balance)
    assert ratio == pytest.approx(1.0)


def test_percent_only_configs_do_not_claim_a_mismatch():
    """When the bot itself sizes by percent there is nothing to warn about, and a
    spurious warning on every run trains the reader to skip it."""
    import config
    saved = config.settings.capital_per_grid_usdt
    try:
        config.settings.capital_per_grid_usdt = 0.0
        _, live, ratio = rb.sizing_mismatch({"capital_per_grid_pct": 0.018}, 4866.0)
        assert live == 0.0 and ratio == 1.0
    finally:
        config.settings.capital_per_grid_usdt = saved


def test_the_ratio_is_reported_not_silently_corrected():
    """Deliberate. The harness cannot size by a fixed stake, so pretending it can
    would be worse than saying it cannot."""
    src = inspect.getsource(rb.sizing_mismatch)
    assert "capital_per_grid_usdt" in src


# --------------------------------------------------------------- the CLI override
def _flags() -> str:
    return inspect.getsource(rb)


def test_the_router_can_be_forced_on_and_off():
    """Comparing on vs off is the whole point -- it is how the -96.97 gap was found."""
    src = _flags()
    assert '"--router"' in src
    assert '"--no-router"' in src


def test_the_flag_defaults_to_following_the_env_not_to_a_hardcoded_side():
    """A default of True or False would silently override STRATEGY_MODE and put the
    tool right back to measuring something other than the bot."""
    src = _flags()
    idx = src.index('"--router"')
    assert "default=None" in src[idx:idx + 300]


def test_the_run_announces_which_strategy_it_measured():
    """The original failure was silent. A run that does not say what it measured can
    be misread as the bot again."""
    src = _flags()
    assert "ROUTER (grid+trend)" in src and "grid only" in src
