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
