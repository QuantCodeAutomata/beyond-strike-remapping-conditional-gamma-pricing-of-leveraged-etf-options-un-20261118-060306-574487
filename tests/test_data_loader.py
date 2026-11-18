"""Tests for data.data_loader: Massive REST API client for QQQ/TQQQ data.

Only two tests touch the network, both gated behind the ``market_data``
marker and skipped unless ``MASSIVE_API_KEY`` is set in the environment (see
conftest.py for the marker registration). The remaining tests are pure unit
tests that never make a network call.
"""
from __future__ import annotations

import os

import pytest

from data.data_loader import MassiveDataError, _api_key, spot_price_on


def test_api_key_raises_informative_error_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.chdir("/tmp")  # avoid picking up a stray .env in the repo root
    with pytest.raises(MassiveDataError, match="MASSIVE_API_KEY"):
        _api_key()


def test_api_key_returns_value_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASSIVE_API_KEY", "dummy-test-key")
    assert _api_key() == "dummy-test-key"


@pytest.mark.market_data
@pytest.mark.skipif(not os.environ.get("MASSIVE_API_KEY"), reason="MASSIVE_API_KEY not set")
def test_spot_price_on_qqq_live() -> None:
    price = spot_price_on("QQQ", "2026-06-22")
    assert isinstance(price, float)
    assert price > 0.0
