"""Tests for src.black_scholes: Black (1976) pricing and implied-vol inversion."""
from __future__ import annotations

import numpy as np
import pytest

from src.black_scholes import black_price, implied_vol, implied_vol_vec


def test_black_price_increasing_in_forward() -> None:
    low = black_price(F=90.0, K=100.0, T=1.0, sigma=0.2)
    high = black_price(F=110.0, K=100.0, T=1.0, sigma=0.2)
    assert high > low


def test_black_price_decreasing_in_strike() -> None:
    low_strike = black_price(F=100.0, K=90.0, T=1.0, sigma=0.2)
    high_strike = black_price(F=100.0, K=110.0, T=1.0, sigma=0.2)
    assert low_strike > high_strike


def test_black_price_non_negative() -> None:
    for K in (50.0, 100.0, 150.0, 200.0):
        price = black_price(F=100.0, K=K, T=1.0, sigma=0.3)
        assert price >= 0.0


def test_black_price_deep_otm_near_zero() -> None:
    price = black_price(F=100.0, K=1000.0, T=0.1, sigma=0.15)
    assert price == pytest.approx(0.0, abs=1e-8)


def test_black_price_put_call_parity() -> None:
    F, K, T, sigma, D = 100.0, 95.0, 0.5, 0.25, 0.97
    call = black_price(F, K, T, sigma, D, option_type="call")
    put = black_price(F, K, T, sigma, D, option_type="put")
    assert call - put == pytest.approx(D * (F - K), rel=1e-8, abs=1e-8)


def test_black_price_zero_vol_or_maturity_is_intrinsic() -> None:
    assert black_price(F=110.0, K=100.0, T=0.0, sigma=0.2) == pytest.approx(10.0)
    assert black_price(F=90.0, K=100.0, T=1.0, sigma=0.0) == pytest.approx(0.0)
    assert black_price(F=110.0, K=100.0, T=1.0, sigma=0.0) == pytest.approx(10.0)


def test_black_price_invalid_option_type_raises() -> None:
    with pytest.raises(ValueError):
        black_price(F=100.0, K=100.0, T=1.0, sigma=0.2, option_type="straddle")


@pytest.mark.parametrize("sigma", [0.05, 0.15, 0.3, 0.6, 1.0])
def test_implied_vol_round_trips(sigma: float) -> None:
    F, K, T, D = 100.0, 105.0, 0.75, 0.98
    price = black_price(F, K, T, sigma, D)
    recovered = implied_vol(price, F, K, T, D)
    assert recovered == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_round_trip_put() -> None:
    F, K, T, D, sigma = 100.0, 95.0, 0.5, 0.95, 0.22
    price = black_price(F, K, T, sigma, D, option_type="put")
    recovered = implied_vol(price, F, K, T, D, option_type="put")
    assert recovered == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_below_intrinsic_is_nan() -> None:
    F, K, T = 100.0, 50.0, 1.0
    # A price at/below the intrinsic forward value has no admissible vol.
    intrinsic = max(F - K, 0.0)
    result = implied_vol(intrinsic - 1.0, F, K, T)
    assert np.isnan(result)


def test_implied_vol_vec_matches_scalar() -> None:
    F, T, D = 100.0, 1.0, 1.0
    strikes = np.array([80.0, 100.0, 120.0])
    sigma = 0.2
    prices = np.array([black_price(F, K, T, sigma, D) for K in strikes])
    recovered = implied_vol_vec(prices, F, strikes, T, D)
    assert recovered == pytest.approx(np.full(3, sigma), abs=1e-6)
