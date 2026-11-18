"""Tests for src.monte_carlo: independent Monte Carlo LETF benchmark."""
from __future__ import annotations

import numpy as np
import pytest

from src.dupire_surface import build_surface
from src.monte_carlo import price_band, price_letf_call_mc, simulate_letf_paths


@pytest.fixture(scope="module")
def surface():
    return build_surface("SPX")


def test_simulate_letf_paths_runs_and_is_finite(surface) -> None:
    mc = simulate_letf_paths(surface, beta=3.0, T=0.25, steps_per_year=50, n_paths=5_000, seed=1234)
    assert mc.n_paths == 5_000
    assert mc.X_T.shape == (5_000,)
    assert mc.Y_T.shape == (5_000,)
    assert np.all(np.isfinite(mc.X_T))
    assert np.all(np.isfinite(mc.Y_T))


def test_price_letf_call_mc_deep_itm_close_to_intrinsic(surface) -> None:
    beta, T = 3.0, 0.25
    mc = simulate_letf_paths(surface, beta=beta, T=T, steps_per_year=50, n_paths=20_000, seed=42)
    F_T_L, D_T = 100.0, 0.99
    K = 10.0  # far below F_T_L for beta > 0: essentially always exercised

    price, stderr = price_letf_call_mc(mc, K, F_T_L, D_T)
    assert stderr > 0.0
    intrinsic_forward = D_T * (F_T_L - K)
    # Deep ITM, the call is close to its discounted intrinsic forward value
    # (residual optionality from remaining tail risk is small but nonzero).
    assert price == pytest.approx(intrinsic_forward, abs=6 * stderr + 0.5)


def test_standard_error_shrinks_with_more_paths(surface) -> None:
    beta, T = 3.0, 0.25
    F_T_L, D_T, K = 100.0, 0.99, 100.0

    mc_small = simulate_letf_paths(surface, beta=beta, T=T, steps_per_year=50, n_paths=5_000, seed=7)
    mc_large = simulate_letf_paths(surface, beta=beta, T=T, steps_per_year=50, n_paths=20_000, seed=7)

    _, se_small = price_letf_call_mc(mc_small, K, F_T_L, D_T)
    _, se_large = price_letf_call_mc(mc_large, K, F_T_L, D_T)

    assert se_large <= se_small * 1.5


def test_price_band_contains_price(surface) -> None:
    beta, T = 3.0, 0.25
    mc = simulate_letf_paths(surface, beta=beta, T=T, steps_per_year=50, n_paths=5_000, seed=99)
    F_T_L, D_T, K = 100.0, 0.99, 100.0

    price, lo, hi = price_band(mc, K, F_T_L, D_T)
    assert lo <= price <= hi
    assert lo < hi
