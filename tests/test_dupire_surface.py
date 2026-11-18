"""Tests for src.dupire_surface: SVI total variance, LocalVolSurface, fitting."""
from __future__ import annotations

import numpy as np
import pytest

from src.dupire_surface import LocalVolSurface, build_surface, fit_svi_slice, svi_total_variance


SAMPLE_PARAMS = {"a": 0.02, "b": 0.15, "rho": -0.4, "m": -0.02, "sigma": 0.12}


def test_svi_total_variance_non_negative() -> None:
    k = np.linspace(-1.0, 1.0, 101)
    w = svi_total_variance(k, SAMPLE_PARAMS)
    assert np.all(w >= 0.0)


def test_svi_total_variance_convex_around_atm() -> None:
    k = np.array([-0.05, 0.0, 0.05])
    w = svi_total_variance(k, SAMPLE_PARAMS)
    # Convexity (discrete second difference >= 0) near the money, a basic
    # SVI smile-shape sanity check.
    second_diff = w[0] - 2 * w[1] + w[2]
    assert second_diff >= -1e-10


@pytest.mark.parametrize("index_name", ["SPX", "NKY"])
def test_build_surface_local_variance_positive_finite(index_name: str) -> None:
    surface = build_surface(index_name)
    assert isinstance(surface, LocalVolSurface)
    x_grid = np.linspace(-1.0, 1.0, 25)
    for t in (0.05, 0.25, 0.5, 1.0, 2.0):
        a = surface.local_variance(t, x_grid)
        assert np.all(np.isfinite(a))
        assert np.all(a > 0.0)


def test_build_surface_unknown_index_raises() -> None:
    with pytest.raises(ValueError):
        build_surface("NOT_AN_INDEX")


def test_build_surface_implied_vol_matches_svi_at_slice_maturity() -> None:
    surface = build_surface("SPX")
    T = 0.25  # the "3M" slice maturity, exactly on the calibrated grid
    k = np.array([-0.1, 0.0, 0.1])
    from src.config import SVI_PARAMS

    expected_w = svi_total_variance(k, SVI_PARAMS["SPX"]["3M"])
    expected_iv = np.sqrt(expected_w / T)
    actual_iv = surface.implied_vol(k, T)
    assert actual_iv == pytest.approx(expected_iv, rel=1e-8)


def test_fit_svi_slice_round_trip_on_synthetic_smile() -> None:
    k = np.linspace(-0.4, 0.4, 25)
    true_params = {"a": 0.03, "b": 0.2, "rho": -0.35, "m": -0.01, "sigma": 0.15}
    total_var = svi_total_variance(k, true_params)

    fitted = fit_svi_slice(k, total_var)
    fitted_w = svi_total_variance(k, fitted)

    assert fitted_w == pytest.approx(total_var, abs=1e-6)


def test_fit_svi_slice_handles_noisy_data() -> None:
    rng = np.random.default_rng(0)
    k = np.linspace(-0.3, 0.3, 30)
    true_params = {"a": 0.02, "b": 0.18, "rho": -0.3, "m": 0.0, "sigma": 0.1}
    total_var = svi_total_variance(k, true_params) + rng.normal(0.0, 1e-4, size=k.shape)

    fitted = fit_svi_slice(k, total_var)
    fitted_w = svi_total_variance(k, fitted)

    assert fitted_w == pytest.approx(total_var, abs=1e-2)
