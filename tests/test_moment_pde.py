"""Tests for src.moment_pde: forward moment PDE system (M_0..M_3)."""
from __future__ import annotations

import numpy as np
import pytest

from src.dupire_surface import build_surface
from src.moment_pde import conditional_moments, solve_moments


@pytest.fixture(scope="module")
def surface():
    return build_surface("SPX")


def test_m0_integrates_to_one(surface, small_moment_grid) -> None:
    result = solve_moments(surface, T=0.25, grid=small_moment_grid, max_n=3)
    M0 = result.M[0]
    mass = np.trapz(M0, result.x_grid)
    assert mass == pytest.approx(1.0, abs=0.1)


def test_higher_moments_non_negative_on_average(surface, small_moment_grid) -> None:
    result = solve_moments(surface, T=0.25, grid=small_moment_grid, max_n=3)
    for n in (1, 2, 3):
        assert result.M[n].sum() >= -1e-6


def test_moment_result_fields(surface, small_moment_grid) -> None:
    result = solve_moments(surface, T=0.25, grid=small_moment_grid, max_n=2)
    assert set(result.M.keys()) == {0, 1, 2}
    assert result.x_grid.shape == (small_moment_grid["n_x"],)
    assert result.T == pytest.approx(0.25)
    assert result.index_name == "SPX"
    assert np.all(np.isfinite(result.M[0]))
    assert np.all(np.isfinite(result.M[1]))
    assert np.all(np.isfinite(result.M[2]))


def test_conditional_moments_returns_expected_keys(surface, small_moment_grid) -> None:
    result = solve_moments(surface, T=0.25, grid=small_moment_grid, max_n=3)
    cm = conditional_moments(result, density_cutoff_relative=1e-6)
    expected_keys = {"p_T", "resolved", "r1", "m", "r2", "v", "r3", "mu3"}
    assert expected_keys.issubset(cm.keys())


def test_conditional_variance_non_negative_on_resolved_set(surface, small_moment_grid) -> None:
    result = solve_moments(surface, T=0.25, grid=small_moment_grid, max_n=3)
    cm = conditional_moments(result, density_cutoff_relative=1e-6)
    v, resolved = cm["v"], cm["resolved"]
    assert np.all(v[resolved] >= -1e-12)


def test_conditional_moments_resolved_mask_near_peak_density(surface, small_moment_grid) -> None:
    result = solve_moments(surface, T=0.25, grid=small_moment_grid, max_n=3)
    cm = conditional_moments(result, density_cutoff_relative=1e-6)
    # The node closest to x=0 (near the peak of the terminal density for a
    # short maturity, roughly centred local vol) must be resolved.
    j0 = int(np.argmin(np.abs(result.x_grid)))
    assert cm["resolved"][j0]
