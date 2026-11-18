"""Tests for src.joint_law_pde: forward joint-law Fokker-Planck PDE solver.

All grids here use the small `small_joint_law_grid` fixture (n_x=80, n_i=40,
steps_per_year=50, n_sigma=6) so each solve runs in well under a second.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.dupire_surface import build_surface
from src.joint_law_pde import (martingale_correction, price_letf_call, solve_joint_law,
                                total_mass, x_domain)


@pytest.fixture(scope="module")
def surface():
    return build_surface("SPX")


def test_x_domain_is_ascending_and_symmetric(surface) -> None:
    grid = x_domain(surface, T=0.5, n_sigma=6.0, n_x=80)
    assert grid.shape == (80,)
    assert np.all(np.diff(grid) > 0)
    assert grid[0] == pytest.approx(-grid[-1], rel=1e-8)


def test_solve_joint_law_total_mass_close_to_one(surface, small_joint_law_grid) -> None:
    result = solve_joint_law(surface, T=0.25, grid=small_joint_law_grid)
    mass = total_mass(result)
    # A coarse unit-test grid (n_x=80, n_i=40) loses some mass at the i_max
    # truncation boundary (see src/joint_law_pde.py docstring); a generous
    # 10% tolerance still catches gross mass-conservation regressions.
    assert mass == pytest.approx(1.0, abs=0.1)


def test_solve_joint_law_grids_and_density_shape(surface, small_joint_law_grid) -> None:
    result = solve_joint_law(surface, T=0.25, grid=small_joint_law_grid)
    assert np.all(np.diff(result.x_grid) > 0)
    assert np.all(np.diff(result.i_grid) > 0)
    assert result.i_grid[0] >= 0.0
    assert result.density.shape == (small_joint_law_grid["n_x"], small_joint_law_grid["n_i"])
    assert np.all(result.density >= -1e-10)
    assert result.T == pytest.approx(0.25)
    assert result.index_name == "SPX"


def test_x_marginal_variance_grows_with_maturity(surface, small_joint_law_grid) -> None:
    result_short = solve_joint_law(surface, T=0.1, grid=small_joint_law_grid)
    result_long = solve_joint_law(surface, T=1.0, grid=small_joint_law_grid)

    def x_marginal_variance(result) -> float:
        marginal = result.density.sum(axis=1) * result.di
        mass = marginal.sum() * result.dx
        mean = (result.x_grid * marginal).sum() * result.dx / mass
        var = (((result.x_grid - mean) ** 2) * marginal).sum() * result.dx / mass
        return float(var)

    var_short = x_marginal_variance(result_short)
    var_long = x_marginal_variance(result_long)
    assert var_long > var_short


def test_martingale_correction_is_finite(surface, small_joint_law_grid) -> None:
    result = solve_joint_law(surface, T=0.25, grid=small_joint_law_grid)
    c_T = martingale_correction(result, beta=3.0, gamma_coef=3.0)
    assert np.isfinite(c_T)


def test_price_letf_call_non_negative_and_decreasing_in_strike(surface, small_joint_law_grid) -> None:
    result = solve_joint_law(surface, T=0.25, grid=small_joint_law_grid)
    beta, gamma_coef = 3.0, 3.0
    c_T = martingale_correction(result, beta, gamma_coef)
    F_T_L, D_T = 100.0, 0.99

    prices = [price_letf_call(result, beta, gamma_coef, K, F_T_L, D_T, c_T)
              for K in (70.0, 100.0, 130.0)]
    assert all(p >= 0.0 for p in prices)
    assert prices[0] >= prices[1] >= prices[2]
