"""Tests for src.gamma_closure: two-moment Gamma and shifted-Gamma closures.

The formula tests use a hand-built synthetic `cm` dict (as returned by
src.moment_pde.conditional_moments) so the expected alpha/theta/zeta values
can be checked against the exact closed-form paper formulas (Eq. 32, 45-46),
independent of any PDE solve. An additional small end-to-end test exercises
the closure against a real (small-grid) moment PDE solve.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.dupire_surface import build_surface
from src.gamma_closure import price_letf_call, shifted_gamma_params, two_moment_params
from src.gamma_closure import martingale_correction as gamma_martingale_correction
from src.moment_pde import conditional_moments, solve_moments


def _synthetic_cm() -> dict:
    """Three nodes: [0] stochastic+admissible, [1] stochastic+non-admissible,
    [2] deterministic (v == 0)."""
    m = np.array([0.01, 0.01, 0.01])
    v = np.array([0.0004, 0.0004, 0.0])
    mu3 = np.array([4e-5, 8e-6, 0.0])  # lambda3 = 5 (admissible), 1 (not), n/a
    resolved = np.array([True, True, True])
    p_T = np.array([1.0, 1.0, 1.0])
    return {"m": m, "v": v, "mu3": mu3, "resolved": resolved, "p_T": p_T}


def test_two_moment_params_formula() -> None:
    cm = _synthetic_cm()
    params = two_moment_params(cm, variance_negligible_relative=1e-6)

    stochastic = params.stochastic
    assert stochastic.tolist() == [True, True, False]
    m, v = cm["m"], cm["v"]
    expected_alpha = m[stochastic] ** 2 / v[stochastic]
    expected_theta = v[stochastic] / m[stochastic]
    assert params.alpha[stochastic] == pytest.approx(expected_alpha)
    assert params.theta[stochastic] == pytest.approx(expected_theta)
    assert np.all(params.zeta == 0.0)
    assert params.deterministic.tolist() == [False, False, True]


def test_shifted_gamma_params_admissible_node() -> None:
    cm = _synthetic_cm()
    params = shifted_gamma_params(cm, variance_negligible_relative=1e-6)

    # Node 0: lambda3 = 5, cv = 2.0, admissible since 5 >= 2*2.
    assert params.alpha[0] == pytest.approx(4.0 / 5.0 ** 2)
    assert params.theta[0] == pytest.approx(5.0 * np.sqrt(0.0004) / 2.0)
    assert params.zeta[0] == pytest.approx(0.01 - 2.0 * np.sqrt(0.0004) / 5.0)
    assert params.zeta[0] >= 0.0


def test_shifted_gamma_params_non_admissible_falls_back_to_two_moment() -> None:
    cm = _synthetic_cm()
    base = two_moment_params(cm, variance_negligible_relative=1e-6)
    shifted = shifted_gamma_params(cm, variance_negligible_relative=1e-6)

    # Node 1: lambda3 = 1, cv = 2.0, 1 < 2*2 -> not admissible -> falls back.
    assert shifted.alpha[1] == pytest.approx(base.alpha[1])
    assert shifted.theta[1] == pytest.approx(base.theta[1])
    assert shifted.zeta[1] == pytest.approx(0.0)


def test_shifted_gamma_params_admissible_fraction_and_gap() -> None:
    cm = _synthetic_cm()
    params = shifted_gamma_params(cm, variance_negligible_relative=1e-6)

    assert 0.0 <= params.admissible_fraction <= 1.0
    assert params.admissible_fraction == pytest.approx(0.5)
    assert np.all(params.lambda3_minus_2cv_on_A >= 0.0)
    assert params.lambda3_minus_2cv_on_A == pytest.approx(np.array([1.0]))


def test_shifted_gamma_params_zeta_non_negative_on_admissible_set() -> None:
    cm = _synthetic_cm()
    params = shifted_gamma_params(cm, variance_negligible_relative=1e-6)
    admissible = params.stochastic & (params.zeta != 0.0)
    assert np.all(params.zeta[admissible] >= 0.0)


def test_closure_end_to_end_price_is_sane(small_moment_grid) -> None:
    surface = build_surface("SPX")
    T = 0.25
    beta, gamma_coef = 3.0, 3.0
    moments = solve_moments(surface, T, small_moment_grid, max_n=2)
    cm = conditional_moments(moments, density_cutoff_relative=1e-6)
    params = two_moment_params(cm, variance_negligible_relative=1e-10)

    c_T = gamma_martingale_correction(moments.x_grid, params, beta, gamma_coef)
    assert np.isfinite(c_T)

    F_T_L, D_T = 100.0, 0.99
    prices = [price_letf_call(moments.x_grid, params, beta, gamma_coef, K, F_T_L, D_T, c_T)
              for K in (70.0, 100.0, 130.0)]
    assert all(p >= 0.0 for p in prices)
    assert prices[0] >= prices[1] >= prices[2]
