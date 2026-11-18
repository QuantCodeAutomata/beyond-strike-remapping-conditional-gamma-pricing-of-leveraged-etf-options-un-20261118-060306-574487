"""Tests for src.pricing_pipeline: shared smile-pricing pipeline helpers."""
from __future__ import annotations

import numpy as np
import pytest

from src.dupire_surface import build_surface
from src.market_inputs import build_inputs
from src.pricing_pipeline import (conditional_gamma_smile, gamma_coef_of, joint_law_smile,
                                   strike_remapping_smile, strikes_from_log_moneyness)


def test_gamma_coef_of_formula() -> None:
    assert gamma_coef_of(3) == pytest.approx(3.0)
    assert gamma_coef_of(2) == pytest.approx(1.0)
    assert gamma_coef_of(-1) == pytest.approx(1.0)
    assert gamma_coef_of(-2) == pytest.approx(3.0)


def test_gamma_coef_of_shared_values_match_paper_claim() -> None:
    # Paper Section 13.1: beta=3 and beta=-2 share gamma=3; beta=2 and
    # beta=-1 share gamma=1.
    assert gamma_coef_of(3) == pytest.approx(gamma_coef_of(-2))
    assert gamma_coef_of(2) == pytest.approx(gamma_coef_of(-1))


def test_strikes_from_log_moneyness_zero_is_forward() -> None:
    F_T_L = 123.45
    strikes = strikes_from_log_moneyness(np.array([0.0]), F_T_L)
    assert strikes[0] == pytest.approx(F_T_L)


def test_strikes_from_log_moneyness_monotone() -> None:
    F_T_L = 100.0
    k = np.array([-0.5, -0.1, 0.0, 0.1, 0.5])
    strikes = strikes_from_log_moneyness(k, F_T_L)
    assert np.all(np.diff(strikes) > 0)
    assert strikes == pytest.approx(F_T_L * np.exp(k))


def test_joint_law_smile_produces_finite_positive_ivs() -> None:
    surface = build_surface("SPX")
    inputs = build_inputs("SPX", T=0.25)
    grid = {"n_x": 80, "n_i": 40, "steps_per_year": 50, "n_sigma": 6.0}
    k_grid = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])

    smile = joint_law_smile(surface, beta=3.0, T=0.25, grid=grid, inputs=inputs, k_grid=k_grid)

    assert smile["strikes"].shape == k_grid.shape
    assert smile["iv"].shape == k_grid.shape
    assert np.isfinite(smile["c_T"])
    finite_ivs = smile["iv"][np.isfinite(smile["iv"])]
    assert finite_ivs.size > 0
    assert np.all(finite_ivs > 0.0)


def test_conditional_gamma_smile_two_moment_and_shifted() -> None:
    surface = build_surface("SPX")
    inputs = build_inputs("SPX", T=0.25)
    moment_grid = {"n_x": 80, "steps_per_year": 50, "n_sigma": 6.0}
    k_grid = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])

    for shifted in (False, True):
        smile = conditional_gamma_smile(surface, beta=3.0, T=0.25, moment_grid=moment_grid,
                                         inputs=inputs, k_grid=k_grid, shifted=shifted,
                                         density_cutoff_relative=1e-6,
                                         variance_negligible_relative=1e-10)
        assert np.isfinite(smile["c_T"])
        finite_ivs = smile["iv"][np.isfinite(smile["iv"])]
        assert finite_ivs.size > 0
        assert np.all(finite_ivs > 0.0)


def test_strike_remapping_smile_shape_and_positivity() -> None:
    surface = build_surface("SPX")
    inputs = build_inputs("SPX", T=0.5)
    k_grid = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])

    smile = strike_remapping_smile(surface, beta=3.0, T=0.5, inputs=inputs, k_grid=k_grid)
    assert smile["iv"].shape == k_grid.shape
    assert np.all(smile["iv"] > 0.0)
    assert np.all(np.isfinite(smile["iv"]))
