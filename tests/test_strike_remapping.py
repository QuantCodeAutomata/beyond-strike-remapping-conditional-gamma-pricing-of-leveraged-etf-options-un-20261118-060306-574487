"""Tests for src.strike_remapping: first-order LETF strike-remapping (Eq. 13-14)."""
from __future__ import annotations

import numpy as np
import pytest

from src.dupire_surface import build_surface
from src.pricing_pipeline import gamma_coef_of
from src.strike_remapping import atmf_variance, remapped_implied_vol, remapped_strike


@pytest.fixture(scope="module")
def surface():
    return build_surface("SPX")


def test_atmf_variance_positive(surface) -> None:
    for T in (0.25, 0.5, 1.0, 2.0):
        a_0 = atmf_variance(surface, T)
        assert a_0 > 0.0
        assert np.isfinite(a_0)


def test_remapped_strike_atmf_maps_back_to_underlying_atmf(surface) -> None:
    T = 0.5
    beta = 3.0
    gamma_coef = gamma_coef_of(beta)
    F_T_S, F_T_L = 5500.0, 100.0
    a_0 = atmf_variance(surface, T)

    K_atmf_letf = F_T_L  # K = F_T^L, i.e. k = 0 in LETF forward log-moneyness
    K_star = remapped_strike(np.array([K_atmf_letf]), beta, gamma_coef, T, F_T_S, F_T_L, a_0)

    V_T_star = np.exp(-gamma_coef * a_0 * T)
    # By construction V_T^* absorbs the drag exactly at the money, so
    # K = F_T^L * V_T^* maps back exactly to K* = F_T^S.
    K_atmf_absorbed = F_T_L * V_T_star
    K_star_absorbed = remapped_strike(np.array([K_atmf_absorbed]), beta, gamma_coef, T, F_T_S, F_T_L, a_0)
    assert K_star_absorbed[0] == pytest.approx(F_T_S, rel=1e-8)


def test_remapped_implied_vol_scales_with_beta(surface) -> None:
    T = 0.5
    F_T_S, F_T_L = 5500.0, 100.0
    strikes = np.array([F_T_L])

    beta2, beta3 = 2.0, 3.0
    iv2 = remapped_implied_vol(surface, strikes, beta2, gamma_coef_of(beta2), T, F_T_S, F_T_L)
    iv3 = remapped_implied_vol(surface, strikes, beta3, gamma_coef_of(beta3), T, F_T_S, F_T_L)

    # The leading-order map multiplies the underlying Black vol by |beta|;
    # evaluated at (nearly) the same underlying strike, iv3/iv2 ~= 3/2.
    ratio = iv3[0] / iv2[0]
    assert ratio == pytest.approx(1.5, rel=0.2)


def test_remapped_implied_vol_matches_abs_beta_times_underlying_vol(surface) -> None:
    T = 0.25
    F_T_S, F_T_L = 5500.0, 100.0
    beta = -2.0
    gamma_coef = gamma_coef_of(beta)
    strikes = np.array([80.0, 100.0, 120.0])

    ivs = remapped_implied_vol(surface, strikes, beta, gamma_coef, T, F_T_S, F_T_L)
    a_0 = atmf_variance(surface, T)
    K_star = remapped_strike(strikes, beta, gamma_coef, T, F_T_S, F_T_L, a_0)
    k_star = np.log(K_star / F_T_S)
    expected = abs(beta) * surface.implied_vol(k_star, T)
    assert ivs == pytest.approx(expected)
    assert np.all(ivs > 0.0)
