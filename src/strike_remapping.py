"""First-order strike remapping for LETF implied volatility (paper Eq. 13-14).

    V_T* = exp(-gamma a_0 T),                 a_0 = sigma_S(T, F_T^S)^2
    K*   = F_T^S * (K / (F_T^L V_T*))^(1/beta)
    sigma_L(T, K) = |beta| * sigma_S(T, K*)

This closed-form map evaluates the underlying Black implied-volatility
surface at a leverage-adjusted strike, using the deterministic base-model
variance a_0 T in place of the true random variance drag I_T (paper Section
4, Eq. 13-14; Section 13.3.1).
"""
from __future__ import annotations

import numpy as np

from src.dupire_surface import LocalVolSurface


def atmf_variance(surface: LocalVolSurface, T: float) -> float:
    """a_0 = sigma_S(T, F_T^S)^2, the ATMF (k=0) Black implied variance."""
    return float(surface.implied_vol(np.array([0.0]), T)[0] ** 2)


def remapped_strike(K: np.ndarray, beta: float, gamma_coef: float, T: float,
                     F_T_S: float, F_T_L: float, a_0: float) -> np.ndarray:
    """K* = F_T^S * (K / (F_T^L * V_T*))^(1/beta), V_T* = exp(-gamma a_0 T)."""
    V_T_star = np.exp(-gamma_coef * a_0 * T)
    return F_T_S * (K / (F_T_L * V_T_star)) ** (1.0 / beta)


def remapped_implied_vol(surface: LocalVolSurface, K: np.ndarray, beta: float, gamma_coef: float,
                          T: float, F_T_S: float, F_T_L: float) -> np.ndarray:
    """sigma_L(T, K) = |beta| * sigma_S(T, K*), the first-order remapped LETF
    Black implied volatility at strikes K (paper Eq. 13-14)."""
    a_0 = atmf_variance(surface, T)
    K_star = remapped_strike(np.asarray(K, dtype=float), beta, gamma_coef, T, F_T_S, F_T_L, a_0)
    k_star = np.log(K_star / F_T_S)
    sigma_S = surface.implied_vol(k_star, T)
    return abs(beta) * sigma_S
