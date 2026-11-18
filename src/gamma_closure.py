"""Conditional-Gamma and conditional shifted-Gamma closed-form LETF pricing.

Implements the two-moment Gamma closure (paper Section 7, equations 31-33) and
its three-moment shifted-Gamma extension (paper Section 8, Theorem 8.1,
equations 45-46), both conditioned on the terminal underlying log-return X_T
and sharing the martingale normalisation of Section 10.1/10.2.

The two closures are unified here by a single per-node pair (zeta(x), and
Gamma parameters alpha(x)/theta(x)): the two-moment Gamma is the special case
zeta(x) = 0 everywhere; the shifted-Gamma sets zeta(x), alpha_s(x), theta_s(x)
on the admissible set A = {x : lambda_3(x) >= 2 cv(x)} and falls back to the
(zeta=0) two-moment parameters elsewhere. At the boundary of A, zeta -> 0 and
alpha_s/theta_s -> alpha/theta, so the hybrid closure is continuous
(implementation_notes, Section 8.2 Remark 8.2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammainc

from src.moment_pde import MomentResult, conditional_moments


@dataclass
class ClosureParams:
    x_grid: np.ndarray
    p_T: np.ndarray
    resolved: np.ndarray
    deterministic: np.ndarray
    stochastic: np.ndarray
    m: np.ndarray
    alpha: np.ndarray
    theta: np.ndarray
    zeta: np.ndarray
    admissible_fraction: float
    lambda3_minus_2cv_on_A: np.ndarray


def two_moment_params(cm: dict, variance_negligible_relative: float) -> ClosureParams:
    """Two-moment Gamma closure parameters (paper Section 7.1, Eq. 32):
    alpha(x) = m(x)^2 / v(x), theta(x) = v(x) / m(x). zeta(x) = 0 everywhere."""
    x_grid = None  # filled by caller via conditional_gamma_prices
    m, v, resolved, p_T = cm["m"], cm["v"], cm["resolved"], cm["p_T"]
    v_cutoff = variance_negligible_relative * max(float(v[resolved].mean()) if resolved.any() else 0.0, 1e-300)
    deterministic = resolved & (v <= v_cutoff) & (m > 0)
    stochastic = resolved & (~deterministic) & (m > 0) & (v > 0)
    alpha = np.zeros_like(m)
    theta = np.zeros_like(m)
    alpha[stochastic] = m[stochastic] ** 2 / v[stochastic]
    theta[stochastic] = v[stochastic] / m[stochastic]
    zeta = np.zeros_like(m)
    return ClosureParams(x_grid=x_grid, p_T=p_T, resolved=resolved, deterministic=deterministic,
                          stochastic=stochastic, m=m, alpha=alpha, theta=theta, zeta=zeta,
                          admissible_fraction=0.0, lambda3_minus_2cv_on_A=np.array([]))


def shifted_gamma_params(cm: dict, variance_negligible_relative: float) -> ClosureParams:
    """Three-moment shifted-Gamma closure (paper Section 8.1, Theorem 8.1):

        lambda_3(x) = mu_3(x) / v(x)^(3/2),  cv(x) = sqrt(v(x)) / m(x)
        admissible set A = {x : lambda_3(x) >= 2 cv(x)}
        alpha_s(x) = 4 / lambda_3(x)^2
        theta_s(x) = lambda_3(x) sqrt(v(x)) / 2
        zeta(x)    = m(x) - 2 sqrt(v(x)) / lambda_3(x)

    On the complement of A (and on deterministic nodes) falls back to the
    two-moment Gamma closure (zeta = 0).
    """
    base = two_moment_params(cm, variance_negligible_relative)
    m, v, mu3, resolved = cm["m"], cm["v"], cm["mu3"], cm["resolved"]
    stochastic = base.stochastic
    with np.errstate(divide="ignore", invalid="ignore"):
        lambda3 = np.where(stochastic, mu3 / np.maximum(v, 1e-300) ** 1.5, 0.0)
        cv = np.where(stochastic, np.sqrt(np.maximum(v, 0.0)) / np.where(m > 0, m, np.nan), 0.0)
    admissible = stochastic & (lambda3 >= 2 * cv) & (lambda3 > 0)

    alpha_s = base.alpha.copy()
    theta_s = base.theta.copy()
    zeta = np.zeros_like(m)
    alpha_s[admissible] = 4.0 / lambda3[admissible] ** 2
    theta_s[admissible] = lambda3[admissible] * np.sqrt(v[admissible]) / 2.0
    zeta[admissible] = m[admissible] - 2.0 * np.sqrt(v[admissible]) / lambda3[admissible]

    gap = (lambda3[admissible] - 2 * cv[admissible]) if admissible.any() else np.array([])
    admissible_fraction = float(admissible.sum()) / max(int(stochastic.sum()), 1)

    return ClosureParams(x_grid=base.x_grid, p_T=base.p_T, resolved=resolved,
                          deterministic=base.deterministic, stochastic=stochastic, m=m,
                          alpha=alpha_s, theta=theta_s, zeta=zeta,
                          admissible_fraction=admissible_fraction, lambda3_minus_2cv_on_A=gap)


def _log_phi(x_grid: np.ndarray, params: ClosureParams, gamma_coef: float) -> np.ndarray:
    """log phi(x) such that the unnormalised forward payoff kernel is
    F_T^L * exp(beta x + log_phi(x)): the per-node moment-generating-function
    factor of -gamma I_T under the fitted (shifted-)Gamma law."""
    log_phi = np.zeros_like(x_grid)
    s, d = params.stochastic, params.deterministic
    log_phi[s] = -params.alpha[s] * np.log1p(gamma_coef * params.theta[s]) - gamma_coef * params.zeta[s]
    log_phi[d] = -gamma_coef * (params.m[d] + params.zeta[d])
    return log_phi


def martingale_correction(x_grid: np.ndarray, params: ClosureParams, beta: float, gamma_coef: float) -> float:
    """c_T = log( integral e^{beta x} phi(x) p_T(x) dx ), paper Section 10.1/10.2."""
    log_phi = _log_phi(x_grid, params, gamma_coef)
    mask = params.resolved
    dx = x_grid[1] - x_grid[0]
    integrand = np.where(mask, np.exp(beta * x_grid + log_phi) * params.p_T, 0.0)
    raw = integrand.sum() * dx
    return float(np.log(max(raw, 1e-300)))


def conditional_call(x_grid: np.ndarray, params: ClosureParams, beta: float, gamma_coef: float,
                      K: float, F_T_L: float, c_T: float) -> np.ndarray:
    """Conditional call value A(x; K) at every x node (paper Eq. 31-33, 45-46),
    with the martingale-normalised exercise boundary
    h_corrected(x) = (beta x - gamma zeta(x) - c_T - k) / gamma, k = log(K/F_T^L)."""
    k = np.log(K / F_T_L)
    h = (beta * x_grid - gamma_coef * params.zeta - c_T - k) / gamma_coef
    out = np.zeros_like(x_grid)

    s = params.stochastic & (h > 0)
    if s.any():
        theta_s, alpha_s = params.theta[s], params.alpha[s]
        one_plus = 1.0 + gamma_coef * theta_s
        forward_term = F_T_L * np.exp(beta * x_grid[s] - gamma_coef * params.zeta[s] - c_T) * one_plus ** (-alpha_s)
        p_upper = gammainc(alpha_s, one_plus * h[s] / theta_s)
        p_lower = gammainc(alpha_s, h[s] / theta_s)
        out[s] = forward_term * p_upper - K * p_lower

    d = params.deterministic & (h > 0)
    if d.any():
        out[d] = F_T_L * np.exp(beta * x_grid[d] - gamma_coef * (params.m[d] + params.zeta[d]) - c_T) - K

    return out


def price_letf_call(x_grid: np.ndarray, params: ClosureParams, beta: float, gamma_coef: float,
                     K: float, F_T_L: float, D_T: float, c_T: float) -> float:
    """C_L(K,T) = D_T * integral A(x;K) p_T(x) dx (paper Eq. 31, step 9 of exp_2)."""
    dx = x_grid[1] - x_grid[0]
    call = conditional_call(x_grid, params, beta, gamma_coef, K, F_T_L, c_T)
    mask = params.resolved
    return D_T * float((np.where(mask, call * params.p_T, 0.0)).sum() * dx)
