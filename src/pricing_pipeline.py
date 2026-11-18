"""Shared LETF smile-pricing pipeline tying together the three methods compared
in the paper: the forward joint-law PDE reference (Section 11), the
conditional-(shifted-)Gamma closure (Sections 7-8), and the first-order strike
remapping (Section 4). Every experiment script calls these functions so that
the grid, normalisation and implied-volatility inversion are identical across
methods and experiments.
"""
from __future__ import annotations

import numpy as np

from src.black_scholes import implied_vol_vec
from src.dupire_surface import LocalVolSurface, build_surface
from src.gamma_closure import (ClosureParams, martingale_correction as gamma_martingale_correction,
                                price_letf_call as gamma_price_letf_call, shifted_gamma_params,
                                two_moment_params)
from src.joint_law_pde import (JointLawResult, martingale_correction as joint_law_martingale_correction,
                                price_letf_call as joint_law_price_letf_call, solve_joint_law)
from src.market_inputs import DeterministicInputs, build_inputs
from src.moment_pde import MomentResult, conditional_moments, solve_moments
from src.strike_remapping import remapped_implied_vol


def gamma_coef_of(beta: float) -> float:
    """gamma = beta (beta - 1) / 2, the variance-drag coefficient (paper Section 2)."""
    return beta * (beta - 1) / 2.0


def strikes_from_log_moneyness(k_grid: np.ndarray, F_T_L: float) -> np.ndarray:
    return F_T_L * np.exp(k_grid)


def joint_law_smile(surface: LocalVolSurface, beta: float, T: float, grid: dict,
                     inputs: DeterministicInputs, k_grid: np.ndarray) -> dict:
    """Full joint-law reference smile: solve the 2D PDE once, price every
    strike in k_grid, invert to Black implied volatility."""
    gamma_coef = gamma_coef_of(beta)
    result = solve_joint_law(surface, T, grid)
    c_T = joint_law_martingale_correction(result, beta, gamma_coef)
    strikes = strikes_from_log_moneyness(k_grid, inputs.F_T_L)
    prices = np.array([joint_law_price_letf_call(result, beta, gamma_coef, K, inputs.F_T_L, inputs.D_T, c_T)
                        for K in strikes])
    ivs = implied_vol_vec(prices, inputs.F_T_L, strikes, T, inputs.D_T)
    return {"k_grid": k_grid, "strikes": strikes, "prices": prices, "iv": ivs, "c_T": c_T,
            "result": result}


def conditional_gamma_smile(surface: LocalVolSurface, beta: float, T: float, moment_grid: dict,
                             inputs: DeterministicInputs, k_grid: np.ndarray, *, shifted: bool,
                             density_cutoff_relative: float, variance_negligible_relative: float) -> dict:
    """Two-moment Gamma (shifted=False) or three-moment shifted-Gamma (shifted=True)
    conditional closure smile, from a single forward moment PDE solve."""
    gamma_coef = gamma_coef_of(beta)
    max_n = 3 if shifted else 2
    moments = solve_moments(surface, T, moment_grid, max_n=max_n)
    cm = conditional_moments(moments, density_cutoff_relative)
    params: ClosureParams = (shifted_gamma_params(cm, variance_negligible_relative) if shifted
                              else two_moment_params(cm, variance_negligible_relative))
    c_T = gamma_martingale_correction(moments.x_grid, params, beta, gamma_coef)
    strikes = strikes_from_log_moneyness(k_grid, inputs.F_T_L)
    prices = np.array([gamma_price_letf_call(moments.x_grid, params, beta, gamma_coef, K, inputs.F_T_L,
                                              inputs.D_T, c_T) for K in strikes])
    ivs = implied_vol_vec(prices, inputs.F_T_L, strikes, T, inputs.D_T)
    return {"k_grid": k_grid, "strikes": strikes, "prices": prices, "iv": ivs, "c_T": c_T,
            "params": params, "moments": moments}


def strike_remapping_smile(surface: LocalVolSurface, beta: float, T: float,
                            inputs: DeterministicInputs, k_grid: np.ndarray) -> dict:
    """First-order remapped LETF implied-volatility smile (closed form, no PDE)."""
    gamma_coef = gamma_coef_of(beta)
    strikes = strikes_from_log_moneyness(k_grid, inputs.F_T_L)
    ivs = remapped_implied_vol(surface, strikes, beta, gamma_coef, T, inputs.F_T_S, inputs.F_T_L)
    return {"k_grid": k_grid, "strikes": strikes, "iv": ivs}
