"""Black (forward measure) option pricing and implied-volatility inversion.

Standard Black formula for a forward price F maturing at T with discount
factor D. Used throughout the repository to express all model and market
prices as Black implied volatilities (paper Section 13.1).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from src.config import IV_BRENTQ_BRACKET, IV_BRENTQ_TOL


def black_price(F: float, K: float, T: float, sigma: float, D: float = 1.0,
                 option_type: str = "call") -> float:
    """Black (1976) forward price of a European option.

    Parameters
    ----------
    F : forward price of the underlying for maturity T
    K : strike
    T : year fraction to maturity
    sigma : Black volatility
    D : discount factor to maturity
    option_type : "call" or "put"
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
        return D * intrinsic
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if option_type == "call":
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    if option_type == "put":
        return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    raise ValueError(f"unknown option_type {option_type!r}")


def implied_vol(price: float, F: float, K: float, T: float, D: float = 1.0,
                 option_type: str = "call") -> float:
    """Black implied volatility solving black_price(..., sigma) == price via brentq."""
    intrinsic = D * (max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0))
    if price <= intrinsic + 1e-14:
        return float("nan")
    lo, hi = IV_BRENTQ_BRACKET

    def objective(sigma: float) -> float:
        return black_price(F, K, T, sigma, D, option_type) - price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    return brentq(objective, lo, hi, xtol=IV_BRENTQ_TOL)


def implied_vol_vec(prices: np.ndarray, F: float, strikes: np.ndarray, T: float,
                     D: float = 1.0, option_type: str = "call") -> np.ndarray:
    """Vectorised wrapper of implied_vol over an array of strikes/prices."""
    out = np.full(len(strikes), np.nan)
    for idx, (price, K) in enumerate(zip(prices, strikes)):
        out[idx] = implied_vol(price, F, K, T, D, option_type)
    return out
