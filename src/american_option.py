"""American-to-European implied-volatility conversion for listed QQQ/TQQQ
option quotes (paper Section 13.5.1: "listed American-style option quotes ...
must be converted to European implied volatilities").

The paper does not specify the conversion method beyond naming
"Bjerksund-Stensland or tree-based early exercise premium" as examples
(flagged uncertainty). This module uses a Cox-Ross-Rubinstein binomial tree
(Cox, Ross & Rubinstein, "Option Pricing: A Simplified Approach", 1979): the
flat volatility that reproduces the American market price in the tree is
found by bisection/Brent root-finding, and that same volatility is then the
"European implied volatility" fed to the Dupire calibration, i.e. the
volatility input for which the tree's American price matches the quote is
used as the corresponding Black-Scholes European volatility. This is the
tree-based early-exercise-premium adjustment referenced in the paper: the
American price is matched by the tree; the European equivalent is read off
the same volatility parameter under the (American-tree-consistent)
Black-Scholes dynamics.

Context7 was consulted for American option pricing (QuantLib wrappers); no
lightweight dependency-free CRR tree is exposed by the installed libraries in
a form usable without a full QuantLib build, so the standard CRR recursion is
implemented directly from Cox-Ross-Rubinstein (1979).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from src.config import IV_BRENTQ_BRACKET, IV_BRENTQ_TOL


def crr_american_price(S0: float, K: float, T: float, r: float, q: float, sigma: float,
                        option_type: str, n_steps: int = 200) -> float:
    """Cox-Ross-Rubinstein binomial-tree price of an American option.

    Vectorised backward induction: terminal payoffs are built once, then one
    array operation per step compares the European continuation value with
    the immediate-exercise payoff (Cox, Ross & Rubinstein, 1979, Eq. 1-7).
    """
    if T <= 0:
        intrinsic = max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
        return intrinsic
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 1e-10), 1 - 1e-10)  # guard against sigma too small/large for this dt

    j = np.arange(n_steps + 1)
    S_T = S0 * u ** (n_steps - j) * d ** j
    if option_type == "call":
        values = np.maximum(S_T - K, 0.0)
    else:
        values = np.maximum(K - S_T, 0.0)

    for step in range(n_steps - 1, -1, -1):
        j = np.arange(step + 1)
        S_t = S0 * u ** (step - j) * d ** j
        continuation = disc * (p * values[:step + 1] + (1 - p) * values[1:step + 2])
        intrinsic = np.maximum(S_t - K, 0.0) if option_type == "call" else np.maximum(K - S_t, 0.0)
        values = np.maximum(continuation, intrinsic)
    return float(values[0])


def american_implied_vol(price: float, S0: float, K: float, T: float, r: float, q: float,
                          option_type: str, n_steps: int = 200) -> float | None:
    """Flat volatility for which the CRR American tree price matches ``price``.

    Returns ``None`` when the quote is below intrinsic value or outside the
    bracketed vol range (stale/crossed quote), rather than raising, so the
    caller can drop that single quote.
    """
    lo, hi = IV_BRENTQ_BRACKET
    intrinsic = max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
    if price < intrinsic - 1e-8:
        return None

    def objective(sigma: float) -> float:
        return crr_american_price(S0, K, T, r, q, sigma, option_type, n_steps) - price

    try:
        f_lo, f_hi = objective(lo), objective(hi)
        if f_lo > 0 or f_hi < 0 or not np.isfinite(f_lo) or not np.isfinite(f_hi):
            return None
        return brentq(objective, lo, hi, xtol=IV_BRENTQ_TOL)
    except (ValueError, RuntimeError):
        return None
