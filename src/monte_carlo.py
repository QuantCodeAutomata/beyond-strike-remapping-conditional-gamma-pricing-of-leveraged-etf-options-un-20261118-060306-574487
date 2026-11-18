"""Independent Monte Carlo benchmark for the LETF dynamics (paper Eq. 71-73).

Simulates the coupled SDE system by log-Euler discretisation (Eq. 72), using
the *same* Brownian increment Z in both the underlying log-return line and the
LETF log-return line, and accumulates Y_T directly as its own process -- it is
never reconstructed from the pathwise factorisation identity Y_T = beta X_T -
gamma I_T. This makes the benchmark independent of the joint-law PDE and the
conditional-Gamma closure, both of which rely on that identity (paper Section
13.2.2).

    X_{t+dt} = X_t - 0.5 a(t,X_t) dt + sqrt(a(t,X_t) dt) Z
    Y_{t+dt} = Y_t - 0.5 beta^2 a(t,X_t) dt + beta sqrt(a(t,X_t) dt) Z
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from src.dupire_surface import LocalVolSurface


@dataclass
class MonteCarloResult:
    X_T: np.ndarray
    Y_T: np.ndarray
    n_paths: int
    n_steps: int
    seed: int


def simulate_letf_paths(surface: LocalVolSurface, beta: float, T: float, steps_per_year: int,
                         n_paths: int, seed: int) -> MonteCarloResult:
    """Simulate n_paths of (X_T, Y_T) via the log-Euler scheme (72)."""
    rng = np.random.default_rng(seed)
    n_steps = max(1, int(round(steps_per_year * T)))
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    X = np.zeros(n_paths)
    Y = np.zeros(n_paths)
    for step in range(n_steps):
        t_mid = (step + 0.5) * dt
        a_x = surface.local_variance(t_mid, X)
        a_x = np.maximum(a_x, 0.0)
        Z = rng.standard_normal(n_paths)
        sqrt_a = np.sqrt(a_x)
        X = X - 0.5 * a_x * dt + sqrt_a * sqrt_dt * Z
        Y = Y - 0.5 * beta ** 2 * a_x * dt + beta * sqrt_a * sqrt_dt * Z
    return MonteCarloResult(X_T=X, Y_T=Y, n_paths=n_paths, n_steps=n_steps, seed=seed)


def price_letf_call_mc(mc: MonteCarloResult, K: float, F_T_L: float, D_T: float) -> tuple[float, float]:
    """Price and Monte Carlo standard error of C_L(K,T) = D_T E[(F_T^L e^{Y_T} - K)^+]
    (paper Eq. 73), estimated by the sample mean over simulated paths."""
    payoff = np.maximum(F_T_L * np.exp(mc.Y_T) - K, 0.0)
    price = D_T * float(payoff.mean())
    stderr = D_T * float(payoff.std(ddof=1) / np.sqrt(mc.n_paths))
    return price, stderr


def price_band(mc: MonteCarloResult, K: float, F_T_L: float, D_T: float,
               z_score: float = 1.96) -> tuple[float, float, float]:
    """Price and its (price - z*se, price + z*se) Monte Carlo sampling band."""
    price, stderr = price_letf_call_mc(mc, K, F_T_L, D_T)
    return price, price - z_score * stderr, price + z_score * stderr
