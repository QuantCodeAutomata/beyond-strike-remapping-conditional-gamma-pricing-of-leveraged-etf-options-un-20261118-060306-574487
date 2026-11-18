"""Triangular forward moment PDE system for the conditional-Gamma closure.

Implements paper equation (55), Section 9.2: for n = 0, 1, 2, 3,

    dM_n/dt = (1/2) d/dx[a(t,x) M_n] + (1/2) d^2/dx^2[a(t,x) M_n] + n a(t,x) M_{n-1}

with M_{-1} = 0, M_0(0,x) = delta(x), M_n(0,x) = 0 for n >= 1. All four
equations share the same forward operator; M_{n-1} is a known source for M_n.
M_0(T,x) is the terminal marginal density p_T(x) of the underlying log-return
X_T (the x-marginal of the joint-law PDE with no i-dependence).

Discretisation: the shared backward-Euler banded operator (I - dt*A) is
assembled and factorised once per time step (paper implementation_notes,
Section 8.2 Remark 8.2) and reused, via back-substitution, for every M_n; the
source term n*a*M_{n-1} is evaluated using the freshly updated M_{n-1} of the
same step (sequential IMEX treatment), since the methodology does not specify
whether the source is frozen at the old or new time level. The x-grid and
time stepping reuse the same construction as the
2D joint-law solver (src.joint_law_pde), so both solves agree on (t,x).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded

from src.dupire_surface import LocalVolSurface
from src.joint_law_pde import diffusion_banded, x_domain


@dataclass
class MomentResult:
    x_grid: np.ndarray
    dx: float
    T: float
    M: dict  # n -> np.ndarray of shape (n_x,), terminal M_n(T, x)
    index_name: str
    density_cutoff: float = field(default=0.0)


def solve_moments(surface: LocalVolSurface, T: float, grid: dict, max_n: int = 3) -> MomentResult:
    """Solve M_0..M_{max_n} on [0, T] with the grid configuration `grid`
    (keys: n_x, steps_per_year, n_sigma)."""
    n_x = grid["n_x"]
    steps_per_year = grid["steps_per_year"]
    n_sigma = grid["n_sigma"]

    x_grid = x_domain(surface, T, n_sigma, n_x)
    dx = x_grid[1] - x_grid[0]
    n_steps = max(1, int(round(steps_per_year * T)))
    dt = T / n_steps

    M = {n: np.zeros(n_x) for n in range(max_n + 1)}
    j0 = int(np.argmin(np.abs(x_grid)))
    M[0][j0] = 1.0 / dx  # discrete approximation to delta(x)

    for step in range(n_steps):
        t_mid = (step + 0.5) * dt
        a_x = surface.local_variance(t_mid, x_grid)
        ab = diffusion_banded(a_x, dx, dt)
        rhs_prev = None
        for n in range(max_n + 1):
            rhs = M[n].copy()
            if n >= 1:
                rhs = rhs + dt * n * a_x * rhs_prev
                rhs[0] = 0.0
                rhs[-1] = 0.0
            M[n] = solve_banded((1, 1), ab, rhs)
            rhs_prev = M[n]

    return MomentResult(x_grid=x_grid, dx=dx, T=T, M=M, index_name=surface.index_name)


def conditional_moments(result: MomentResult, density_cutoff_relative: float) -> dict:
    """Conditional moments of integrated variance I_T given X_T = x, formed from
    the raw moments r_n(x) = M_n(T,x) / M_0(T,x) (paper Section 9.2):

        m(x)   = r_1(x)                         conditional mean
        v(x)   = r_2(x) - r_1(x)^2               conditional variance
        mu_3(x)= r_3(x) - 3 r_1(x) r_2(x) + 2 r_1(x)^3   conditional 3rd central moment

    Nodes where M_0(T,x) falls below `density_cutoff_relative` times the peak
    of M_0(T,x) are marked unresolved (the paper does not give the exact
    cutoff value) and excluded from the `resolved` mask.
    """
    M0 = result.M[0]
    cutoff = density_cutoff_relative * M0.max()
    resolved = M0 > cutoff

    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(resolved, result.M[1] / M0, 0.0)
        out = {"p_T": M0, "resolved": resolved, "r1": r1, "m": r1}
        if 2 in result.M:
            r2 = np.where(resolved, result.M[2] / M0, 0.0)
            v = r2 - r1 ** 2
            out["r2"] = r2
            out["v"] = np.maximum(v, 0.0)
        if 3 in result.M:
            r3 = np.where(resolved, result.M[3] / M0, 0.0)
            mu3 = r3 - 3 * r1 * r2 + 2 * r1 ** 3
            out["r3"] = r3
            out["mu3"] = mu3
    return out
