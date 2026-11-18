"""Forward joint-law Fokker-Planck PDE for (X_t, I_t) under Dupire local volatility.

Implements paper equation (66):

    dP/dt = (1/2) d/dx[a(t,x) P] - d/di[a(t,x) P] + (1/2) d^2/dx^2[a(t,x) P]

with initial condition p(0, x, i) = delta(x) delta(i) (Section 11). The x
direction diffuses (resolves the terminal underlying log-return density) and
is discretised with a conservative central-difference finite-volume operator,
solved implicitly (backward Euler in time) via a banded tridiagonal solve
batched over every i-column at once. The i direction is a pure transport of
the monotone process I_t at local speed a(t, x); since a does not depend on i,
each row (fixed x) obeys the exact linear advection equation dp/dt = -a dp/di,
whose solution for a time step dt is the exact shift p(i, t+dt) = p(i - a dt,
t). This is evaluated by a fully vectorised linear interpolation across the
whole 2D grid (a semi-Lagrangian / method-of-characteristics transport step),
which is consistent with "upwind... since I_t is monotone increasing"
(implementation_notes) while avoiding any Python-level loop over the i grid.

The exact finite-difference/time-stepping scheme and the boundary conditions
are not specified in the paper; both choices are
documented here and the grid-refinement study of experiment 1 checks that the
scheme converges under Table 1's nested configurations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from src.config import I_DOMAIN_SAFETY_MULTIPLE
from src.dupire_surface import LocalVolSurface


@dataclass
class JointLawResult:
    x_grid: np.ndarray
    i_grid: np.ndarray
    density: np.ndarray  # shape (n_x, n_i), p(T, x, i) on the tensor grid
    dx: float
    di: float
    T: float
    index_name: str


def x_domain(surface: LocalVolSurface, T: float, n_sigma: float, n_x: int) -> np.ndarray:
    """Public alias of the x-grid construction, shared with the 1D moment solver."""
    return _x_domain(surface, T, n_sigma, n_x)


def diffusion_banded(a_x: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """Public alias of the banded backward-Euler diffusion operator, shared with
    the 1D moment solver (same forward operator, paper Section 9.2)."""
    return _build_diffusion_banded(a_x, dx, dt)


def _x_domain(surface: LocalVolSurface, T: float, n_sigma: float, n_x: int) -> np.ndarray:
    """Truncate the x domain at n_sigma standard deviations of X_T, estimated
    from the ATMF local variance integrated over [0, T] (flat proxy, refined
    automatically by the PDE's own diffusion once solved)."""
    atm_var = surface.local_variance(T / 2, np.array([0.0]))[0]
    sigma_x = np.sqrt(atm_var * T) if atm_var * T > 0 else 0.2 * np.sqrt(T)
    half_width = n_sigma * max(sigma_x, 1e-3)
    return np.linspace(-half_width, half_width, n_x)


def _build_diffusion_banded(a_x: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """Banded matrix (I - dt * A) for backward-Euler diffusion, where A is the
    conservative central-difference discretisation of
    L_x[P] = (1/2) d/dx[a P] + (1/2) d^2/dx^2[a P] on a uniform grid, i.e.

        A P |_j = P_{j+1} * (a_{j+1}/(4 dx) + a_{j+1}/(2 dx^2))
                + P_j     * (-a_j / dx^2)
                + P_{j-1} * (-a_{j-1}/(4 dx) + a_{j-1}/(2 dx^2))

    Absorbing (Dirichlet, P = 0) boundary conditions are applied at both ends
    of the truncated x domain (not specified in the paper; justified because
    n_sigma is chosen so that the probability mass beyond the truncation is
    negligible).
    """
    n_x = len(a_x)
    upper = np.zeros(n_x)
    diag = np.zeros(n_x)
    lower = np.zeros(n_x)
    upper[1:] = a_x[1:] / (4 * dx) + a_x[1:] / (2 * dx ** 2)
    diag[:] = -a_x / dx ** 2
    lower[:-1] = -a_x[:-1] / (4 * dx) + a_x[:-1] / (2 * dx ** 2)
    # banded = I - dt * A
    ab = np.zeros((3, n_x))
    ab[0, 1:] = -dt * upper[1:]
    ab[1, :] = 1.0 - dt * diag
    ab[2, :-1] = -dt * lower[:-1]
    # Dirichlet boundaries: rows 0 and n_x-1 become identity rows
    ab[1, 0] = 1.0
    ab[0, 1] = 0.0
    ab[1, -1] = 1.0
    ab[2, -2] = 0.0
    return ab


def _semi_lagrangian_transport(density: np.ndarray, a_x: np.ndarray, di: float, dt: float) -> np.ndarray:
    """Exact-advection transport step for all x-rows at once: shifts each row
    by a_x[j] * dt / di grid cells using vectorised linear interpolation.
    Mass shifted past the i = 0 boundary is impossible (a_x >= 0, I_t only
    increases) and mass shifted past the truncated i_max boundary is dropped
    (documented assumption; i_max is chosen generously, see config)."""
    n_x, n_i = density.shape
    shift_cells = a_x * dt / di  # shape (n_x,)
    k = np.arange(n_i)
    src = k[None, :] - shift_cells[:, None]  # shape (n_x, n_i)
    floor_idx = np.floor(src).astype(int)
    frac = src - floor_idx
    valid_lo = (floor_idx >= 0) & (floor_idx <= n_i - 1)
    valid_hi = (floor_idx + 1 >= 0) & (floor_idx + 1 <= n_i - 1)
    floor_clipped = np.clip(floor_idx, 0, n_i - 1)
    ceil_clipped = np.clip(floor_idx + 1, 0, n_i - 1)
    row_idx = np.arange(n_x)[:, None]
    v_lo = density[row_idx, floor_clipped] * valid_lo
    v_hi = density[row_idx, ceil_clipped] * valid_hi
    return (1 - frac) * v_lo + frac * v_hi


def solve_joint_law(surface: LocalVolSurface, T: float, grid: dict) -> JointLawResult:
    """Solve the forward joint-law PDE (66) from t=0 to t=T on the grid
    configuration `grid` (keys: n_x, n_i, steps_per_year, n_sigma)."""
    n_x, n_i = grid["n_x"], grid["n_i"]
    steps_per_year = grid["steps_per_year"]
    n_sigma = grid["n_sigma"]

    x_grid = _x_domain(surface, T, n_sigma, n_x)
    dx = x_grid[1] - x_grid[0]

    atm_var_max = surface.local_variance(T / 2, np.array([0.0]))[0]
    local_var_sample = surface.local_variance(T / 2, x_grid)
    a_max_est = max(atm_var_max, local_var_sample.max())
    i_max = I_DOMAIN_SAFETY_MULTIPLE * a_max_est * T
    i_grid = np.linspace(0.0, i_max, n_i)
    di = i_grid[1] - i_grid[0]

    n_steps = max(1, int(round(steps_per_year * T)))
    dt = T / n_steps

    density = np.zeros((n_x, n_i))
    j0 = np.argmin(np.abs(x_grid))
    density[j0, 0] = 1.0 / (dx * di)  # discrete approximation to delta(x) delta(i)

    for step in range(n_steps):
        t_mid = (step + 0.5) * dt
        a_x = surface.local_variance(t_mid, x_grid)
        ab = _build_diffusion_banded(a_x, dx, dt)
        density = solve_banded((1, 1), ab, density)
        density = _semi_lagrangian_transport(density, a_x, di, dt)

    return JointLawResult(x_grid=x_grid, i_grid=i_grid, density=density, dx=dx, di=di, T=T,
                           index_name=surface.index_name)


def total_mass(result: JointLawResult) -> float:
    return result.density.sum() * result.dx * result.di


def martingale_correction(result: JointLawResult, beta: float, gamma_coef: float) -> float:
    """c_T^num = log( integral integral e^(beta x - gamma i) p(T,x,i) dx di ),
    the scalar martingale normalisation of Section 10.1 applied to the
    joint-law reference."""
    x = result.x_grid[:, None]
    i = result.i_grid[None, :]
    integrand = np.exp(beta * x - gamma_coef * i) * result.density
    raw = integrand.sum() * result.dx * result.di
    return float(np.log(max(raw, 1e-300)))


def price_letf_call(result: JointLawResult, beta: float, gamma_coef: float, K: float,
                     F_T_L: float, D_T: float, c_T: float) -> float:
    """C_L(K,T) = D_T * integral integral (F_T^L e^(beta x - gamma i - c_T) - K)^+ p(x,i) dx di,
    paper equations (step 4-5 of exp_2 methodology)."""
    x = result.x_grid[:, None]
    i = result.i_grid[None, :]
    payoff = np.maximum(F_T_L * np.exp(beta * x - gamma_coef * i - c_T) - K, 0.0)
    return D_T * float((payoff * result.density).sum() * result.dx * result.di)
