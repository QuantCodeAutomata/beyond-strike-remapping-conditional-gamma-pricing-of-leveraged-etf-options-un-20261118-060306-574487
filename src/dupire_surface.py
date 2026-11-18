"""Parametric Dupire local-volatility surface from an SVI implied-variance smile.

The paper (Section 13.1) calibrates SPX and NKY Dupire local-volatility
surfaces to the respective index option markets on 22 June 2026, but does not
give the calibration methodology or parametrisation. This module builds each
surface from a raw SVI
(Gatheral, "A parsimonious arbitrage-free implied volatility parameterization
with application to the valuation of volatility derivatives", 2004) total
implied variance smile per maturity, linearly interpolated/extrapolated in
total variance across maturities (the standard no-calendar-arbitrage
interpolation rule for a term structure of SVI slices), and converts it to
Dupire local variance with the closed-form local-variance-from-total-variance
formula (Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces", 2014,
Eq. 2.10), evaluated by finite differences of the fitted surface in (k, T).

No maintained Python library implements the SVI-to-Dupire local-variance map
directly (py_vollib and QuantLib cover Black-Scholes/Black IV inversion
only), so this closed-form transform is implemented from the cited formula.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.config import MATURITY_YEARS, SVI_PARAMS


def svi_total_variance(k: np.ndarray, params: dict) -> np.ndarray:
    """Raw SVI total implied variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    a, b, rho, m, sigma = params["a"], params["b"], params["rho"], params["m"], params["sigma"]
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


@dataclass
class LocalVolSurface:
    """Dupire local-variance surface a(t, x) = sigma^2(t, F_t e^x) built from SVI slices.

    `x` is the forward log-moneyness log(K/F_t), the same coordinate as the
    terminal log-return X_t = log(S_t / F_t^S) used throughout the paper.
    """

    index_name: str
    maturity_grid: np.ndarray  # sorted maturities (years) of the calibrated SVI slices
    svi_param_grid: list

    def total_variance(self, k: np.ndarray, T: float) -> np.ndarray:
        """Total implied variance w(k, T), linearly interpolated in total variance
        between the two bracketing SVI slices (flat extrapolation outside)."""
        T = max(T, 1e-6)
        grid = self.maturity_grid
        if T <= grid[0]:
            return svi_total_variance(k, self.svi_param_grid[0]) * (T / grid[0])
        if T >= grid[-1]:
            return svi_total_variance(k, self.svi_param_grid[-1]) * (T / grid[-1])
        j = np.searchsorted(grid, T) - 1
        j = max(0, min(j, len(grid) - 2))
        T0, T1 = grid[j], grid[j + 1]
        w0 = svi_total_variance(k, self.svi_param_grid[j])
        w1 = svi_total_variance(k, self.svi_param_grid[j + 1])
        weight = (T - T0) / (T1 - T0)
        return (1 - weight) * w0 + weight * w1

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        return np.sqrt(np.maximum(self.total_variance(k, T), 1e-12) / max(T, 1e-12))

    def local_variance(self, t: float, x: np.ndarray) -> np.ndarray:
        """Dupire local variance a(t, x) via the Gatheral-Jacquier (2014) Eq. 2.10
        local-variance-from-total-variance formula, with partial derivatives of
        w(k, T) estimated by central finite differences in (k, T)."""
        t = max(t, 1e-4)
        dk = 1e-4
        dT = max(1e-4, 1e-3 * t)
        w = self.total_variance(x, t)
        w_up = self.total_variance(x + dk, t)
        w_dn = self.total_variance(x - dk, t)
        dw_dk = (w_up - w_dn) / (2 * dk)
        d2w_dk2 = (w_up - 2 * w + w_dn) / (dk ** 2)
        w_tp = self.total_variance(x, t + dT)
        w_tm = self.total_variance(x, max(t - dT, 1e-6))
        dw_dT = (w_tp - w_tm) / (t + dT - max(t - dT, 1e-6))

        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = 1.0 - (x / np.maximum(w, 1e-12)) * dw_dk
            term2 = 0.25 * (-0.25 - 1.0 / np.maximum(w, 1e-12) + (x ** 2) / np.maximum(w ** 2, 1e-24)) * dw_dk ** 2
            term3 = 0.5 * d2w_dk2
            denom = term1 + term2 + term3
            local_var = dw_dT / denom
        # Guard against ill-conditioned denominators (deep wings): fall back to
        # the raw-maturity Black variance derivative, which is always positive
        # by construction of the SVI slices used here.
        fallback = np.maximum(dw_dT, 1e-8)
        local_var = np.where((denom > 1e-6) & np.isfinite(local_var), local_var, fallback)
        return np.maximum(local_var, 1e-8)


def fit_svi_slice(k: np.ndarray, total_var: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Least-squares raw-SVI fit of one maturity slice (Gatheral, 2004) to
    listed total-implied-variance observations ``total_var`` at log-moneyness
    ``k``. Used in experiment 4 to calibrate the QQQ surface to listed market
    quotes (paper Section 13.5.1), the same SVI parametrisation used for the
    synthetic SPX/NKY surfaces in experiments 1-3 (``src.config.SVI_PARAMS``).
    """
    weights = np.ones_like(k) if weights is None else weights
    a0 = float(np.min(total_var))
    b0 = 0.1
    rho0 = -0.3
    m0 = 0.0
    sigma0 = 0.1

    def residuals(p: np.ndarray) -> np.ndarray:
        params = {"a": p[0], "b": abs(p[1]), "rho": np.clip(p[2], -0.999, 0.999),
                  "m": p[3], "sigma": max(abs(p[4]), 1e-4)}
        return weights * (svi_total_variance(k, params) - total_var)

    fit = least_squares(residuals, x0=[a0, b0, rho0, m0, sigma0], method="lm", max_nfev=5000)
    p = fit.x
    return {"a": float(p[0]), "b": float(abs(p[1])), "rho": float(np.clip(p[2], -0.999, 0.999)),
            "m": float(p[3]), "sigma": float(max(abs(p[4]), 1e-4))}


def build_surface(index_name: str) -> LocalVolSurface:
    """Build the LocalVolSurface for 'SPX' or 'NKY' from src.config.SVI_PARAMS."""
    if index_name not in SVI_PARAMS:
        raise ValueError(f"no SVI parameters configured for index {index_name!r}")
    labels = sorted(SVI_PARAMS[index_name], key=lambda lbl: MATURITY_YEARS[lbl])
    maturity_grid = np.array([MATURITY_YEARS[lbl] for lbl in labels])
    svi_param_grid = [SVI_PARAMS[index_name][lbl] for lbl in labels]
    return LocalVolSurface(index_name=index_name, maturity_grid=maturity_grid, svi_param_grid=svi_param_grid)
