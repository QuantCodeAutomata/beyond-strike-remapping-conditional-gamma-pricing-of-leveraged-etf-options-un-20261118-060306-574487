"""Experiment 1: forward joint-law PDE solver and Monte Carlo validation.

Reproduces paper Section 13.2: the 2D forward Fokker-Planck PDE (Eq. 66) for
(X_t, I_t) is solved on the four nested grid configurations of Table 1
(Section 13.2.1) and checked for grid-refinement convergence, then compared
against an independent Monte Carlo benchmark (Section 13.2.2, Eq. 71-73) that
evolves the LETF log-return Y_t as its own process and never uses the
pathwise factorisation identity Y_T = beta X_T - gamma I_T.

Run: python -m exp.exp1_joint_law_validation [--full]
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.black_scholes import implied_vol_vec
from src.config import (BETAS, GRID_CONFIGS_PAPER, GRID_CONFIGS_RUN, INDEX_NAMES,
                         MATURITY_LABELS, MATURITY_YEARS, MC_PATHS_PAPER, MC_PATHS_RUN,
                         MC_SEED, MC_STEPS_PER_YEAR, STRIKE_K_RANGE,
                         STRIKE_LOG_MONEYNESS_GRID_PAPER)
from src.dupire_surface import build_surface
from src.joint_law_pde import (martingale_correction as joint_law_martingale_correction,
                                price_letf_call as joint_law_price_letf_call, solve_joint_law,
                                total_mass)
from src.market_inputs import build_inputs
from src.monte_carlo import price_band, simulate_letf_paths
from src.pricing_pipeline import gamma_coef_of, strikes_from_log_moneyness

RESULTS_DIR = "results"
GRID_ORDER = ["Coarse", "Baseline", "Mid", "Fine"]


def _k_grid() -> np.ndarray:
    return np.linspace(STRIKE_K_RANGE[0], STRIKE_K_RANGE[1], STRIKE_LOG_MONEYNESS_GRID_PAPER)


def run(full: bool) -> None:
    grid_configs = GRID_CONFIGS_PAPER if full else GRID_CONFIGS_RUN
    mc_paths = MC_PATHS_PAPER if full else MC_PATHS_RUN
    k_grid = _k_grid()

    convergence_rows = []
    mc_rows = []
    t_start = time.time()

    for index_name in INDEX_NAMES:
        surface = build_surface(index_name)
        for T_label in MATURITY_LABELS:
            T = MATURITY_YEARS[T_label]
            inputs = build_inputs(index_name, T)
            strikes = strikes_from_log_moneyness(k_grid, inputs.F_T_L)

            # The 2D density p(T, x, i) does not depend on beta: solve once per
            # (index, maturity, grid) and reuse it for every leverage ratio.
            joint_results = {}
            masses = {}
            for grid_name, grid in grid_configs.items():
                t0 = time.time()
                result = solve_joint_law(surface, T, grid)
                joint_results[grid_name] = result
                masses[grid_name] = total_mass(result)
                print(f"[exp1] {index_name} {T_label} {grid_name}: "
                      f"{time.time() - t0:.2f}s, total mass={masses[grid_name]:.4f}")

            for beta in BETAS:
                gamma_coef = gamma_coef_of(beta)
                grid_ivs = {}
                for grid_name in GRID_ORDER:
                    if grid_name not in joint_results:
                        continue
                    result = joint_results[grid_name]
                    c_T = joint_law_martingale_correction(result, beta, gamma_coef)
                    prices = np.array([
                        joint_law_price_letf_call(result, beta, gamma_coef, K, inputs.F_T_L,
                                                   inputs.D_T, c_T)
                        for K in strikes
                    ])
                    ivs = implied_vol_vec(prices, inputs.F_T_L, strikes, T, inputs.D_T)
                    grid_ivs[grid_name] = ivs

                present_grids = [g for g in GRID_ORDER if g in grid_ivs]
                for prev, cur in zip(present_grids[:-1], present_grids[1:]):
                    diff = grid_ivs[cur] - grid_ivs[prev]
                    finite = np.isfinite(diff)
                    convergence_rows.append({
                        "index": index_name, "beta": beta, "maturity": T_label,
                        "from_grid": prev, "to_grid": cur,
                        "max_abs_iv_diff": float(np.nanmax(np.abs(diff[finite]))) if finite.any() else np.nan,
                        "mean_abs_iv_diff": float(np.nanmean(np.abs(diff[finite]))) if finite.any() else np.nan,
                    })

                # Monte Carlo benchmark, drawn independently of the PDE grid.
                t0 = time.time()
                mc = simulate_letf_paths(surface, beta, T, MC_STEPS_PER_YEAR, mc_paths, MC_SEED)
                mc_time = time.time() - t0
                baseline_iv = grid_ivs.get("Baseline")
                for idx_k, K in enumerate(strikes):
                    price, lo, hi = price_band(mc, K, inputs.F_T_L, inputs.D_T)
                    mc_iv = implied_vol_vec(np.array([price]), inputs.F_T_L, np.array([K]), T,
                                             inputs.D_T)[0]
                    mc_iv_lo = implied_vol_vec(np.array([max(lo, 1e-12)]), inputs.F_T_L,
                                                np.array([K]), T, inputs.D_T)[0]
                    mc_iv_hi = implied_vol_vec(np.array([hi]), inputs.F_T_L, np.array([K]), T,
                                                inputs.D_T)[0]
                    row = {
                        "index": index_name, "beta": beta, "maturity": T_label,
                        "k": float(k_grid[idx_k]), "strike": float(K),
                        "mc_price": price, "mc_iv": mc_iv, "mc_iv_lo": mc_iv_lo, "mc_iv_hi": mc_iv_hi,
                        "mc_time_s": mc_time, "mc_paths": mc_paths,
                    }
                    if baseline_iv is not None:
                        b_iv = baseline_iv[idx_k]
                        row["baseline_iv"] = b_iv
                        lo_v, hi_v = sorted((mc_iv_lo, mc_iv_hi))
                        row["baseline_in_band"] = bool(np.isfinite(b_iv) and lo_v <= b_iv <= hi_v)
                    mc_rows.append(row)
                print(f"[exp1] {index_name} {T_label} beta={beta}: MC {mc_time:.2f}s "
                      f"({mc_paths} paths)")

    convergence = pd.DataFrame(convergence_rows)
    mc_validation = pd.DataFrame(mc_rows)
    convergence.to_csv(f"{RESULTS_DIR}/exp1_convergence.csv", index=False)
    mc_validation.to_csv(f"{RESULTS_DIR}/exp1_mc_validation.csv", index=False)

    summary = {
        "grid_configs_used": "paper" if full else "reduced",
        "n_x": {k: v["n_x"] for k, v in grid_configs.items()},
        "n_i": {k: v["n_i"] for k, v in grid_configs.items()},
        "mc_paths": mc_paths,
        "coverage_fraction_baseline_in_mc_band": (
            float(mc_validation["baseline_in_band"].mean())
            if "baseline_in_band" in mc_validation else None
        ),
        "wall_clock_seconds": time.time() - t_start,
    }
    pd.Series(summary).to_json(f"{RESULTS_DIR}/exp1_summary.json", indent=2)
    print("[exp1] summary:", summary)

    _plot_examples(mc_validation, full)


def _plot_examples(mc_validation: pd.DataFrame, full: bool) -> None:
    """Smile plots for the largest-|beta| case of each index at T=1Y (paper
    Section 13.2.3: residuals are largest in the wings and grow with |beta|)."""
    convergence = pd.read_csv(f"{RESULTS_DIR}/exp1_convergence.csv")
    for index_name in INDEX_NAMES:
        beta = max(BETAS, key=abs)
        sub = mc_validation[(mc_validation["index"] == index_name) &
                             (mc_validation["beta"] == beta) &
                             (mc_validation["maturity"] == "1Y")].sort_values("k")
        if sub.empty or "baseline_iv" not in sub:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sub["k"], sub["baseline_iv"] * 100, label="Baseline joint-law PDE", color="C0")
        ax.fill_between(sub["k"], sub["mc_iv_lo"] * 100, sub["mc_iv_hi"] * 100, color="C1",
                         alpha=0.25, label="Monte Carlo 95% band")
        ax.plot(sub["k"], sub["mc_iv"] * 100, color="C1", linestyle="--", label="Monte Carlo mean")
        ax.set_xlabel("LETF forward log-moneyness k = log(K/F_T^L)")
        ax.set_ylabel("Black implied volatility (%)")
        ax.set_title(f"{index_name} beta={beta}, T=1Y: joint-law PDE vs Monte Carlo (Eq. 66 vs 71-73)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/exp1_smile_{index_name}_beta{beta}_1Y.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for grid_name, marker in zip(GRID_ORDER, ["o", "s", "^", "d"]):
        sub = convergence[convergence["to_grid"] == grid_name]
        if sub.empty:
            continue
        ax.plot(range(len(sub)), sub["max_abs_iv_diff"] * 1e4, marker=marker,
                label=f"-> {grid_name}", linestyle="none")
    ax.set_xlabel("(index, beta, maturity) combination")
    ax.set_ylabel("Max abs implied-vol change vs previous grid (bp)")
    ax.set_title("Grid-refinement convergence (Table 1 nested configurations)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's Table 1 grids and Monte Carlo path count")
    args = parser.parse_args()
    run(full=args.full)
