"""Experiment 2: conditional-Gamma method vs first-order strike remapping.

Reproduces paper Section 13.3: the two-moment conditional-Gamma closure
(Section 7, Eq. 31-33, martingale normalisation Section 10.1) and the
first-order strike remapping (Section 4, Eq. 13-14) are compared against the
Baseline forward joint-law PDE reference validated in experiment 1, for every
(index, beta, maturity) combination.

Run: python -m exp.exp2_conditional_gamma_vs_remapping [--full]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (BETAS, DENSITY_CUTOFF_RELATIVE, GRID_CONFIGS_PAPER, GRID_CONFIGS_RUN,
                         INDEX_NAMES, MATURITY_LABELS, MATURITY_YEARS, MOMENT_GRID_PAPER,
                         MOMENT_GRID_RUN, STRIKE_K_RANGE, STRIKE_LOG_MONEYNESS_GRID_PAPER,
                         VARIANCE_NEGLIGIBLE_RELATIVE)
from src.dupire_surface import build_surface
from src.market_inputs import build_inputs
from src.pricing_pipeline import conditional_gamma_smile, joint_law_smile, strike_remapping_smile

RESULTS_DIR = "results"


def _k_grid() -> np.ndarray:
    return np.linspace(STRIKE_K_RANGE[0], STRIKE_K_RANGE[1], STRIKE_LOG_MONEYNESS_GRID_PAPER)


def run(full: bool) -> None:
    grid_configs = GRID_CONFIGS_PAPER if full else GRID_CONFIGS_RUN
    moment_grid = MOMENT_GRID_PAPER if full else MOMENT_GRID_RUN
    baseline_grid = grid_configs["Baseline"]
    k_grid = _k_grid()

    rows = []
    for index_name in INDEX_NAMES:
        surface = build_surface(index_name)
        for T_label in MATURITY_LABELS:
            T = MATURITY_YEARS[T_label]
            inputs = build_inputs(index_name, T)
            for beta in BETAS:
                reference = joint_law_smile(surface, beta, T, baseline_grid, inputs, k_grid)
                gamma = conditional_gamma_smile(surface, beta, T, moment_grid, inputs, k_grid,
                                                 shifted=False,
                                                 density_cutoff_relative=DENSITY_CUTOFF_RELATIVE,
                                                 variance_negligible_relative=VARIANCE_NEGLIGIBLE_RELATIVE)
                remap = strike_remapping_smile(surface, beta, T, inputs, k_grid)

                ref_iv, gamma_iv, remap_iv = reference["iv"], gamma["iv"], remap["iv"]
                gamma_err = gamma_iv - ref_iv
                remap_err = remap_iv - ref_iv
                finite = np.isfinite(gamma_err) & np.isfinite(remap_err)
                for idx_k in range(len(k_grid)):
                    rows.append({
                        "index": index_name, "beta": beta, "maturity": T_label,
                        "k": float(k_grid[idx_k]), "strike": float(reference["strikes"][idx_k]),
                        "reference_iv": ref_iv[idx_k], "gamma_iv": gamma_iv[idx_k],
                        "remap_iv": remap_iv[idx_k],
                        "gamma_abs_err": abs(gamma_err[idx_k]) if finite[idx_k] else np.nan,
                        "remap_abs_err": abs(remap_err[idx_k]) if finite[idx_k] else np.nan,
                    })
                print(f"[exp2] {index_name} beta={beta} {T_label}: "
                      f"max|gamma_err|={np.nanmax(np.abs(gamma_err[finite])) * 1e4:.2f}bp "
                      f"max|remap_err|={np.nanmax(np.abs(remap_err[finite])) * 1e4:.2f}bp")

    detail = pd.DataFrame(rows)
    detail.to_csv(f"{RESULTS_DIR}/exp2_smile_detail.csv", index=False)

    summary = (detail.groupby(["index", "beta", "maturity"])
               .agg(max_gamma_abs_err=("gamma_abs_err", "max"),
                    mean_gamma_abs_err=("gamma_abs_err", "mean"),
                    max_remap_abs_err=("remap_abs_err", "max"),
                    mean_remap_abs_err=("remap_abs_err", "mean"))
               .reset_index())
    summary["gamma_beats_remap"] = summary["max_gamma_abs_err"] < summary["max_remap_abs_err"]
    gamma_val = {2: 1, -1: 1, 3: 3, -2: 3}
    summary["gamma_coef"] = summary["beta"].map(gamma_val)
    summary.to_csv(f"{RESULTS_DIR}/exp2_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[exp2] fraction of cases where conditional-Gamma max error < "
          f"remapping max error: {summary['gamma_beats_remap'].mean():.3f}")

    _plot_examples(detail)


def _plot_examples(detail: pd.DataFrame) -> None:
    for index_name in INDEX_NAMES:
        beta = max(BETAS, key=abs)
        sub = detail[(detail["index"] == index_name) & (detail["beta"] == beta) &
                     (detail["maturity"] == "2Y")].sort_values("k")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sub["k"], sub["reference_iv"] * 100, label="Joint-law reference (Baseline)",
                color="k", linewidth=2)
        ax.plot(sub["k"], sub["gamma_iv"] * 100, label="Conditional-Gamma (2nd order)", color="C0")
        ax.plot(sub["k"], sub["remap_iv"] * 100, label="First-order strike remapping",
                color="C3", linestyle="--")
        ax.set_xlabel("LETF forward log-moneyness k = log(K/F_T^L)")
        ax.set_ylabel("Black implied volatility (%)")
        ax.set_title(f"{index_name} beta={beta}, T=2Y: conditional-Gamma vs strike remapping (Eq. 13-14, 31-33)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/exp2_smile_{index_name}_beta{beta}_2Y.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's Table 1 Baseline grid at full resolution")
    args = parser.parse_args()
    run(full=args.full)
