"""Experiment 3: two-moment vs three-moment (shifted-Gamma) conditional closure.

Reproduces paper Section 13.4: the three-moment shifted-Gamma closure
(Section 8, Theorem 8.1, Eq. 45-46, martingale normalisation Section 10.2) is
compared against the two-moment conditional-Gamma closure and the Baseline
joint-law PDE reference, for every (index, beta, maturity) combination, with
the admissibility diagnostics of Section 8.1 (lambda_3(x) >= 2 cv(x)).

Run: python -m exp.exp3_shifted_gamma_closure [--full]
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
from src.pricing_pipeline import conditional_gamma_smile, joint_law_smile

RESULTS_DIR = "results"


def _k_grid() -> np.ndarray:
    return np.linspace(STRIKE_K_RANGE[0], STRIKE_K_RANGE[1], STRIKE_LOG_MONEYNESS_GRID_PAPER)


def run(full: bool) -> None:
    grid_configs = GRID_CONFIGS_PAPER if full else GRID_CONFIGS_RUN
    moment_grid = MOMENT_GRID_PAPER if full else MOMENT_GRID_RUN
    baseline_grid = grid_configs["Baseline"]
    k_grid = _k_grid()

    rows, diagnostics = [], []
    for index_name in INDEX_NAMES:
        surface = build_surface(index_name)
        for T_label in MATURITY_LABELS:
            T = MATURITY_YEARS[T_label]
            inputs = build_inputs(index_name, T)
            for beta in BETAS:
                reference = joint_law_smile(surface, beta, T, baseline_grid, inputs, k_grid)
                two_moment = conditional_gamma_smile(
                    surface, beta, T, moment_grid, inputs, k_grid, shifted=False,
                    density_cutoff_relative=DENSITY_CUTOFF_RELATIVE,
                    variance_negligible_relative=VARIANCE_NEGLIGIBLE_RELATIVE)
                three_moment = conditional_gamma_smile(
                    surface, beta, T, moment_grid, inputs, k_grid, shifted=True,
                    density_cutoff_relative=DENSITY_CUTOFF_RELATIVE,
                    variance_negligible_relative=VARIANCE_NEGLIGIBLE_RELATIVE)

                ref_iv = reference["iv"]
                two_err = two_moment["iv"] - ref_iv
                three_err = three_moment["iv"] - ref_iv
                finite_two = np.isfinite(two_err)
                finite_three = np.isfinite(three_err)
                for idx_k in range(len(k_grid)):
                    rows.append({
                        "index": index_name, "beta": beta, "maturity": T_label,
                        "k": float(k_grid[idx_k]),
                        "reference_iv": ref_iv[idx_k],
                        "two_moment_iv": two_moment["iv"][idx_k],
                        "three_moment_iv": three_moment["iv"][idx_k],
                        "two_moment_abs_err": abs(two_err[idx_k]) if finite_two[idx_k] else np.nan,
                        "three_moment_abs_err": abs(three_err[idx_k]) if finite_three[idx_k] else np.nan,
                    })

                params = three_moment["params"]
                gap = params.lambda3_minus_2cv_on_A
                diagnostics.append({
                    "index": index_name, "beta": beta, "maturity": T_label,
                    "admissible_fraction": params.admissible_fraction,
                    "mean_lambda3_minus_2cv": float(np.mean(gap)) if gap.size else np.nan,
                    "max_two_moment_abs_err_bp": float(np.nanmax(np.abs(two_err[finite_two]))) * 1e4,
                    "max_three_moment_abs_err_bp": float(np.nanmax(np.abs(three_err[finite_three]))) * 1e4,
                })
                print(f"[exp3] {index_name} beta={beta} {T_label}: "
                      f"admissible={params.admissible_fraction:.2f} "
                      f"max2bp={diagnostics[-1]['max_two_moment_abs_err_bp']:.2f} "
                      f"max3bp={diagnostics[-1]['max_three_moment_abs_err_bp']:.2f}")

    detail = pd.DataFrame(rows)
    diag = pd.DataFrame(diagnostics)
    detail.to_csv(f"{RESULTS_DIR}/exp3_smile_detail.csv", index=False)
    diag.to_csv(f"{RESULTS_DIR}/exp3_diagnostics.csv", index=False)
    print(diag.to_string(index=False))

    _plot_examples(detail)


def _plot_examples(detail: pd.DataFrame) -> None:
    """beta=-2 (inverse, gamma=3) at the long end versus beta=3 (same gamma=3,
    long product), where the paper expects the largest two-moment/three-moment
    gap difference despite the shared drag coefficient (Section 13.4.1)."""
    for index_name in INDEX_NAMES:
        for beta in (-2, 3):
            sub = detail[(detail["index"] == index_name) & (detail["beta"] == beta) &
                         (detail["maturity"] == "2Y")].sort_values("k")
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(sub["k"], sub["reference_iv"] * 100, label="Joint-law reference", color="k",
                    linewidth=2)
            ax.plot(sub["k"], sub["two_moment_iv"] * 100, label="Two-moment Gamma", color="C0")
            ax.plot(sub["k"], sub["three_moment_iv"] * 100, label="Three-moment shifted-Gamma",
                    color="C2", linestyle="--")
            ax.set_xlabel("LETF forward log-moneyness k = log(K/F_T^L)")
            ax.set_ylabel("Black implied volatility (%)")
            ax.set_title(f"{index_name} beta={beta}, T=2Y: two- vs three-moment closure (Eq. 31-33 vs 45-46)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{RESULTS_DIR}/exp3_smile_{index_name}_beta{beta}_2Y.png", dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's Table 1 Baseline grid at full resolution")
    args = parser.parse_args()
    run(full=args.full)
