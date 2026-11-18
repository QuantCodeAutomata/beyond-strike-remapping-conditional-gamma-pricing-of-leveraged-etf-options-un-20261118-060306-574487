"""Experiment 5: computational cost comparison.

Reproduces paper Section 13.6 (Table 2): the wall-clock cost of producing a
full LETF smile is measured for the conditional-Gamma method (M_0, M_1, M_2
forward moment solves plus conditional quadrature, Section 7), the
shifted-Gamma method (adds the M_3 solve, Section 8), and the forward
joint-law PDE reference (2D solve on the Baseline grid of Table 1, Section 11)
for every (index, beta, maturity) combination.

Run: python -m exp.exp5_computational_cost [--full]
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src.config import (BETAS, DENSITY_CUTOFF_RELATIVE, GRID_CONFIGS_PAPER, GRID_CONFIGS_RUN,
                         INDEX_NAMES, MATURITY_LABELS, MATURITY_YEARS, MOMENT_GRID_PAPER,
                         MOMENT_GRID_RUN, STRIKE_K_RANGE, STRIKE_LOG_MONEYNESS_GRID_PAPER,
                         VARIANCE_NEGLIGIBLE_RELATIVE)
from src.dupire_surface import build_surface
from src.market_inputs import build_inputs
from src.pricing_pipeline import conditional_gamma_smile, joint_law_smile

RESULTS_DIR = "results"


def _k_grid():
    import numpy as np
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
                t0 = time.perf_counter()
                conditional_gamma_smile(surface, beta, T, moment_grid, inputs, k_grid, shifted=False,
                                         density_cutoff_relative=DENSITY_CUTOFF_RELATIVE,
                                         variance_negligible_relative=VARIANCE_NEGLIGIBLE_RELATIVE)
                t_gamma = time.perf_counter() - t0

                t0 = time.perf_counter()
                conditional_gamma_smile(surface, beta, T, moment_grid, inputs, k_grid, shifted=True,
                                         density_cutoff_relative=DENSITY_CUTOFF_RELATIVE,
                                         variance_negligible_relative=VARIANCE_NEGLIGIBLE_RELATIVE)
                t_shifted = time.perf_counter() - t0

                t0 = time.perf_counter()
                joint_law_smile(surface, beta, T, baseline_grid, inputs, k_grid)
                t_joint = time.perf_counter() - t0

                rows.append({
                    "index": index_name, "beta": beta, "maturity": T_label,
                    "conditional_gamma_s": t_gamma, "shifted_gamma_s": t_shifted,
                    "joint_law_pde_s": t_joint,
                    "speedup_gamma_vs_joint": t_joint / t_gamma,
                    "speedup_shifted_vs_joint": t_joint / t_shifted,
                    "incremental_m3_cost_s": t_shifted - t_gamma,
                })
                print(f"[exp5] {index_name} beta={beta} {T_label}: "
                      f"gamma={t_gamma:.3f}s shifted={t_shifted:.3f}s joint={t_joint:.3f}s "
                      f"speedup(gamma)={rows[-1]['speedup_gamma_vs_joint']:.1f}x")

    timing = pd.DataFrame(rows)
    timing.to_csv(f"{RESULTS_DIR}/exp5_timing.csv", index=False)

    summary = timing.groupby("index")[
        ["conditional_gamma_s", "shifted_gamma_s", "joint_law_pde_s",
         "speedup_gamma_vs_joint", "speedup_shifted_vs_joint", "incremental_m3_cost_s"]
    ].agg(["mean", "min", "max"])
    summary.to_csv(f"{RESULTS_DIR}/exp5_summary_by_index.csv")
    print(summary)

    overall = timing[["conditional_gamma_s", "shifted_gamma_s", "joint_law_pde_s",
                       "speedup_gamma_vs_joint", "speedup_shifted_vs_joint",
                       "incremental_m3_cost_s"]].agg(["mean", "min", "max"])
    overall.to_csv(f"{RESULTS_DIR}/exp5_summary_overall.csv")
    print(overall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's Table 1 Baseline grid at full resolution")
    args = parser.parse_args()
    run(full=args.full)
