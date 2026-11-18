"""Experiment parameters as module-level variables.

Every variable carries a comment stating its meaning, unit and source. Where
the source paper (Beyond Strike Remapping: Conditional-Gamma Pricing of
Leveraged ETF Options under Local Volatility) does not specify a value, the
comment says so explicitly and the value is a documented modelling choice
recorded in results/RESULTS.md under "Deviations from the paper".
"""

# ---------------------------------------------------------------------------
# Common test configuration (paper Section 13.1)
# ---------------------------------------------------------------------------

VALUATION_DATE = "2026-06-22"  # paper Section 13.1: the common valuation date for SPX/NKY tests

INDEX_NAMES = ["SPX", "NKY"]  # paper Section 13.1: the two index markets tested

BETAS = [-2, -1, 2, 3]  # paper Section 13.1: leverage ratios tested (two inverse, two long)

MATURITY_LABELS = ["3M", "6M", "1Y", "2Y"]  # paper Section 13.1: four maturities per (index, beta)
MATURITY_YEARS = {"3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0}  # year fractions, act/365 convention (not specified in the paper; standard option-market convention used)

STRIKE_LOG_MONEYNESS_GRID_PAPER = 41  # number of LETF forward log-moneyness points plotted per smile (not specified in the paper; chosen for smooth figures)
STRIKE_K_RANGE = (-1.2, 1.2)  # LETF forward log-moneyness k = log(K/F_T^L) range quoted in the smile plots (not specified in the paper; chosen to cover the wings referenced in Section 13.2.3)

# ---------------------------------------------------------------------------
# Deterministic forward/discount inputs (paper Section 2, deterministic rates)
# ---------------------------------------------------------------------------

RISK_FREE_RATE = {"SPX": 0.045, "NKY": 0.010}  # flat continuously-compounded risk-free rate, per annum (not specified in the paper; representative USD/JPY short-rate levels used as a documented modelling input)
DIVIDEND_YIELD = {"SPX": 0.018, "NKY": 0.020}  # flat continuous dividend yield of the underlying index, per annum (not specified in the paper; representative index dividend yields used as a documented modelling input)
SPOT_LEVEL = {"SPX": 5500.0, "NKY": 39500.0}  # index spot level on the valuation date, index points (not specified in the paper; representative levels used as a documented modelling input)

# ---------------------------------------------------------------------------
# Parametric Dupire local-volatility surface (SVI implied-variance parametrisation)
# ---------------------------------------------------------------------------
# The paper states that SPX and NKY local-volatility surfaces are calibrated to
# the respective index option markets on 22 June 2026 (Section 13.1) but does
# not give the calibration methodology or parametrisation (flagged
# uncertainty). This repository builds each surface from a raw SVI
# (Gatheral (2004)) total implied variance smile per maturity, interpolated in
# total variance across maturities, and converts it to Dupire local variance
# with the closed-form Gatheral-Jacquier (2014) local-variance-from-SVI
# formula. Parameters are representative equity-index skew levels (SPX more
# skewed short end typical of index markets; NKY calibrated with a somewhat
# steeper skew than SPX, consistent with the paper's statement in exp_2 that
# "the NKY skew is steeper than SPX").
#
# SVI raw parametrisation: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
# with k the log-moneyness log(K/F) and w the total implied variance (sigma_BS^2 * T).
SVI_PARAMS = {
    "SPX": {
        "3M": {"a": 0.0070, "b": 0.14, "rho": -0.55, "m": -0.02, "sigma": 0.09},
        "6M": {"a": 0.0140, "b": 0.15, "rho": -0.52, "m": -0.03, "sigma": 0.11},
        "1Y": {"a": 0.0280, "b": 0.165, "rho": -0.48, "m": -0.045, "sigma": 0.14},
        "2Y": {"a": 0.0560, "b": 0.185, "rho": -0.42, "m": -0.06, "sigma": 0.18},
    },
    "NKY": {
        "3M": {"a": 0.0085, "b": 0.17, "rho": -0.62, "m": -0.025, "sigma": 0.095},
        "6M": {"a": 0.0170, "b": 0.185, "rho": -0.60, "m": -0.04, "sigma": 0.12},
        "1Y": {"a": 0.0340, "b": 0.205, "rho": -0.56, "m": -0.06, "sigma": 0.155},
        "2Y": {"a": 0.0680, "b": 0.230, "rho": -0.50, "m": -0.08, "sigma": 0.20},
    },
}  # not specified in the paper (flagged uncertainty); representative SVI smiles, steeper for NKY than SPX

# ---------------------------------------------------------------------------
# Table 1 (paper Section 13.2.1): nested grid configurations for the forward
# joint-law PDE solve. n_x: nodes in terminal log-return x; n_i: nodes in
# integrated variance i; steps_per_year: time steps per year; n_sigma: x-domain
# truncation in standard deviations of the terminal underlying log-return.
# ---------------------------------------------------------------------------

GRID_CONFIGS_PAPER = {
    "Coarse": {"n_x": 1600, "n_i": 800, "steps_per_year": 250, "n_sigma": 8.0},
    "Baseline": {"n_x": 2000, "n_i": 1000, "steps_per_year": 250, "n_sigma": 10.0},
    "Mid": {"n_x": 2400, "n_i": 1200, "steps_per_year": 250, "n_sigma": 12.0},
    "Fine": {"n_x": 3000, "n_i": 1500, "steps_per_year": 250, "n_sigma": 15.0},
}  # paper Table 1, verbatim

# Reduced grid configurations used for the executed runs in this repository,
# preserving Table 1's nesting ratios and the baseline's role as the converged
# reference. Use --full on exp_1/exp_5 to run GRID_CONFIGS_PAPER instead; see
# results/RESULTS.md for the measured runtime that motivates this reduction.
GRID_CONFIGS_RUN = {
    "Coarse": {"n_x": 400, "n_i": 200, "steps_per_year": 250, "n_sigma": 8.0},
    "Baseline": {"n_x": 500, "n_i": 250, "steps_per_year": 250, "n_sigma": 10.0},
    "Mid": {"n_x": 600, "n_i": 300, "steps_per_year": 250, "n_sigma": 12.0},
    "Fine": {"n_x": 750, "n_i": 375, "steps_per_year": 250, "n_sigma": 15.0},
}

# i-domain truncation (not specified in the paper, flagged uncertainty): the
# upper bound of the integrated-variance grid is set as a multiple of the
# maximum local variance on the surface times the maturity, generous enough
# that the probability mass beyond it is negligible.
I_DOMAIN_SAFETY_MULTIPLE = 6.0  # i_max = I_DOMAIN_SAFETY_MULTIPLE * a_max * T

# Density cutoff below which M_0(T, x) is treated as numerically unresolved
# and conditional-moment ratios are not formed (not specified in the paper,
# flagged uncertainty; chosen relative to the peak density on each grid).
DENSITY_CUTOFF_RELATIVE = 1e-6  # fraction of the maximum of M_0(T, x) on the grid
VARIANCE_NEGLIGIBLE_RELATIVE = 1e-10  # v(x) treated as 0 below this fraction of mean(v) on the resolved set

# ---------------------------------------------------------------------------
# Monte Carlo benchmark (paper Section 13.2.2, equations (71)-(73))
# ---------------------------------------------------------------------------

MC_PATHS_PAPER = 2_000_000  # number of simulated paths (not specified in the paper, flagged uncertainty)
MC_PATHS_RUN = 200_000  # reduced path count used for the executed runs; use --full for MC_PATHS_PAPER
MC_SEED = 20260622  # fixed seed for reproducibility (valuation date used as seed, arbitrary choice)
MC_STEPS_PER_YEAR = 250  # same time discretisation as the PDE grids (paper Table 1)

# ---------------------------------------------------------------------------
# 1D forward moment PDE system (paper Section 9.2, equation (55)) grid
# ---------------------------------------------------------------------------
# Not specified in the paper (flagged uncertainty): the 1D grid for M_0..M_3
# uses the same n_x and n_sigma as the Baseline 2D configuration, since the x
# direction plays the same diffusive role in both solves.
MOMENT_GRID_PAPER = {"n_x": GRID_CONFIGS_PAPER["Baseline"]["n_x"], "steps_per_year": 250,
                      "n_sigma": GRID_CONFIGS_PAPER["Baseline"]["n_sigma"]}
MOMENT_GRID_RUN = {"n_x": GRID_CONFIGS_RUN["Baseline"]["n_x"], "steps_per_year": 250,
                    "n_sigma": GRID_CONFIGS_RUN["Baseline"]["n_sigma"]}

# ---------------------------------------------------------------------------
# Experiment 4: empirical validation against listed TQQQ market quotes
# (paper Section 13.5.1)
# ---------------------------------------------------------------------------

EXP4_UNDERLYING_TICKER = "QQQ"  # paper Section 13.5.1
EXP4_LETF_TICKER = "TQQQ"  # paper Section 13.5.1
EXP4_BETA = 3  # paper Section 13.5.1: beta = 3, gamma = 3
EXP4_MARKET_DATES = ["2026-03-20", "2026-06-22", "2026-09-25"]  # paper Section 13.5.1
EXP4_TARGET_TENORS_MONTHS = {
    "2026-03-20": [1, 3, 6, 10],
    "2026-06-22": [1, 3, 6, 18],
    "2026-09-25": [1, 3, 6, 16],
}  # paper Section 13.5.1: nearest listed expiries to 1M/3M/6M plus the stated long-dated expiry
EXP4_STRIKE_MONEYNESS_BAND = (0.75, 1.30)  # underlying spot-moneyness band of strikes retained (not specified in the paper; limits the number of contracts queried from the options market data provider)

# ---------------------------------------------------------------------------
# Numerical/root-finding tolerances (not specified in the paper)
# ---------------------------------------------------------------------------

IV_BRENTQ_BRACKET = (1e-4, 5.0)  # Black implied-volatility search bracket
IV_BRENTQ_TOL = 1e-10
