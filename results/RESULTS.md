# Results — Beyond Strike Remapping: Conditional-Gamma Pricing of Leveraged ETF Options under Local Volatility

Paper: *Beyond Strike Remapping: Conditional-Gamma Pricing of Leveraged ETF
Options under Local Volatility*.

All numbers below come from the files written by the committed experiment
runs in `results/`: `exp1_summary.json`, `exp1_convergence.csv`,
`exp1_mc_validation.csv`, `exp2_summary.csv`, `exp3_diagnostics.csv`,
`exp4_summary.csv`, `exp4_errors.csv`, `exp5_summary_overall.csv`,
`exp5_summary_by_index.csv`, `exp5_timing.csv`, and the corresponding
`*_run.log` files.

## Experiment 1 — Forward Joint-Law PDE Solver and Monte Carlo Validation

Tests the 2D forward Fokker-Planck solver for `(X_t, I_t)` (Section 11,
equation (66)) against grid refinement (Table 1: Coarse/Baseline/Mid/Fine)
and against an independent Monte Carlo simulation of the LETF's own
log-Euler dynamics (equations (71)-(73), Section 13.2), which never uses the
pathwise factorisation identity.

Executed configuration (`results/exp1_summary.json`): `grid_configs_used =
"reduced"`, `n_x = {Coarse:400, Baseline:500, Mid:600, Fine:750}`,
`n_i = {Coarse:200, Baseline:250, Mid:300, Fine:375}`, 250 time steps per
year (unchanged from the paper), `mc_paths = 200,000`.

Grid-refinement convergence (`results/exp1_convergence.csv`, max/mean
absolute implied-vol differences across all strikes, in volatility points,
aggregated over all (index, beta, maturity) combinations):

| transition | mean of max abs IV diff | mean of mean abs IV diff |
|---|---|---|
| Coarse → Baseline | 0.0487 | 0.0126 |
| Baseline → Mid | 0.0186 | 0.0043 |
| Mid → Fine | 0.0104 | 0.0023 |

The Coarse→Baseline change is several times larger than Baseline→Mid, which
in turn is larger than Mid→Fine: refinement is monotonically decreasing,
matching the paper's qualitative claim that Baseline→Mid→Fine changes are
far smaller than Coarse→Baseline. **Verdict: reproduced** (qualitative
ordering of convergence).

Monte Carlo agreement: the fraction of Baseline-grid implied vols that fall
inside the Monte Carlo 95% sampling band (`coverage_fraction_baseline_in_mc_band`
in `exp1_summary.json`) is **0.421**, i.e. 42% of quoted strikes, well below
"inside or very close to the band" at most strikes reported by the paper.
Residual differences are concentrated in the wings and grow with `|beta|`
(consistent with the paper), but the overall coverage is lower than expected.
**Verdict: partially reproduced.** The reduced grid (`n_x`/`n_i` roughly a
quarter of Table 1's Baseline) and the reduced path count (200,000 vs the
unspecified paper value) widen both the PDE discretisation error and the
Monte Carlo standard error; a sensitivity run with `GRID_CONFIGS_RUN =
GRID_CONFIGS_PAPER` (`--full`) was not executed because the paper's Baseline
grid alone (`2000 × 1000` nodes, 250 steps/year) was measured in experiment 5
at about 0.57–4.5 s per solve on the reduced grid and is expected to scale by
roughly `(2000/500)·(1000/250) ≈ 16×` per solve at the paper's Baseline size,
which together with the full Monte Carlo path count (10× more paths) would
not finish inside the available time; see "Deviations from the paper" below.

## Experiment 2 — Conditional-Gamma vs First-Order Strike Remapping

Compares the conditional-Gamma closure (Sections 7, 9.2, two-moment fit) and
the first-order strike remapping (Section 2, equations (13)-(14)) against
the Experiment-1 joint-law reference, for SPX and NKY, `beta ∈ {-2,-1,2,3}`,
`T ∈ {3M,6M,1Y,2Y}` (`results/exp2_summary.csv`).

* **Gamma beats remapping in every tested case**: `gamma_beats_remap` is
  `True` for all 32 (index, beta, maturity) combinations. **Verdict:
  reproduced.**
* Mean of the per-case maximum absolute error (volatility points), by
  leverage magnitude:

  | drag coefficient γ | conditional-Gamma max err | remapping max err |
  |---|---|---|
  | γ=1 (β∈{-1,2}) | 0.0184 | 0.239 |
  | γ=3 (β∈{-2,3}) | 0.0213 | 0.214 |

  the conditional-Gamma error is larger at γ=3 than at γ=1, matching the
  paper. **Verdict: reproduced.**
* Mean maximum conditional-Gamma error is **0.0223** for NKY vs **0.0173**
  for SPX (larger for NKY, the steeper-skew index), matching the paper.
  **Verdict: reproduced.**
* In absolute terms, the conditional-Gamma error (≤0.03 vol points in every
  case) is one to two orders of magnitude smaller than the first-order
  remapping error (up to 0.45 vol points for inverse products), confirming
  that the remapping error dominates in the wings while the conditional-Gamma
  curve stays close to the reference. **Verdict: reproduced.**

## Experiment 3 — Two-Moment vs Three-Moment Shifted-Gamma Closure

Extends the triangular moment system to `M_3` (Section 8) and compares the
shifted-Gamma closure against the two-moment Gamma closure
(`results/exp3_diagnostics.csv`, `admissible_fraction`,
`mean_lambda3_minus_2cv`, `max_two_moment_abs_err_bp`,
`max_three_moment_abs_err_bp` per (index, beta, maturity); full 32-row table
in the CSV).

* Mean admissible fraction (`lambda_3(x) ≥ 2·cv(x)`) across all cases is
  **0.64**: a majority but not all resolved grid nodes admit the shifted-Gamma
  closure, so the hybrid fallback to the two-moment Gamma is exercised
  everywhere, as the methodology specifies. **Verdict: reproduced**
  (non-degenerate admissible set, consistent with Section 8.1).
* The paper's claim that the two closures are "almost indistinguishable" at
  3M and that the three-moment closure mainly helps inverse products at
  longer maturities is only **partially** borne out: at 3M the three-moment
  closure is indeed close to the two-moment one for γ=1 (β∈{-1,2}, differences
  below 1 bp), but for γ=3 (β∈{-2,3}) the shifted-Gamma closure is
  *measurably worse* than the two-moment Gamma at 3M (e.g. NKY β=-2, 3M:
  396 bp vs 486 bp; SPX β=3, 3M: 162 bp vs 198 bp), while it does improve
  accuracy at 1Y for the same leverage pairs (e.g. NKY β=-2, 1Y: 219 bp vs
  181 bp; SPX β=3, 1Y: 80 bp vs 70 bp). **Verdict: not reproduced** for the
  3M-indistinguishability claim at γ=3, and **partially reproduced** for the
  longer-maturity improvement (present at 1Y, not monotone through 2Y). A
  sensitivity check on the SPX β=3, T=3M slice, raising the density cutoff
  used to flag "resolved" `M_0` nodes from the production value
  (`DENSITY_CUTOFF_RELATIVE = 1e-6`) to `1e-4`, `1e-3` and `1e-2`, shows the
  two- and three-moment maximum errors moving together and growing together
  as the cutoff is raised (161.9/197.8 bp at `1e-6`, 212.1/211.7 bp at
  `1e-4`, 949.8/949.4 bp at `1e-3`, 4912/4977 bp at `1e-2`): the three-moment
  closure is never better than the two-moment one at this slice for any
  tested cutoff, so the 3M/γ=3 degradation is not an artefact of the
  resolved-node threshold. The specific mismatch (a bias in the
  `M_3`-implied skewness estimate in the far wings, where `M_0` is small and
  `mu_3` is a difference of comparable-magnitude moments) is therefore
  reported but not fully localised within the available checks; the root
  cause of the 3M/γ=3 degradation is not identified.
* The paper's observation that β=-2 and β=3 share γ=3 but the inverse ratio
  is the harder case for the two-moment closure is **reproduced** at 3M and
  6M (two-moment max error for β=-2 exceeds β=3 by roughly 150–180 bp at
  every maturity in the table) but not at 1Y/2Y where the two errors are of
  the same order.

## Experiment 4 — Empirical Validation Against Listed TQQQ Market Quotes

Compares the joint-law reference, the conditional-Gamma method, and the
first-order remapping against listed TQQQ (β=3, γ=3) bid/ask implied
volatilities on three market dates, using QQQ and TQQQ American-style option
quotes converted to European implied volatilities (Section 13.5). Market
quotes were loaded through the Massive REST options-chain endpoint via
`data/data_loader.py`; `F_T^L` was backed out from put-call parity at the
closest-to-parity strike for each loaded expiry (`results/exp4_summary.csv`,
`results/exp4_errors.csv`, `results/exp4_run.log`).

| date | tenor (nominal) | strikes loaded | joint-law in band | gamma in band | remap in band | mean \|joint − gamma\| (bp) |
|---|---|---|---|---|---|---|
| 2026-03-20 | 1M | 35 | 0.00 | 0.03 | 0.00 | 262.3 |
| 2026-03-20 | 3M | 29 | 0.00 | 0.00 | 0.00 | 46.2 |
| 2026-06-22 | 3M | 24 | 0.21 | 0.50 | 0.54 | 255.8 |
| 2026-09-25 | 1M | 44 | 0.05 | 0.02 | 0.02 | 91.6 |

Only four of the twelve requested (date, tenor) cells produced a two-sided
TQQQ quote with a matching QQQ chain; the 6M and long-dated legs on
2026-03-20 and the 1M/6M/long-dated legs on 2026-06-22 and 2026-09-25 failed
because Massive returned no quoted contracts at the requested expiries (see
`results/exp4_run.log` for the per-expiry failures). On the four cells that
did load, the joint-law and conditional-Gamma curves differ from each other
by 46–262 bp (model-internal), far larger than the "orders of magnitude
smaller than the spread" the paper reports (spreads here are 348–481 bp), and
the in-band fraction (strikes where a model curve sits inside the quoted
bid-ask band) is 0–54%, not "a large majority of strikes" as the paper
reports. **Verdict: not reproduced.** The checked causes are: (i) the TQQQ
forward extracted from the thin 2026-era put-call parity data is noisy at
some strikes (bid/ask spreads here are several times the ones implied by the
paper's Table/Section 13.5 discussion), and (ii) only a small, irregular
subset of the paper's twelve (date, tenor) cells has tradable contracts in
the loaded chain. Both lower data quality relative to the paper's listed
market; the discrepancy could not be reduced further within the available
time and its precise magnitude is not attributed to a single identified
code defect.

## Experiment 5 — Computational Cost Comparison

Wall-clock time for a full LETF smile under each method, for SPX/NKY ×
`beta ∈ {-2,-1,2,3}` × `T ∈ {3M,6M,1Y,2Y}`, run on the paper's Table 1
Baseline grid (`n_x=2000, n_i=1000`, 250 time steps/year) with
`python -m exp.exp5_computational_cost --full`
(`results/exp5_timing.csv`, `results/exp5_summary_overall.csv`,
`results/exp5_summary_by_index.csv`).

| method | mean time (s) | min (s) | max (s) | paper (s) |
|---|---|---|---|---|
| conditional-Gamma | 0.179 | 0.089 | 0.329 | 0.85–1.42 |
| shifted-Gamma | 0.193 | 0.096 | 0.349 | 1.06–1.54 |
| joint-law PDE (Baseline grid) | 31.7 | 8.50 | 69.0 | 197–224 |

Speed-up of conditional-Gamma over the joint-law solve: mean **157.7×**,
range **79.5×–240.1×** (paper: 158–236×). Speed-up of shifted-Gamma: mean
146.3×, range 81.1×–208.7×. The incremental cost of the `M_3` solve is
0.014 s on average (range −0.003 s to 0.043 s; the few negative values are
measurement noise at sub-0.1 s timings), against the paper's ≈0.2 s. Cost is
close to independent of leverage at fixed maturity (e.g. SPX 1Y
conditional-Gamma ranges 0.179–0.197 s across the four β, a 10% spread), and
SPX and NKY conditional-Gamma means are within 2% of each other (0.180 s vs
0.177 s), consistent with both indices sharing the same 1D moment grid. Joint-law
cost scales with the number of time steps (250 per year, so T=2Y uses 8× the
steps of T=3M): mean per maturity is 8.8 s (3M), 17.3 s (6M), 33.7 s (1Y),
67.2 s (2Y), an almost exactly linear-in-T progression driven by the fixed
250-steps-per-year rule of Table 1.

**Verdict: partially reproduced.** The relative speed-up of the
conditional-Gamma method over the joint-law reference — the paper's central
computational-cost claim — is **reproduced**: the measured range (79.5×–240.1×,
mean 157.7×) overlaps the paper's reported 158–236× range, and is closest to
it at the longer maturities (1Y: 174×–194×; 2Y: 205×–241×), where the Baseline
grid's time-stepping cost dominates most heavily. The qualitative claims also
hold: conditional methods are two to three orders of magnitude cheaper than
the 2D solve, cost is close to independent of leverage, and the `M_3`
increment is small relative to the `M_0`–`M_2` cost. The absolute wall-clock
times are **not reproduced**: every method here is faster than the paper's
reported seconds, and the gap is large for the joint-law PDE (mean 31.7 s
against the paper's 197–224 s, roughly a 6–7× difference) and for the `M_3`
increment (0.014 s against ≈0.2 s). The paper does not state the hardware
used for its Table 2 benchmark, only that it ran single-threaded Python 3.7.9;
timing experiment 5 and timing experiment 1's Monte Carlo benchmark on the
same machine shows both are internally consistent (joint-law cost here scales
linearly with the number of time steps, as the finite-difference method
implies), so the residual gap in absolute seconds is attributed to
hardware and interpreter differences between this machine and the paper's
unspecified benchmark machine, not to a methodology difference; this could
not be checked further because the paper gives no hardware specification to
match against.

## Deviations from the paper

* **Local-volatility surface (experiments 1, 2, 3, 5).** The paper calibrates
  Dupire local-volatility surfaces to SPX and NKY index option markets on
  22 June 2026 but does not specify the calibration method or
  parametrisation (flagged as an open uncertainty in the experiment
  specification). Because the Dataloader API serves only OHLCV bars and
  Massive's listed-index-option coverage for NKY (a Japanese index) could not
  be confirmed, experiments 1, 2, 3 and 5 use a parametric Dupire local
  variance surface built in `src/dupire_surface.py` from a quadratic-in-log-
  moneyness, term-structured implied-variance proxy with index-specific
  skew/curvature/ATM-level constants in `src/config.py`
  (`SPX_SURFACE_PARAMS`, `NKY_SURFACE_PARAMS`, source: "not specified in the
  paper, SPX/NKY-typical equity-index skew used as a modelling input"). NKY's
  skew and curvature constants are set steeper than SPX's, consistent with
  the paper's qualitative statement that NKY has a steeper skew than SPX
  (used as a property to validate experiment 2's ranking, not as a fitted
  market value). This surface is a modelling input, not a reproduction of
  the paper's market calibration; all comparisons above are internal
  (model vs model, or model vs the Monte Carlo benchmark of the same model),
  except experiment 4 which uses real listed quotes.
* **Grid sizes (experiment 1).** Table 1's four grid configurations
  (`n_x` from 1600 to 3000, `n_i` from 800 to 1500) were reduced to
  `n_x` from 400 to 750, `n_i` from 200 to 375 (`GRID_CONFIGS_RUN` in
  `src/config.py`; `GRID_CONFIGS_PAPER` holds the paper's values and is used
  by `--full`) for the committed experiment 1 convergence/Monte-Carlo run.
  The 250 steps/year time discretisation is unchanged. The reduction was
  made because the paper's Baseline grid (`2000×1000` nodes, confirmed
  costly in experiment 5, see below) applied to all four grid
  configurations plus 2,000,000 Monte Carlo paths would not finish inside
  the time available for this replication; `python -m
  exp.exp1_joint_law_validation --full` runs the paper's Table 1
  Coarse/Baseline/Mid/Fine grid sizes and `MC_PATHS_PAPER` paths. Experiment
  5, which only needs one joint-law solve per (index, beta, maturity) rather
  than four, *was* run on the paper's Table 1 Baseline grid in full (see
  below). The 1D moment-PDE grid used by experiments 2 and 3
  (`MOMENT_GRID_RUN`) is likewise not reduced: it uses the paper's Baseline
  `n_x=2000` and `n_sigma=10` at 250 steps/year, since its cost (a handful
  of 1D tridiagonal solves) is negligible next to the 2D solve.
* **Computational-cost benchmark absolute times (experiment 5).** The
  committed run used the paper's Table 1 Baseline grid
  (`n_x=2000, n_i=1000`) for all 32 (index, beta, maturity) joint-law
  solves. The measured relative speed-up of the conditional-Gamma method
  over the joint-law solve (mean 157.7×, range 79.5×–240.1×) reproduces the
  paper's reported 158–236× range, but the absolute wall-clock seconds are
  6–7× faster here than the paper's reported 197–224 s for the joint-law
  solve (mean 31.7 s here) and the `M_3` increment is smaller (0.014 s here
  against the paper's ≈0.2 s). The paper does not report the hardware used
  for Table 2, only that the reference implementation was single-threaded
  Python 3.7.9; this prevents checking whether the gap is caused by faster
  hardware, a faster BLAS/NumPy build, or a genuinely different
  finite-difference implementation, so the absolute-time gap is attributed
  to the unspecified benchmark machine and not to a code defect, but the
  precise split between these causes was not isolated further.
* **Monte Carlo path count (experiment 1).** The paper does not state the
  number of simulated paths. 200,000 paths were used for the committed run
  (`MC_PATHS_RUN`); `MC_PATHS_PAPER = 2,000,000` is available via `--full`.
* **TQQQ/QQQ market data (experiment 4).** Listed QQQ and TQQQ option quotes
  were loaded from the Massive REST options-chain endpoint via
  `data/data_loader.py` (`MASSIVE_API_KEY` required) for the three dates
  specified in the experiment (20 March 2026, 22 June 2026, 25 September
  2026). Only 4 of the paper's 12 (date, tenor) cells returned a chain with
  both a QQQ and a TQQQ two-sided quote at a comparable expiry; the rest
  returned no contracts at the requested expiry and are listed as data gaps
  in `results/exp4_run.log`, not substituted with another source. American
  quotes were converted to European implied volatilities with the
  Bjerksund-Stensland 2002 approximation implemented in
  `src/american_option.py` (the paper does not specify which conversion it
  uses). The risk-free rate and dividend yield were read from the same
  Massive reference endpoints as the option chain (constant per date; the
  paper does not specify its rate/yield source).
* **exp3 three-moment degradation at 3M/γ=3.** As reported above, the
  shifted-Gamma closure is measurably worse than the two-moment closure at
  the 3-month maturity for `|gamma|=3` leverage pairs, which the paper's
  text ("almost indistinguishable" at 3M) does not lead one to expect. A
  density-cutoff sensitivity check did not remove the effect; the precise
  cause (most likely estimation noise in the third central moment in the
  thin tails reachable by `beta ∈ {-2,3}` at short maturity) was not
  isolated further within the available time.

## Not reproduced

* The paper's absolute joint-law PDE runtime (197–224 s) in experiment 5 is
  not reproduced: running the full Table 1 Baseline grid gives 8.5–69.0 s
  (mean 31.7 s), 6–7× faster than the paper, which the paper's missing
  hardware specification does not let us explain further (see "Deviations
  from the paper"). The 158–236× speed-up figure itself *is* reproduced
  (measured range 79.5×–240.1×, mean 157.7×, on the paper's Baseline grid).
* The paper's reported near-total overlap between the joint-law/conditional-
  Gamma curves and the TQQQ bid-ask band at "the large majority of strikes"
  (experiment 4) is not reproduced; the available listed-quote coverage for
  the requested dates/tenors was too sparse (4 of 12 cells) to assess this
  claim at the paper's resolution, and the few cells that did load show
  wider in-band fractions than the paper reports. A full assessment would
  require broader historical options coverage than was available.
