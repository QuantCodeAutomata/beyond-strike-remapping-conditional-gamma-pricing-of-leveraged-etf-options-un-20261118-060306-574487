# Beyond Strike Remapping: Conditional-Gamma Pricing of Leveraged ETF Options under Local Volatility

Replication of selected numerical experiments from *Beyond Strike Remapping:
Conditional-Gamma Pricing of Leveraged ETF Options under Local Volatility*.

This repository implements, for leveraged and inverse ETFs (LETFs) under a
Dupire local-volatility model of the underlying index:

1. the two-dimensional forward joint-law Fokker-Planck PDE for the augmented
   process `(X_t, I_t)` (terminal log-return, realised integrated variance),
   validated by grid refinement and by an independent Monte Carlo simulation
   of the LETF's own log-Euler dynamics (Section 13.2, equations (66), (71)-(73));
2. the conditional-Gamma closure, which conditions on `X_T`, matches the first
   two moments of `I_T | X_T` from a triangular system of 1D forward moment
   PDEs, and prices in closed form with the regularised incomplete Gamma
   function (Sections 7, 9.2);
3. the first-order strike remapping baseline (Section 2, equations (13)-(14));
4. the three-moment shifted-Gamma closure, which also matches conditional
   skewness subject to an admissibility constraint (Section 8);
5. empirical validation against listed TQQQ/QQQ option quotes (Section 13.5);
6. a wall-clock cost comparison of the three pricing methods (Section 13.6).

## Repository layout

```
src/    Dupire surface, Black-Scholes/IV, joint-law PDE, moment PDE system,
        conditional-Gamma and shifted-Gamma closures, strike remapping,
        Monte Carlo benchmark, American-exercise adjustment, pricing pipeline,
        config.py (all paper-derived parameters)
exp/    exp1_joint_law_validation.py, exp2_conditional_gamma_vs_remapping.py,
        exp3_shifted_gamma_closure.py, exp4_tqqq_market_validation.py,
        exp5_computational_cost.py
data/   data_loader.py: Massive REST option-chain loader for experiment 4
tests/  pytest suite (unit tests on synthetic/small grids; market_data tests
        marked and skipped without API keys)
results/ RESULTS.md and every table/plot produced by the experiment scripts
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY for experiment 4 only
```

Experiments 1-3 and 5 use a parametric Dupire local-volatility surface (see
"Deviations from the paper" in `results/RESULTS.md`) and require no API keys.
Experiment 4 queries listed QQQ/TQQQ option quotes through the Massive REST
API and requires `MASSIVE_API_KEY`.

## Running

```bash
python -m pytest -m "not market_data"   # unit tests (no network)
python -m pytest                         # full suite, needs MASSIVE_API_KEY

python -m exp.exp1_joint_law_validation
python -m exp.exp2_conditional_gamma_vs_remapping
python -m exp.exp3_shifted_gamma_closure
python -m exp.exp4_tqqq_market_validation
python -m exp.exp5_computational_cost
```

Each experiment accepts `--full` to run the paper's Table 1 grid sizes and
Monte Carlo path count instead of the reduced configuration used for the
results committed in this repository (see `results/RESULTS.md` for the
runtime that motivates the reduction, and `src/config.py` for the paper and
executed values side by side, e.g. `GRID_CONFIGS_PAPER` / `GRID_CONFIGS_RUN`
and `MC_PATHS_PAPER` / `MC_PATHS_RUN`).

`reproduce.sh` runs the test suite and all five experiments in sequence.

Every run overwrites the files in `results/`. The numbers and verdicts in
`results/RESULTS.md` were checked against the files produced by the runs
described there and must be re-checked by hand after a rerun.
