# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A reproduction of selected numerical experiments from *Beyond Strike
Remapping: Conditional-Gamma Pricing of Leveraged ETF Options under Local
Volatility*: the forward joint-law Fokker-Planck PDE for leveraged ETF
(LETF) options under Dupire local volatility (Section 11), its Monte Carlo
validation (Section 13.2), the conditional-Gamma closure (Section 7) and its
three-moment shifted-Gamma extension (Section 8), the first-order strike
remapping baseline (Section 2), an empirical comparison against listed
TQQQ/QQQ quotes (Section 13.5), and a wall-clock cost comparison of the three
pricing methods (Section 13.6).

## Layout

- `src/` — flat library modules: `dupire_surface.py`, `black_scholes.py`,
  `joint_law_pde.py` (2D Fokker-Planck solver), `moment_pde.py` (triangular
  1D forward moment system), `gamma_closure.py` (two- and three-moment
  closures), `strike_remapping.py`, `monte_carlo.py`, `american_option.py`,
  `market_inputs.py`, `pricing_pipeline.py` (shared entry points called by
  every `exp/` script), `config.py` (every experiment parameter as a module
  level UPPER_CASE variable, with a comment on its meaning, unit and the
  paper section it comes from, or stating that the paper does not specify
  it).
- `exp/` — one script per experiment (`exp1_joint_law_validation.py` ...
  `exp5_computational_cost.py`), calling only the `src/` functions as written.
- `data/` — `data_loader.py`: the Massive REST option-chain loader used by
  experiment 4. Experiments 1-3 and 5 use a parametric local-volatility
  surface built in `src/dupire_surface.py` from `src/config.py` and need no
  network access.
- `tests/` — pytest suite. `tests/conftest.py` registers the `market_data`
  marker and provides fixtures; tests of experiment 4's data path are marked
  `@pytest.mark.market_data` and are skipped automatically when
  `MASSIVE_API_KEY` is not set.
- `results/` — every table, plot and log produced by the `exp/` scripts, plus
  `results/RESULTS.md` with the comparison against the paper.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"   # unit tests, no network, default CI command
python -m pytest                         # full suite including experiment-4 data tests

python -m exp.exp1_joint_law_validation
python -m exp.exp2_conditional_gamma_vs_remapping
python -m exp.exp3_shifted_gamma_closure
python -m exp.exp4_tqqq_market_validation
python -m exp.exp5_computational_cost

# paper's Table 1 grid sizes and Monte Carlo path count instead of the
# reduced configuration committed in results/:
python -m exp.exp1_joint_law_validation --full
python -m exp.exp5_computational_cost --full
```

`reproduce.sh` runs the above test and experiment commands in sequence.

## Conventions

- All experiment parameters live only in `src/config.py` as module-level
  UPPER_CASE variables (no classes or dicts of parameters); every variable
  has a one-line comment giving its meaning, unit and source (a paper
  section/equation or "not specified in the paper").
- Market data is loaded only through `data/data_loader.py`; `src/` and `exp/`
  never call a data API directly. `DATALOADER_API_KEY` and `MASSIVE_API_KEY`
  are read from the environment via `python-dotenv` and must only ever live
  in an uncommitted `.env` (see `.env.example` for the placeholders).
- Experiments are run as `python -m exp.<name>` from the repository root and
  tests as `python -m pytest`; no `sys.path` edits in code.
- Running an experiment script overwrites its files in `results/`. The
  figures, numbers and verdicts written in `results/RESULTS.md` reflect the
  run described there; after a rerun, re-check and update them by hand.
- Deviations from the paper (parametric local-volatility surface, reduced
  grid/path-count configuration, market-data coverage gaps) are recorded only
  in `results/RESULTS.md`, not scattered across code comments beyond the
  one-line source tag in `src/config.py`.
