# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this repository implements

A replication of Johansen & Sornette, "Large Stock Market Price Drawdowns Are
Outliers" (*Journal of Risk*, 2001; arXiv:cond-mat/0010050): the toy
nonlinear-moving-average model (Section 2), stretched-exponential fits of
drawdown/drawup distributions for world indices, currencies and gold (Section
3), the robustness of the fitted outliers to added noise (Section 3.4), the
normalized-distribution and frequency-regression analysis for individual US
companies (Section 4), and the surrogate-data significance test (Sections
3.5, 4).

## Layout

- `src/` — flat library modules: `toy_model.py` (Section 2), `drawdown.py`
  (run detection and amplitude computation, Section 3),
  `stretched_exponential.py` (the fit and its error bars),
  `robustness.py` (noise injection, Section 3.4), `surrogate.py`
  (reshuffled-return significance test), `pipeline.py` (the single
  per-instrument analysis shared by experiments 2-6), `plotting.py`, and
  `config.py` holding every paper-derived parameter as module-level
  `UPPER_CASE` variables.
- `exp/` — one script per experiment, `exp1.py` .. `exp6.py`, each calling
  only the `src/` functions.
- `data/` — `data_loader.py`, the only place market data is fetched: company
  stocks through the QuantX Dataloader REST API, indices through Yahoo
  Finance, currencies/gold through FRED (see the module docstring and
  `results/RESULTS.md` for why each instrument family needs a different
  source). No loader code lives in `src/` or `exp/`.
- `tests/` — pytest tests of `src/` components (synthetic fixtures) and of
  the data path (`tests/conftest.py`, marked `market_data`, skipped without
  `DATALOADER_API_KEY`/`MASSIVE_API_KEY`).
- `results/` — every table, metric and plot the experiments produce, plus
  `RESULTS.md` with the comparison against the paper.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env   # DATALOADER_API_KEY, MASSIVE_API_KEY

python -m pytest -m "not market_data"   # unit tests, no network
python -m pytest                        # full suite, needs the API keys

python -m exp.exp1   # .. python -m exp.exp6
./reproduce.sh        # tests + all six experiments
```

## Conventions

- All experiment parameters live only in `src/config.py` as module-level
  variables with a comment on meaning, unit and paper source; no parameter is
  hardcoded inside `src/` or `exp/`.
- Market data is loaded only through `data/data_loader.py`; downloaded bars
  are never committed (`data/cache/` is in `.gitignore`).
- API keys are read from the environment via `python-dotenv`/`os.environ` and
  never appear in code, logs or results; `.env` is git-ignored and
  `.env.example` holds only empty placeholders.
- Tests run from the repository root as `python -m pytest`; experiments run
  as `python -m exp.<name>`. No `sys.path` edits.
- Running an experiment overwrites its files in `results/`. After a rerun,
  check the numbers and verdicts in `results/RESULTS.md` against the new
  files and update them by hand; they are not regenerated automatically.
