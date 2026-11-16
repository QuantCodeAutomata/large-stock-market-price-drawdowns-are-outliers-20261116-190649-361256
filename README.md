# Large Stock Market Price Drawdowns Are Outliers — Replication

Paper: Johansen, A. and Sornette, D., "Large Stock Market Price Drawdowns Are
Outliers", *Journal of Risk*, 2001 (arXiv:cond-mat/0010050).

This repository reproduces the paper's drawdown/drawup analysis: a toy model
showing that drawdowns detect higher-order correlations invisible to
two-point statistics (Section 2), stretched-exponential fits of drawdown and
drawup distributions for world indices, currencies and gold (Section 3),
robustness of the fitted outliers to added noise (Section 3.4), the
normalized-distribution analysis and drawdown/drawup frequency regression for
individual US companies (Section 4), and the surrogate-data significance test
for the outlier hypothesis (Sections 3.5 and 4).

## Layout

```
src/    toy model, drawdown/drawup computation, stretched-exponential fit,
        robustness and surrogate routines, plotting, and src/config.py with
        every paper-derived parameter
exp/    one script per experiment (exp1 .. exp6), run as python -m exp.expN
data/   data/data_loader.py: all market-data access (QuantX Dataloader for
        company stocks, Yahoo Finance for indices, FRED for FX series)
tests/  pytest unit tests of src/ components and experiment-level tests
results/  RESULTS.md and every table/plot/metric produced by the experiments
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

`DATALOADER_API_KEY` and `MASSIVE_API_KEY` are required for Experiments 5 and
6 (US company stocks, loaded through the QuantX Dataloader). Experiments 1-4
also use Yahoo Finance and FRED, which need no key.

## Running

```bash
python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                        # full suite, needs the API keys above

python -m exp.exp1   # toy model (Section 2)
python -m exp.exp2   # 12 world indices, stretched-exponential fit (Section 3)
python -m exp.exp3   # currencies and gold (Section 3)
python -m exp.exp4   # robustness to added noise (Section 3.4)
python -m exp.exp5   # 29 US companies, normalized distributions (Section 4)
python -m exp.exp6   # surrogate-data significance test (Sections 3.5, 4)
```

or, to run the unit tests and every experiment in sequence:

```bash
./reproduce.sh
```

Each experiment script overwrites its result files in `results/` on every
run.

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for the full comparison of
every fitted parameter and reported statistic against the paper's tables and
figures, with a verdict (reproduced / partially reproduced / not reproduced)
for each metric, and the "Deviations from the paper" and "Not reproduced"
sections documenting every data-source substitution and gap.
