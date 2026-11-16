#!/usr/bin/env bash
# Reproduce every experiment of the drawdown-outlier replication.
#
# Johansen, A. and Sornette, D., "Large Stock Market Price Drawdowns Are
# Outliers", Journal of Risk, 2001 (arXiv:cond-mat/0010050).
set -uo pipefail

cd "$(dirname "$0")"

echo "== Unit tests (no market data) =="
python -m pytest -m "not market_data"

echo "== Experiment 1: toy model =="
python -m exp.exp1

echo "== Experiment 2: world indices =="
python -m exp.exp2

echo "== Experiment 3: currencies and gold =="
python -m exp.exp3

echo "== Experiment 4: robustness to added noise =="
python -m exp.exp4

echo "== Experiment 5: US companies =="
python -m exp.exp5

echo "== Experiment 6: surrogate significance test =="
python -m exp.exp6

echo "All experiments finished. See results/RESULTS.md for the paper comparison."
