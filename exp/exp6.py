"""Experiment 6: surrogate (reshuffled-returns) significance test for
drawdown outliers (paper Section 3.5, Tables 6-7).

For each instrument, 1000 reshufflings of the daily log-returns destroy
serial correlation while preserving the marginal return distribution; the
number of large drawdowns in the real series is compared with the
distribution of counts across the 1000 surrogates to obtain a non-parametric
confidence level for the outlier hypothesis.

Run: python -m exp.exp6
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.data_loader import (
    DataLoaderError,
    company_close_series,
    load_company_bars,
    load_fred_series,
    load_index_series,
)
from src.config import (
    COMPANIES,
    CURRENCIES,
    N_SURROGATES,
    SURROGATE_SEED,
    SURROGATE_THRESHOLDS_PAPER,
    WORLD_INDICES,
)
from src.drawdown import compute_drawdowns_drawups
from src.pipeline import analyze_instrument
from src.stretched_exponential import relative_error_curve
from src.surrogate import confidence_level, surrogate_drawdown_counts

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Companies tested for the surrogate significance test (paper Table 7: 25 of
# the 29 companies of experiment 5; Oracle, AOL, JDS Uniphase and American
# Home Products are not listed in Table 7 and are skipped here, matching the
# paper's own selection). Of the 25, Lucent, Sun Microsystems, SBC
# Communications, EMC and MCI WorldCom could not be loaded (see
# "Not reproduced" in results/RESULTS.md), leaving 20 companies tested here.
SURROGATE_COMPANIES = [
    "MSFT", "CSCO", "GE", "INTC", "XOM", "WMT", "IBM", "T", "C",
    "TXN", "HPQ", "MRK", "PFE", "KO", "QCOM", "AMAT", "PG", "GM", "MDT", "F",
]


def _auto_threshold(amplitudes: np.ndarray, fit) -> float | None:
    """Smallest amplitude at which the relative error of the stretched-
    exponential fit departs persistently above zero, taken as the empirical
    bulk/outlier crossover (paper Section 3 error plots). Returns the integer
    percentage just above the crossover, matching the paper's Table 6/7
    thresholds (e.g. "DJIA -18%"). None if no crossover is found."""
    if fit is None or len(amplitudes) < 10:
        return None
    x_sorted, rel_err = relative_error_curve(amplitudes, fit)
    # Scan from the largest event down, extending the outlier run while the
    # relative error stays persistently positive; the crossover is the
    # smallest amplitude still inside that run (paper Section 3 error plots:
    # bulk errors fluctuate near zero, outliers depart systematically upward).
    crossover_idx = None
    for i in range(len(x_sorted) - 1, -1, -1):
        if rel_err[i] > 0.1:
            crossover_idx = i
        else:
            break
    if crossover_idx is None or (len(x_sorted) - crossover_idx) < 2:
        return None
    return float(np.floor(x_sorted[crossover_idx]))


def _run_one(key: str, name: str, prices, thresholds_paper: dict) -> dict:
    result = analyze_instrument(prices, name)
    threshold = thresholds_paper.get(key)
    auto_threshold = _auto_threshold(result.drawdowns, result.fit_drawdown)
    if threshold is None:
        threshold = auto_threshold
    if threshold is None:
        return {"label": name, "n_obs": result.n_obs, "error": "no outlier crossover detected"}

    real_count = int(np.sum(result.drawdowns >= threshold))
    counts = surrogate_drawdown_counts(prices, threshold, N_SURROGATES, SURROGATE_SEED)
    cl = confidence_level(real_count, counts)
    hist = {int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}
    return {
        "label": name, "n_obs": result.n_obs, "threshold_pct": threshold,
        "auto_detected_threshold_pct": auto_threshold,
        "real_count_above_threshold": real_count,
        "surrogate_count_histogram": hist,
        "surrogate_mean": round(float(np.mean(counts)), 3),
        "confidence_level": round(cl, 4),
    }


def run() -> dict:
    results: dict = {"indices": {}, "currencies": {}, "companies": {}}

    for key, (symbol, start, end, name) in WORLD_INDICES.items():
        try:
            prices = load_index_series(symbol, start, end)
        except DataLoaderError as exc:
            results["indices"][key] = {"label": name, "error": str(exc)}
            continue
        results["indices"][key] = _run_one(key, name, prices, SURROGATE_THRESHOLDS_PAPER)
        print(f"  [index] {key}: {results['indices'][key]}")

    for key, (series_id, start, end, name) in CURRENCIES.items():
        try:
            prices = load_fred_series(series_id, start, end)
        except DataLoaderError as exc:
            results["currencies"][key] = {"label": name, "error": str(exc)}
            continue
        results["currencies"][key] = _run_one(key, name, prices, SURROGATE_THRESHOLDS_PAPER)
        print(f"  [currency] {key}: {results['currencies'][key]}")

    company_tickers = [COMPANIES[k][0] for k in SURROGATE_COMPANIES if k in COMPANIES]
    try:
        bars, _meta = load_company_bars(company_tickers, "1962-01-01", "2000-07-01",
                                         identity_date="2000-07-01")
    except DataLoaderError as exc:
        results["companies"] = {"error": str(exc)}
        bars = None

    if bars is not None:
        for key in SURROGATE_COMPANIES:
            symbol, start, end, name = COMPANIES[key]
            try:
                prices = company_close_series(bars, symbol)
            except DataLoaderError as exc:
                results["companies"][key] = {"label": name, "error": str(exc)}
                continue
            prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
            if len(prices) < 30:
                results["companies"][key] = {"label": name, "error": "too few observations"}
                continue
            results["companies"][key] = _run_one(key, name, prices, {})
            print(f"  [company] {key}: {results['companies'][key]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp6_surrogate_significance.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Experiment 6 complete. Table written to results/exp6_surrogate_significance.json")
    return results


if __name__ == "__main__":
    run()
