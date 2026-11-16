"""Experiment 5: drawdown/drawup analysis for individual US companies (paper
Section 4, Tables 4-5, Figures 22-23).

Builds the stretched-exponential fit for each company, the normalized
collapse plot Nc/A vs y = sign(x)*(|x|/chi)^z (paper Figure 22), a qualitative
outlier classification, the annual frequency of events above
``COMPANY_DRAWDOWN_THRESHOLD_PCT`` and the linear regression between
drawup and drawdown frequencies (paper Figure 23).

Run: python -m exp.exp5
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from data.data_loader import DataLoaderError, company_close_series, load_company_bars
from src.config import COMPANIES, COMPANY_DRAWDOWN_THRESHOLD_PCT
from src.pipeline import analyze_instrument
from src.plotting import plot_normalized_collapse

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Outlier-classification cutoff on the relative error curve (Nc_emp-Nc_fit)/Nc_fit
# averaged over the largest 2% of events (paper Section 3 defines the outlier
# regime as the ~2% largest events departing from the fit; the paper classifies
# companies by visual inspection of this departure, so a fixed numeric cutoff
# on the same quantity is an interpreted, reproducible stand-in - see
# results/RESULTS.md, "Deviations from the paper").
OUTLIER_RELATIVE_ERROR_CUTOFF = 1.0
OUTLIER_TOP_FRACTION = 0.02


def _years_span(start: str, end: str) -> float:
    start_dt, end_dt = np.datetime64(start), np.datetime64(end)
    return max((end_dt - start_dt).astype("timedelta64[D]").astype(float) / 365.25, 1e-6)


def _has_outlier(amplitudes: np.ndarray, fit) -> bool | None:
    """True if the top OUTLIER_TOP_FRACTION of events show a mean positive
    relative departure from the stretched-exponential fit above the cutoff
    (paper Section 3: outliers are the largest ~2% of events departing from
    the bulk fit); None if too few events to judge."""
    if fit is None or len(amplitudes) < 10:
        return None
    x_sorted = np.sort(amplitudes)
    n_top = max(1, int(round(OUTLIER_TOP_FRACTION * len(x_sorted))))
    top = x_sorted[-n_top:]
    nc_top = np.arange(n_top, 0, -1, dtype=float)  # Nc for the largest n_top events
    nc_fit_top = fit.A * np.exp(-(top / fit.chi) ** fit.z)
    rel_err = (nc_top - nc_fit_top) / nc_fit_top
    return bool(np.mean(rel_err) > OUTLIER_RELATIVE_ERROR_CUTOFF)


def run() -> dict:
    tickers = [symbol for symbol, _, _, _ in COMPANIES.values()]
    try:
        bars, meta = load_company_bars(tickers, "1962-01-01", "2000-07-01",
                                        identity_date="2000-07-01")
    except DataLoaderError as exc:
        print(f"FAILED to load company bars: {exc}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "exp5_companies.json", "w") as f:
            json.dump({"error": str(exc)}, f, indent=2)
        return {"error": str(exc)}

    results: dict = {}
    collapse_entries = []
    freq_dd, freq_du, freq_labels = [], [], []
    for key, (symbol, start, end, name) in COMPANIES.items():
        try:
            prices = company_close_series(bars, symbol)
        except DataLoaderError as exc:
            results[key] = {"label": name, "error": str(exc)}
            print(f"  {key} ({name}): FAILED to load - {exc}")
            continue
        prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
        if len(prices) < 30:
            results[key] = {"label": name, "error": "fewer than 30 observations in window"}
            continue
        result = analyze_instrument(prices, name)
        row = result.to_summary()

        years = _years_span(result.start, result.end)
        n_dd_15 = int(np.sum(result.drawdowns >= COMPANY_DRAWDOWN_THRESHOLD_PCT))
        n_du_15 = int(np.sum(result.drawups >= COMPANY_DRAWDOWN_THRESHOLD_PCT))
        row["years_span"] = round(years, 2)
        row["drawdowns_per_year_gt15"] = round(n_dd_15 / years, 4)
        row["drawups_per_year_gt15"] = round(n_du_15 / years, 4)
        row["drawdown_outlier"] = _has_outlier(result.drawdowns, result.fit_drawdown)
        row["drawup_outlier"] = _has_outlier(result.drawups, result.fit_drawup)
        results[key] = row
        print(f"  {key} ({name}): n_obs={result.n_obs} dd/yr>15%={row['drawdowns_per_year_gt15']:.2f} "
              f"du/yr>15%={row['drawups_per_year_gt15']:.2f} "
              f"dd_outlier={row['drawdown_outlier']} du_outlier={row['drawup_outlier']}")

        if result.fit_drawdown is not None:
            collapse_entries.append((result.drawdowns, result.fit_drawdown, -1, name))
        if result.fit_drawup is not None:
            collapse_entries.append((result.drawups, result.fit_drawup, 1, name))
        freq_dd.append(row["drawdowns_per_year_gt15"])
        freq_du.append(row["drawups_per_year_gt15"])
        freq_labels.append(key)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Normalized collapse plot (paper Figure 22): all companies on one axis.
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_normalized_collapse(ax, collapse_entries)
    ax.set_title("Normalized drawdown/drawup collapse, 29 US companies (paper Fig. 22)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_normalized_collapse.png", dpi=120)
    plt.close(fig)

    # Linear regression of drawup frequency on drawdown frequency (paper
    # Figure 23): scipy.stats.linregress, ordinary least squares with a free
    # intercept, as the paper reports an intercept found to be negligible.
    regression = {}
    if len(freq_dd) >= 3:
        lr = stats.linregress(freq_dd, freq_du)
        regression = {
            "slope": round(float(lr.slope), 4),
            "intercept": round(float(lr.intercept), 4),
            "r_value": round(float(lr.rvalue), 4),
            "r_squared": round(float(lr.rvalue ** 2), 4),
            "p_value": float(lr.pvalue),
            "n_companies": len(freq_dd),
            "mean_drawdowns_per_year_gt15": round(float(np.mean(freq_dd)), 4),
            "mean_drawups_per_year_gt15": round(float(np.mean(freq_du)), 4),
        }
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(freq_dd, freq_du, s=20)
        x_grid = np.linspace(0, max(freq_dd) * 1.1 + 1e-6, 50)
        ax.plot(x_grid, lr.intercept + lr.slope * x_grid, "r-",
                label=f"fit: slope={lr.slope:.2f}, r={lr.rvalue:.2f}")
        ax.set_xlabel("drawdowns/year with amplitude > 15%")
        ax.set_ylabel("drawups/year with amplitude > 15%")
        ax.set_title("Drawup vs drawdown frequency, 29 US companies (paper Fig. 23)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "exp5_dd_du_regression.png", dpi=120)
        plt.close(fig)

    table = {
        "companies": results,
        "regression": regression,
        "non_massive_source": meta.get("non_massive_source", {}),
        "failed_tickers": meta.get("failed", {}),
    }
    with open(RESULTS_DIR / "exp5_companies.json", "w") as f:
        json.dump(table, f, indent=2, default=str)
    print("Experiment 5 complete. Table written to results/exp5_companies.json")
    return table


if __name__ == "__main__":
    run()
