"""Experiment 2: stretched-exponential fit of drawdown/drawup distributions
for 12 major world financial indices (paper Section 3, Tables 2-3, Figures
2-4 and the error plots of Section 3).

Run: python -m exp.exp2
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.data_loader import DataLoaderError, load_index_series
from src.config import WORLD_INDICES
from src.pipeline import analyze_instrument
from src.plotting import plot_relative_error, plot_survival_with_fit

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# A handful of indices shown in detail plots (bulk vs outlier regime contrast,
# paper Figures 2-4): DJIA and S&P500 (clear outliers), CAC40 (paper's
# exception with a near-perfect exponential fit and no visible outliers).
HIGHLIGHT = ["DJIA", "SP500", "CAC40"]


def run() -> dict:
    results = {}
    for key, (symbol, start, end, name) in WORLD_INDICES.items():
        try:
            prices = load_index_series(symbol, start, end)
        except DataLoaderError as exc:
            results[key] = {"label": name, "error": str(exc)}
            print(f"  {key} ({name}): FAILED to load - {exc}")
            continue
        result = analyze_instrument(prices, name)
        results[key] = result.to_summary()
        results[key]["_result_obj"] = result  # kept only in-memory for plotting
        print(f"  {key} ({name}): n_obs={result.n_obs} [{result.start}..{result.end}] "
              f"n_drawdowns={len(result.drawdowns)} n_drawups={len(result.drawups)}")
        if result.fit_drawdown is not None:
            f = result.fit_drawdown
            print(f"    drawdown fit: chi={f.chi:.3f}% z={f.z:.3f} (A={f.A})")
        if result.fit_drawup is not None:
            f = result.fit_drawup
            print(f"    drawup   fit: chi={f.chi:.3f}% z={f.z:.3f} (A={f.A})")

    _plot_highlights(results)

    table = {k: {kk: vv for kk, vv in v.items() if kk != "_result_obj"} for k, v in results.items()}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp2_indices.json", "w") as f:
        json.dump(table, f, indent=2, default=str)

    print("Experiment 2 complete. Table written to results/exp2_indices.json")
    return table


def _plot_highlights(results: dict) -> None:
    present = [k for k in HIGHLIGHT if k in results and "error" not in results[k]]
    if not present:
        return
    fig, axes = plt.subplots(len(present), 2, figsize=(11, 3.3 * len(present)), squeeze=False)
    for i, key in enumerate(present):
        r = results[key]["_result_obj"]
        if r.fit_drawdown is not None:
            plot_survival_with_fit(axes[i][0], r.drawdowns, r.fit_drawdown, color="C0", label="drawdown")
        if r.fit_drawup is not None:
            plot_survival_with_fit(axes[i][0], r.drawups, r.fit_drawup, color="C3", label="drawup")
        axes[i][0].set_title(f"{r.label}: Nc(x) and stretched-exponential fit")
        axes[i][0].legend(fontsize=7)

        if r.fit_drawdown is not None:
            plot_relative_error(axes[i][1], r.drawdowns, r.fit_drawdown, color="C0", label="drawdown")
        if r.fit_drawup is not None:
            plot_relative_error(axes[i][1], r.drawups, r.fit_drawup, color="C3", label="drawup")
        axes[i][1].set_title(f"{r.label}: relative error (Nc_emp-Nc_fit)/Nc_fit")
        axes[i][1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_survival_and_error_plots.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
