"""Experiment 3: stretched-exponential fit of drawdown/drawup distributions
for currencies (paper Section 3, Table 2-3) and the US$/Yen asymmetry test.

Run: python -m exp.exp3
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.data_loader import DataLoaderError, load_fred_series
from src.config import CURRENCIES
from src.pipeline import analyze_instrument
from src.plotting import plot_relative_error, plot_survival_with_fit

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> dict:
    results = {}
    for key, (series_id, start, end, name) in CURRENCIES.items():
        try:
            prices = load_fred_series(series_id, start, end)
        except DataLoaderError as exc:
            results[key] = {"label": name, "error": str(exc)}
            print(f"  {key} ({name}): FAILED to load - {exc}")
            continue
        result = analyze_instrument(prices, name)
        results[key] = result.to_summary()
        results[key]["_result_obj"] = result
        print(f"  {key} ({name}): n_obs={result.n_obs} [{result.start}..{result.end}] "
              f"n_drawdowns={len(result.drawdowns)} n_drawups={len(result.drawups)}")
        if result.fit_drawdown is not None:
            f = result.fit_drawdown
            print(f"    drawdown fit: chi={f.chi:.3f}% z={f.z:.3f} (A={f.A})")
        if result.fit_drawup is not None:
            f = result.fit_drawup
            print(f"    drawup   fit: chi={f.chi:.3f}% z={f.z:.3f} (A={f.A})")

    _plot_all(results)
    _yen_asymmetry(results)

    table = {k: {kk: vv for kk, vv in v.items() if kk != "_result_obj"} for k, v in results.items()}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp3_currencies.json", "w") as f:
        json.dump(table, f, indent=2, default=str)

    print("Experiment 3 complete. Table written to results/exp3_currencies.json")
    return table


def _plot_all(results: dict) -> None:
    present = [k for k in results if "error" not in results[k]]
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
    fig.savefig(RESULTS_DIR / "exp3_survival_and_error_plots.png", dpi=120)
    plt.close(fig)


def _yen_asymmetry(results: dict) -> None:
    """US$/Yen drawdown vs drawup characteristic scale chi, same exponent z
    (paper Section 3: "a strong asymmetry ... same z but different chi")."""
    if "JPY" not in results or "error" in results["JPY"]:
        return
    r = results["JPY"]["_result_obj"]
    if r.fit_drawdown is None or r.fit_drawup is None:
        return
    summary = {
        "chi_drawdown": r.fit_drawdown.chi, "chi_drawup": r.fit_drawup.chi,
        "z_drawdown": r.fit_drawdown.z, "z_drawup": r.fit_drawup.z,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp3_yen_asymmetry.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  US$/Yen asymmetry: chi_dd={summary['chi_drawdown']:.3f}% "
          f"chi_du={summary['chi_drawup']:.3f}% z_dd={summary['z_drawdown']:.3f} "
          f"z_du={summary['z_drawup']:.3f}")


if __name__ == "__main__":
    run()
