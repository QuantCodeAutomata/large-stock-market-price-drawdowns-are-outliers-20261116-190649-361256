"""Experiment 4: robustness of the drawdown/drawup outlier regime to added
noise of different amplitudes (paper Section 3.4).

Run: python -m exp.exp4
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.data_loader import DataLoaderError, load_index_series
from src.config import NOISE_AMPLITUDES, NOISE_SEED, WORLD_INDICES
from src.drawdown import compute_drawdowns_drawups
from src.plotting import plot_noise_overlay
from src.robustness import noisy_price_series

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Robustness test run only on the three indices the paper names for this test
# (paper Section 3.4): DJIA, S&P500, Nasdaq.
NOISE_TEST_INDICES = ["DJIA", "SP500", "NASDAQ"]


def run() -> dict:
    results: dict = {}
    for key in NOISE_TEST_INDICES:
        symbol, start, end, name = WORLD_INDICES[key]
        try:
            prices = load_index_series(symbol, start, end)
        except DataLoaderError as exc:
            results[key] = {"label": name, "error": str(exc)}
            print(f"  {key} ({name}): FAILED to load - {exc}")
            continue

        drawdown_sets: dict[float, np.ndarray] = {}
        drawup_sets: dict[float, np.ndarray] = {}
        counts = {}
        for amplitude in NOISE_AMPLITUDES:
            if amplitude == 0.0:
                noisy = prices
            else:
                # Independent seed per noise level so realizations do not
                # repeat the same uniform draw across amplitudes.
                noisy = noisy_price_series(prices, amplitude, seed=NOISE_SEED + int(amplitude * 10))
            drawdowns, drawups = compute_drawdowns_drawups(noisy)
            drawdown_sets[amplitude] = drawdowns
            drawup_sets[amplitude] = drawups
            counts[str(amplitude)] = {
                "n_drawdowns": int(len(drawdowns)),
                "n_drawups": int(len(drawups)),
                "max_drawdown_pct": float(drawdowns.max()) if len(drawdowns) else None,
                "max_drawup_pct": float(drawups.max()) if len(drawups) else None,
            }
        results[key] = {"label": name, "n_obs": len(prices), "counts": counts}
        print(f"  {key} ({name}): " + ", ".join(
            f"A={a}: n_dd={counts[str(a)]['n_drawdowns']} n_du={counts[str(a)]['n_drawups']}"
            for a in NOISE_AMPLITUDES))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_noise_overlay(axes[0], drawdown_sets, f"{name}: drawdowns under added noise")
        plot_noise_overlay(axes[1], drawup_sets, f"{name}: drawups under added noise")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"exp4_noise_{key}.png", dpi=120)
        plt.close(fig)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp4_noise_robustness.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Experiment 4 complete. Table written to results/exp4_noise_robustness.json")
    return results


if __name__ == "__main__":
    run()
