"""Shared matplotlib helpers for drawdown/drawup cumulative-distribution plots
(paper Figures 2-4, 6-21 and 23).

Custom implementation: these are thin wrappers around matplotlib, not a
quantitative-analytics library function.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.stretched_exponential import (
    StretchedExponentialFit,
    empirical_complementary_cdf,
    relative_error_curve,
)


def plot_survival_with_fit(ax: plt.Axes, amplitudes: np.ndarray,
                            fit: StretchedExponentialFit, color: str = "C0",
                            label: str = "data") -> None:
    """Semi-log plot of the empirical Nc(x) with the stretched-exponential
    fit overlaid (paper Figure 2/3)."""
    x, nc = empirical_complementary_cdf(amplitudes)
    ax.semilogy(x, nc, ".", color=color, label=f"{label} (empirical, N={fit.A})")
    x_fit = np.linspace(max(x.min(), 1e-6), x.max(), 200)
    nc_fit = fit.A * np.exp(-(x_fit / fit.chi) ** fit.z)
    ax.semilogy(x_fit, nc_fit, "-", color=color, alpha=0.7,
                label=f"{label} fit (z={fit.z:.2f}, chi={fit.chi:.2f}%)")
    ax.set_xlabel("amplitude |x| (%)")
    ax.set_ylabel("Nc(x)")


def plot_relative_error(ax: plt.Axes, amplitudes: np.ndarray,
                         fit: StretchedExponentialFit, color: str = "C0",
                         label: str = "data") -> None:
    """(Nc_emp - Nc_fit)/Nc_fit vs |x|, used to locate the bulk/outlier
    crossover (paper Section 3 error plots)."""
    x, err = relative_error_curve(amplitudes, fit)
    ax.plot(x, err, ".", color=color, label=label, ms=4)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("amplitude |x| (%)")
    ax.set_ylabel("(Nc_emp - Nc_fit) / Nc_fit")


def plot_normalized_collapse(ax: plt.Axes,
                              entries: Iterable[tuple[np.ndarray, StretchedExponentialFit, int, str]]) -> None:
    """Normalized collapse plot: Nc/A vs y = sign(x)*(|x|/chi)^z for several
    instruments, with the universal curve exp(-|y|) overlaid (paper Section 4,
    Figure 22)."""
    for amplitudes, fit, sign, _label in entries:
        x_sorted, nc = empirical_complementary_cdf(amplitudes)
        y_sorted = sign * (x_sorted / fit.chi) ** fit.z
        ax.semilogy(y_sorted, nc / fit.A, ".", ms=2, alpha=0.4, color="C0" if sign < 0 else "C3")
    y_grid = np.linspace(-20, 20, 400)
    ax.semilogy(y_grid, np.exp(-np.abs(y_grid)), "k-", lw=1.5, label="universal exp(-|y|)")
    ax.set_xlabel(r"$y = \mathrm{sign}(x)\,(|x|/\chi)^z$")
    ax.set_ylabel("Nc/A")
    ax.legend()


def plot_noise_overlay(ax: plt.Axes, amplitude_sets: dict[float, np.ndarray],
                        title: str) -> None:
    """Overlay the empirical Nc(x) for several noise amplitudes on one
    semi-log plot (paper Section 3.4)."""
    for noise_level, amplitudes in sorted(amplitude_sets.items()):
        if len(amplitudes) == 0:
            continue
        x, nc = empirical_complementary_cdf(amplitudes)
        ax.semilogy(x, nc, ".-", ms=3, lw=0.8, label=f"A={noise_level}")
    ax.set_xlabel("amplitude |x| (%)")
    ax.set_ylabel("Nc(x)")
    ax.set_title(title)
    ax.legend()
