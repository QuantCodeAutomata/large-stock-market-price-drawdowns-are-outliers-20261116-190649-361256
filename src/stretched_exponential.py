"""Stretched-exponential fit of drawdown/drawup cumulative distributions.

Custom implementation (paper Eq. 4 and Section 3): the stretched-exponential
survival model Nc(x) = A*exp(-(|x|/chi)^z) is specific to this paper; no
general quantitative library (scipy.stats, statsmodels) implements this
particular fit with A fixed to the observed count, so the nonlinear
least-squares fit is coded directly with scipy.optimize.curve_fit, and the
asymptotic error-bar formulas are taken from Thoman et al. (1970) as cited in
the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class StretchedExponentialFit:
    A: int        # total number of events (fixed, not fitted)
    B: float      # fitted scale parameter; chi = 1/B**(1/z)
    z: float      # fitted stretching exponent
    chi: float    # characteristic scale (same unit as the input amplitudes, %)
    sigma_chi: float
    sigma_z: float
    N: int        # number of events used in the fit (== A)


def empirical_complementary_cdf(amplitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical Nc(x): number of events with amplitude >= x, for x = each
    observed amplitude, sorted ascending. Standard rank-based survival
    function estimator (paper Section 3, Figure 2/3)."""
    x_sorted = np.sort(amplitudes)
    n = len(x_sorted)
    # Nc(x_sorted[i]) = number of events with amplitude >= x_sorted[i] = n - i
    nc = n - np.arange(n)
    return x_sorted, nc.astype(float)


def fit_stretched_exponential(amplitudes: np.ndarray) -> StretchedExponentialFit:
    """Fit Nc(x) = A*exp(-(|x|/chi)^z) with A fixed to the event count.

    The fit works in log-space: log(Nc) = log(A) - B*x^z, with B, z free
    parameters (paper Section 3, implementation notes). chi = 1/B**(1/z).
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    if len(amplitudes) < 5:
        raise ValueError("Need at least 5 events to fit a stretched exponential")
    x_sorted, nc = empirical_complementary_cdf(amplitudes)
    n = len(x_sorted)
    log_nc = np.log(nc)

    def model(x: np.ndarray, B: float, z: float) -> np.ndarray:
        return np.log(n) - B * x ** z

    popt, _ = curve_fit(model, x_sorted, log_nc, p0=[1.0, 0.9], bounds=([1e-6, 0.05], [1e3, 3.0]), maxfev=20000)
    B, z = popt
    chi = (1.0 / B) ** (1.0 / z)

    # Thoman, Bain and Antle (1970) asymptotic relative errors, as cited in
    # the paper, for the Weibull/stretched-exponential shape and scale.
    sigma_chi = chi * 1.053 / (z * np.sqrt(n))
    sigma_z = z * 0.78 / np.sqrt(n)

    return StretchedExponentialFit(A=n, B=float(B), z=float(z), chi=float(chi),
                                    sigma_chi=float(sigma_chi), sigma_z=float(sigma_z), N=n)


def relative_error_curve(amplitudes: np.ndarray, fit: StretchedExponentialFit) -> tuple[np.ndarray, np.ndarray]:
    """Relative error (Nc_emp - Nc_fit)/Nc_fit vs |x| used to locate the
    bulk/outlier crossover (paper Section 3, error plots)."""
    x_sorted, nc_emp = empirical_complementary_cdf(amplitudes)
    nc_fit = fit.A * np.exp(-(x_sorted / fit.chi) ** fit.z)
    rel_err = (nc_emp - nc_fit) / nc_fit
    return x_sorted, rel_err


def normalized_variable(amplitudes: np.ndarray, fit: StretchedExponentialFit, sign: int) -> np.ndarray:
    """y = sign*(x/chi)^z, the collapse variable of paper Section 4 (Eq. for
    the universal curve exp(-|y|))."""
    return sign * (amplitudes / fit.chi) ** fit.z
