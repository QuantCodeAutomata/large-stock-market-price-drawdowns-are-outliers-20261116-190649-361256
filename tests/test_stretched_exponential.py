"""Unit tests for the stretched-exponential fit (paper Eq. 4, Section 3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.stretched_exponential import (
    empirical_complementary_cdf,
    fit_stretched_exponential,
    normalized_variable,
    relative_error_curve,
)


def test_empirical_cdf_is_monotone_decreasing():
    amplitudes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x, nc = empirical_complementary_cdf(amplitudes)
    assert np.all(np.diff(x) >= 0)
    assert np.all(np.diff(nc) <= 0)
    assert nc[0] == len(amplitudes)


def test_fit_recovers_known_stretched_exponential_parameters():
    rng = np.random.default_rng(0)
    true_chi, true_z = 2.0, 0.85
    n = 5000
    # Inverse-CDF sampling from Nc(x) = N*exp(-(x/chi)^z) => x = chi*(-ln(U))^(1/z)
    u = rng.uniform(1e-6, 1.0, size=n)
    amplitudes = true_chi * (-np.log(u)) ** (1.0 / true_z)
    fit = fit_stretched_exponential(amplitudes)
    assert fit.A == n
    assert fit.chi == pytest.approx(true_chi, rel=0.15)
    assert fit.z == pytest.approx(true_z, rel=0.15)
    assert fit.sigma_chi > 0 and fit.sigma_z > 0


def test_fit_requires_minimum_events():
    with pytest.raises(ValueError):
        fit_stretched_exponential(np.array([1.0, 2.0]))


def test_relative_error_curve_small_near_fit():
    rng = np.random.default_rng(1)
    true_chi, true_z = 1.5, 0.9
    n = 3000
    u = rng.uniform(1e-6, 1.0, size=n)
    amplitudes = true_chi * (-np.log(u)) ** (1.0 / true_z)
    fit = fit_stretched_exponential(amplitudes)
    x, err = relative_error_curve(amplitudes, fit)
    # bulk (below the 90th percentile) should fluctuate near zero
    bulk = x < np.percentile(x, 90)
    assert abs(np.mean(err[bulk])) < 0.5


def test_normalized_variable_sign_and_scale():
    fit_amplitudes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fit = fit_stretched_exponential(fit_amplitudes)
    y = normalized_variable(fit_amplitudes, fit, sign=-1)
    assert np.all(y <= 0)
    y_pos = normalized_variable(fit_amplitudes, fit, sign=1)
    assert np.all(y_pos >= 0)
