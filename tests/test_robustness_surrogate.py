"""Unit tests for the noise-robustness transform (Section 3.4) and the
surrogate significance test (Section 3.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.robustness import noisy_price_series
from src.surrogate import confidence_level, surrogate_drawdown_counts


def _sample_prices(n: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))


def test_noisy_price_series_zero_amplitude_matches_original_returns():
    prices = _sample_prices()
    noisy = noisy_price_series(prices, amplitude=0.0, seed=1)
    log_r_orig = np.diff(np.log(prices.values))
    log_r_noisy = np.diff(np.log(noisy.values))
    assert np.allclose(log_r_orig, log_r_noisy, atol=1e-10)


def test_noisy_price_series_positive_amplitude_changes_path():
    prices = _sample_prices()
    noisy = noisy_price_series(prices, amplitude=1.0, seed=1)
    assert len(noisy) == len(prices)
    assert not np.allclose(noisy.values, prices.values)
    assert (noisy.values > 0).all()


def test_surrogate_counts_are_nonnegative_and_correct_length():
    prices = _sample_prices(n=300, seed=2)
    counts = surrogate_drawdown_counts(prices, threshold_pct=2.0, n_surrogates=20, seed=5)
    assert len(counts) == 20
    assert (counts >= 0).all()


def test_confidence_level_known_values():
    surrogate_counts = np.array([0, 1, 1, 2, 2, 2, 3, 3, 4, 5])
    # real_count = 4: surrogates strictly below 4 -> 8 of 10
    assert confidence_level(4, surrogate_counts) == 0.8
    # real_count = 0: none strictly below -> 0
    assert confidence_level(0, surrogate_counts) == 0.0
    # real_count = 100: all strictly below -> 1
    assert confidence_level(100, surrogate_counts) == 1.0
