"""Noise-robustness transform of returns (paper Section 3.4).

r_A(i) = r(i) + A * Ran * sigma, Ran ~ Uniform[-1, +1], sigma = std(r).
Custom implementation (paper Section 3.4): a specific perturbation defined by
the paper, not a generic library transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def noisy_price_series(prices: pd.Series, amplitude: float, seed: int) -> pd.Series:
    """Build the noisy price series p_A(i) = p_0 * exp(cumsum(r_A)) (paper
    Section 3.4, "time series neighborhood")."""
    log_prices = np.log(prices.astype(float))
    returns = log_prices.diff().dropna()
    sigma = returns.std()
    rng = np.random.default_rng(seed)
    ran = rng.uniform(-1.0, 1.0, size=len(returns))
    r_noisy = returns.values + amplitude * ran * sigma
    p0 = prices.iloc[0]
    noisy_log = np.log(p0) + np.cumsum(r_noisy)
    noisy = pd.Series(np.concatenate([[p0], np.exp(noisy_log)]), index=prices.index)
    return noisy
