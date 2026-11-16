"""Surrogate (reshuffled-returns) significance test for drawdown outliers
(paper Section 3.5 / Table 6-7).

Custom implementation (paper Section 3.5): a non-parametric permutation test
specific to the paper's drawdown definition; scipy.stats offers generic
permutation tests but not this exact reconstruction-from-shuffled-returns
procedure, so it is coded directly against src.drawdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.drawdown import compute_drawdowns_drawups


def surrogate_drawdown_counts(prices: pd.Series, threshold_pct: float,
                               n_surrogates: int, seed: int) -> np.ndarray:
    """For each of ``n_surrogates`` random permutations of the daily returns,
    reconstruct a price path and count drawdowns with amplitude >=
    threshold_pct. Returns an array of length n_surrogates."""
    log_prices = np.log(prices.astype(float))
    returns = log_prices.diff().dropna().to_numpy()
    p0 = prices.iloc[0]
    rng = np.random.default_rng(seed)
    counts = np.empty(n_surrogates, dtype=int)
    for k in range(n_surrogates):
        r_perm = rng.permutation(returns)
        perm_log = np.log(p0) + np.cumsum(r_perm)
        perm_prices = pd.Series(np.concatenate([[p0], np.exp(perm_log)]))
        drawdowns, _ = compute_drawdowns_drawups(perm_prices)
        counts[k] = int(np.sum(drawdowns >= threshold_pct))
    return counts


def confidence_level(real_count: int, surrogate_counts: np.ndarray) -> float:
    """Confidence level that the real count of large drawdowns exceeds what
    reshuffled (correlation-free) surrogates produce: the one-sided fraction
    of surrogates with a count strictly below the real data's count (paper
    Section 3.5, Table 6-7)."""
    n = len(surrogate_counts)
    return float(np.mean(surrogate_counts < real_count))
