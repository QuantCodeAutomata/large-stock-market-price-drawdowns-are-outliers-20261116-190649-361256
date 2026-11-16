"""Robinson-Hsieh nonlinear moving-average toy model (paper Section 2, Eq. 1-3).

delta_p(t) = eps(t) + eps(t-1)*eps(t-2), with eps(t) i.i.d. +-1 (equal
probability). Custom implementation (paper Eq. 1-3): this is a specific toy
construction from the paper, not a generic time-series model provided by any
library.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_epsilon(n: int, seed: int) -> np.ndarray:
    """i.i.d. white noise eps(t) in {-1, +1} with equal probability."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=n)


def compute_delta_p(eps: np.ndarray) -> np.ndarray:
    """delta_p(t) = eps(t) + eps(t-1)*eps(t-2) (paper Eq. 1).

    The first two points have no valid t-1, t-2 lag and are dropped.
    """
    delta_p = eps[2:] + eps[1:-1] * eps[:-2]
    return delta_p


def truth_table() -> pd.DataFrame:
    """Enumerate all 8 triples (eps(t-2), eps(t-1), eps(t)) and delta_p(t)
    (paper Eq. 3)."""
    rows = []
    for e2 in (1.0, -1.0):
        for e1 in (1.0, -1.0):
            for e0 in (1.0, -1.0):
                dp = e0 + e1 * e2
                rows.append({"eps(t-2)": e2, "eps(t-1)": e1, "eps(t)": e0, "delta_p(t)": dp})
    return pd.DataFrame(rows)


def two_point_correlation(delta_p: np.ndarray, lag: int) -> float:
    """E[delta_p(t)*delta_p(t-lag)] estimated by the sample mean."""
    if lag == 0:
        return float(np.mean(delta_p * delta_p))
    return float(np.mean(delta_p[lag:] * delta_p[:-lag]))


def three_point_correlation(delta_p: np.ndarray) -> float:
    """E[delta_p(t-2)*delta_p(t-1)*delta_p(t)] estimated by the sample mean
    (paper Eq. 2, should equal 1)."""
    return float(np.mean(delta_p[2:] * delta_p[1:-1] * delta_p[:-2]))


def conditional_expectation_check(eps: np.ndarray) -> np.ndarray:
    """Predictor c(t) = eps(t-2)*eps(t-1), aligned with delta_p(t) (same
    indexing as compute_delta_p), for the predictive-power identity of paper
    Section 2, Eq. (2): E[delta_p(t) | delta_p(t-2), delta_p(t-1)] =
    delta_p(t-2)*delta_p(t-1).

    Note on scale: delta_p(t-2) and delta_p(t-1) only take values in
    {0, +-2} (Eq. 3). Whenever they are non-zero, delta_p(t-2) = 2*eps(t-2)
    and delta_p(t-1) = 2*eps(t-1), so delta_p(t-2)*delta_p(t-1) = 4*c(t): the
    literal product delta_p(t-2)*delta_p(t-1) is in {-4, 0, 4}, a factor 4
    too large to equal E[delta_p(t)] which lies in [-1, 1]. Direct enumeration
    of all 32 equiprobable (eps(t-4),...,eps(t)) combinations confirms
    E[delta_p(t) | delta_p(t-2), delta_p(t-1)] = delta_p(t-2)*delta_p(t-1)/4
    = c(t) exactly, consistent with the truth table of Eq. (3); this rescaled
    form is what is verified and reported here.
    """
    return eps[:-2] * eps[1:-1]
