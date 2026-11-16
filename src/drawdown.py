"""Drawdown and drawup identification (paper Section 3).

Johansen & Sornette define a drawdown as "a persistent decrease in the price
... over consecutive days", i.e. the cumulative percentage loss from the last
local maximum to the next local minimum of the closing price; a drawup is the
symmetric cumulative percentage gain from a local minimum to the next local
maximum.

Custom implementation (paper Section 3, run-statistics definition): no
library in the environment implements this specific run-based event
definition, so it is coded directly from the paper's text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ZERO_RETURN_BREAKS_RUN


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Daily log-returns r(i) = ln[price(i)] - ln[price(i-1)] (paper Section 3)."""
    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive to take logarithms")
    log_prices = np.log(prices.astype(float))
    returns = log_prices.diff().dropna()
    returns.name = "log_return"
    return returns


def _run_signs(returns: np.ndarray) -> np.ndarray:
    """Assign +1/-1 to each return, propagating the sign of the last non-zero
    return across exact zeros so that a flat day does not break a run (see
    ``ZERO_RETURN_BREAKS_RUN`` in src/config.py)."""
    signs = np.sign(returns)
    if ZERO_RETURN_BREAKS_RUN:
        signs[signs == 0] = 0
        return signs
    last = 0
    for i, s in enumerate(signs):
        if s == 0:
            signs[i] = last
        else:
            last = s
    return signs


def identify_runs(prices: pd.Series) -> pd.DataFrame:
    """Identify runs of consecutive same-sign daily returns in ``prices``.

    Returns a DataFrame with one row per run: ``start`` and ``end`` (index
    labels of the local extremum that bounds the run), ``sign`` (+1 for a
    drawup, -1 for a drawdown), ``length`` (number of days in the run) and
    ``amplitude_pct`` (positive cumulative percentage change over the run:
    (Pmax-Pmin)/Pmax*100 for a drawdown, (Pmax-Pmin)/Pmin*100 for a drawup).
    """
    prices = prices.astype(float)
    returns = prices.values[1:] / prices.values[:-1] - 1.0
    if len(returns) == 0:
        return pd.DataFrame(columns=["start", "end", "sign", "length", "amplitude_pct"])

    signs = _run_signs(returns.copy())
    index = prices.index

    runs = []
    run_start_pos = 0  # position (in `prices`) of the local extremum starting the run
    current_sign = signs[0]
    for i in range(1, len(signs) + 1):
        if i == len(signs) or (signs[i] != current_sign and current_sign != 0):
            run_end_pos = i  # position in `prices` of the local extremum ending the run
            if current_sign != 0 and run_end_pos > run_start_pos:
                p_start = prices.iloc[run_start_pos]
                p_end = prices.iloc[run_end_pos]
                if current_sign < 0:
                    amplitude = (p_start - p_end) / p_start * 100.0
                else:
                    amplitude = (p_end - p_start) / p_start * 100.0
                runs.append({
                    "start": index[run_start_pos],
                    "end": index[run_end_pos],
                    "sign": int(current_sign),
                    "length": run_end_pos - run_start_pos,
                    "amplitude_pct": amplitude,
                })
            run_start_pos = run_end_pos
            if i < len(signs):
                current_sign = signs[i]
    return pd.DataFrame(runs)


def compute_drawdowns_drawups(prices: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (drawdown_amplitudes, drawup_amplitudes) as positive percentages.

    A drawdown amplitude is (Pmax-Pmin)/Pmax*100 over a run of consecutive
    negative returns; a drawup amplitude is (Pmax-Pmin)/Pmin*100 over a run of
    consecutive positive returns (paper Section 3).
    """
    runs = identify_runs(prices)
    if runs.empty:
        return np.array([]), np.array([])
    drawdowns = runs.loc[runs["sign"] < 0, "amplitude_pct"].to_numpy()
    drawups = runs.loc[runs["sign"] > 0, "amplitude_pct"].to_numpy()
    assert (drawdowns >= 0).all() and (drawups >= 0).all()
    return drawdowns, drawups


def max_run_length(values: np.ndarray) -> int:
    """Longest run of consecutive strictly negative values in ``values``
    (paper Section 2, toy-model drawdown-duration demonstration).

    A value of exactly zero is non-negative and therefore breaks a drawdown
    run: the paper's bound ("any three consecutive epsilon values produce at
    least one non-negative delta_p") relies on this strict definition.
    """
    best = cur = 0
    for v in values:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best
