"""Unit tests for drawdown/drawup identification (paper Section 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drawdown import (
    compute_drawdowns_drawups,
    compute_log_returns,
    identify_runs,
    max_run_length,
)


def test_log_returns_simple_series():
    prices = pd.Series([100.0, 110.0, 99.0])
    r = compute_log_returns(prices)
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(np.log(1.1))


def test_log_returns_rejects_non_positive_prices():
    with pytest.raises(ValueError):
        compute_log_returns(pd.Series([100.0, -1.0, 50.0]))


def test_single_drawdown_run():
    # 100 -> 90 -> 80 -> 85 (one drawdown run of length 2, one drawup of length 1)
    prices = pd.Series([100.0, 90.0, 80.0, 85.0])
    drawdowns, drawups = compute_drawdowns_drawups(prices)
    assert len(drawdowns) == 1
    assert drawdowns[0] == pytest.approx(20.0)  # (100-80)/100*100
    assert len(drawups) == 1
    assert drawups[0] == pytest.approx((85.0 - 80.0) / 80.0 * 100.0)


def test_drawdown_amplitudes_are_nonnegative():
    rng = np.random.default_rng(0)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000))))
    drawdowns, drawups = compute_drawdowns_drawups(prices)
    assert (drawdowns >= 0).all()
    assert (drawups >= 0).all()
    assert len(drawdowns) > 0 and len(drawups) > 0


def test_monotone_increasing_series_has_no_drawdowns():
    prices = pd.Series(np.linspace(100, 200, 50))
    drawdowns, drawups = compute_drawdowns_drawups(prices)
    assert len(drawdowns) == 0
    assert len(drawups) == 1


def test_identify_runs_lengths_sum_to_total_returns():
    rng = np.random.default_rng(1)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
    runs = identify_runs(prices)
    assert runs["length"].sum() == len(prices) - 1


def test_max_run_length_bounds_toy_model_delta_p():
    from src.toy_model import compute_delta_p, generate_epsilon

    eps = generate_epsilon(50_000, seed=42)
    delta_p = compute_delta_p(eps)
    assert max_run_length(delta_p) <= 2


def test_max_run_length_empty():
    assert max_run_length(np.array([])) == 0


def test_max_run_length_all_negative():
    assert max_run_length(np.array([-1.0, -2.0, -3.0])) == 3
