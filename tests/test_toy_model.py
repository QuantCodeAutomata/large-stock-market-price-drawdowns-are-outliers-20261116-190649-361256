"""Unit tests for the Robinson-Hsieh toy model (paper Section 2, Eq. 1-3)."""

from __future__ import annotations

import numpy as np

from src.toy_model import (
    compute_delta_p,
    conditional_expectation_check,
    generate_epsilon,
    three_point_correlation,
    truth_table,
    two_point_correlation,
)


def test_generate_epsilon_values_are_plus_minus_one():
    eps = generate_epsilon(5000, seed=1)
    assert set(np.unique(eps)) <= {-1.0, 1.0}
    # equal probability: sample mean close to zero
    assert abs(np.mean(eps)) < 0.05


def test_delta_p_takes_only_three_values():
    eps = generate_epsilon(5000, seed=2)
    delta_p = compute_delta_p(eps)
    assert set(np.unique(delta_p)) <= {-2.0, 0.0, 2.0}
    assert len(delta_p) == len(eps) - 2


def test_truth_table_matches_equation_3():
    table = truth_table()
    assert len(table) == 8
    expected = {
        (1, 1, 1): 2.0, (1, 1, -1): 0.0, (1, -1, 1): 0.0, (1, -1, -1): -2.0,
        (-1, 1, 1): 0.0, (-1, 1, -1): -2.0, (-1, -1, 1): 2.0, (-1, -1, -1): 0.0,
    }
    for _, row in table.iterrows():
        key = (int(row["eps(t-2)"]), int(row["eps(t-1)"]), int(row["eps(t)"]))
        assert row["delta_p(t)"] == expected[key]


def test_two_point_correlation_vanishes_for_nonzero_lag():
    eps = generate_epsilon(200_000, seed=3)
    delta_p = compute_delta_p(eps)
    assert abs(two_point_correlation(delta_p, 0) - 2.0) < 0.1
    for lag in range(1, 6):
        assert abs(two_point_correlation(delta_p, lag)) < 0.05


def test_three_point_correlation_equals_one():
    eps = generate_epsilon(200_000, seed=4)
    delta_p = compute_delta_p(eps)
    assert abs(three_point_correlation(delta_p) - 1.0) < 0.05


def test_mean_delta_p_is_zero():
    eps = generate_epsilon(200_000, seed=5)
    delta_p = compute_delta_p(eps)
    assert abs(np.mean(delta_p)) < 0.02


def test_conditional_expectation_matches_eps_product():
    eps = generate_epsilon(1000, seed=6)
    pred = conditional_expectation_check(eps)
    assert len(pred) == len(eps) - 2
    assert np.array_equal(pred, eps[:-2] * eps[1:-1])
