"""Experiment 1: Toy model demonstrating that drawdowns detect higher-order
correlations invisible to two-point statistics (paper Section 2, Eq. 1-3).

Run: python -m exp.exp1
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import TOY_MODEL_N, TOY_MODEL_SEED
from src.drawdown import max_run_length
from src.toy_model import (
    compute_delta_p,
    conditional_expectation_check,
    generate_epsilon,
    three_point_correlation,
    truth_table,
    two_point_correlation,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> dict:
    eps = generate_epsilon(TOY_MODEL_N, TOY_MODEL_SEED)
    delta_p = compute_delta_p(eps)

    mean_delta_p = float(np.mean(delta_p))
    two_point = {lag: two_point_correlation(delta_p, lag) for lag in range(0, 6)}
    three_point = three_point_correlation(delta_p)

    predictor = conditional_expectation_check(eps)  # c(t) = eps(t-2)*eps(t-1)
    # Paper Eq. (2): E[delta_p(t) | delta_p(t-2), delta_p(t-1)] = eps(t-2)*eps(t-1)
    # (rescaled form, see src/toy_model.conditional_expectation_check). Verify by
    # grouping delta_p(t) on the predictor value and checking the sample mean,
    # since delta_p(t) itself is random (= predictor + eps(t)) for each t.
    mean_given_plus1 = float(np.mean(delta_p[predictor == 1.0]))
    mean_given_minus1 = float(np.mean(delta_p[predictor == -1.0]))
    predictive_match = bool(
        abs(mean_given_plus1 - 1.0) < 0.1 and abs(mean_given_minus1 - (-1.0)) < 0.1
    )

    table = truth_table()

    # Cumulative price paths.
    p_delta = np.concatenate([[0.0], np.cumsum(delta_p)])
    p_eps = np.concatenate([[0.0], np.cumsum(eps)])

    # Drawdown duration: longest run of consecutive non-positive delta_p/eps values.
    max_dd_duration_delta_p = max_run_length(delta_p)
    max_dd_duration_eps = max_run_length(eps)

    # Longest run of consecutive negative eps-process increments across many
    # independent longer realizations, to show it can exceed the delta_p bound.
    rng = np.random.default_rng(TOY_MODEL_SEED + 1)
    longest_eps_runs = []
    for trial in range(200):
        eps_trial = rng.choice([-1.0, 1.0], size=TOY_MODEL_N)
        longest_eps_runs.append(max_run_length(eps_trial))
    max_eps_run_over_trials = int(np.max(longest_eps_runs))

    results = {
        "n": TOY_MODEL_N,
        "seed": TOY_MODEL_SEED,
        "mean_delta_p": mean_delta_p,
        "two_point_correlation_by_lag": two_point,
        "three_point_correlation": three_point,
        "mean_delta_p_given_predictor_plus1": mean_given_plus1,
        "mean_delta_p_given_predictor_minus1": mean_given_minus1,
        "predictive_identity_holds": predictive_match,
        "truth_table": table.to_dict(orient="records"),
        "max_drawdown_duration_delta_p": int(max_dd_duration_delta_p),
        "max_drawdown_duration_eps_single_path": int(max_dd_duration_eps),
        "max_drawdown_duration_eps_over_200_trials": max_eps_run_over_trials,
    }

    assert abs(mean_delta_p) < 0.05, "delta_p should have zero mean"
    for lag, value in two_point.items():
        if lag == 0:
            # Var[delta_p] = E[eps(t)^2] + E[eps(t-1)^2 eps(t-2)^2] + 2E[eps(t)eps(t-1)eps(t-2)] = 1 + 1 + 0 = 2.
            assert abs(value - 2.0) < 0.1, "delta_p should have variance 2"
        else:
            assert abs(value) < 0.1, f"two-point correlation at lag {lag} should vanish"
    assert abs(three_point - 1.0) < 0.1, "three-point correlation should equal 1"
    assert predictive_match, "E[delta_p(t)|delta_p(t-2),delta_p(t-1)] should track the predictor (Eq. 2)"
    assert max_dd_duration_delta_p <= 2, "delta_p drawdowns must be bounded by 2 steps"
    assert max_eps_run_over_trials > 2, "eps-process drawdowns are not bounded"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp1_toy_model.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot(p_delta, p_eps, delta_p, eps)

    print("Experiment 1 complete.")
    print(f"  mean(delta_p) = {mean_delta_p:.4f} (expect ~0)")
    print(f"  two-point correlations (lag 1-5): {[round(two_point[l], 4) for l in range(1, 6)]}")
    print(f"  three-point correlation = {three_point:.4f} (expect 1.0)")
    print(f"  max drawdown duration of delta_p: {max_dd_duration_delta_p} (expect <= 2)")
    print(f"  max drawdown duration of eps (200 trials of length {TOY_MODEL_N}): "
          f"{max_eps_run_over_trials} (unbounded, not capped at 2)")
    return results


def _plot(p_delta: np.ndarray, p_eps: np.ndarray, delta_p: np.ndarray, eps: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(500, len(p_delta))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(p_eps[:n_show], label="cumsum(eps(t))  [unbounded drawdowns]")
    axes[0].plot(p_delta[:n_show], label="cumsum(delta_p(t))  [drawdowns bounded at 2 steps]")
    axes[0].set_ylabel("cumulative price")
    axes[0].legend()
    axes[0].set_title("Toy model: price paths (paper Section 2)")

    axes[1].hist(delta_p, bins=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], rwidth=0.8)
    axes[1].set_xlabel("delta_p(t) value")
    axes[1].set_ylabel("count")
    axes[1].set_title("delta_p(t) takes only values {-2, 0, +2}")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_toy_model.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
