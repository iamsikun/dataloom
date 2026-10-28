"""Focused tests for the targeted rebuttal experiment infrastructure."""

from __future__ import annotations

import numpy as np
import pytest

from dataloom.estimators import get
from dataloom.experiments.expR3_hierarchical import _one_replication
from dataloom.io.seeds import make_rng
from dataloom.oracle import (
    R_profile,
    R_profile_curve,
    oracle_grid,
    oracle_grid_curve,
)
from dataloom.rebuttal.adult_auxiliary import (
    estimand,
    influence_values,
    variance_constant,
)
from dataloom.rebuttal.bounds import (
    buffered_safe_condition,
    gaussian_variance_lower,
    gaussian_variance_upper,
    mean_upper,
)


def test_general_curve_risk_matches_power_law_api():
    n, a, v_n, c, beta = 2000, 1.2, 0.001, 0.8, 1.0
    x = np.asarray([0, 5, 100, n])
    curve = lambda z: v_n + c * np.asarray(z, dtype=float) ** (-2 * beta)
    np.testing.assert_allclose(
        R_profile_curve(x, n=n, a=a, B_curve=curve),
        R_profile(x, n=n, a=a, v_n=v_n, c=c, beta=beta),
    )


def test_general_curve_oracle_reports_full_knife_edge_tie():
    n = 100
    curve = lambda x: np.asarray(x, dtype=float) ** -1
    result = oracle_grid_curve(n=n, a=1.0, B_curve=curve)
    assert result.x_star == 0
    assert result.minimizers == tuple(range(n + 1))


def test_equality_boundary_three_cases():
    n = 50_000
    beta, rho = 1.0, 1.5
    expected = {0.75: 1.0, np.sqrt(2.0): 1.0, 2.0: 2 ** (-1 / 3)}
    for v0, limit in expected.items():
        result = oracle_grid(
            n=n,
            a=1.0,
            v_n=v0 * n ** (-rho),
            c=1.0,
            beta=beta,
        )
        assert abs(result.x_star / n - limit) < 0.01


def test_product_surrogate_compatibility_alias():
    assert get("old_fixed_share_oracle_alpha") is not get(
        "product_surrogate_fixed_share_oracle_alpha"
    )
    assert get("old_fixed_share_oracle_alpha").needs_synth
    assert get("product_surrogate_fixed_share_oracle_alpha").needs_synth


def test_bounds_are_ordered_and_fail_closed():
    lower = gaussian_variance_lower(1.1, 100, 0.05)
    upper = gaussian_variance_upper(1.1, 99, 0.05)
    assert 0 < lower < 1.1 < upper
    assert np.isinf(mean_upper(np.asarray([1.0]), 0.05))
    assert not buffered_safe_condition(10, float("inf"), 1.0)
    assert buffered_safe_condition(10, 0.05, 1.0)


def test_hierarchical_untruncated_score_is_centered():
    scores = []
    target = None
    for seed in range(500):
        _, diagnostics = _one_replication(
            rng=make_rng(seed),
            seed=seed,
            n=5000,
            beta=1.0,
            rho=1.0,
            validation_size=5000,
            pilot_fits=16,
            redraws=8,
            alpha_level=0.05,
        )
        row = next(item for item in diagnostics if item["x_selected"] == 40)
        scores.append(row["score_untruncated"])
        target = row["bias2_true"]
    assert target is not None
    assert abs(np.mean(scores) - target) < 0.35 * target


def test_adult_gap_influence_is_centered_and_variance_positive():
    values = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    sex = np.asarray([0, 0, 1, 1, 1, 0])
    influence = influence_values(values, sex, "subgroup_gap")
    assert float(influence.mean()) == pytest.approx(0.0)
    assert variance_constant(values, sex, "subgroup_gap") > 0
    assert estimand(values, sex, "subgroup_gap") == pytest.approx(1 / 3)
