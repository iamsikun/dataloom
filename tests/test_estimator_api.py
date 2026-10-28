"""Estimator API + registry sanity tests."""

from __future__ import annotations

import numpy as np
import pytest

from dataloom.dgp.gaussian import (
    GaussianParams,
    estimand_mean,
    make_synth_fn,
    sample_real,
)
from dataloom.estimators import REGISTRY, get
from dataloom.estimators.api import EstimatorResult


EXP1_ESTIMATORS = [
    "real_only_all",
    "synthetic_only_full_calibration",
    "naive_pooling",
    "fixed_half_split_oracle_alpha",
    "old_fixed_share_oracle_alpha",
    "corrected_oracle_gn",
    "safe_corrected_oracle_gn",
]


@pytest.mark.parametrize("name", EXP1_ESTIMATORS)
def test_registry_contains(name):
    assert name in REGISTRY
    assert get(name).name == name


@pytest.mark.parametrize("name", EXP1_ESTIMATORS)
def test_estimator_returns_estimator_result(name):
    rng = np.random.default_rng(7)
    p = GaussianParams(n=200, beta=1.0, rho=1.0)
    X = sample_real(p, rng)
    synth_fn = make_synth_fn(p)
    truth_params = {"a": p.a, "v_n": p.v_n, "c": p.c, "beta": p.beta,
                    "sigma_s2": p.sigma_s2, "B0": p.B0, "m": p.m}
    est = get(name)
    result = est(
        X, synth_fn,
        n=p.n, rng=rng, truth_params=truth_params,
        estimand=estimand_mean, config={},
    )
    assert isinstance(result, EstimatorResult)
    assert np.isfinite(result.theta_hat)
    assert not result.failure_flag


def test_real_only_all_does_not_call_synth_fn():
    """real_only_all must work even when synth_fn is None."""
    rng = np.random.default_rng(1)
    p = GaussianParams(n=200, beta=1.0, rho=1.0)
    X = sample_real(p, rng)
    truth_params = {"a": p.a, "v_n": p.v_n, "c": p.c, "beta": p.beta,
                    "sigma_s2": p.sigma_s2, "B0": p.B0, "m": p.m}
    res = get("real_only_all")(
        X, None,
        n=p.n, rng=rng, truth_params=truth_params,
        estimand=estimand_mean, config={},
    )
    assert res.theta_hat == pytest.approx(float(np.mean(X)))


def test_corrected_oracle_picks_grid_argmin():
    """corrected_oracle_gn's x_selected must equal oracle_grid's argmin."""
    from dataloom.oracle import oracle_grid

    rng = np.random.default_rng(42)
    p = GaussianParams(n=2000, beta=1.0, rho=1.0)
    X = sample_real(p, rng)
    synth_fn = make_synth_fn(p)
    truth_params = {"a": p.a, "v_n": p.v_n, "c": p.c, "beta": p.beta,
                    "sigma_s2": p.sigma_s2, "B0": p.B0, "m": p.m}
    res = get("corrected_oracle_gn")(
        X, synth_fn,
        n=p.n, rng=rng, truth_params=truth_params,
        estimand=estimand_mean, config={},
    )
    expected = oracle_grid(n=p.n, a=p.a, v_n=p.v_n, c=p.c, beta=p.beta).x_star
    assert res.x_selected == expected
