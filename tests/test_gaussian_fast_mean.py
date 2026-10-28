"""Tests for the fast_mean=True path in the Gaussian DGP.

Verifies that synthetic_only MC MSE with fast_mean=True matches the analytic
formula v_n + c * n^{-2*beta} within MC sampling error, same sanity check as
test_exp1_sanity.test_synthetic_only_mc_mse_matches_analytic.
"""

from __future__ import annotations

import numpy as np

from dataloom.dgp.gaussian import (
    GaussianParams,
    estimand_mean,
    make_synth_fn,
    sample_real,
)
from dataloom.estimators import get
from dataloom.io.seeds import make_rng, replication_seed


def _mc_mse_fast(name: str, params: GaussianParams, R: int, seed_root: int = 42) -> float:
    """MC MSE for one estimator using fast_mean params."""
    truth_params = {"a": params.a, "v_n": params.v_n, "c": params.c,
                    "beta": params.beta, "sigma_s2": params.sigma_s2,
                    "B0": params.B0, "m": params.m}
    est = get(name)
    cell_key = (params.n, int(params.beta * 1000), int(params.rho * 1000))
    sq_errors = np.empty(R)
    for r in range(R):
        s = replication_seed(seed_root, "test_fast_mean", cell_key, r)
        rng = make_rng(s)
        X = sample_real(params, rng)
        synth_fn = make_synth_fn(params)
        res = est(X, synth_fn, n=params.n, rng=rng,
                  truth_params=truth_params, estimand=estimand_mean,
                  config={"x_strategy": "half_split"})
        sq_errors[r] = (res.theta_hat - params.theta_star) ** 2
    return float(np.mean(sq_errors))


def test_fast_mean_synth_fn_returns_length_one():
    """fast_mean=True produces a length-1 array; fast_mean=False produces length m."""
    params_fast = GaussianParams(n=500, beta=1.0, rho=1.0, fast_mean=True)
    params_slow = GaussianParams(n=500, beta=1.0, rho=1.0, fast_mean=False)
    rng = np.random.default_rng(0)
    Z_fast = make_synth_fn(params_fast)(10, rng)
    Z_slow = make_synth_fn(params_slow)(10, rng)
    assert len(Z_fast) == 1, f"fast_mean should return length-1, got {len(Z_fast)}"
    assert len(Z_slow) == params_slow.m, f"slow path should return length m={params_slow.m}"


def test_fast_mean_nan_fill_on_x_le_zero():
    """fast_mean=True still returns NaNs for x <= 0."""
    params = GaussianParams(n=500, beta=1.0, rho=1.0, fast_mean=True)
    rng = np.random.default_rng(1)
    Z = make_synth_fn(params)(0, rng)
    assert np.all(np.isnan(Z)), "x=0 should return NaN array even with fast_mean=True"


def test_synthetic_only_mc_mse_fast_mean_matches_analytic():
    """synthetic_only MSE with fast_mean=True should match v_n + c * n^{-2*beta}.

    Cell: n=2000, beta=1, rho=1 — same as test_synthetic_only_mc_mse_matches_analytic
    but with params.fast_mean=True.
    """
    p = GaussianParams(n=2000, beta=1.0, rho=1.0, fast_mean=True)
    R = 2000
    mc = _mc_mse_fast("synthetic_only_full_calibration", p, R=R)
    expected = p.v_n + p.c * (p.n ** (-2 * p.beta))
    se = np.sqrt(2.0 / R) * p.v_n + 0.05 * expected
    assert abs(mc - expected) < 5 * se, (
        f"fast_mean MC MSE {mc:.6f} vs analytic {expected:.6f}, SE ~ {se:.6f}"
    )
