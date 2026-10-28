"""Phase 2 §14 sanity tests for Experiment 1.

These run small Monte Carlo simulations and check the empirical MC MSE
against the analytic risk model.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataloom.dgp.gaussian import (
    GaussianParams,
    estimand_mean,
    make_synth_fn,
    sample_real,
)
from dataloom.estimators import get
from dataloom.io.seeds import make_rng, replication_seed
from dataloom.oracle import oracle_grid


def _mc_mse(name: str, params: GaussianParams, R: int, seed_root: int = 1) -> float:
    """Run R replications of one estimator on the Gaussian DGP and return MC MSE."""
    truth_params = {"a": params.a, "v_n": params.v_n, "c": params.c,
                    "beta": params.beta, "sigma_s2": params.sigma_s2,
                    "B0": params.B0, "m": params.m}
    est = get(name)
    cell_key = (params.n, int(params.beta * 1000), int(params.rho * 1000))
    sq_errors = np.empty(R)
    for r in range(R):
        s = replication_seed(seed_root, "test", cell_key, r)
        rng = make_rng(s)
        X = sample_real(params, rng)
        synth_fn = make_synth_fn(params)
        res = est(X, synth_fn, n=params.n, rng=rng,
                  truth_params=truth_params, estimand=estimand_mean,
                  config={"x_strategy": "half_split"})
        sq_errors[r] = (res.theta_hat - params.theta_star) ** 2
    return float(np.mean(sq_errors))


# ---------------------------------------------------------------------------
# §14 Priority 2 sanity checks
# ---------------------------------------------------------------------------


def test_real_only_mc_mse_matches_a_over_n():
    """MC MSE of real_only_all should be close to a / n."""
    p = GaussianParams(n=2000, beta=1.0, rho=1.0, a=1.0)
    mc = _mc_mse("real_only_all", p, R=2000)
    expected = p.a / p.n
    # MC SE of MSE estimator for chi-squared sample variance is ~ sqrt(2/R) * expected.
    se = np.sqrt(2.0 / 2000) * expected
    assert abs(mc - expected) < 4 * se, (
        f"MC MSE {mc:.6f} vs analytic a/n = {expected:.6f}, SE ~ {se:.6f}"
    )


def test_synthetic_only_mc_mse_matches_analytic():
    """MC MSE of synthetic_only_full_calibration should be close to v_n + c * n^{-2 beta}."""
    p = GaussianParams(n=2000, beta=1.0, rho=1.0)
    mc = _mc_mse("synthetic_only_full_calibration", p, R=2000)
    expected = p.v_n + p.c * (p.n ** (-2 * p.beta))
    # Bias-only MSE = bias^2 + variance; bias = c * n^{-2β} contributes deterministically,
    # variance contribution has MC SE ~ sqrt(2/R) * v_n.
    se = np.sqrt(2.0 / 2000) * p.v_n + 0.05 * expected
    assert abs(mc - expected) < 5 * se, (
        f"MC MSE {mc:.6f} vs analytic {expected:.6f}, SE ~ {se:.6f}"
    )


def test_corrected_oracle_mc_mse_matches_R_n_at_xstar():
    """MC MSE of corrected_oracle_gn should match R_n(x*) within MC SE."""
    p = GaussianParams(n=2000, beta=1.0, rho=1.0)
    res = oracle_grid(n=p.n, a=p.a, v_n=p.v_n, c=p.c, beta=p.beta)
    expected = res.risk_star
    mc = _mc_mse("corrected_oracle_gn", p, R=3000)
    se = np.sqrt(2.0 / 3000) * expected
    assert abs(mc - expected) < 5 * se, (
        f"MC MSE {mc:.6f} vs R_n(x*) {expected:.6f}, x* {res.x_star}, SE ~ {se:.6f}"
    )


def test_old_fixed_share_overcalibrates_in_common_regime():
    """For m≍n, β>1/2: old fixed-share rule's x_old should exceed corrected x*."""
    p = GaussianParams(n=10000, beta=1.0, rho=1.0)
    lam_old = (2.0 * p.beta) / (1.0 + 2.0 * p.beta)  # = 2/3 for beta=1
    x_old = int(np.floor(p.n * lam_old))
    res = oracle_grid(n=p.n, a=p.a, v_n=p.v_n, c=p.c, beta=p.beta)
    assert x_old > res.x_star, (
        f"x_old={x_old} should exceed corrected x*={res.x_star} "
        f"in the common fast-learning regime"
    )


def test_old_fixed_share_mc_mse_higher_than_corrected_oracle():
    """The same regime: old fixed-share MC MSE should exceed corrected_oracle_gn MC MSE."""
    p = GaussianParams(n=10000, beta=1.0, rho=1.0)
    mc_old = _mc_mse("old_fixed_share_oracle_alpha", p, R=2000)
    mc_new = _mc_mse("corrected_oracle_gn", p, R=2000)
    assert mc_old > mc_new, (
        f"old fixed-share MC MSE {mc_old:.6f} should exceed corrected "
        f"{mc_new:.6f} in the common regime"
    )
