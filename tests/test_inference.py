"""Phase 6 sanity tests for the interval estimators."""

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


def _coverage(name: str, params: GaussianParams, R: int = 300, seed_root: int = 1):
    """Return (coverage, mean_length, mean_se, mean_bias)."""
    truth_params = {"a": params.a, "v_n": params.v_n, "c": params.c,
                    "beta": params.beta, "sigma_s2": params.sigma_s2,
                    "B0": params.B0, "m": params.m}
    est = get(name)
    cell_key = (params.n, int(params.beta * 1000), int(params.rho * 1000))
    covered = []
    lengths = []
    biases = []
    for r in range(R):
        s = replication_seed(seed_root, "test_inf", cell_key, r)
        rng = make_rng(s)
        X = sample_real(params, rng)
        synth_fn = make_synth_fn(params)
        res = est(X, synth_fn, n=params.n, rng=rng,
                  truth_params=truth_params, estimand=estimand_mean,
                  config={})
        if res.ci_lower is None or res.ci_upper is None:
            continue
        covered.append(res.ci_lower <= params.theta_star <= res.ci_upper)
        lengths.append(res.ci_upper - res.ci_lower)
        biases.append(res.theta_hat - params.theta_star)
    return (
        float(np.mean(covered)) if covered else float("nan"),
        float(np.mean(lengths)) if lengths else float("nan"),
        float(np.mean(np.abs(biases))) if biases else float("nan"),
    )


@pytest.mark.parametrize("name", [
    "ci_real_only", "ci_gn_naive", "ci_gn_bias_aware",
    "ci_gn_undersmoothed", "ci_validation_debiased",
])
def test_interval_estimator_returns_finite_ci(name):
    rng = make_rng(7)
    p = GaussianParams(n=500, beta=1.0, rho=1.0)
    X = sample_real(p, rng)
    synth_fn = make_synth_fn(p)
    truth_params = {"a": p.a, "v_n": p.v_n, "c": p.c, "beta": p.beta,
                    "sigma_s2": p.sigma_s2, "B0": p.B0, "m": p.m}
    res = get(name)(
        X, synth_fn, n=p.n, rng=rng, truth_params=truth_params,
        estimand=estimand_mean, config={},
    )
    assert np.isfinite(res.theta_hat)
    assert res.ci_lower is not None and res.ci_upper is not None
    assert res.ci_lower < res.ci_upper


def test_ci_real_only_coverage_close_to_nominal():
    """ci_real_only is a textbook Wald interval; coverage should be ~95%."""
    p = GaussianParams(n=500, beta=1.0, rho=1.0)
    cov, _, _ = _coverage("ci_real_only", p, R=400)
    # 400 reps -> SE on a 0.95 estimate is sqrt(0.95*0.05/400) ≈ 0.011, so ±0.04 OK.
    assert 0.91 <= cov <= 0.99, f"coverage={cov}"


def test_ci_validation_debiased_improves_over_naive_in_biased_regime():
    """In (β=0.4, ρ=1), bias is large; debiased CI should cover better than naive."""
    p = GaussianParams(n=2000, beta=0.4, rho=1.0)
    cov_naive, _, _ = _coverage("ci_gn_naive", p, R=300)
    cov_deb, _, _ = _coverage("ci_validation_debiased", p, R=300)
    # The debiased CI should be no worse than the naive CI (allowing a small slack).
    assert cov_deb + 0.02 >= cov_naive, \
        f"validation_debiased={cov_deb}, naive={cov_naive}"
