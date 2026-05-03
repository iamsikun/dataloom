"""Phase 3 sanity tests for the adaptive estimators."""

from __future__ import annotations

import numpy as np
import pytest

from dataloom.adaptive.bias_curve import (
    default_pilot_grid,
    estimate_bias_curve,
)
from dataloom.dgp.gaussian import (
    GaussianParams,
    estimand_mean,
    make_synth_fn,
    sample_real,
)
from dataloom.estimators import get
from dataloom.io.seeds import make_rng, replication_seed


def _setup(n=2000, beta=1.0, rho=1.0, seed=42):
    rng = make_rng(seed)
    p = GaussianParams(n=n, beta=beta, rho=rho)
    X = sample_real(p, rng)
    truth_params = {"a": p.a, "v_n": p.v_n, "c": p.c, "beta": p.beta,
                    "sigma_s2": p.sigma_s2, "B0": p.B0, "m": p.m}
    return p, X, make_synth_fn(p), rng, truth_params


def test_default_pilot_grid_intersects_with_n():
    g = default_pilot_grid(2000)
    assert (g >= 1).all() and (g <= 1000).all()
    assert 5 in g and 640 in g
    assert len(g) >= 3


def test_estimate_bias_curve_recovers_beta_in_clean_regime():
    """With a clean Gaussian DGP and large n, β̂ should be near the true β."""
    rng_master = make_rng(123)
    beta_true = 1.0
    p = GaussianParams(n=20000, beta=beta_true, rho=1.0)
    truth_params = {"m": p.m}

    # Average β̂ across a few replications to reduce MC noise.
    betas = []
    for r in range(20):
        rng = make_rng(replication_seed(123, "test_bias", (p.n,), r))
        X = sample_real(p, rng)
        synth_fn = make_synth_fn(p)
        fit = estimate_bias_curve(
            n=p.n, n_v=p.n // 5, X=X, synth_fn=synth_fn, rng=rng,
            estimand=estimand_mean, m=p.m,
        )
        if np.isfinite(fit.beta_hat) and fit.beta_hat < 5.0:
            betas.append(fit.beta_hat)
    mean_beta = float(np.mean(betas))
    assert abs(mean_beta - beta_true) < 0.3, (
        f"mean β̂={mean_beta:.3f}, true β={beta_true}"
    )


@pytest.mark.parametrize("name", [
    "corrected_adaptive_gn",
    "adaptive_parametric_foc",
    "adaptive_parametric_grid",
    "adaptive_nonparametric_grid",
    "safe_corrected_adaptive_gn",
])
def test_adaptive_estimator_basic_call(name):
    """Each adaptive estimator should run without error and return finite theta_hat."""
    p, X, synth_fn, rng, truth_params = _setup()
    res = get(name)(
        X, synth_fn, n=p.n, rng=rng, truth_params=truth_params,
        estimand=estimand_mean, config={},
    )
    assert np.isfinite(res.theta_hat)
    assert not res.failure_flag
    if res.x_selected is not None:
        assert 0 <= res.x_selected <= p.n


def test_adaptive_allocation_tracks_oracle_in_common_regime():
    """In (β=1, ρ=1), corrected_adaptive_gn's mean x̂ should be close to oracle x*."""
    from dataloom.oracle import oracle_grid
    n, beta, rho = 5000, 1.0, 1.0
    p = GaussianParams(n=n, beta=beta, rho=rho)
    truth_params = {"a": p.a, "v_n": p.v_n, "c": p.c, "beta": p.beta,
                    "sigma_s2": p.sigma_s2, "B0": p.B0, "m": p.m}
    oracle_x = oracle_grid(n=n, a=p.a, v_n=p.v_n, c=p.c, beta=p.beta).x_star

    xs = []
    for r in range(40):
        rng = make_rng(replication_seed(7, "test_track", (n,), r))
        X = sample_real(p, rng)
        synth_fn = make_synth_fn(p)
        res = get("corrected_adaptive_gn")(
            X, synth_fn, n=n, rng=rng, truth_params=truth_params,
            estimand=estimand_mean, config={},
        )
        if res.x_selected is not None:
            xs.append(res.x_selected)
    mean_x = float(np.mean(xs))
    # Allow generous tolerance: adaptive on n=5000 with R=40 reps.
    # oracle_x for (n=5000, β=1, ρ=1) is around 360.
    assert 0.3 * oracle_x <= mean_x <= 3.0 * oracle_x, (
        f"mean x̂={mean_x:.1f} vs oracle x*={oracle_x}"
    )


def test_safe_fallback_activates_when_synthetic_useless():
    """In a regime where synthetic is hopeless (rho=0, fixed tiny m),
    the safe variant should fall back to real-only most of the time."""
    n, beta, rho = 1000, 0.4, 0.0
    p = GaussianParams(n=n, beta=beta, rho=rho, sigma_s2=1.0, kappa=1.0)
    truth_params = {"a": p.a, "v_n": p.v_n, "c": p.c, "beta": p.beta,
                    "sigma_s2": p.sigma_s2, "B0": p.B0, "m": p.m}

    fallbacks = []
    for r in range(40):
        rng = make_rng(replication_seed(11, "test_fb", (n,), r))
        X = sample_real(p, rng)
        synth_fn = make_synth_fn(p)
        res = get("safe_corrected_adaptive_gn")(
            X, synth_fn, n=n, rng=rng, truth_params=truth_params,
            estimand=estimand_mean, config={},
        )
        fallbacks.append(bool(res.fallback_used))
    rate = float(np.mean(fallbacks))
    # Heuristic: in this hopeless regime fallback should fire at least often.
    assert rate >= 0.5, f"safe fallback rate {rate:.2f} too low for hopeless regime"
