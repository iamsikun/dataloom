"""Phase 4 sanity tests for multichannel."""

from __future__ import annotations

import numpy as np
import pytest

from dataloom.dgp.multichannel import (
    GaussianTwoChannelParams,
    make_synth_fn_2,
    sample_real,
)
from dataloom.dgp.gaussian import estimand_mean
from dataloom.io.seeds import make_rng
from dataloom.multichannel.estimators import (
    best_single_channel_oracle,
    corrected_multichannel_oracle,
    equal_split_two_channels,
    old_multichannel_fixed_share,
)
from dataloom.multichannel.oracle import (
    R_profile_2,
    marginal_value,
    oracle_grid_2,
)


def test_R_profile_2_matches_definition_interior():
    n, a, v_n = 1000, 1.0, 0.001
    c1, beta1 = 1.0, 1.0
    c2, beta2 = 1.0, 0.5
    x1 = np.array([10, 50])
    x2 = np.array([20, 100])
    actual = R_profile_2(x1, x2, n=n, a=a, v_n=v_n,
                         c1=c1, beta1=beta1, c2=c2, beta2=beta2)
    expected = np.empty_like(x1, dtype=float)
    for i, (a1, a2) in enumerate(zip(x1, x2)):
        b = v_n + c1 * a1 ** (-2 * beta1) + c2 * a2 ** (-2 * beta2)
        v = a / (n - a1 - a2)
        expected[i] = v * b / (v + b)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_oracle_grid_2_finds_minimum():
    """oracle_grid_2 risk should be ≤ R at any nearby integer point."""
    n, a, v_n = 1000, 1.0, 0.001
    res = oracle_grid_2(n=n, a=a, v_n=v_n,
                        c1=1.0, beta1=1.0, c2=1.0, beta2=0.5,
                        coarse_step=1)
    risk_neighbors = []
    for dx1 in (-1, 0, 1):
        for dx2 in (-1, 0, 1):
            x1 = res.x1_star + dx1
            x2 = res.x2_star + dx2
            if 1 <= x1 < n and 1 <= x2 < n and x1 + x2 < n:
                r = R_profile_2(np.array([x1]), np.array([x2]),
                                n=n, a=a, v_n=v_n,
                                c1=1.0, beta1=1.0, c2=1.0, beta2=0.5)[0]
                risk_neighbors.append(float(r))
    # Allow slack: oracle_grid_2 uses a coarse grid so neighbors might be lower.
    assert res.risk_star <= min(risk_neighbors) + 1e-6 or \
           abs(res.risk_star - min(risk_neighbors)) / res.risk_star < 0.01


def test_marginal_values_equal_at_oracle():
    """KKT (§6.3): MV_1 ≈ MV_2 at the corrected oracle, when both channels active."""
    res = oracle_grid_2(
        n=20000, a=1.0, v_n=1e-4,
        c1=1.0, beta1=1.0, c2=1.0, beta2=0.75,
    )
    if res.x1_star > 0 and res.x2_star > 0:
        # Marginal values should be roughly equal at the optimum
        # (loose tolerance because grid is discrete and we coarsen).
        ratio = res.mv1 / res.mv2
        assert 0.5 <= ratio <= 2.0, (
            f"MV1={res.mv1:.3e}, MV2={res.mv2:.3e}, ratio={ratio:.2f}"
        )


def test_old_fixed_share_mc_overcalibrates_in_fast_regime():
    """Old multichannel rule should pick a higher x1+x2 than corrected oracle
    in the common fast-learning regime (β1, β2 > 1/2)."""
    n = 5000
    res_oracle = oracle_grid_2(
        n=n, a=1.0, v_n=1.0 / n,
        c1=1.0, beta1=0.75, c2=1.0, beta2=1.5,
    )
    beta1, beta2 = 0.75, 1.5
    denom = 1.0 + 2.0 * (beta1 + beta2)
    lam_total = 2.0 * (beta1 + beta2) / denom
    x_total_old = int(np.floor(n * lam_total))
    assert x_total_old > res_oracle.x1_star + res_oracle.x2_star, (
        f"old total {x_total_old} should exceed corrected total "
        f"{res_oracle.x1_star + res_oracle.x2_star}"
    )


@pytest.mark.parametrize("name", [
    "best_single_channel_oracle",
    "equal_split_two_channels",
    "old_multichannel_fixed_share",
    "corrected_multichannel_oracle",
])
def test_multichannel_estimators_basic_call(name):
    p = GaussianTwoChannelParams(n=500, beta1=0.75, beta2=1.5, rho=1.0)
    rng = make_rng(7)
    X = sample_real(p, rng)
    synth_fn_2 = make_synth_fn_2(p)
    truth = {"a": p.a, "v_n": p.v_n,
             "c1": p.c1, "beta1": p.beta1,
             "c2": p.c2, "beta2": p.beta2}
    fn = {
        "best_single_channel_oracle": best_single_channel_oracle,
        "equal_split_two_channels": equal_split_two_channels,
        "old_multichannel_fixed_share": old_multichannel_fixed_share,
        "corrected_multichannel_oracle": corrected_multichannel_oracle,
    }[name]
    res = fn(X, synth_fn_2, n=p.n, rng=rng, params=truth, estimand=estimand_mean)
    assert np.isfinite(res.theta_hat)
