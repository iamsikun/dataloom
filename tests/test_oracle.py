"""Phase 1 sanity tests for the deterministic oracle calculator.

Maps to the §14 Priority 1 checklist in docs/experiments.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataloom.notation import classify_regime, theory_slope
from dataloom.oracle import (
    B_eff,
    OracleResult,
    R_profile,
    V_real,
    foc_residual,
    m_of_n,
    oracle_grid,
    oracle_root,
    safe_condition,
    v_of_n,
)


# ---------------------------------------------------------------------------
# Notation / regime classifier (§4.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "beta,rho,expected",
    [
        (1.0, 0.0, "persistent_variance"),
        (0.4, 1.0, "slow_learning"),
        (0.5, 1.0, "parametric_knife_edge"),
        (1.0, 1.0, "sublinear_fast_learning"),    # rho=1 < beta+0.5=1.5
        (1.0, 2.0, "boundary_full_calibration"),  # rho=2 > beta+0.5=1.5
        (1.0, 1.5, "boundary_knife_edge"),        # rho == beta+0.5
    ],
)
def test_classify_regime(beta, rho, expected):
    assert classify_regime(beta, rho) == expected


def test_theory_slope_known_cases():
    # rho=0: bounded calibration -> slope 0
    assert theory_slope(1.0, 0.0) == 0.0
    # beta < 1/2: slow learning -> slope 0
    assert theory_slope(0.4, 1.0) == 0.0
    # interior fast-learning: 2 rho / (2 beta + 1)
    assert theory_slope(1.0, 1.0) == pytest.approx(2.0 / 3.0)
    # boundary full calibration -> slope 1
    assert theory_slope(1.0, 2.0) == 1.0
    # knife edges: None
    assert theory_slope(0.5, 1.0) is None
    assert theory_slope(1.0, 1.5) is None


# ---------------------------------------------------------------------------
# Risk model identities
# ---------------------------------------------------------------------------


def test_R_profile_matches_definition_interior():
    n, a, v_n, c, beta = 1000, 1.0, 0.001, 1.0, 1.0
    x = np.array([10, 50, 100, 500])
    b = B_eff(x, v_n, c, beta)
    v = V_real(x, n, a)
    expected = v * b / (v + b)
    actual = R_profile(x, n=n, a=a, v_n=v_n, c=c, beta=beta)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_R_profile_boundary_x_zero_equals_a_over_n():
    """At x = 0, no synthetic data is used: risk should be exactly a / n."""
    n, a = 500, 2.5
    r = R_profile(0, n=n, a=a, v_n=0.1, c=1.0, beta=1.0)
    assert r == pytest.approx(a / n)


def test_R_profile_boundary_x_n_equals_B_eff_n():
    """At x = n, no real data is left: risk should equal B_eff(n)."""
    n, v_n, c, beta = 500, 0.01, 1.0, 1.0
    expected = v_n + c * n ** (-2 * beta)
    r = R_profile(n, n=n, a=1.0, v_n=v_n, c=c, beta=beta)
    assert r == pytest.approx(expected)


def test_B_eff_x_zero_is_inf():
    assert np.isinf(B_eff(0, v_n=0.01, c=1.0, beta=1.0))


def test_V_real_x_ge_n_is_inf():
    assert np.isinf(V_real(1000, n=1000, a=1.0))
    assert np.isinf(V_real(1500, n=1000, a=1.0))


# ---------------------------------------------------------------------------
# FOC residual and safe condition
# ---------------------------------------------------------------------------


def test_safe_condition_matches_inequality():
    n, a, v_n, c, beta = 1000, 1.0, 0.001, 1.0, 1.0
    x = np.array([1, 10, 100, 500])
    expected = x * (v_n + c * np.power(x, -2.0 * beta)) < a
    actual = safe_condition(x, a=a, v_n=v_n, c=c, beta=beta)
    np.testing.assert_array_equal(actual, expected)


def test_foc_residual_zero_at_grid_argmin_interior():
    """At the interior grid argmin, the FOC residual should be ~0."""
    # Choose params guaranteed to give an interior optimum.
    n, a, c, beta = 50000, 1.0, 1.0, 1.0
    rho = 1.0
    v_n = v_of_n(n, kappa=1.0, rho=rho, sigma_s2=1.0)

    res = oracle_grid(n=n, a=a, v_n=v_n, c=c, beta=beta)
    assert not res.boundary, "expected interior optimum for sanity test"

    # The discrete grid argmin won't make the continuous FOC exactly zero,
    # but the magnitude should be small relative to either side of FOC.
    fr = foc_residual(res.x_star, a=a, v_n=v_n, c=c, beta=beta)
    rhs = 2.0 * beta * a * c * res.x_star ** (-(2.0 * beta + 1.0))
    assert abs(fr) < 0.05 * rhs


def test_oracle_grid_matches_oracle_root_in_interior():
    """Brent root and grid argmin agree to within 1 step in interior regime."""
    n, a, c, beta = 50000, 1.0, 1.0, 1.0
    rho = 1.0
    v_n = v_of_n(n, kappa=1.0, rho=rho, sigma_s2=1.0)

    grid_x = oracle_grid(n=n, a=a, v_n=v_n, c=c, beta=beta).x_star
    root_x = oracle_root(n=n, a=a, v_n=v_n, c=c, beta=beta)
    assert root_x is not None
    assert abs(grid_x - root_x) <= 1.0


# ---------------------------------------------------------------------------
# Oracle scaling: log x* ~ slope * log n in the interior fast-learning regime
# ---------------------------------------------------------------------------


def test_oracle_scaling_interior_regime():
    """For beta=1, rho=1 (sublinear fast learning), x* should scale as
    n^(2 rho / (2 beta + 1)) = n^(2/3)."""
    beta, rho = 1.0, 1.0
    a, c, sigma_s2, kappa = 1.0, 1.0, 1.0, 1.0

    ns = np.array([2000, 5000, 10000, 20000, 50000])
    xs = np.empty_like(ns, dtype=float)
    for i, n in enumerate(ns):
        v_n = v_of_n(int(n), kappa=kappa, rho=rho, sigma_s2=sigma_s2)
        xs[i] = oracle_grid(n=int(n), a=a, v_n=v_n, c=c, beta=beta).x_star

    log_n = np.log(ns)
    log_x = np.log(xs)
    slope, _ = np.polyfit(log_n, log_x, 1)

    expected = theory_slope(beta, rho)
    assert expected is not None
    # Discrete grid + finite-n constants -> allow ~10% slack.
    assert abs(slope - expected) < 0.05, (
        f"empirical slope {slope:.4f} vs theory {expected:.4f}"
    )


def test_oracle_scaling_boundary_full_calibration():
    """For beta=1, rho=2 (boundary regime), oracle should hit x = n."""
    beta, rho = 1.0, 2.0
    a, c, sigma_s2, kappa = 1.0, 1.0, 1.0, 1.0

    for n in (200, 1000, 5000):
        v_n = v_of_n(n, kappa=kappa, rho=rho, sigma_s2=sigma_s2)
        res = oracle_grid(n=n, a=a, v_n=v_n, c=c, beta=beta)
        # In the boundary regime, optimum is x = n (full calibration).
        assert res.x_star == n, (
            f"n={n}: expected boundary x*=n, got {res.x_star}; "
            f"v_n={v_n:.3e}, risk={res.risk_star:.3e}"
        )
        assert res.boundary


def test_oracle_persistent_variance_regime_bounded():
    """For rho = 0, m_n stays constant: x* should not grow with n."""
    beta, rho = 1.0, 0.0
    a, c, sigma_s2, kappa = 1.0, 1.0, 1.0, 100.0  # m fixed at 100

    xs = []
    for n in (1000, 5000, 20000):
        v_n = v_of_n(n, kappa=kappa, rho=rho, sigma_s2=sigma_s2)
        xs.append(oracle_grid(n=n, a=a, v_n=v_n, c=c, beta=beta).x_star)

    # x* should be bounded as n grows (theory slope = 0).
    assert max(xs) - min(xs) < 50, f"expected bounded x*, got {xs}"


def test_oracle_real_only_dominates_when_synthetic_useless():
    """When v_n is huge (synthetic noise dominates), x* should be 0."""
    n, a, c, beta = 500, 1.0, 1.0, 1.0
    v_n = 1000.0  # synthetic data is hopeless
    res = oracle_grid(n=n, a=a, v_n=v_n, c=c, beta=beta)
    assert res.x_star == 0
    assert res.alpha_star == 1.0
    assert res.boundary


# ---------------------------------------------------------------------------
# OracleResult invariants
# ---------------------------------------------------------------------------


def test_oracle_result_fields_consistent_interior():
    n, a, c, beta = 10000, 1.0, 1.0, 1.0
    v_n = v_of_n(n, kappa=1.0, rho=1.0, sigma_s2=1.0)
    res = oracle_grid(n=n, a=a, v_n=v_n, c=c, beta=beta)
    assert isinstance(res, OracleResult)
    assert not res.boundary
    # alpha = B / (V_R + B)
    assert res.alpha_star == pytest.approx(
        res.B_eff_star / (res.V_R_star + res.B_eff_star)
    )
    # safe_margin = a - x * B
    assert res.safe_margin == pytest.approx(a - res.x_star * res.B_eff_star)
    # safe_pass agrees with safe_margin > 0
    assert res.safe_pass == (res.safe_margin > 0)


def test_m_of_n_matches_spec():
    # m_n = ceil(kappa * n^rho)
    assert m_of_n(1000, kappa=1.0, rho=1.0) == 1000
    assert m_of_n(1000, kappa=2.0, rho=1.0) == 2000
    assert m_of_n(100, kappa=1.0, rho=1.5) == int(np.ceil(100 ** 1.5))
    assert m_of_n(1000, kappa=1.0, rho=0.0) == 1
