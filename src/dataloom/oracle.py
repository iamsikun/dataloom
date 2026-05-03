"""Deterministic oracle calculator for the corrected risk model.

Implements the §1 risk model from docs/experiments.md:

    V_R(x) = a / (n - x)
    B_n(x) = v_n + c * x ** (-2 * beta)
    R_n(x) = V_R(x) * B_n(x) / (V_R(x) + B_n(x))

and the Priority 1 API in §14:

    B_eff, V_real, R_profile, foc_residual, safe_condition,
    oracle_grid, oracle_root.

All functions accept scalars or numpy arrays for x and broadcast naturally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import brentq


def B_eff(x, v_n: float, c: float, beta: float):
    """Effective synthetic error v_n + c * x^(-2 beta).

    For x = 0 the bias term diverges; return +inf so that downstream
    boundary handling treats x = 0 as the "use all real data" branch.
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.inf, dtype=float)
    pos = x > 0
    out[pos] = v_n + c * np.power(x[pos], -2.0 * beta)
    return out if out.shape else float(out)


def V_real(x, n: int, a: float):
    """Real-estimator variance a / (n - x).

    For x >= n returns +inf (no real data left for direct estimation).
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.inf, dtype=float)
    pos = x < n
    out[pos] = a / (n - x[pos])
    return out if out.shape else float(out)


def R_profile(x, n: int, a: float, v_n: float, c: float, beta: float):
    """Profiled risk V_R(x) * B(x) / (V_R(x) + B(x)).

    Handles boundary cases:
    - x = 0:           returns a / n           (real_only_all)
    - x = n:           returns B_eff(n)        (synthetic_only_full_calibration)
    - x in (0, n):     returns V_R * B / (V_R + B)
    """
    x_arr = np.asarray(x, dtype=float)
    scalar = x_arr.shape == ()
    x_arr = np.atleast_1d(x_arr)
    out = np.empty_like(x_arr, dtype=float)

    # x = 0: only real data is used -> risk is a / n
    zero = x_arr == 0
    out[zero] = a / n

    # x = n: only synthetic is used -> risk is B_eff(n)
    full = x_arr == n
    out[full] = v_n + c * np.power(n, -2.0 * beta)

    interior = (~zero) & (~full) & (x_arr > 0) & (x_arr < n)
    if np.any(interior):
        x_in = x_arr[interior]
        b = v_n + c * np.power(x_in, -2.0 * beta)
        v = a / (n - x_in)
        out[interior] = v * b / (v + b)

    invalid = ~(zero | full | interior)
    out[invalid] = np.inf

    return float(out[0]) if scalar else out


def foc_residual(x, a: float, v_n: float, c: float, beta: float):
    """First-order condition residual:

        (v_n + c x^{-2 beta})^2 - 2 beta a c x^{-(2 beta + 1)}.

    A root corresponds to a stationary point of R_n(x). Combine with
    grid minimization (oracle_grid) to confirm it's a minimum vs. boundary.
    """
    x = np.asarray(x, dtype=float)
    lhs = (v_n + c * np.power(x, -2.0 * beta)) ** 2
    rhs = 2.0 * beta * a * c * np.power(x, -(2.0 * beta + 1.0))
    return lhs - rhs


def safe_condition(x, a: float, v_n: float, c: float, beta: float):
    """Safe-improvement condition x * B_n(x) < a (§1).

    Returns boolean (or boolean array). True means the synthetic-augmented
    estimator strictly improves on real-only at this x.
    """
    x = np.asarray(x, dtype=float)
    return x * (v_n + c * np.power(x, -2.0 * beta)) < a


@dataclass
class OracleResult:
    """Result of grid minimization of R_n over x."""

    x_star: int
    risk_star: float
    alpha_star: float           # B / (V_R + B) at x_star; 1.0 at the x=0 boundary
    B_eff_star: float | None    # None at x = 0 boundary (no synthetic used)
    V_R_star: float | None      # None at x = n boundary (no real used)
    safe_pass: bool             # x_star * B(x_star) < a (False at x=0 by convention)
    safe_margin: float          # a - x_star * B(x_star)
    boundary: bool              # True iff x_star in {0, n}


def oracle_grid(
    n: int,
    a: float,
    v_n: float,
    c: float,
    beta: float,
    grid: Sequence[int] | np.ndarray | None = None,
    include_boundaries: bool = True,
) -> OracleResult:
    """Grid-minimize R_n(x) over x and return the oracle allocation.

    Default grid is the integer set {1, ..., n-1}. If include_boundaries=True,
    x=0 (real_only_all) and x=n (synthetic_only_full_calibration) are also
    considered.
    """
    if grid is None:
        x_grid = np.arange(1, n, dtype=int)
    else:
        x_grid = np.asarray(grid, dtype=int)

    if include_boundaries:
        x_grid = np.unique(np.concatenate([[0], x_grid, [n]]))

    risks = R_profile(x_grid, n=n, a=a, v_n=v_n, c=c, beta=beta)
    idx = int(np.argmin(risks))
    x_star = int(x_grid[idx])
    r_star = float(risks[idx])

    if x_star == 0:
        return OracleResult(
            x_star=0,
            risk_star=r_star,
            alpha_star=1.0,
            B_eff_star=None,
            V_R_star=a / n,
            safe_pass=False,
            safe_margin=a,
            boundary=True,
        )
    if x_star == n:
        b = v_n + c * np.power(n, -2.0 * beta)
        return OracleResult(
            x_star=n,
            risk_star=r_star,
            alpha_star=0.0,
            B_eff_star=float(b),
            V_R_star=None,
            safe_pass=bool(n * b < a),
            safe_margin=float(a - n * b),
            boundary=True,
        )

    b = float(v_n + c * x_star ** (-2.0 * beta))
    v = float(a / (n - x_star))
    alpha = b / (v + b)
    return OracleResult(
        x_star=x_star,
        risk_star=r_star,
        alpha_star=float(alpha),
        B_eff_star=b,
        V_R_star=v,
        safe_pass=bool(x_star * b < a),
        safe_margin=float(a - x_star * b),
        boundary=False,
    )


def oracle_root(
    n: int,
    a: float,
    v_n: float,
    c: float,
    beta: float,
    bracket: tuple[float, float] | None = None,
) -> float | None:
    """Brent root of foc_residual on the open interval (1, n-1).

    Returns the root if one exists with a sign change inside the bracket,
    else None. Used as a cross-check against oracle_grid in interior regimes.
    """
    lo, hi = bracket if bracket is not None else (1.0, float(n - 1))
    f_lo = float(foc_residual(lo, a, v_n, c, beta))
    f_hi = float(foc_residual(hi, a, v_n, c, beta))
    if not np.isfinite(f_lo) or not np.isfinite(f_hi):
        return None
    if f_lo * f_hi > 0:
        return None
    return float(brentq(foc_residual, lo, hi, args=(a, v_n, c, beta)))


def m_of_n(n: int, kappa: float, rho: float) -> int:
    """Synthetic sample size m_n = ceil(kappa * n^rho) per §4.2."""
    return int(np.ceil(kappa * n ** rho))


def v_of_n(n: int, kappa: float, rho: float, sigma_s2: float) -> float:
    """Synthetic estimator variance v_n = sigma_s2 / m_n."""
    return float(sigma_s2 / m_of_n(n, kappa, rho))
