"""Corrected oracle Generalized Neyman estimators (§2.7, §2.9 + §4.5).

`corrected_oracle_gn`        — minimize R_n(x) over the integer grid (oracle)
`safe_corrected_oracle_gn`   — same, but fall back to real_only_all if the
                                safety check x * B(x) < a fails.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..oracle import oracle_grid
from .api import EstimatorResult, register, split_indices


def _run_at(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    x: int,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
) -> EstimatorResult:
    """Standard GN combination at a fixed x with oracle alpha."""
    a = truth_params["a"]
    v_n = truth_params["v_n"]
    c = truth_params["c"]
    beta = truth_params["beta"]

    if x == 0:
        # collapse to real_only_all
        return EstimatorResult(
            theta_hat=estimand(X),
            x_selected=0,
            alpha_selected=1.0,
            V_R_selected=a / n,
            estimated_risk_selected=a / n,
            safe_pass=False,  # x*B = 0 trivially; we just record the fallback
            fallback_used=False,
        )
    if x >= n:
        Z = synth_fn(n, rng)
        b = v_n + c * (n ** (-2.0 * beta))
        return EstimatorResult(
            theta_hat=estimand(Z),
            x_selected=n,
            alpha_selected=0.0,
            B_eff_selected=b,
            estimated_risk_selected=b,
            safe_pass=bool(n * b < a),
            safe_margin=a - n * b,
            fallback_used=False,
        )

    _, est_idx = split_indices(n, x, rng)
    theta_R = estimand(X[est_idx])
    Z = synth_fn(x, rng)
    theta_S = estimand(Z)
    b = v_n + c * (x ** (-2.0 * beta))
    v = a / (n - x)
    alpha = b / (v + b)
    return EstimatorResult(
        theta_hat=alpha * theta_R + (1.0 - alpha) * theta_S,
        x_selected=x,
        alpha_selected=alpha,
        B_eff_selected=b,
        V_R_selected=v,
        estimated_risk_selected=v * b / (v + b),
        safe_pass=bool(x * b < a),
        safe_margin=a - x * b,
        fallback_used=False,
    )


def corrected_oracle_gn(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """Pick x* = argmin_x R_n(x) over the integer grid (incl. boundaries),
    then combine real and synthetic with the oracle mixing weight.
    Benchmark for all adaptive methods."""
    a = truth_params["a"]
    v_n = truth_params["v_n"]
    c = truth_params["c"]
    beta = truth_params["beta"]
    res = oracle_grid(n=n, a=a, v_n=v_n, c=c, beta=beta)
    out = _run_at(
        X, synth_fn, x=res.x_star, n=n, rng=rng,
        truth_params=truth_params, estimand=estimand,
    )
    return out


corrected_oracle_gn.name = "corrected_oracle_gn"
corrected_oracle_gn.needs_synth = True


def safe_corrected_oracle_gn(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """Corrected oracle with the §2.9 safety check: if x* * B(x*) >= a, fall
    back to real_only_all. Resolves the §4.5 reference to safe_corrected_oracle_gn.
    """
    a = truth_params["a"]
    v_n = truth_params["v_n"]
    c = truth_params["c"]
    beta = truth_params["beta"]
    res = oracle_grid(n=n, a=a, v_n=v_n, c=c, beta=beta)

    if res.x_star == 0 or not res.safe_pass:
        # fall back to real-only
        return EstimatorResult(
            theta_hat=estimand(X),
            x_selected=0,
            alpha_selected=1.0,
            V_R_selected=a / n,
            estimated_risk_selected=a / n,
            safe_pass=False,
            safe_margin=res.safe_margin,
            fallback_used=True,
        )

    return _run_at(
        X, synth_fn, x=res.x_star, n=n, rng=rng,
        truth_params=truth_params, estimand=estimand,
    )


safe_corrected_oracle_gn.name = "safe_corrected_oracle_gn"
safe_corrected_oracle_gn.needs_synth = True


register(corrected_oracle_gn)
register(safe_corrected_oracle_gn)
