"""Fixed-share estimators: §2.5 (50/50 split) and §2.6 (old fixed-share rule).

Both are baselines, not the proposed method. The plug-in versions
(`fixed_half_split_plugin_alpha`, `old_fixed_share_plugin_alpha`) live here
but call into adaptive plug-in estimators for alpha and so are gated on the
adaptive module being importable.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .api import EstimatorResult, register, split_indices


def _gn_combine(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    x: int,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
) -> tuple[float, float, float, float]:
    """Run the standard (calibration / estimation / synthetic) GN pipeline at
    the given x with the oracle alpha. Returns (theta_hat, alpha, B, V_R)."""
    a = truth_params["a"]
    v_n = truth_params["v_n"]
    c = truth_params["c"]
    beta = truth_params["beta"]

    _, est_idx = split_indices(n, x, rng)
    theta_R = estimand(X[est_idx])
    Z = synth_fn(x, rng)
    theta_S = estimand(Z)

    b = v_n + c * (x ** (-2.0 * beta))
    v = a / max(n - x, 1)
    alpha = b / (v + b)
    return alpha * theta_R + (1.0 - alpha) * theta_S, alpha, b, v


def fixed_half_split_oracle_alpha(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """50/50 calibration with oracle MSE-optimal mixing weight (§2.5)."""
    x = n // 2
    theta_hat, alpha, b, v = _gn_combine(
        X, synth_fn, x=x, n=n, rng=rng,
        truth_params=truth_params, estimand=estimand,
    )
    a = truth_params["a"]
    return EstimatorResult(
        theta_hat=theta_hat,
        x_selected=x,
        alpha_selected=alpha,
        B_eff_selected=b,
        V_R_selected=v,
        estimated_risk_selected=v * b / (v + b),
        safe_pass=bool(x * b < a),
        safe_margin=a - x * b,
        fallback_used=False,
    )


fixed_half_split_oracle_alpha.name = "fixed_half_split_oracle_alpha"
fixed_half_split_oracle_alpha.needs_synth = True


def old_fixed_share_oracle_alpha(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """Old fixed-share rule lambda_old = 2 beta / (1 + 2 beta) (§2.6).

    Stress-test of the old theory; expected to over-calibrate when m≍n, β>1/2.
    """
    beta = truth_params["beta"]
    lam = (2.0 * beta) / (1.0 + 2.0 * beta)
    x = max(1, min(n - 1, int(np.floor(n * lam))))
    theta_hat, alpha, b, v = _gn_combine(
        X, synth_fn, x=x, n=n, rng=rng,
        truth_params=truth_params, estimand=estimand,
    )
    a = truth_params["a"]
    return EstimatorResult(
        theta_hat=theta_hat,
        x_selected=x,
        alpha_selected=alpha,
        B_eff_selected=b,
        V_R_selected=v,
        estimated_risk_selected=v * b / (v + b),
        safe_pass=bool(x * b < a),
        safe_margin=a - x * b,
        fallback_used=False,
    )


old_fixed_share_oracle_alpha.name = "old_fixed_share_oracle_alpha"
old_fixed_share_oracle_alpha.needs_synth = True


register(fixed_half_split_oracle_alpha)
register(old_fixed_share_oracle_alpha)
