"""Interval procedures for Experiment 4 (§7.3).

Locked-in convention (per the plan):
    Ω̂_n = α̂² · â / (n − x̂) + (1 − α̂)² · v̂_n
for ci_gn_naive and ci_gn_bias_aware.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..adaptive.bias_curve import (
    default_pilot_grid,
    estimate_bias_curve,
)
from ..adaptive.parametric import _resolve_n_v, _run_at_with_plugins
from ..estimators.api import EstimatorResult, register, split_indices
from ..oracle import oracle_grid

Z975 = 1.959963984540054   # scipy.stats.norm.ppf(0.975)


def ci_real_only(
    X: np.ndarray,
    synth_fn=None,
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """θ̂_R,all ± 1.96 sqrt(â / n) (§7.3.A)."""
    theta_hat = estimand(X)
    a_hat = float(np.var(X, ddof=1))
    se = (a_hat / n) ** 0.5
    return EstimatorResult(
        theta_hat=theta_hat,
        x_selected=0,
        alpha_selected=1.0,
        V_R_selected=a_hat / n,
        a_hat=a_hat,
        ci_lower=theta_hat - Z975 * se,
        ci_upper=theta_hat + Z975 * se,
        fallback_used=False,
    )


ci_real_only.name = "ci_real_only"
ci_real_only.needs_synth = False


def _adaptive_select(X, synth_fn, n, rng, truth_params, estimand, config):
    """Run the standard adaptive pipeline and return (fit, n_eff, x_hat, b_hat,
    v_R_hat, alpha_hat, theta_hat). Used by the GN-based interval methods."""
    n_v = _resolve_n_v(n, config)
    m = truth_params["m"]
    fit = estimate_bias_curve(
        n=n, n_v=n_v, X=X, synth_fn=synth_fn, rng=rng, estimand=estimand,
        pilot_grid=default_pilot_grid(n), m=m,
    )
    n_eff = n - n_v
    res = oracle_grid(n=n_eff, a=fit.a_hat, v_n=fit.v_hat,
                      c=fit.c_hat, beta=fit.beta_hat)
    x_hat = res.x_star
    if x_hat <= 0:
        # Real-only fallback: use whole X with plug-in variance.
        return None, n_eff, 0, None, fit.a_hat / n, 1.0, estimand(X), fit
    if x_hat >= n_eff:
        x_hat = n_eff - 1
    cal_idx, est_idx = split_indices(n_eff, x_hat, rng)
    Z = synth_fn(int(x_hat), rng)
    theta_R = estimand(X[: n_eff][est_idx])
    theta_S = estimand(Z)
    b_hat = float(fit.v_hat + fit.c_hat * (x_hat ** (-2.0 * fit.beta_hat)))
    v_R_hat = float(fit.a_hat / max(n_eff - x_hat, 1))
    alpha_hat = b_hat / (v_R_hat + b_hat)
    theta_hat = alpha_hat * theta_R + (1.0 - alpha_hat) * theta_S
    return fit, n_eff, x_hat, b_hat, v_R_hat, alpha_hat, theta_hat, fit


def _omega_n(alpha: float, a_hat: float, n_eff: int, x_hat: int,
             v_hat: float) -> float:
    """Plug-in variance per the locked-in convention."""
    n_e = max(n_eff - x_hat, 1)
    return alpha ** 2 * (a_hat / n_e) + (1.0 - alpha) ** 2 * v_hat


def ci_gn_naive(
    X: np.ndarray,
    synth_fn,
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """Wald interval ignoring synthetic bias (§7.3.B). Expected to undercover."""
    out = _adaptive_select(X, synth_fn, n, rng, truth_params, estimand, config)
    if out[0] is None:
        # Real-only fallback path
        _, _, _, _, v_R, alpha, theta_hat, fit = out
        se = v_R ** 0.5
        return EstimatorResult(
            theta_hat=theta_hat, x_selected=0, alpha_selected=1.0,
            V_R_selected=v_R, a_hat=fit.a_hat, v_hat=fit.v_hat,
            beta_hat=fit.beta_hat, c_hat=fit.c_hat,
            ci_lower=theta_hat - Z975 * se, ci_upper=theta_hat + Z975 * se,
            fallback_used=True,
        )
    fit, n_eff, x_hat, b_hat, v_R, alpha, theta_hat, _ = out
    om = _omega_n(alpha, fit.a_hat, n_eff, x_hat, fit.v_hat)
    se = om ** 0.5
    return EstimatorResult(
        theta_hat=theta_hat, x_selected=x_hat, alpha_selected=alpha,
        B_eff_selected=b_hat, V_R_selected=v_R,
        a_hat=fit.a_hat, v_hat=fit.v_hat,
        beta_hat=fit.beta_hat, c_hat=fit.c_hat,
        estimated_risk_selected=v_R * b_hat / (v_R + b_hat),
        safe_pass=bool(x_hat * b_hat < fit.a_hat),
        safe_margin=fit.a_hat - x_hat * b_hat,
        fallback_used=False,
        ci_lower=theta_hat - Z975 * se,
        ci_upper=theta_hat + Z975 * se,
    )


ci_gn_naive.name = "ci_gn_naive"
ci_gn_naive.needs_synth = True


def ci_gn_bias_aware(
    X: np.ndarray,
    synth_fn,
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """θ̂ ± (1.96 sqrt(Ω̂) + |(1-α̂) b̂(x̂)|), where b̂ = sqrt(positive bias²)."""
    out = _adaptive_select(X, synth_fn, n, rng, truth_params, estimand, config)
    if out[0] is None:
        _, _, _, _, v_R, alpha, theta_hat, fit = out
        se = v_R ** 0.5
        return EstimatorResult(
            theta_hat=theta_hat, x_selected=0, alpha_selected=1.0,
            V_R_selected=v_R, a_hat=fit.a_hat, v_hat=fit.v_hat,
            beta_hat=fit.beta_hat, c_hat=fit.c_hat,
            ci_lower=theta_hat - Z975 * se, ci_upper=theta_hat + Z975 * se,
            fallback_used=True,
        )
    fit, n_eff, x_hat, b_hat, v_R, alpha, theta_hat, _ = out
    om = _omega_n(alpha, fit.a_hat, n_eff, x_hat, fit.v_hat)
    se = om ** 0.5
    # Bias is the synthetic bias in the *combined* estimator: (1 - alpha) * b
    b_signed = float(fit.c_hat) ** 0.5 * (x_hat ** (-fit.beta_hat))   # |b|
    bias_pad = abs(1.0 - alpha) * b_signed
    return EstimatorResult(
        theta_hat=theta_hat, x_selected=x_hat, alpha_selected=alpha,
        B_eff_selected=b_hat, V_R_selected=v_R,
        a_hat=fit.a_hat, v_hat=fit.v_hat,
        beta_hat=fit.beta_hat, c_hat=fit.c_hat,
        estimated_risk_selected=v_R * b_hat / (v_R + b_hat),
        ci_lower=theta_hat - (Z975 * se + bias_pad),
        ci_upper=theta_hat + (Z975 * se + bias_pad),
        fallback_used=False,
    )


ci_gn_bias_aware.name = "ci_gn_bias_aware"
ci_gn_bias_aware.needs_synth = True


def ci_gn_undersmoothed(
    X: np.ndarray,
    synth_fn,
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """Pick larger x so |(1-α) b(x)| ≤ ε √Ω(x). Default ε=0.1.

    Implementation: scan x along the integer grid; keep the smallest x that
    satisfies the undersmoothing constraint AND has profiled risk ≤ 1.5 *
    optimal (so we don't entirely abandon MSE). If none satisfies, use the
    bias-aware interval at the corrected x̂.
    """
    cfg = config or {}
    eps = float(cfg.get("epsilon", 0.1))

    out = _adaptive_select(X, synth_fn, n, rng, truth_params, estimand, config)
    if out[0] is None:
        _, _, _, _, v_R, alpha, theta_hat, fit = out
        se = v_R ** 0.5
        return EstimatorResult(
            theta_hat=theta_hat, x_selected=0, alpha_selected=1.0,
            V_R_selected=v_R, a_hat=fit.a_hat, v_hat=fit.v_hat,
            beta_hat=fit.beta_hat, c_hat=fit.c_hat,
            ci_lower=theta_hat - Z975 * se, ci_upper=theta_hat + Z975 * se,
            fallback_used=True,
        )
    fit, n_eff, x_hat_mse, _, _, _, _, _ = out

    # Search grid larger than x_hat_mse for an undersmoothed point.
    grid = np.unique(np.concatenate([
        np.arange(x_hat_mse, n_eff, max(1, (n_eff - x_hat_mse) // 100)),
        [n_eff - 1],
    ])).astype(int)
    chosen = None
    for x in grid:
        if x <= 0 or x >= n_eff:
            continue
        b = fit.v_hat + fit.c_hat * (x ** (-2.0 * fit.beta_hat))
        v = fit.a_hat / max(n_eff - x, 1)
        alpha = b / (v + b)
        b_abs = float(fit.c_hat) ** 0.5 * (x ** (-fit.beta_hat))
        bias_combined = abs(1.0 - alpha) * b_abs
        om = _omega_n(alpha, fit.a_hat, n_eff, int(x), fit.v_hat)
        se = om ** 0.5
        if bias_combined <= eps * se:
            chosen = (int(x), alpha, b, v, om)
            break
    if chosen is None:
        # Fallback to bias-aware at the MSE-optimal x.
        return ci_gn_bias_aware(
            X, synth_fn, n=n, rng=rng,
            truth_params=truth_params, estimand=estimand, config=config,
        )
    x, alpha, b, v, om = chosen
    cal_idx, est_idx = split_indices(n_eff, x, rng)
    Z = synth_fn(x, rng)
    theta_R = estimand(X[: n_eff][est_idx])
    theta_S = estimand(Z)
    theta_hat = alpha * theta_R + (1.0 - alpha) * theta_S
    se = om ** 0.5
    return EstimatorResult(
        theta_hat=theta_hat, x_selected=x, alpha_selected=alpha,
        B_eff_selected=b, V_R_selected=v,
        a_hat=fit.a_hat, v_hat=fit.v_hat,
        beta_hat=fit.beta_hat, c_hat=fit.c_hat,
        estimated_risk_selected=v * b / (v + b),
        ci_lower=theta_hat - Z975 * se, ci_upper=theta_hat + Z975 * se,
        fallback_used=False,
    )


ci_gn_undersmoothed.name = "ci_gn_undersmoothed"
ci_gn_undersmoothed.needs_synth = True


def ci_validation_debiased(
    X: np.ndarray,
    synth_fn,
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """Held-out validation debiasing (§2.10 / §7.3.E)."""
    n_v = _resolve_n_v(n, config)
    m = truth_params["m"]
    n_eff = n - n_v
    val = X[n - n_v :]
    rest = X[: n_eff]

    fit = estimate_bias_curve(
        n=n, n_v=n_v, X=X, synth_fn=synth_fn, rng=rng, estimand=estimand,
        pilot_grid=default_pilot_grid(n), m=m,
    )
    res = oracle_grid(n=n_eff, a=fit.a_hat, v_n=fit.v_hat,
                      c=fit.c_hat, beta=fit.beta_hat)
    x_hat = res.x_star
    if x_hat <= 0:
        theta_hat = estimand(X)
        se = (fit.a_hat / n) ** 0.5
        return EstimatorResult(
            theta_hat=theta_hat, x_selected=0, alpha_selected=1.0,
            V_R_selected=fit.a_hat / n, a_hat=fit.a_hat, v_hat=fit.v_hat,
            beta_hat=fit.beta_hat, c_hat=fit.c_hat,
            ci_lower=theta_hat - Z975 * se, ci_upper=theta_hat + Z975 * se,
            fallback_used=True,
        )
    if x_hat >= n_eff:
        x_hat = n_eff - 1

    cal_idx, est_idx = split_indices(n_eff, x_hat, rng)
    theta_R = estimand(rest[est_idx])
    Z = synth_fn(int(x_hat), rng)
    theta_S = estimand(Z)

    # Validation-based bias estimate at this x:
    Z_v = synth_fn(int(x_hat), rng)  # fresh synthetic for validation step
    theta_R_V = estimand(val)
    theta_S_V = estimand(Z_v)
    bias_hat = theta_S_V - theta_R_V    # estimate of synthetic - real bias
    theta_S_tilde = theta_S - bias_hat

    # Combine with the same alpha.
    b_hat = float(fit.v_hat + fit.c_hat * (x_hat ** (-2.0 * fit.beta_hat)))
    v_R = float(fit.a_hat / max(n_eff - x_hat, 1))
    alpha = b_hat / (v_R + b_hat)
    theta_hat = alpha * theta_R + (1.0 - alpha) * theta_S_tilde

    # Variance: alpha² Var(theta_R) + (1-alpha)² (Var(theta_S) + Var(bias_hat)).
    # Var(bias_hat) ≈ Var(theta_S_V) + Var(theta_R_V) ≈ v_n + a/n_v.
    var_bias = fit.v_hat + fit.a_hat / max(n_v, 1)
    om = alpha ** 2 * v_R + (1.0 - alpha) ** 2 * (fit.v_hat + var_bias)
    se = om ** 0.5
    return EstimatorResult(
        theta_hat=theta_hat, x_selected=x_hat, alpha_selected=alpha,
        B_eff_selected=b_hat, V_R_selected=v_R,
        a_hat=fit.a_hat, v_hat=fit.v_hat,
        beta_hat=fit.beta_hat, c_hat=fit.c_hat,
        estimated_risk_selected=v_R * b_hat / (v_R + b_hat),
        ci_lower=theta_hat - Z975 * se,
        ci_upper=theta_hat + Z975 * se,
        fallback_used=False,
    )


ci_validation_debiased.name = "ci_validation_debiased"
ci_validation_debiased.needs_synth = True


# Register them all.
register(ci_real_only)
register(ci_gn_naive)
register(ci_gn_bias_aware)
register(ci_gn_undersmoothed)
register(ci_validation_debiased)


# Also register a non-CI alias for `validation_debiased_gn` (§2.10 point estimator).
def validation_debiased_gn(*args, **kwargs):
    return ci_validation_debiased(*args, **kwargs)


validation_debiased_gn.name = "validation_debiased_gn"
validation_debiased_gn.needs_synth = True
register(validation_debiased_gn)
