"""Parametric adaptive estimators (docs/experiments.md §5.5 A and B).

Both estimate (â, v̂_n, ĉ, β̂) from the bias curve and pick x̂:
    A. adaptive_parametric_foc:   Brent root of FOC residual on (1, n_eff-1).
    B. adaptive_parametric_grid:  argmin over the integer grid of R̂_n(x).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..estimators.api import EstimatorResult, register, split_indices
from ..oracle import foc_residual, oracle_grid
from .bias_curve import default_pilot_grid, estimate_bias_curve


def _resolve_n_v(n: int, config: dict[str, Any] | None) -> int:
    """Resolve validation size.

    By default adaptive point estimation uses all real observations as the
    bias-curve reference and reserves no separate validation block. If callers
    request the original validation-reference protocol, fall back to the
    docs/experiments.md default.
    """
    cfg = config or {}
    if "n_v" in cfg:
        return max(0, min(int(cfg["n_v"]), n - 2))
    if cfg.get("bias_reference", "all") == "validation":
        return max(1, min(int(np.floor(0.2 * n)), n // 4))
    return 0


def _bias_reference(config: dict[str, Any] | None) -> str:
    return str((config or {}).get("bias_reference", "all"))


def _pilot_repeats(config: dict[str, Any] | None) -> int | None:
    cfg = config or {}
    return int(cfg["pilot_repeats"]) if "pilot_repeats" in cfg else None


def _min_positive(config: dict[str, Any] | None) -> int:
    return int((config or {}).get("min_positive_bias_points", 3))


def _min_beta_for_synthetic(config: dict[str, Any] | None) -> float:
    # The β=1/2 parametric edge is constants-driven. With the paper's default
    # a=c=1 it should fall back to real-only, and noisy β̂ just above 1/2 was
    # a recurrent harm mode. Use a conservative guard band by default.
    return float((config or {}).get("min_beta_for_synthetic", 0.60))


def _fit_allows_synthetic(fit, config: dict[str, Any] | None) -> bool:
    return bool(fit.fit_reliable and fit.beta_hat > _min_beta_for_synthetic(config))


def _candidate_grid(n_eff: int, fit, config: dict[str, Any] | None) -> np.ndarray:
    """Adaptive grid for plug-in minimization.

    We do not allow x below the smallest observed pilot size, because that was
    the catastrophic extrapolation path in the failed runs. Extrapolation above
    the pilot grid is allowed for the parametric power-law fit so boundary/full
    calibration regimes remain reachable.
    """
    cfg = config or {}
    min_x = int(cfg.get("min_candidate_x", int(np.min(fit.pilot_x))))
    min_x = max(1, min(min_x, n_eff))
    interior = np.arange(min_x, n_eff, dtype=int)
    return np.unique(np.concatenate([[0], interior, [n_eff]]))


def _real_only_from_fit(
    X: np.ndarray,
    *,
    n: int,
    fit,
    estimand: Callable[[np.ndarray], float],
    fallback_used: bool,
    safe_margin: float | None = None,
) -> EstimatorResult:
    return EstimatorResult(
        theta_hat=estimand(X),
        x_selected=0,
        alpha_selected=1.0,
        V_R_selected=fit.a_hat / n,
        beta_hat=fit.beta_hat,
        c_hat=fit.c_hat,
        a_hat=fit.a_hat,
        v_hat=fit.v_hat,
        estimated_risk_selected=fit.a_hat / n,
        safe_pass=False,
        safe_margin=fit.a_hat if safe_margin is None else safe_margin,
        fallback_used=fallback_used,
    )


def _combine(
    X_rest: np.ndarray,
    Z: np.ndarray,
    estimand: Callable[[np.ndarray], float],
    alpha: float,
    est_idx: np.ndarray,
) -> float:
    theta_R = estimand(X_rest[est_idx])
    theta_S = estimand(Z)
    return alpha * theta_R + (1.0 - alpha) * theta_S


def adaptive_parametric_grid(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """Estimate (â, v̂, ĉ, β̂) and grid-minimize R̂_n(x) over x ∈ [1, n_eff)."""
    n_v = _resolve_n_v(n, config)
    m = truth_params["m"]
    fit = estimate_bias_curve(
        n=n, n_v=n_v, X=X, synth_fn=synth_fn, rng=rng, estimand=estimand,
        pilot_grid=default_pilot_grid(n), m=m,
        pilot_repeats=_pilot_repeats(config),
        min_positive=_min_positive(config),
        reference=_bias_reference(config),
    )
    if not _fit_allows_synthetic(fit, config):
        return _real_only_from_fit(
            X, n=n, fit=fit, estimand=estimand, fallback_used=True
        )

    # Available real for cal+est is X[: n - n_v]. The grid x runs over
    # [0, n_eff], where n_eff = n - n_v.
    n_eff = n - n_v
    grid = _candidate_grid(n_eff, fit, config)
    res = oracle_grid(
        n=n_eff, a=fit.a_hat, v_n=fit.v_hat, c=fit.c_hat, beta=fit.beta_hat,
        grid=grid, include_boundaries=False,
    )
    x_hat = res.x_star
    return _run_at_with_plugins(
        X=X, synth_fn=synth_fn, x=x_hat, n=n, n_v=n_v, n_eff=n_eff,
        rng=rng, fit=fit, estimand=estimand,
    )


adaptive_parametric_grid.name = "adaptive_parametric_grid"
adaptive_parametric_grid.needs_synth = True


def adaptive_parametric_foc(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """Estimate (â, v̂, ĉ, β̂) and Brent-root the FOC on (1, n_eff-1)."""
    n_v = _resolve_n_v(n, config)
    m = truth_params["m"]
    fit = estimate_bias_curve(
        n=n, n_v=n_v, X=X, synth_fn=synth_fn, rng=rng, estimand=estimand,
        pilot_grid=default_pilot_grid(n), m=m,
        pilot_repeats=_pilot_repeats(config),
        min_positive=_min_positive(config),
        reference=_bias_reference(config),
    )
    if not _fit_allows_synthetic(fit, config):
        return _real_only_from_fit(
            X, n=n, fit=fit, estimand=estimand, fallback_used=True
        )
    n_eff = n - n_v
    # Try Brent on the FOC; fall back to grid argmin if no sign change.
    lo, hi = 1.0, float(n_eff - 1)
    f_lo = float(foc_residual(lo, fit.a_hat, fit.v_hat, fit.c_hat, fit.beta_hat))
    f_hi = float(foc_residual(hi, fit.a_hat, fit.v_hat, fit.c_hat, fit.beta_hat))
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        # Boundary regime: FOC has no interior root. Use grid argmin.
        grid = _candidate_grid(n_eff, fit, config)
        res = oracle_grid(n=n_eff, a=fit.a_hat, v_n=fit.v_hat,
                          c=fit.c_hat, beta=fit.beta_hat,
                          grid=grid, include_boundaries=False)
        x_hat = res.x_star
    else:
        from scipy.optimize import brentq
        root = brentq(
            foc_residual, lo, hi,
            args=(fit.a_hat, fit.v_hat, fit.c_hat, fit.beta_hat),
        )
        x_hat = int(np.round(root))
        min_x = int(np.min(_candidate_grid(n_eff, fit, config)[1:]))
        x_hat = int(np.clip(x_hat, min_x, n_eff - 1))

    return _run_at_with_plugins(
        X=X, synth_fn=synth_fn, x=x_hat, n=n, n_v=n_v, n_eff=n_eff,
        rng=rng, fit=fit, estimand=estimand,
    )


adaptive_parametric_foc.name = "adaptive_parametric_foc"
adaptive_parametric_foc.needs_synth = True


def _run_at_with_plugins(
    *,
    X: np.ndarray,
    synth_fn,
    x: int,
    n: int,
    n_v: int,
    n_eff: int,
    rng: np.random.Generator,
    fit,
    estimand: Callable[[np.ndarray], float],
) -> EstimatorResult:
    """Standard combine using plug-in (â, v̂, ĉ, β̂) for alpha and risk."""
    if x <= 0:
        # Real-only: use ALL real data (validation set included).
        return EstimatorResult(
            theta_hat=estimand(X),
            x_selected=0,
            alpha_selected=1.0,
            V_R_selected=fit.a_hat / n,
            beta_hat=fit.beta_hat,
            c_hat=fit.c_hat,
            a_hat=fit.a_hat,
            v_hat=fit.v_hat,
            estimated_risk_selected=fit.a_hat / n,
            safe_pass=False,
            fallback_used=False,
        )
    if x >= n_eff:
        x = n_eff
        Z = synth_fn(int(x), rng)
        b = float(fit.v_hat + fit.c_hat * (x ** (-2.0 * fit.beta_hat)))
        return EstimatorResult(
            theta_hat=estimand(Z),
            x_selected=x,
            alpha_selected=0.0,
            B_eff_selected=b,
            V_R_selected=None,
            beta_hat=fit.beta_hat,
            c_hat=fit.c_hat,
            a_hat=fit.a_hat,
            v_hat=fit.v_hat,
            estimated_risk_selected=b,
            safe_pass=bool(x * b < fit.a_hat),
            safe_margin=fit.a_hat - x * b,
            fallback_used=False,
        )
    X_rest = X[: n_eff]
    cal_idx, est_idx = split_indices(n_eff, x, rng)
    Z = synth_fn(int(x), rng)
    b = float(fit.v_hat + fit.c_hat * (x ** (-2.0 * fit.beta_hat)))
    v = float(fit.a_hat / max(n_eff - x, 1))
    alpha = b / (v + b)
    theta_hat = _combine(X_rest, Z, estimand, alpha, est_idx)
    return EstimatorResult(
        theta_hat=theta_hat,
        x_selected=x,
        alpha_selected=alpha,
        B_eff_selected=b,
        V_R_selected=v,
        beta_hat=fit.beta_hat,
        c_hat=fit.c_hat,
        a_hat=fit.a_hat,
        v_hat=fit.v_hat,
        estimated_risk_selected=v * b / (v + b),
        safe_pass=bool(x * b < fit.a_hat),
        safe_margin=fit.a_hat - x * b,
        fallback_used=False,
    )


register(adaptive_parametric_grid)
register(adaptive_parametric_foc)


# Alias: the spec uses both `corrected_adaptive_gn` (§2.8, headline name) and
# `adaptive_parametric_grid` (§5.5 B). We register the same callable under
# the headline name so output rows can be identified consistently.

def corrected_adaptive_gn(*args, **kwargs):
    return adaptive_parametric_grid(*args, **kwargs)


corrected_adaptive_gn.name = "corrected_adaptive_gn"
corrected_adaptive_gn.needs_synth = True
register(corrected_adaptive_gn)
