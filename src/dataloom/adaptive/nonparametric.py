"""Nonparametric adaptive estimator (docs/experiments.md §5.5 C).

Estimate B̂_n(x) directly with a monotone smoother on the pilot bias
estimates, then minimize the empirical profiled risk over a fine grid.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..estimators.api import EstimatorResult, register
from .bias_curve import (
    B_eff_from_smoother,
    default_pilot_grid,
    estimate_bias_curve,
)
from .parametric import (
    _bias_reference,
    _candidate_grid,
    _fit_allows_synthetic,
    _min_positive,
    _pilot_repeats,
    _real_only_from_fit,
    _resolve_n_v,
    _run_at_with_plugins,
)


def adaptive_nonparametric_grid(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """Estimate the bias curve nonparametrically (PAV smoother) and minimize
    R̂_n(x) = â B̂(x) / (â + B̂(x) (n_eff − x)) over the integer grid."""
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
    x_all = _candidate_grid(n_eff, fit, config)
    x_grid = x_all[(x_all > 0) & (x_all < n_eff)]
    b_grid = B_eff_from_smoother(x_grid, fit)
    v_grid = fit.a_hat / np.maximum(n_eff - x_grid, 1)
    risk = v_grid * b_grid / (v_grid + b_grid)

    # Include x=0 (real-only) and x=n_eff (synth-only) boundary.
    boundary_real = fit.a_hat / n
    boundary_synth = float(B_eff_from_smoother(n_eff, fit))
    risks_all = np.concatenate([[boundary_real], risk, [boundary_synth]])
    x_all = np.concatenate([[0], x_grid, [n_eff]])
    idx = int(np.argmin(risks_all))
    x_hat = int(x_all[idx])

    res = _run_at_with_plugins(
        X=X, synth_fn=synth_fn, x=x_hat, n=n, n_v=n_v, n_eff=n_eff,
        rng=rng, fit=fit, estimand=estimand,
    )
    # Override estimated risk with the smoother-based value at x_hat.
    res.estimated_risk_selected = float(risks_all[idx])
    return res


adaptive_nonparametric_grid.name = "adaptive_nonparametric_grid"
adaptive_nonparametric_grid.needs_synth = True
register(adaptive_nonparametric_grid)
