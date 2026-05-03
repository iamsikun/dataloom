"""Safe adaptive estimator (docs/experiments.md §2.9).

Wraps `corrected_adaptive_gn` with the safety check x̂ · B̂(x̂) < â.
On failure, falls back to `real_only_all` using all n real observations.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..estimators.api import EstimatorResult, register
from .bias_curve import default_pilot_grid, estimate_bias_curve
from .parametric import _resolve_n_v, _run_at_with_plugins
from ..oracle import oracle_grid


def safe_corrected_adaptive_gn(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
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

    # Safety check (using plug-ins).
    b_at_x = float(fit.v_hat + fit.c_hat * (x_hat ** (-2.0 * fit.beta_hat))) \
        if x_hat > 0 else float("inf")
    safe_pass = bool(x_hat > 0 and x_hat * b_at_x < fit.a_hat)
    safe_margin = fit.a_hat - x_hat * b_at_x if x_hat > 0 else fit.a_hat

    if not safe_pass:
        # Fall back to real_only_all using all real data (validation set OK
        # to use here because no synthetic-derived plug-in goes into the estimate).
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
            safe_margin=safe_margin,
            fallback_used=True,
        )

    return _run_at_with_plugins(
        X=X, synth_fn=synth_fn, x=x_hat, n=n, n_v=n_v, n_eff=n_eff,
        rng=rng, fit=fit, estimand=estimand,
    )


safe_corrected_adaptive_gn.name = "safe_corrected_adaptive_gn"
safe_corrected_adaptive_gn.needs_synth = True
register(safe_corrected_adaptive_gn)
