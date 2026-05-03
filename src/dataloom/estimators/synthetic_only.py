"""Synthetic-only estimator (§2.3 #2).

Note: §2.3 #1 `synthetic_only_oracle_x` is not implemented. Under the §1
risk model the synthetic-only risk v_n + c x^{-2 beta} is monotonically
decreasing in x, so its oracle is x = n which collapses into
`synthetic_only_full_calibration`. See plan file for rationale.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .api import EstimatorResult, register


def synthetic_only_full_calibration(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """Calibrate using all n real observations, then estimate from synthetic only."""
    v_n = truth_params["v_n"]
    c = truth_params["c"]
    beta = truth_params["beta"]
    Z = synth_fn(n, rng)
    b = v_n + c * (n ** (-2.0 * beta))
    return EstimatorResult(
        theta_hat=estimand(Z),
        x_selected=n,
        alpha_selected=0.0,
        B_eff_selected=b,
        V_R_selected=None,
        estimated_risk_selected=b,
        safe_pass=bool(n * b < truth_params["a"]),
        safe_margin=truth_params["a"] - n * b,
        fallback_used=False,
    )


synthetic_only_full_calibration.name = "synthetic_only_full_calibration"
synthetic_only_full_calibration.needs_synth = True

register(synthetic_only_full_calibration)
