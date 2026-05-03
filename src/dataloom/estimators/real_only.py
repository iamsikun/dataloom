"""Real-only estimators (§2.1, §2.2)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .api import EstimatorResult, register, split_indices


def real_only_all(
    X: np.ndarray,
    synth_fn=None,
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """All n real observations used for direct estimation. The main baseline."""
    a = truth_params["a"]
    return EstimatorResult(
        theta_hat=estimand(X),
        x_selected=0,
        alpha_selected=1.0,
        V_R_selected=a / n,
        B_eff_selected=None,
        estimated_risk_selected=a / n,
        safe_pass=None,
        fallback_used=False,
    )


real_only_all.name = "real_only_all"
real_only_all.needs_synth = False


def real_only_split(
    X: np.ndarray,
    synth_fn=None,
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    """Use only the estimation subset of size n - x. Diagnostic only.

    `config['x']` controls the calibration size; defaults to floor(n/2).
    """
    cfg = config or {}
    x = int(cfg.get("x", n // 2))
    a = truth_params["a"]
    _, est_idx = split_indices(n, x, rng)
    return EstimatorResult(
        theta_hat=estimand(X[est_idx]),
        x_selected=x,
        alpha_selected=1.0,
        V_R_selected=a / max(n - x, 1),
        B_eff_selected=None,
        estimated_risk_selected=a / max(n - x, 1),
        safe_pass=None,
        fallback_used=False,
    )


real_only_split.name = "real_only_split"
real_only_split.needs_synth = False


register(real_only_all)
register(real_only_split)
