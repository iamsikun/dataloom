"""Naive pooling estimator (§2.4).

Convention (locked in by the plan): for non-mean estimands, this stacks
real-estimation rows and synthetic rows and re-fits the estimand. For the
default sample-mean estimand, this is exactly the §2.4 weighted-average
formula:
    theta_pool = (n_e * theta_R + m * theta_S) / (n_e + m).

The calibration size x is determined by `config['x_strategy']`:
- 'half_split' (default): x = floor(n / 2)
- 'fixed':                x = config['x_value']
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .api import EstimatorResult, register, split_indices


def naive_pooling(
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    *,
    n: int,
    rng: np.random.Generator,
    truth_params: dict[str, Any],
    estimand: Callable[[np.ndarray], float],
    config: dict[str, Any] | None = None,
) -> EstimatorResult:
    cfg = config or {}
    strategy = cfg.get("x_strategy", "half_split")
    if strategy == "half_split":
        x = n // 2
    elif strategy == "fixed":
        x = int(cfg["x_value"])
    else:
        raise ValueError(f"unknown naive_pooling x_strategy={strategy!r}")

    _, est_idx = split_indices(n, x, rng)
    Z = synth_fn(x, rng)
    pooled = np.concatenate([X[est_idx], Z])
    theta_hat = estimand(pooled)

    a = truth_params["a"]
    v_n = truth_params["v_n"]
    c = truth_params["c"]
    beta = truth_params["beta"]
    b = v_n + c * (x ** (-2.0 * beta)) if x > 0 else np.inf
    n_e = n - x
    m = len(Z)
    # alpha implied by sample-size weighting (informational only)
    alpha_implied = n_e / (n_e + m) if (n_e + m) > 0 else None
    return EstimatorResult(
        theta_hat=theta_hat,
        x_selected=x,
        alpha_selected=alpha_implied,
        B_eff_selected=b,
        V_R_selected=a / max(n_e, 1),
        fallback_used=False,
    )


naive_pooling.name = "naive_pooling"
naive_pooling.needs_synth = True

register(naive_pooling)
