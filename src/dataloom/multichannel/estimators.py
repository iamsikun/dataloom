"""Multichannel estimators (§6.4).

These run inside a dedicated multichannel runner (not the §11 master runner)
because they take a 2-tuple `(x1, x2)` as the allocation. Outputs use the
same EstimatorResult dataclass and store (x1, x2) in `extras`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..estimators.api import EstimatorResult, split_indices
from ..oracle import oracle_grid
from .oracle import marginal_value, oracle_grid_2


def _combine_2(
    X: np.ndarray,
    synth_fn_2: Callable[[tuple[int, int], np.random.Generator], np.ndarray],
    *,
    n: int, x1: int, x2: int,
    a: float, v_n: float,
    c1: float, beta1: float, c2: float, beta2: float,
    estimand: Callable[[np.ndarray], float],
    rng: np.random.Generator,
) -> EstimatorResult:
    n_e = max(n - x1 - x2, 1)
    # Partition real into (cal_1, cal_2, est) by random permutation.
    perm = rng.permutation(n)
    cal1 = perm[:x1]
    cal2 = perm[x1:x1 + x2]
    est = perm[x1 + x2:]
    theta_R = estimand(X[est])
    Z = synth_fn_2((x1, x2), rng)
    theta_S = estimand(Z)
    b = v_n + (c1 * (x1 ** (-2 * beta1)) if x1 > 0 else 1e30) \
            + (c2 * (x2 ** (-2 * beta2)) if x2 > 0 else 1e30)
    v = a / n_e
    alpha = b / (v + b)
    theta_hat = alpha * theta_R + (1.0 - alpha) * theta_S
    extras = {
        "x1": x1, "x2": x2,
        "lambda1": x1 / n, "lambda2": x2 / n,
        "mv1": marginal_value(x1, a=a, beta_k=beta1, c_k=c1) if x1 > 0 else float("inf"),
        "mv2": marginal_value(x2, a=a, beta_k=beta2, c_k=c2) if x2 > 0 else float("inf"),
    }
    return EstimatorResult(
        theta_hat=theta_hat,
        x_selected=x1 + x2,
        alpha_selected=alpha,
        B_eff_selected=b,
        V_R_selected=v,
        estimated_risk_selected=v * b / (v + b),
        safe_pass=bool((x1 + x2) * b < a),
        safe_margin=a - (x1 + x2) * b,
        fallback_used=False,
        extras=extras,
    )


def best_single_channel_oracle(
    X, synth_fn_2, *,
    n, rng, params, estimand,
) -> EstimatorResult:
    """Pick the better of two single-channel allocations: (x1*, 0) or (0, x2*)."""
    a, v_n = params["a"], params["v_n"]
    c1, beta1 = params["c1"], params["beta1"]
    c2, beta2 = params["c2"], params["beta2"]

    res1 = oracle_grid(n=n, a=a, v_n=v_n, c=c1, beta=beta1)
    res2 = oracle_grid(n=n, a=a, v_n=v_n, c=c2, beta=beta2)
    if res1.risk_star <= res2.risk_star:
        return _combine_2(
            X, synth_fn_2, n=n, x1=res1.x_star, x2=0,
            a=a, v_n=v_n, c1=c1, beta1=beta1, c2=c2, beta2=beta2,
            estimand=estimand, rng=rng,
        )
    return _combine_2(
        X, synth_fn_2, n=n, x1=0, x2=res2.x_star,
        a=a, v_n=v_n, c1=c1, beta1=beta1, c2=c2, beta2=beta2,
        estimand=estimand, rng=rng,
    )


def equal_split_two_channels(
    X, synth_fn_2, *,
    n, rng, params, estimand,
) -> EstimatorResult:
    x = max(1, n // 4)
    return _combine_2(
        X, synth_fn_2, n=n, x1=x, x2=x,
        a=params["a"], v_n=params["v_n"],
        c1=params["c1"], beta1=params["beta1"],
        c2=params["c2"], beta2=params["beta2"],
        estimand=estimand, rng=rng,
    )


def old_multichannel_fixed_share(
    X, synth_fn_2, *,
    n, rng, params, estimand,
) -> EstimatorResult:
    """λ_k^old = 2 β_k / (1 + 2 Σ_j β_j)."""
    beta1, beta2 = params["beta1"], params["beta2"]
    denom = 1.0 + 2.0 * (beta1 + beta2)
    lam1 = 2.0 * beta1 / denom
    lam2 = 2.0 * beta2 / denom
    x1 = max(1, int(np.floor(n * lam1)))
    x2 = max(1, int(np.floor(n * lam2)))
    if x1 + x2 >= n:
        # Scale down to keep at least 1 estimation obs.
        scale = (n - 1) / (x1 + x2)
        x1 = max(1, int(np.floor(x1 * scale)))
        x2 = max(1, int(np.floor(x2 * scale)))
    return _combine_2(
        X, synth_fn_2, n=n, x1=x1, x2=x2,
        a=params["a"], v_n=params["v_n"],
        c1=params["c1"], beta1=params["beta1"],
        c2=params["c2"], beta2=params["beta2"],
        estimand=estimand, rng=rng,
    )


def corrected_multichannel_oracle(
    X, synth_fn_2, *,
    n, rng, params, estimand,
) -> EstimatorResult:
    """Grid search over (x1, x2) with x1 + x2 < n."""
    res = oracle_grid_2(
        n=n, a=params["a"], v_n=params["v_n"],
        c1=params["c1"], beta1=params["beta1"],
        c2=params["c2"], beta2=params["beta2"],
        coarse_step=max(1, n // 200),
    )
    return _combine_2(
        X, synth_fn_2, n=n, x1=res.x1_star, x2=res.x2_star,
        a=params["a"], v_n=params["v_n"],
        c1=params["c1"], beta1=params["beta1"],
        c2=params["c2"], beta2=params["beta2"],
        estimand=estimand, rng=rng,
    )


METHODS_2: dict[str, Callable] = {
    "best_single_channel_oracle": best_single_channel_oracle,
    "equal_split_two_channels": equal_split_two_channels,
    "old_multichannel_fixed_share": old_multichannel_fixed_share,
    "corrected_multichannel_oracle": corrected_multichannel_oracle,
}
