"""Two-channel oracle calculator and KKT diagnostics (§6.2-§6.3).

Risk model:
    B_n(x_1, x_2) = v_n + c_1 x_1^(-2 β_1) + c_2 x_2^(-2 β_2)
    V_R(x_1, x_2) = a / (n - x_1 - x_2)
    R_n(x_1, x_2) = V_R B_n / (V_R + B_n)

KKT (active channels): B_n^2 = 2 β_k a c_k x_k^(-(2 β_k + 1)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def B_eff_2(x1, x2, *, v_n: float,
            c1: float, beta1: float, c2: float, beta2: float):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    out = np.full(np.broadcast_shapes(x1.shape, x2.shape), v_n)
    pos1 = x1 > 0
    pos2 = x2 > 0
    bias1 = np.zeros_like(out)
    bias2 = np.zeros_like(out)
    bias1[pos1] = c1 * np.power(x1[pos1] if x1.shape else x1, -2.0 * beta1)
    bias2[pos2] = c2 * np.power(x2[pos2] if x2.shape else x2, -2.0 * beta2)
    return out + bias1 + bias2


def V_real_2(x1, x2, *, n: int, a: float):
    s = np.asarray(x1, dtype=float) + np.asarray(x2, dtype=float)
    out = np.full_like(s, np.inf)
    feas = s < n
    out[feas] = a / (n - s[feas])
    return out


def R_profile_2(x1, x2, *, n: int, a: float, v_n: float,
                c1: float, beta1: float, c2: float, beta2: float):
    s = np.asarray(x1, dtype=float) + np.asarray(x2, dtype=float)
    if s.shape == ():
        s = np.array([s])
    feas = s < n
    out = np.full(s.shape, np.inf)
    if np.any(feas):
        x1a = np.broadcast_to(x1, s.shape)[feas]
        x2a = np.broadcast_to(x2, s.shape)[feas]
        b = v_n + c1 * np.power(np.maximum(x1a, 1.0), -2.0 * beta1) + \
                  c2 * np.power(np.maximum(x2a, 1.0), -2.0 * beta2)
        # If a channel has x_k = 0, its bias term is +inf
        zero1 = x1a == 0
        zero2 = x2a == 0
        b = np.where(zero1, np.inf, b)
        b = np.where(zero2, np.inf, b)
        v = a / (n - x1a - x2a)
        out[feas] = v * b / (v + b)
    return out


@dataclass
class Oracle2Result:
    x1_star: int
    x2_star: int
    risk_star: float
    alpha_star: float
    B_eff_star: float
    V_R_star: float
    mv1: float
    mv2: float


def marginal_value(xk: float, *, a: float, beta_k: float, c_k: float) -> float:
    """MV_k = 2 β_k a c_k x_k^{-(2 β_k + 1)} from §6.3."""
    if xk <= 0:
        return float("inf")
    return float(2.0 * beta_k * a * c_k * xk ** (-(2.0 * beta_k + 1.0)))


def oracle_grid_2(
    n: int, a: float, v_n: float,
    c1: float, beta1: float, c2: float, beta2: float,
    coarse_step: int | None = None,
) -> Oracle2Result:
    """Grid-minimize R_n(x_1, x_2) over feasible integer pairs.

    For large n the full grid is O(n^2). The default `coarse_step` is set to
    cap the grid at ~200 × 200 cells (i.e. step ≈ max(1, n // 200)).
    """
    step = coarse_step if coarse_step is not None else max(1, n // 200)
    step = max(1, int(step))
    xs = np.arange(1, n, step, dtype=int)
    X1, X2 = np.meshgrid(xs, xs, indexing="ij")
    flat_x1 = X1.ravel()
    flat_x2 = X2.ravel()
    feas = flat_x1 + flat_x2 < n
    flat_x1 = flat_x1[feas]
    flat_x2 = flat_x2[feas]
    risks = R_profile_2(
        flat_x1, flat_x2, n=n, a=a, v_n=v_n,
        c1=c1, beta1=beta1, c2=c2, beta2=beta2,
    )
    idx = int(np.argmin(risks))
    x1_star = int(flat_x1[idx])
    x2_star = int(flat_x2[idx])
    r_star = float(risks[idx])

    b = v_n + c1 * (x1_star ** (-2.0 * beta1)) + c2 * (x2_star ** (-2.0 * beta2))
    v = a / (n - x1_star - x2_star)
    alpha = b / (v + b)
    return Oracle2Result(
        x1_star=x1_star, x2_star=x2_star, risk_star=r_star,
        alpha_star=float(alpha), B_eff_star=float(b), V_R_star=float(v),
        mv1=marginal_value(x1_star, a=a, beta_k=beta1, c_k=c1),
        mv2=marginal_value(x2_star, a=a, beta_k=beta2, c_k=c2),
    )
