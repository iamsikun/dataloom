"""Pilot bias-curve estimation (docs/experiments.md §5.4 / §8.6).

For pilot calibration sizes x_j ∈ G_pilot, the squared-bias estimator is

    b̂²(x_j) = max(
        (θ̂_S(x_j) − θ̂_R^V)² − Var̂(θ̂_S(x_j)) − Var̂(θ̂_R^V),
        0
    ).

The power-law fit log b̂²(x_j) = log c − 2β log x_j + ε_j gives plug-in
(β̂, ĉ). A monotone PAV smoother is also returned for the nonparametric
variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import isotonic_regression


@dataclass
class BiasCurveFit:
    pilot_x: np.ndarray            # ints
    bias2_hat: np.ndarray          # raw squared-bias estimates (positive part)
    var_S_hat: np.ndarray          # variance of synthetic estimator at each x_j
    var_R_V_hat: float             # variance of validation real estimator
    bias2_hat_smooth: np.ndarray   # monotone (decreasing) smoothed b̂²(x_j)
    beta_hat: float                # power-law slope (-2β recovered as -slope/2)
    c_hat: float                   # power-law intercept exp
    powerlaw_r2: float
    monotonicity_violations: int
    a_hat: float
    sigma_s2_hat: float
    v_hat: float                   # plug-in v_n = sigma_s2_hat / m
    n_positive: int = 0
    pilot_repeats: int = 1
    fit_reliable: bool = True
    fit_status: str = "ok"


def default_pilot_grid(n: int) -> np.ndarray:
    """Pilot grid §5.3: {5, 10, 20, 40, 80, 160, 320, 640} ∩ [1, n/2].

    Falls back to {n/20, n/10, n/5, n/3} for very small n.
    """
    base = np.array([5, 10, 20, 40, 80, 160, 320, 640], dtype=int)
    grid = base[(base >= 1) & (base <= n // 2)]
    if len(grid) >= 3:
        return np.unique(grid)
    fallback = np.array([n // 20, n // 10, n // 5, n // 3], dtype=int)
    fallback = fallback[(fallback >= 1) & (fallback <= n // 2)]
    return np.unique(fallback)


def estimate_bias_curve(
    *,
    n: int,
    n_v: int,
    X: np.ndarray,
    synth_fn: Callable[[int, np.random.Generator], np.ndarray],
    rng: np.random.Generator,
    estimand: Callable[[np.ndarray], float],
    pilot_grid: np.ndarray | None = None,
    m: int,
    pilot_repeats: int | None = None,
    min_positive: int = 3,
    reference: str = "validation",
) -> BiasCurveFit:
    """Run the pilot protocol and fit (β̂, ĉ).

    Convention: the validation set is the last n_v observations of X (after
    the caller has already shuffled or otherwise made order arbitrary).
    """
    if pilot_grid is None:
        pilot_grid = default_pilot_grid(n)

    a_hat = float(np.var(X, ddof=1))   # use all real for the variance plug-in
    if reference == "validation":
        if n_v <= 0:
            raise ValueError("n_v must be positive when reference='validation'")
        val = X[n - n_v :]
        theta_R_V = estimand(val)
        var_R_V = a_hat / n_v
    elif reference == "all":
        theta_R_V = estimand(X)
        var_R_V = a_hat / n
    else:
        raise ValueError(f"unknown bias-curve reference={reference!r}")

    bias2 = np.empty(len(pilot_grid))
    var_S = np.empty(len(pilot_grid))
    sigma_s2_estimates = []

    _fn_fast_mean = getattr(synth_fn, "fast_mean", False)
    _fn_m = getattr(synth_fn, "m", None)
    _fn_sigma_s2 = getattr(synth_fn, "sigma_s2", None)
    if pilot_repeats is None:
        # Repeating fast-mean pilots is cheap and reduces the main failure mode:
        # pilot bias estimates collapsing to zero due to synthetic/validation noise.
        pilot_repeats = 8 if _fn_fast_mean else 1
    pilot_repeats = max(1, int(pilot_repeats))

    for i, x_j in enumerate(pilot_grid):
        theta_s_repeats = np.empty(pilot_repeats, dtype=float)
        var_s_repeats = np.empty(pilot_repeats, dtype=float)
        s2_repeats = np.empty(pilot_repeats, dtype=float)
        for k in range(pilot_repeats):
            Z = synth_fn(int(x_j), rng)
            theta_s_repeats[k] = estimand(Z)
            if len(Z) >= 2:
                s2 = float(np.var(Z, ddof=1))
                var_S_hat = s2 / len(Z)
            elif _fn_fast_mean and _fn_m is not None and _fn_sigma_s2 is not None:
                # Z is a length-1 draw representing the mean of _fn_m full observations.
                # Its variance is sigma_s2/_fn_m, not sigma_s2.  Recover s2=sigma_s2
                # for the sigma_s2_hat estimator and set var_S_hat correctly.
                s2 = _fn_sigma_s2
                var_S_hat = _fn_sigma_s2 / _fn_m
            else:
                # m=1 (persistent_variance regime) or no metadata available.
                s2 = a_hat
                var_S_hat = s2 / max(len(Z), 1)
            s2_repeats[k] = s2
            var_s_repeats[k] = var_S_hat
        theta_S = float(theta_s_repeats.mean())
        # The pilot estimate averages independent synthetic estimates.
        var_S_hat = float(var_s_repeats.mean() / pilot_repeats)
        s2 = float(np.median(s2_repeats))
        sigma_s2_estimates.append(s2)
        var_S[i] = var_S_hat
        diff2 = float((theta_S - theta_R_V) ** 2)
        bias2[i] = max(diff2 - var_S_hat - var_R_V, 0.0)

    sigma_s2_hat = float(np.median(sigma_s2_estimates))
    v_hat = sigma_s2_hat / m

    # Monotone (non-increasing) smoother on bias2 vs. pilot_x.
    # isotonic_regression fits non-decreasing; flip sign for non-increasing.
    iso = isotonic_regression(-bias2, increasing=True)
    bias2_smooth = -iso.x

    # Count locations where raw bias2 violates monotonicity.
    monotonicity_violations = int(np.sum(np.diff(bias2) > 0))

    # Power-law fit on positive bias2 only. Too few positive points is not
    # evidence of zero bias; it means the pilot is underpowered. Fail closed.
    pos = bias2 > 0
    n_positive = int(pos.sum())
    fit_reliable = True
    fit_status = "ok"
    if n_positive >= min_positive:
        log_x = np.log(pilot_grid[pos].astype(float))
        log_b2 = np.log(bias2[pos])
        slope, intercept = np.polyfit(log_x, log_b2, 1)
        beta_hat = float(-slope / 2.0)
        c_hat = float(np.exp(intercept))
        # Coefficient of determination
        pred = slope * log_x + intercept
        ss_res = float(np.sum((log_b2 - pred) ** 2))
        ss_tot = float(np.sum((log_b2 - log_b2.mean()) ** 2))
        powerlaw_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        if not np.isfinite(beta_hat) or not np.isfinite(c_hat):
            fit_reliable = False
            fit_status = "nonfinite_powerlaw_fit"
        elif beta_hat <= 0.0 or c_hat <= 0.0:
            fit_reliable = False
            fit_status = "nondecreasing_or_nonpositive_powerlaw_fit"
    else:
        # Conservative fallback. With beta=0 the estimated bias curve is flat,
        # so grid minimization selects real-only unless the data strongly support
        # synthetic use through a reliable fit.
        beta_hat = 0.0
        c_hat = max(a_hat, float(bias2.max()), 1e-12)
        powerlaw_r2 = float("nan")
        fit_reliable = False
        fit_status = f"too_few_positive_bias_points:{n_positive}"

    return BiasCurveFit(
        pilot_x=pilot_grid.astype(int),
        bias2_hat=bias2,
        var_S_hat=var_S,
        var_R_V_hat=var_R_V,
        bias2_hat_smooth=bias2_smooth,
        beta_hat=max(beta_hat, 0.0),
        c_hat=max(c_hat, 1e-15),
        powerlaw_r2=powerlaw_r2,
        monotonicity_violations=monotonicity_violations,
        a_hat=a_hat,
        sigma_s2_hat=sigma_s2_hat,
        v_hat=v_hat,
        n_positive=n_positive,
        pilot_repeats=pilot_repeats,
        fit_reliable=fit_reliable,
        fit_status=fit_status,
    )


def B_eff_from_powerlaw(
    x: np.ndarray | int, fit: BiasCurveFit
) -> np.ndarray | float:
    """B(x) = v̂ + ĉ x^(-2 β̂)."""
    x = np.asarray(x, dtype=float)
    out = fit.v_hat + fit.c_hat * np.power(x, -2.0 * fit.beta_hat)
    return float(out) if out.shape == () else out


def B_eff_from_smoother(
    x: np.ndarray | int, fit: BiasCurveFit
) -> np.ndarray | float:
    """Interpolated B(x) from the monotone smoother of pilot b̂²(x_j).

    Linear interpolation on log-x for in-range x; constant extrapolation
    outside the pilot grid.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    log_x_pilot = np.log(fit.pilot_x.astype(float))
    log_b2_smooth = np.log(np.maximum(fit.bias2_hat_smooth, 1e-30))
    log_x_q = np.log(np.maximum(x_arr, 1.0))
    log_b2 = np.interp(log_x_q, log_x_pilot, log_b2_smooth)
    out = fit.v_hat + np.exp(log_b2)
    if out.shape == () or out.shape == (1,) and np.isscalar(x):
        return float(out[0])
    return out
