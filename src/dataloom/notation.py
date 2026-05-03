"""Canonical names, constants, and regime classifier.

All variable names match docs/experiments.md §1. The regime classifier
implements §4.3.
"""

from __future__ import annotations

from typing import Literal

Regime = Literal[
    "persistent_variance",
    "slow_learning",
    "parametric_knife_edge",
    "sublinear_fast_learning",
    "boundary_full_calibration",
    "boundary_knife_edge",
]


def classify_regime(beta: float, rho: float) -> Regime:
    """Classify a (beta, rho) pair into one of the §4.3 regimes."""
    if rho == 0:
        return "persistent_variance"
    if beta < 0.5:
        return "slow_learning"
    if beta == 0.5:
        return "parametric_knife_edge"
    threshold = beta + 0.5
    if rho == threshold:
        return "boundary_knife_edge"
    if rho < threshold:
        return "sublinear_fast_learning"
    return "boundary_full_calibration"


def theory_slope(beta: float, rho: float) -> float | None:
    """Theoretical slope of log x* vs log n (§3.3).

    Returns None on knife-edge cases where the slope is not universal.
    """
    if rho == 0 or beta < 0.5:
        return 0.0
    if beta == 0.5:
        return None
    threshold = beta + 0.5
    if rho == threshold:
        return None
    if rho < threshold:
        return 2.0 * rho / (2.0 * beta + 1.0)
    return 1.0


ESTIMATOR_NAMES: tuple[str, ...] = (
    "real_only_all",
    "real_only_split",
    "synthetic_only_oracle_x",
    "synthetic_only_full_calibration",
    "naive_pooling",
    "fixed_half_split_oracle_alpha",
    "fixed_half_split_plugin_alpha",
    "old_fixed_share_oracle_alpha",
    "old_fixed_share_plugin_alpha",
    "corrected_oracle_gn",
    "safe_corrected_oracle_gn",
    "corrected_adaptive_gn",
    "safe_corrected_adaptive_gn",
    "validation_debiased_gn",
    "adaptive_parametric_foc",
    "adaptive_parametric_grid",
    "adaptive_nonparametric_grid",
    "best_single_channel_oracle",
    "equal_split_two_channels",
    "old_multichannel_fixed_share",
    "corrected_multichannel_oracle",
    "corrected_multichannel_adaptive",
    "ci_real_only",
    "ci_gn_naive",
    "ci_gn_bias_aware",
    "ci_gn_undersmoothed",
    "ci_validation_debiased",
)
