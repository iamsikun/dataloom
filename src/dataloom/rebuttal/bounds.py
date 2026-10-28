"""One-sided uncertainty bounds for conservative allocation safety checks."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, t


def gaussian_variance_lower(
    sample_variance: float, sample_size: int, alpha: float
) -> float:
    """Exact one-sided lower confidence bound for a Gaussian variance."""
    if sample_size < 2 or not np.isfinite(sample_variance) or sample_variance < 0:
        return 0.0
    df = sample_size - 1
    return float(df * sample_variance / chi2.ppf(1.0 - alpha, df))


def gaussian_variance_upper(
    sample_variance: float, degrees_freedom: int, alpha: float
) -> float:
    """Exact one-sided upper confidence bound for a Gaussian variance."""
    if (
        degrees_freedom < 1
        or not np.isfinite(sample_variance)
        or sample_variance < 0
    ):
        return float("inf")
    quantile = chi2.ppf(alpha, degrees_freedom)
    if not np.isfinite(quantile) or quantile <= 0:
        return float("inf")
    return float(degrees_freedom * sample_variance / quantile)


def mean_upper(values: np.ndarray, alpha: float) -> float:
    """Student-t one-sided upper bound for a mean; fail closed for n<2."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("inf")
    se = float(np.std(values, ddof=1) / np.sqrt(values.size))
    critical = float(t.ppf(1.0 - alpha, values.size - 1))
    return float(np.mean(values) + critical * se)


def buffered_safe_condition(x: int, U_B: float, L_a: float) -> bool:
    """Return whether the conservative sufficient condition can be certified."""
    return bool(
        x > 0
        and np.isfinite(U_B)
        and np.isfinite(L_a)
        and U_B >= 0
        and L_a > 0
        and x * U_B < L_a
    )
