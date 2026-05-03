"""Adaptive estimators (docs/experiments.md §5).

Importing this package registers:
    corrected_adaptive_gn, safe_corrected_adaptive_gn,
    adaptive_parametric_foc, adaptive_parametric_grid,
    adaptive_nonparametric_grid.
"""

from . import bias_curve, parametric, nonparametric, safety  # noqa: F401
