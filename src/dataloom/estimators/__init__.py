"""Estimator registry. Importing this package populates REGISTRY by
side-effect via the @register decorator on each estimator module."""

from __future__ import annotations

from .api import (
    REGISTRY,
    Estimator,
    EstimatorResult,
    get,
    register,
    split_indices,
)

# Importing the modules below populates REGISTRY for the in-package estimators.
# Subpackages (adaptive, inference) register themselves when their __init__ is
# loaded — that import lives in the top-level dataloom/__init__.py to avoid
# the circular import that occurred when both subpackages and these in-package
# modules referenced one another at import time.
from . import (  # noqa: E402, F401
    real_only,
    synthetic_only,
    pooling,
    fixed_share,
    corrected_oracle,
)

__all__ = [
    "Estimator",
    "EstimatorResult",
    "REGISTRY",
    "get",
    "register",
    "split_indices",
]
