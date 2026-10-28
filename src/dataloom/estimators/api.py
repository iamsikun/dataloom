"""Uniform API for all estimators in §2 of docs/experiments.md.

Every estimator is a callable with signature

    (X, synth_fn, *, n, rng, truth_params, estimand, config) -> EstimatorResult

and exposes `name` (the canonical method string) and `needs_synth`
(whether the runner must materialize synthetic data for it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


@dataclass
class EstimatorResult:
    """Result of one estimator run. Maps onto the §11 master schema columns.

    Fields default to None so estimators only set what's applicable.
    """

    theta_hat: float
    x_selected: int | None = None
    alpha_selected: float | None = None
    B_eff_selected: float | None = None
    V_R_selected: float | None = None
    beta_hat: float | None = None
    c_hat: float | None = None
    a_hat: float | None = None
    v_hat: float | None = None
    estimated_risk_selected: float | None = None
    safe_pass: bool | None = None
    safe_margin: float | None = None
    fallback_used: bool | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    failure_flag: bool = False
    failure_reason: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class Estimator(Protocol):
    name: str
    needs_synth: bool

    def __call__(
        self,
        X: np.ndarray,
        synth_fn: Callable[[int, np.random.Generator], np.ndarray] | None,
        *,
        n: int,
        rng: np.random.Generator,
        truth_params: dict[str, Any],
        estimand: Callable[[np.ndarray], float],
        config: dict[str, Any] | None = None,
    ) -> EstimatorResult: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


REGISTRY: dict[str, Estimator] = {}


def register(estimator: Estimator) -> Estimator:
    """Decorator: add an estimator to the global registry under its `name`."""
    name = estimator.name
    if name in REGISTRY:
        raise ValueError(f"estimator already registered: {name}")
    REGISTRY[name] = estimator
    return estimator


def get(name: str) -> Estimator:
    if name not in REGISTRY:
        raise KeyError(
            f"unknown estimator '{name}'. Registered: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_indices(
    n: int, x: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random partition of {0,...,n-1} into a calibration set of size x
    and an estimation set of size n - x. Both arrays are returned sorted."""
    if x < 0 or x > n:
        raise ValueError(f"invalid split x={x} for n={n}")
    perm = rng.permutation(n)
    cal = np.sort(perm[:x])
    est = np.sort(perm[x:])
    return cal, est
