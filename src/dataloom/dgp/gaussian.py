"""Gaussian DGP for Experiments 1, 2, and 4 (docs/experiments.md §4.2).

Real observations:
    X_i = theta_star + eps_i,         eps_i ~ N(0, a)

Synthetic observations after calibration size x:
    Z_j(x) = theta_star + B0 * x^(-beta) + u_j,    u_j ~ N(0, sigma_s2)

The synthetic sample size is m_n = ceil(kappa * n^rho).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..oracle import m_of_n


@dataclass(frozen=True)
class GaussianParams:
    n: int
    beta: float
    rho: float
    a: float = 1.0
    B0: float = 1.0
    sigma_s2: float = 1.0
    kappa: float = 1.0
    theta_star: float = 0.0

    @property
    def m(self) -> int:
        return m_of_n(self.n, kappa=self.kappa, rho=self.rho)

    @property
    def v_n(self) -> float:
        return self.sigma_s2 / self.m

    @property
    def c(self) -> float:
        return self.B0 ** 2


def sample_real(params: GaussianParams, rng: np.random.Generator) -> np.ndarray:
    """Draw n iid Gaussian real observations."""
    return params.theta_star + rng.normal(
        loc=0.0, scale=np.sqrt(params.a), size=params.n
    )


def make_synth_fn(params: GaussianParams):
    """Closure (x: int, rng) -> ndarray of m synthetic Gaussian observations.

    The synthetic mean is shifted by B0 * x^(-beta); when x = 0 we treat the
    bias as +infinite (synthetic data uninformative) and return NaNs so any
    estimator that uses it will fail loudly. Estimators that don't need
    synthetic data must declare needs_synth=False so the runner skips this.
    """
    def synth_fn(x: int, rng: np.random.Generator) -> np.ndarray:
        if x <= 0:
            return np.full(params.m, np.nan)
        bias = params.B0 * (x ** (-params.beta))
        return (
            params.theta_star
            + bias
            + rng.normal(loc=0.0, scale=np.sqrt(params.sigma_s2), size=params.m)
        )

    return synth_fn


def estimand_mean(sample: np.ndarray) -> float:
    """Default scalar estimand: sample mean."""
    return float(np.mean(sample))
