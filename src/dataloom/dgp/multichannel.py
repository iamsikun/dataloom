"""Two-channel Gaussian DGP for Experiment 3 (§6.2).

Real X_i ~ N(theta_star, a).
Synthetic Z_j(x_1, x_2) = theta_star + B0_1 x_1^(-beta_1) + B0_2 x_2^(-beta_2)
                          + u_j,  u_j ~ N(0, sigma_s2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..oracle import m_of_n


@dataclass(frozen=True)
class GaussianTwoChannelParams:
    n: int
    beta1: float
    beta2: float
    rho: float = 1.0
    a: float = 1.0
    B0_1: float = 1.0
    B0_2: float = 1.0
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
    def c1(self) -> float:
        return self.B0_1 ** 2

    @property
    def c2(self) -> float:
        return self.B0_2 ** 2


def sample_real(params: GaussianTwoChannelParams,
                rng: np.random.Generator) -> np.ndarray:
    return params.theta_star + rng.normal(
        loc=0.0, scale=np.sqrt(params.a), size=params.n,
    )


def make_synth_fn_2(params: GaussianTwoChannelParams):
    """Closure ((x1, x2), rng) -> ndarray of m synthetic obs."""
    def synth_fn(x_pair: tuple[int, int],
                 rng: np.random.Generator) -> np.ndarray:
        x1, x2 = int(x_pair[0]), int(x_pair[1])
        bias = 0.0
        if x1 > 0:
            bias += params.B0_1 * (x1 ** (-params.beta1))
        if x2 > 0:
            bias += params.B0_2 * (x2 ** (-params.beta2))
        return (
            params.theta_star + bias
            + rng.normal(loc=0.0, scale=np.sqrt(params.sigma_s2), size=params.m)
        )
    return synth_fn
