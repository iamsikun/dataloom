"""IHDP-style semi-synthetic causal dataset (Hill, 2011).

The genuine IHDP simulation uses 25 covariates and is widely benchmarked.
Loading the canonical CSVs requires network access; if unavailable we
generate a deterministic IHDP-like pseudo-population on the fly. This stand-in
preserves: continuous + binary covariate mix, treatment imbalance, and a
known true ATE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IHDPSpec:
    n: int = 5000
    p_treated: float = 0.30
    true_ate: float = 4.0
    seed: int = 0


def make_ihdp(spec: IHDPSpec | None = None) -> pd.DataFrame:
    """Generate an IHDP-style population with known potential outcomes.

    Columns:
        x1..x6  — continuous covariates
        z1..z3  — binary covariates
        A       — treatment (Bernoulli, propensity depends on covariates)
        Y0, Y1  — potential outcomes (recorded for evaluation only)
        Y       — observed outcome A * Y1 + (1-A) * Y0
    """
    spec = spec or IHDPSpec()
    rng = np.random.default_rng(spec.seed)
    n = spec.n
    X = rng.normal(size=(n, 6))
    Z = rng.integers(0, 2, size=(n, 3))

    # Propensity: imbalanced, depends on covariates.
    z = -1.0 + 0.5 * X[:, 0] + 0.4 * Z[:, 0] - 0.3 * X[:, 1] + 0.2 * Z[:, 1]
    propensity = 1.0 / (1.0 + np.exp(-z))
    A = (rng.uniform(size=n) < propensity).astype(int)

    # Potential outcomes (Hill response surface B-style):
    Y0 = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * Z[:, 0] + rng.normal(0, 1, n)
    tau_x = spec.true_ate + 0.5 * Z[:, 1] - 0.3 * X[:, 2]
    Y1 = Y0 + tau_x

    Y = A * Y1 + (1 - A) * Y0
    df = pd.DataFrame({
        "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2], "x4": X[:, 3],
        "x5": X[:, 4], "x6": X[:, 5],
        "z1": Z[:, 0], "z2": Z[:, 1], "z3": Z[:, 2],
        "A": A, "Y": Y, "Y0": Y0, "Y1": Y1,
    })
    df.attrs["true_ate"] = float((Y1 - Y0).mean())
    return df


def true_ate(df: pd.DataFrame) -> float:
    return float((df["Y1"] - df["Y0"]).mean())
