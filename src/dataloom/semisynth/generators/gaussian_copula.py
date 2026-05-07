"""Gaussian-copula generator with a pure-numpy implementation.

For each column:
- Estimate the empirical CDF (with linear interpolation between order stats).
- Map data → standard normal via Φ⁻¹(F_emp).
- Estimate the Gaussian correlation matrix.
- Sample from the multivariate normal, map back via F_emp⁻¹(Φ).

This avoids the SDV dependency for the first end-to-end pass. SDV's
GaussianCopula can be plugged in later by extending GENERATORS in
__init__.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import TabularGenerator


class GaussianCopulaGenerator(TabularGenerator):
    name: str = "gaussian_copula"

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge
        self._cols: list[str] = []
        self._sorted: dict[str, np.ndarray] = {}
        self._L: np.ndarray | None = None  # Cholesky of corr

    def _regularized_corr(self, Z: np.ndarray) -> np.ndarray:
        p = Z.shape[1]
        if Z.shape[0] < 2:
            return np.eye(p)
        corr = np.corrcoef(Z, rowvar=False)
        corr = np.asarray(corr, dtype=float)
        if corr.shape == ():
            return np.eye(p)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, self.ridge)
        corr = (eigvecs * eigvals) @ eigvecs.T
        scale = np.sqrt(np.maximum(np.diag(corr), self.ridge))
        corr = corr / np.outer(scale, scale)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0 + self.ridge)
        return corr

    def fit(self, df: pd.DataFrame) -> "GaussianCopulaGenerator":
        # Auto-select numeric columns; bootstrap the rest.
        self._numeric_cols = list(df.select_dtypes(include="number").columns)
        self._other_cols = [c for c in df.columns if c not in self._numeric_cols]
        self._cols = list(df.columns)
        self._df_full = df.reset_index(drop=True)
        self._sorted = {}
        n = len(df)
        Z = np.empty((n, len(self._numeric_cols)))
        for j, c in enumerate(self._numeric_cols):
            x = df[c].to_numpy(dtype=float)
            order = np.argsort(x)
            sorted_x = x[order]
            self._sorted[c] = sorted_x
            ranks = np.empty(n)
            ranks[order] = (np.arange(n) + 0.5) / n
            Z[:, j] = norm.ppf(ranks)
        if len(self._numeric_cols) >= 2:
            corr = self._regularized_corr(Z)
            self._L = np.linalg.cholesky(corr)
        elif len(self._numeric_cols) == 1:
            self._L = np.array([[1.0]])
        else:
            self._L = np.zeros((0, 0))
        return self

    def sample(self, m: int, rng: np.random.Generator) -> pd.DataFrame:
        if self._L is None:
            raise RuntimeError("GaussianCopulaGenerator.fit must be called before sample")
        out: dict[str, np.ndarray] = {}
        if len(self._numeric_cols) > 0:
            eps = rng.normal(size=(m, len(self._numeric_cols)))
            Z = eps @ self._L.T
            u = norm.cdf(Z)
            for j, c in enumerate(self._numeric_cols):
                sorted_x = self._sorted[c]
                n = len(sorted_x)
                idx_f = u[:, j] * n - 0.5
                idx_low = np.clip(np.floor(idx_f).astype(int), 0, n - 1)
                idx_high = np.clip(idx_low + 1, 0, n - 1)
                frac = np.clip(idx_f - idx_low, 0.0, 1.0)
                out[c] = sorted_x[idx_low] * (1 - frac) + sorted_x[idx_high] * frac
        if self._other_cols:
            n = len(self._df_full)
            idx = rng.integers(0, n, size=m)
            for c in self._other_cols:
                out[c] = self._df_full[c].to_numpy()[idx]
        return pd.DataFrame(out, columns=self._cols)
