"""Bootstrap-smoothed generator (debugging baseline, §8.5 #1 alternative).

Resamples rows with replacement and adds Gaussian noise to continuous
columns. Categorical / binary columns are passed through unchanged. This
is intentionally a weak generator so the calibration / risk machinery is
exercised end-to-end without depending on SDV.

The bias of estimators based on this generator vs. the population mean
should still decrease in the calibration size x via the bandwidth schedule
that scales with x^{-2β} (we hard-code β=0.5 for diagnostic clarity, but
the adaptive estimator does not need to know this).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import TabularGenerator


CONTINUOUS_HINT = ("age", "education_num", "hours_per_week",
                   "fnlwgt", "capital_gain", "capital_loss")


class BootstrapSmoothed(TabularGenerator):
    name: str = "bootstrap_smoothed"

    def __init__(self, smoothing_scale: float = 0.5):
        self.smoothing_scale = smoothing_scale
        self._df: pd.DataFrame | None = None
        self._continuous: list[str] = []
        self._continuous_std: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "BootstrapSmoothed":
        self._df = df.reset_index(drop=True)
        self._continuous = [c for c in df.columns if c in CONTINUOUS_HINT]
        self._continuous_std = {
            c: float(df[c].std(ddof=1)) for c in self._continuous
        }
        return self

    def sample(self, m: int, rng: np.random.Generator) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("BootstrapSmoothed.fit must be called before sample")
        n_calib = len(self._df)
        idx = rng.integers(0, n_calib, size=m)
        out = self._df.iloc[idx].copy().reset_index(drop=True)
        # The bandwidth shrinks with calibration size to mimic learning:
        # h ~ scale * n_calib^{-1/2}.  Synthetic mean bias on continuous
        # columns then decreases as n_calib^{-1/2} (β = 0.5).
        h = self.smoothing_scale * n_calib ** (-0.5)
        for c in self._continuous:
            sd = self._continuous_std.get(c, 1.0)
            noise = rng.normal(0.0, h * sd, size=m)
            out[c] = out[c].astype(float) + noise
        return out
