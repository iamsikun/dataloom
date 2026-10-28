"""Abstract tabular generator interface.

Implementations must support fit(df) and sample(m, rng) → DataFrame.
The sampled DataFrame must have the same columns as the calibration data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class TabularGenerator(ABC):
    name: str = "abstract"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "TabularGenerator":
        ...

    @abstractmethod
    def sample(self, m: int, rng: np.random.Generator) -> pd.DataFrame:
        ...
