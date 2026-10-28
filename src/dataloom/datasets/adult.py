"""UCI Adult / Census Income dataset loader.

Falls back to a synthetic stand-in pseudo-population if the dataset is not
already cached and the network isn't available. The synthetic stand-in
preserves the rough column shape so downstream code is exercised either way.

Cache path: ~/.cache/dataloom/adult.parquet
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(os.path.expanduser("~/.cache/dataloom"))
CACHE_FILE = CACHE_DIR / "adult.parquet"

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income",
]


def _download_adult() -> pd.DataFrame:
    df = pd.read_csv(
        ADULT_URL, header=None, names=COLUMN_NAMES,
        na_values="?", skipinitialspace=True,
    )
    df = df.dropna().reset_index(drop=True)
    df["income_binary"] = (df["income"].str.strip() == ">50K").astype(int)
    df["sex_binary"] = (df["sex"].str.strip() == "Male").astype(int)
    return df


def _synthetic_stand_in(n: int = 30000, seed: int = 0) -> pd.DataFrame:
    """Synthetic 'Adult-like' population. Used when the network is unavailable."""
    rng = np.random.default_rng(seed)
    age = rng.integers(17, 90, size=n)
    education_num = rng.integers(1, 17, size=n)
    hours_per_week = np.clip(rng.normal(40, 12, n), 1, 99).astype(int)
    sex_binary = rng.integers(0, 2, size=n)
    # Income probability: increasing in age, education, hours; higher for sex=1.
    z = (
        -3.5
        + 0.03 * (age - 38)
        + 0.30 * (education_num - 10)
        + 0.02 * (hours_per_week - 40)
        + 0.40 * sex_binary
    )
    p = 1.0 / (1.0 + np.exp(-z))
    income_binary = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({
        "age": age,
        "education_num": education_num,
        "hours_per_week": hours_per_week,
        "sex_binary": sex_binary,
        "income_binary": income_binary,
    })


def load_adult(use_cache: bool = True, force_download: bool = False) -> pd.DataFrame:
    """Return the Adult pseudo-population. Cached on disk."""
    if use_cache and CACHE_FILE.exists() and not force_download:
        return pd.read_parquet(CACHE_FILE)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = _download_adult()
    except Exception as e:  # noqa: BLE001
        # Network failure: use a deterministic synthetic stand-in.
        df = _synthetic_stand_in()
        df.attrs["source"] = f"synthetic_stand_in (download failed: {e})"
    else:
        df.attrs["source"] = "uci_adult"

    df.to_parquet(CACHE_FILE)
    return df


# ---------------------------------------------------------------------------
# Estimands
# ---------------------------------------------------------------------------


def estimand_income_mean(df: pd.DataFrame) -> float:
    """Mean of income_binary (§8.3 estimand 1)."""
    return float(df["income_binary"].mean())


def estimand_subgroup_gap(df: pd.DataFrame) -> float:
    """E[Y | sex=1] − E[Y | sex=0] (§8.3 estimand 2)."""
    g1 = df[df["sex_binary"] == 1]["income_binary"]
    g0 = df[df["sex_binary"] == 0]["income_binary"]
    if len(g1) == 0 or len(g0) == 0:
        return float("nan")
    return float(g1.mean() - g0.mean())


def estimand_age_coefficient(df: pd.DataFrame) -> float:
    """Linear regression coefficient on age in y ~ age + education_num + hours_per_week."""
    if len(df) < 4:
        return float("nan")
    cols = ["age", "education_num", "hours_per_week"]
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(dtype=float) for c in cols])
    y = df["income_binary"].to_numpy(dtype=float)
    # OLS via lstsq.
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])  # age coefficient


ESTIMANDS = {
    "income_mean": estimand_income_mean,
    "subgroup_gap": estimand_subgroup_gap,
    "age_coefficient": estimand_age_coefficient,
}
