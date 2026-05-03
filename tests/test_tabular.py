"""Phase 5 sanity tests for tabular semi-synthetic pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataloom.datasets.adult import (
    estimand_age_coefficient,
    estimand_income_mean,
    estimand_subgroup_gap,
    load_adult,
)
from dataloom.semisynth.generators import GENERATORS, get_generator


def test_load_adult_returns_dataframe():
    df = load_adult()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100
    assert "income_binary" in df.columns


@pytest.mark.parametrize("name", list(GENERATORS))
def test_generator_fit_sample_roundtrip(name):
    # Use the numeric subset that the tabular runner actually feeds to generators.
    df = load_adult().head(500).select_dtypes(include="number")
    gen = get_generator(name).fit(df)
    rng = np.random.default_rng(0)
    Z = gen.sample(200, rng)
    assert len(Z) == 200
    for c in df.columns:
        assert c in Z.columns


def test_estimand_income_mean_well_defined():
    df = load_adult()
    val = estimand_income_mean(df)
    assert 0.0 <= val <= 1.0


def test_estimand_subgroup_gap_within_unit_interval():
    df = load_adult()
    val = estimand_subgroup_gap(df)
    assert -1.0 <= val <= 1.0


def test_estimand_age_coefficient_finite():
    df = load_adult().head(500)
    coef = estimand_age_coefficient(df)
    assert np.isfinite(coef)


def test_real_only_recovers_population_mean_to_mc_tolerance():
    """real_only_all on an n=2000 subsample should match the population mean
    up to MC SE."""
    df = load_adult()
    rng = np.random.default_rng(42)
    truth = estimand_income_mean(df)
    n = 2000
    estimates = []
    for _ in range(50):
        idx = rng.choice(len(df), size=n, replace=False)
        sub = df.iloc[idx]
        estimates.append(estimand_income_mean(sub))
    se = np.std(estimates, ddof=1)
    # MC SE bound: sqrt(p*(1-p)/n) * sqrt(2/50) ~ slack.
    assert abs(np.mean(estimates) - truth) < 5 * se / np.sqrt(50)
