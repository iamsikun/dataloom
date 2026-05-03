"""Phase 6 (causal) sanity tests."""

from __future__ import annotations

import numpy as np
import pytest

from dataloom.datasets.ihdp import IHDPSpec, make_ihdp, true_ate
from dataloom.semisynth.causal_runner import _ate_aipw, _ate_diff_in_means


def test_make_ihdp_has_required_columns():
    df = make_ihdp(IHDPSpec(n=500, seed=0))
    for c in ["x1", "x2", "z1", "A", "Y", "Y0", "Y1"]:
        assert c in df.columns


def test_true_ate_close_to_spec():
    df = make_ihdp(IHDPSpec(n=20000, true_ate=4.0, seed=0))
    assert abs(true_ate(df) - 4.0) < 0.5


def test_aipw_unbiased_on_synthetic_dgp():
    """AIPW with a correctly-specified linear outcome model should recover
    the true ATE on average (within MC noise)."""
    truths = []
    estimates = []
    for s in range(10):
        df = make_ihdp(IHDPSpec(n=2000, true_ate=4.0, seed=s))
        truths.append(true_ate(df))
        estimates.append(_ate_aipw(df))
    assert abs(np.mean(estimates) - np.mean(truths)) < 0.4


def test_diff_in_means_biased_under_confounding():
    """In the IHDP-like DGP, propensity depends on covariates -> diff-in-means
    is generally biased; AIPW should be closer to truth on average."""
    truths = []
    aipw = []
    dim = []
    for s in range(8):
        df = make_ihdp(IHDPSpec(n=2000, true_ate=4.0, seed=s))
        truths.append(true_ate(df))
        aipw.append(_ate_aipw(df))
        dim.append(_ate_diff_in_means(df))
    assert abs(np.mean(aipw) - np.mean(truths)) <= abs(np.mean(dim) - np.mean(truths)) + 0.1
