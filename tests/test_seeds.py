"""Seed reproducibility tests."""

from __future__ import annotations

import numpy as np

from dataloom.io.seeds import make_rng, replication_seed


def test_replication_seed_deterministic():
    s1 = replication_seed(20260502, "exp1_phase_diagram", (1000, 1000, 1000), 7)
    s2 = replication_seed(20260502, "exp1_phase_diagram", (1000, 1000, 1000), 7)
    assert s1 == s2


def test_replication_seed_distinct_across_reps():
    seeds = {
        replication_seed(20260502, "exp1_phase_diagram", (1000, 1000, 1000), r)
        for r in range(100)
    }
    # No collisions in 100 reps.
    assert len(seeds) == 100


def test_replication_seed_distinct_across_cells():
    s1 = replication_seed(20260502, "exp1_phase_diagram", (1000, 1000, 1000), 0)
    s2 = replication_seed(20260502, "exp1_phase_diagram", (2000, 1000, 1000), 0)
    assert s1 != s2


def test_make_rng_reproducible():
    s = replication_seed(20260502, "exp1_phase_diagram", (1000, 1000, 1000), 0)
    a = make_rng(s).normal(size=10)
    b = make_rng(s).normal(size=10)
    np.testing.assert_array_equal(a, b)
