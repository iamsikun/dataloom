"""Reproducible seed scheme using numpy.random.SeedSequence.

Each (experiment_id, cell_key, replication) maps deterministically to a
single uint64 seed; the materialized seed is stored in the row so a single
output line can be reproduced from the config alone.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _hash_str(s: str) -> int:
    """Stable 32-bit hash for spawn keys (Python's hash() is salt-randomized)."""
    h = 1469598103934665603  # FNV-1a 64-bit offset basis
    prime = 1099511628211
    for ch in s.encode("utf-8"):
        h = ((h ^ ch) * prime) & 0xFFFFFFFFFFFFFFFF
    return h & 0xFFFFFFFF


def replication_seed(
    seed_root: int,
    experiment_id: str,
    cell_key: Sequence[int],
    rep: int,
) -> int:
    """Deterministic per-replication seed.

    Storing (seed_root, experiment_id, cell_key, rep) plus the returned int
    in each output row guarantees per-row reproducibility.
    """
    spawn = (_hash_str(experiment_id), *map(int, cell_key), int(rep))
    ss = np.random.SeedSequence(entropy=int(seed_root), spawn_key=spawn)
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))
