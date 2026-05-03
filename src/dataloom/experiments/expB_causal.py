"""Experiment B: causal digital-twin benchmark (§9)."""

from __future__ import annotations

from pathlib import Path

from ..semisynth.causal_runner import run as _run


def run(config_path: str | Path) -> Path:
    return _run(config_path)
