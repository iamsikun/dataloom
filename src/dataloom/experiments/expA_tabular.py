"""Experiment A: tabular semi-synthetic benchmark (§8)."""

from __future__ import annotations

from pathlib import Path

from ..semisynth.tabular_runner import run as _run


def run(config_path: str | Path) -> Path:
    return _run(config_path)
