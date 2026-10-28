"""Prepare genuine UCI Adult data for Experiments A and R4.

Reuse a validated cache, or download the public training table. Download failures
are fatal: this workflow never substitutes a synthetic population.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataloom.datasets.adult import CACHE_FILE, _download_adult
from dataloom.rebuttal.adult_auxiliary import (
    REQUIRED_REAL_COLUMNS,
    load_required_adult,
)


def prepare_adult(*, check_only: bool = False) -> Path:
    if CACHE_FILE.exists():
        frame, digest = load_required_adult(CACHE_FILE)
    elif check_only:
        raise FileNotFoundError(
            "Adult cache is missing; run: uv run --frozen python scripts/prepare_adult.py"
        )
    else:
        frame = _download_adult()
        missing = REQUIRED_REAL_COLUMNS.difference(frame.columns)
        if missing or len(frame) != 30162:
            raise ValueError(f"Unexpected Adult training table: {len(frame)} rows; missing {sorted(missing)}")
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        frame.attrs["source"] = "uci_adult"
        frame.to_parquet(CACHE_FILE)
        frame, digest = load_required_adult(CACHE_FILE)
    if len(frame) != 30162:
        raise ValueError(f"Expected 30162 cleaned Adult training rows, found {len(frame)}")
    print(f"Validated genuine Adult cache: {len(frame)} rows, SHA-256 {digest}")
    return CACHE_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate the existing cache without downloading.")
    args = parser.parse_args()
    try:
        prepare_adult(check_only=args.check_only)
    except Exception as exc:
        parser.exit(1, f"Adult data preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
