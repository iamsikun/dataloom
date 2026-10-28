"""Long-format Parquet writer for the master schema (§11)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..schema import rows_to_dataframe


def cell_dir_name(cell: dict[str, Any]) -> str:
    """Hive-style partition directory name from cell parameters."""
    parts = [f"{k}={cell[k]}" for k in sorted(cell)]
    return "cell=" + "_".join(parts)


class ResultsWriter:
    """Append rows to results/{run_id}/raw/cell=.../part-{worker_id}.parquet."""

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def write_cell(
        self,
        cell: dict[str, Any],
        rows: list[dict[str, Any]],
        worker_id: int = 0,
    ) -> Path:
        """Write a parquet part-file for the given cell."""
        cell_path = self.raw_dir / cell_dir_name(cell)
        cell_path.mkdir(parents=True, exist_ok=True)
        df = rows_to_dataframe(rows)
        out = cell_path / f"part-{worker_id}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
        return out


def read_run(raw_dir: str | Path) -> pd.DataFrame:
    """Read all parquet parts under raw_dir into a single DataFrame."""
    dataset = pq.ParquetDataset(str(raw_dir))
    return dataset.read().to_pandas()
