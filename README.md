# Dataloom

Research code for **Optimal Real Data Allocation for Synthetic Data Augmented
Inference**, by **Sikun Xu, Zineng Xu, and Mingduo Zhao** (NeurIPS 2026).

Real observations can estimate a target directly or calibrate a generator that
produces synthetic observations. This project studies how to divide that real data
budget, accounting for synthetic bias, sampling variance, and the opportunity cost
of calibration. It includes allocation oracles, adaptive procedures, inference
diagnostics, and synthetic and semi-synthetic experiments.

**[Camera-ready release and replication archive](https://github.com/iamsikun/dataloom/releases/tag/v0.1.0)**
· [Experiment guide](docs/experiments.md)
· [Exact replication instructions](docs/replication.md)

## Quickstart

Use **Python 3.13** and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The committed lockfile records the dependency versions. The experiments run on CPUs;
no GPU, model credentials, or private data are required.

```sh
git clone https://github.com/iamsikun/dataloom.git
cd dataloom
uv sync --frozen --group dev
uv run --frozen pytest -q

# A small simulation and its analysis; no dataset download is needed.
RUN_DIR=$(uv run --frozen python scripts/run_experiment.py --config configs/expR1_boundary_smoke.yaml)
uv run --frozen python scripts/analyze_run.py "$RUN_DIR"
```

The runner prints the newly created directory under `results/`. Analysis writes
aggregate Parquet tables and figures inside that directory. Smoke runs check the
pipeline; their small replication counts are not intended to reproduce paper values.
See the [experiment guide](docs/experiments.md) for all ten experiment families,
full configurations, seeds, CPU settings, and Adult data preparation.

## Reproduce the camera-ready results

Download the versioned ZIP from the
[release](https://github.com/iamsikun/dataloom/releases/tag/v0.1.0) and follow
[the replication instructions](docs/replication.md). The archive is the unchanged
camera-ready snapshot and includes its own code, configs, checksums, and saved outputs.

| Experiments | What is available |
| --- | --- |
| R1–R4 | Retained full-run raw outputs, aggregate tables, and numerical verification scripts. |
| 4, A, B | Full-profile reproductions using the historical source revision `70ac9da` included in the archive, with per-cell results and figure builders. These are reproduced outputs, not recovered original raw files. |
| 1–3 | Code, configurations, saved notebook outputs, and summaries. Original full-run raw files are unavailable. |

Use the archive's historical source to reproduce the reported legacy numbers.
The current implementation supports new runs, but its full configurations alone
do not establish numerical equivalence to every historical result.

The saved [analysis notebook](notebooks/exp_analysis.ipynb) and
[full-run summary](results/full_run_summary.md) are historical records. The notebook
selects runs by modification time and requires raw inputs that are not all available;
it is not the camera-ready figure-rebuilding workflow.

## Repository layout

| Directory | Contents |
| --- | --- |
| `src/dataloom/` | Allocation, estimation, inference, simulation, and analysis implementations. |
| `configs/` | Paired smoke and full YAML profiles for Experiments 1–4, A, B, and R1–R4. |
| `scripts/` | Experiment runners, analysis, smoke checks, and camera-ready verification/builders. |
| `tests/` | Deterministic and statistical unit tests; unit tests use local fixtures without downloading data. |
| `docs/` | Experiment and replication instructions. |
| `notebooks/` | Historical analysis notebook with saved outputs. |
| `results/` | Historical summary; newly generated runs are ignored by Git. |

## Data and license

Experiments A and R4 use the public
[UCI Adult dataset](https://doi.org/10.24432/C5XW20), credited to Barry Becker and
Ronny Kohavi and distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The Adult source table
is downloaded separately and is not included in this repository or the release.
The official workflow requires the genuine cleaned table, with 30,162 rows.

Experiment B generates an **IHDP-style synthetic population**; it does not load the
canonical IHDP benchmark. Experiments 1–4 and R1–R3 use simulated or analytic data.

The authors' code is licensed under the [MIT License](LICENSE), including their code
in the camera-ready archive. Dataset and dependency licenses remain their own.

## Citation

```bibtex
@inproceedings{xu2026optimal,
  title     = {Optimal Real Data Allocation for Synthetic Data Augmented Inference},
  author    = {Xu, Sikun and Xu, Zineng and Zhao, Mingduo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://github.com/iamsikun/dataloom}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).
