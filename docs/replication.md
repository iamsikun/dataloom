# Camera-ready replication

The [v0.1.0 release](https://github.com/iamsikun/dataloom/releases/tag/v0.1.0)
provides the unchanged camera-ready archive, `dataloom-replication-corrected.zip`,
its external `SHA256SUMS`, and `LICENSE`. The archive is 118,129,076 bytes.
Its SHA-256 is:

```text
97f0576ffed9e72bda4538156f240d8f55a4aa54bb236a2bb77c537cf1b0e11b
```

The authors' code in the archive is covered by the release's MIT license. The
archive itself is preserved byte-for-byte to match the camera-ready supplement;
its internal README predates the repository cleanup. Its 422 supplied files have
an internal checksum manifest. The ZIP and raw outputs are release assets, not
files stored in Git.

## Download and verify

The commands below use a new working directory, separate from a development
checkout. Install Python 3.13 and uv first.

```sh
mkdir dataloom-replication-work
cd dataloom-replication-work
curl -fL -O https://github.com/iamsikun/dataloom/releases/download/v0.1.0/dataloom-replication-corrected.zip
curl -fL -O https://github.com/iamsikun/dataloom/releases/download/v0.1.0/SHA256SUMS
curl -fL -O https://github.com/iamsikun/dataloom/releases/download/v0.1.0/LICENSE
shasum -a 256 -c SHA256SUMS
unzip dataloom-replication-corrected.zip
cd dataloom-replication
shasum -a 256 -c SHA256SUMS
uv sync --frozen --group dev
```

On systems without `shasum`, use `sha256sum -c SHA256SUMS`. Verify internal
checksums before editing or generating files in the extracted package.

## Verify retained R1–R4 outputs

Run these commands **inside the extracted `dataloom-replication/` directory**:

```sh
uv run --frozen python scripts/verify_camera_ready_revision.py --output verification.json
uv run --frozen python scripts/build_camera_ready_assets.py --output rebuilt-assets
```

The verifier recomputes reported quantities from raw outputs, including the paired
R4 bootstrap intervals and deterministic working-risk identities. The builder
checks row counts and headline numbers, then writes figures, LaTeX tables, and an
evidence report. Neither command reruns the simulations or changes retained data.
The builder preserves the numerical content; final manuscript typography and
layout adjustments are not guaranteed to produce byte-identical PDFs.

| Study | Retained raw rows | Contents |
| --- | ---: | --- |
| R1 | 480,000 | Method evaluations for boundary comparisons. |
| R2 | 63 | Deterministic risk-curve evaluations. |
| R3 | 2,520,000 | Method evaluations and grid diagnostics. |
| R4 | 116,000 | External pilot targets and method evaluations. |

These studies used seed root `20260727` and local revisions based on `c54575d`.
The archive includes those revisions; its recorded dirty state is historical
provenance, not a requirement to reconstruct an unknown working tree.

Revision identifiers such as `c54575d` and `70ac9da` refer to pre-release
development commits. They are not part of this repository's public history; the
corresponding source is included in the release archive.

## Rebuild historical figures and tables

```sh
uv run --frozen python scripts/reconcile_camera_ready_figures.py --data-root legacy-reproduction --output rebuilt-legacy
```

This reads the included full-profile reproductions of Experiments 4, A, and B
and verifies their coverage and MSE summaries at the reported precision. It
writes per-cell CSVs, table rows, figures, and a hash/provenance audit. The corrected
camera-ready Figures 6 and 9–13 and Tables 4, 6, and 7 use these common per-cell
results. Figures 7–8 are deterministic allocation-grid diagnostics. Missing
replication counts and the finite-estimate aggregation of the historical
product-surrogate baseline are recorded in the audit.

The reproduced outputs used source commit `70ac9da`, seed root `20260502`, and
four workers. They are **new reproductions**, not recovered original raw files.
Use the included historical source when repeating those computations:

```sh
# Download and validate genuine Adult data using the archive's original helpers.
uv run --frozen python -c 'from dataloom.datasets.adult import load_adult; from dataloom.rebuttal.adult_auxiliary import load_required_adult; load_adult(force_download=True); frame, _ = load_required_adult(); assert len(frame) == 30162'

uv run --frozen python legacy-reproduction/run_reproduction.py exp4 --output fresh-legacy
uv run --frozen python legacy-reproduction/run_reproduction.py expA --output fresh-legacy
uv run --frozen python legacy-reproduction/run_reproduction.py expB --output fresh-legacy
uv run --frozen python scripts/reconcile_camera_ready_figures.py --data-root fresh-legacy --output rebuilt-fresh
```

The last four commands perform full simulations and can take substantially longer
than checking the retained outputs. The source snapshot is pinned inside
`legacy-reproduction/historical-source/`; current development code is not a
substitute for that snapshot when matching historical numbers.

## Availability and limits

Original full-run raw outputs for Experiments 1–3 remain unavailable. Their code,
full configurations, saved notebook outputs, and historical summaries are supplied.
The notebook is evidence of saved analysis, not a fully executable replacement for
missing raw files. New simulations can be run using the [experiment guide](experiments.md),
but this release does not claim that every historical number has been rerun.

The genuine [UCI Adult data](https://doi.org/10.24432/C5XW20) must be obtained
separately for fresh A or R4 simulations. The retained-result checks above need
only the supplied outputs, not the Adult source table. Experiment B is an
IHDP-style synthetic construction rather than the canonical IHDP benchmark.

The compact code-and-summaries companion to the paper omits raw Parquet files.
Use the complete GitHub release ZIP for the verification and figure-building
commands on this page.
