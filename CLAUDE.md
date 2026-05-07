# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

`dataloom` is a simulation-studies codebase for the paper *Optimal Real-Data Allocation for Synthetic-Data-Augmented Inference* (working title). The package is library-only — there is no CLI, no application, no service. Code here exists to validate the theoretical claims in `docs/theory.md` via Monte Carlo experiments specified in `docs/model.md`.

`docs/model.md` is the single source of truth for the experimental design (estimator names, DGPs, parameter grids, output schemas, figure/table specs). `docs/theory.md` is the single source of truth for the math. **Read both before adding code** — they fix conventions that other Claude instances and downstream analyses depend on.

## Python tooling

- Always use `uv` (per user global preference). Never invoke bare `python` or `pip`.
- Python 3.13 is required (`.python-version`).
- Common commands:
  - `uv sync` — install/refresh dependencies and the editable `dataloom` package
  - `uv run pytest` — run the test suite (pytest is configured with `pythonpath = ["src"]` so tests can `import dataloom` directly)
  - `uv run pytest tests/path/to/test_x.py::test_name` — run a single test
  - `uv run jupyter lab` — launch JupyterLab for notebook work
  - `uv add <pkg>` / `uv add --dev <pkg>` — add runtime/dev dependencies
- Notebooks under `notebooks/` rely on `notebooks/__dev_setup.py` to insert `src/` onto `sys.path` and enable IPython autoreload. New notebooks should `%run __dev_setup.py` (or import it) at the top.

## Layout

- `src/dataloom/` — the package. Currently nearly empty; experiment code lives here.
- `tests/` — pytest tests (also nearly empty).
- `docs/theory.md`, `docs/model.md` — authoritative spec. Edit only when the design genuinely changes.
- `configs/` — YAML run configs (one per experiment run). Save the config alongside results.
- `results/` — long-format replication output and aggregated tables. Treat as build artifacts; do not hand-edit.
- `scripts/` — entry-point scripts that read a config from `configs/` and write to `results/`.
- `notebooks/` — exploration and figure rendering only; not a place for reusable logic.

## Implementation conventions baked into the spec

These come from `docs/model.md` and apply across all experiment code. Follow them so estimator names, columns, and metrics line up across synthetic and semi-synthetic experiments.

- **Notation matches `docs/model.md` §1**: variable names in code (`n`, `m`, `rho`, `beta`, `a`, `sigma_s2`, `v_n`, `B0`, `c`, `x`/`n_f`, `n_e`, `lambda_f`, `alpha`, `B_eff`, `R_profile`) must match the table. Use those exact names.
- **Estimator names are stable identifiers**: `real_only_all`, `real_only_split`, `synthetic_only_oracle_x`, `synthetic_only_full_calibration`, `naive_pooling`, `fixed_half_split_oracle_alpha`, `fixed_half_split_plugin_alpha`, `old_fixed_share_oracle_alpha`, `old_fixed_share_plugin_alpha`, `corrected_oracle_gn`, `corrected_adaptive_gn`, `safe_corrected_adaptive_gn`, `validation_debiased_gn`, plus the multichannel and inference variants. Reuse these strings everywhere — output rows, plot legends, log lines.
- **Master output schema**: every experiment writes a long-format file with one row per `(replication, method, estimand)` using the columns listed in `docs/model.md` §11. Fill non-applicable columns with `NA` rather than dropping them.
- **Implementation priority order** (`docs/model.md` §14): (1) deterministic oracle calculator with the `B_eff` / `V_real` / `R_profile` / `foc_residual` / `safe_condition` / `oracle_grid` API; (2) Experiment 1 Gaussian Monte Carlo; (3) adaptive estimator; (4) multichannel; (5) tabular; (6) causal/coverage. Do not skip ahead — later experiments reuse the earlier primitives.
- **Sanity checks on Experiment 1** (`docs/model.md` §14, Priority 2): real-only MSE ≈ `a/n`; synthetic-only MSE ≈ `v_n + c x^{-2β}`; corrected-oracle MC MSE matches `R_n(x*)`; old fixed-share over-calibrates when `m ≍ n` and `β > 1/2`.
- **Reproducibility** (`docs/model.md` §17): fix and store all random seeds in output rows; save the config used for every run; save raw long-format results before any aggregation; save figures as both `.pdf` and `.png`. Never hard-code analytic results into simulation outputs — compute them through the same reusable functions.
- **Failure logging, not hiding** (`docs/model.md` §16): generator failures, negative debiased squared-bias estimates, multiple FOC roots, boundary optima, and safe-fallback activations must be recorded in the output rows (`failure_flag`, `failure_reason`, `safe_pass`, `fallback_used`, etc.), not silently dropped.

## Theory pitfalls to avoid

These are claims the paper deliberately rejects; do not reintroduce them in code or comments:

- The optimal calibration share is **not** the fixed-share rule `λ = 2β/(1+2β)`. That estimator exists only as a baseline (`old_fixed_share_*`) to demonstrate over-calibration.
- The proposed method minimizes the corrected profiled risk `R_n(x) = V_R(x) B_n(x) / (V_R(x) + B_n(x))` over a grid that includes the boundaries `x=0` and `x=n`, then applies the safety check `x · B_n(x) < a`. It does not pick a fixed fraction.
- MSE-optimal point estimation and valid confidence intervals are separate problems — naive Wald intervals around the MSE-optimal estimator may undercover by design (see Experiment 4).

## Paper writing

When asked to draft or edit the paper itself (not the code), use the Pepper agent (per user global preference). The theory and methods draft is `docs/theory.md`; the experimental spec is `docs/model.md`.

<!-- pepper:start -->
# Academic Paper Writing System

Pepper is a project-local academic paper writing framework for machine learning,
economics, marketing, operations research, and quant finance.

The canonical interface is the `pepper` CLI. Runtime adapters such as Claude Code
and Codex consume the same workflow and role definitions rendered into their
preferred local files.

## Canonical Workflows

- `pepper new-paper`
- `pepper import-paper`
- `pepper literature-search`
- `pepper draft-paper`
- `pepper draft-section`
- `pepper edit-section`
- `pepper review-paper`
- `pepper revise-paper`
- `pepper set-target`
- `pepper create-journal-version`
- `pepper assemble`
- `pepper polish`
- `pepper camera-ready`

## Context Resolution

All workflows and roles resolve context through:

1. `paper/state.yaml` for the active target and stage
2. `paper/shared/context.md` for title, topic, contributions, and source map
3. `paper/<active_target>/target.yaml` for venue metadata
4. Repo-local source paths from the source map

Deterministic repo operations belong in the CLI. Role-driven work is reserved for
literature synthesis, outlining, drafting, review, and revision planning.


## Runtime Adapter

This file is the Claude Code adapter. The canonical workflow contract still lives in the
Pepper core spec and the `pepper` CLI.

Use repo-local slash commands as convenience wrappers around the canonical CLI workflows.
When a workflow requires deterministic repository changes, prefer the CLI over ad hoc manual edits.

## Key References

- `.pepper/config.yaml`
- `.pepper/shared-agent-protocols.md`
- `.pepper/writing-style.md`
- `paper/state.yaml`
- `paper/shared/context.md`
- `paper/shared/session-log.md`
- `paper/<active_target>/target.yaml`
<!-- pepper:end -->
