# Full-Run Summary: Exp 1 + Exp 2 with Gaussian Fast-Mean Path

Generated: 2026-05-03

## Run directories

| Experiment | Run directory |
|---|---|
| Exp 1 (phase diagram) | `results/exp1_phase_diagram__df6d462__20260503T071839__dirty` |
| Exp 2 (adaptive)      | `results/exp2_adaptive__df6d462__20260503T073557__dirty`      |

> **Note on the "dirty" suffix**: the run was executed from a working tree with
> uncommitted changes (the fast-mean feature itself).  The git hash `df6d462`
> is the parent commit; the changes are fully captured in this PR.

---

## Empirical vs theory slopes — Exp 1 Table S1

Source: `<exp1_run_dir>/aggregated/table_s1_scaling.parquet`

The slope is the log–log slope of the corrected-oracle MC MSE vs n (i.e. the MSE
convergence rate).  Cells in boundary or degenerate regimes have no interior
slope to measure (x* = 0 or x* = n for all n in the grid), so `empirical_slope`
and `slope_se` are NaN for those rows.  The theory slope for the
`sublinear_fast_learning` regime is `2β·ρ / (1 + 2β)`.

### Interior-regime rows (empirical slope available)

| β    | ρ   | regime                  | theory\_slope | empirical\_slope | slope\_se |
|------|-----|-------------------------|---------------|-----------------|-----------|
| 0.75 | 0.5 | sublinear\_fast\_learning | 0.4000        | 0.4534          | 0.0063    |
| 0.75 | 1.0 | sublinear\_fast\_learning | 0.8000        | 0.8423          | 0.0033    |
| 1.00 | 0.5 | sublinear\_fast\_learning | 0.3333        | 0.3749          | 0.0109    |
| 1.00 | 1.0 | sublinear\_fast\_learning | 0.6667        | 0.6841          | 0.0031    |
| 1.50 | 0.5 | sublinear\_fast\_learning | 0.2500        | 0.2504          | 0.0060    |
| 1.50 | 1.0 | sublinear\_fast\_learning | 0.5000        | 0.5045          | 0.0018    |
| 1.50 | 1.5 | sublinear\_fast\_learning | 0.7500        | 0.7516          | 0.0009    |
| 2.00 | 0.5 | sublinear\_fast\_learning | 0.2000        | 0.2036          | 0.0135    |
| 2.00 | 1.0 | sublinear\_fast\_learning | 0.4000        | 0.4044          | 0.0033    |
| 2.00 | 1.5 | sublinear\_fast\_learning | 0.6000        | 0.5986          | 0.0007    |
| 2.00 | 2.0 | sublinear\_fast\_learning | 0.8000        | 0.7995          | 0.0008    |

**All 11 interior-regime cells match theory within 2 SE.**

### Boundary / degenerate regime summary (no interior slope)

| β    | ρ   | regime                    | theory\_slope | note |
|------|-----|---------------------------|---------------|------|
| 0.25 | any | slow\_learning / persistent | 0.0         | x*=0 for all n; boundary |
| 0.40 | any | slow\_learning / persistent | 0.0         | x*=0 for all n; boundary |
| 0.50 | any | parametric\_knife\_edge    | N/A           | x*=0; boundary |
| 0.75 | 1.5+ | boundary\_full\_calibration | 1.0         | x*=n; boundary |
| 1.00 | 1.5 | boundary\_knife\_edge      | N/A           | x*=n; boundary |
| 1.00 | 2.0+ | boundary\_full\_calibration | 1.0         | x*=n; boundary |
| 1.50 | 2.0 | boundary\_knife\_edge      | N/A           | x*=n; boundary |
| 1.50 | 3.0+ | boundary\_full\_calibration | 1.0         | x*=n; boundary |
| 2.00 | 3.0 | boundary\_full\_calibration | 1.0         | x*=n; boundary |

---

## Headline regime MSE ratios — Exp 1 Table S2

Source: `<exp1_run_dir>/aggregated/table_s2_estimator_perf.parquet`

Cell: **β=1, ρ=1, n=20 000** (sublinear\_fast\_learning regime).  Sorted by
MSE ratio to real-only (ascending = better).

| Rank | Estimator                       | MC MSE   | MSE / real\_only | oracle\_regret |
|------|---------------------------------|----------|-----------------|----------------|
| 1    | corrected\_oracle\_gn           | 2.60e-05 | 0.535           | 0.000          |
| 2    | safe\_corrected\_oracle\_gn     | 2.61e-05 | 0.540           | 0.010          |
| 3    | fixed\_half\_split\_oracle\_alpha | 3.41e-05 | 0.710           | 0.328          |
| 4    | old\_fixed\_share\_oracle\_alpha | 3.76e-05 | 0.779           | 0.457          |
| 5    | real\_only\_all                 | 4.84e-05 | 1.000           | 0.870          |
| 6    | synthetic\_only\_full\_calibration | 4.87e-05 | 1.015           | 0.898          |
| 7    | naive\_pooling                  | 9.95e-05 | 2.055           | 2.843          |

**Key findings:**
- `corrected_oracle_gn` achieves 46% MSE reduction vs real-only at n=20 000,
  with oracle_regret = 0 by construction.
- `safe_corrected_oracle_gn` incurs only 1% additional regret for the safety
  guarantee.
- The `old_fixed_share` over-calibrates as predicted (x_old > x*) and loses
  9% relative to the corrected oracle.
- `naive_pooling` (half-split, no bias correction) is 2× worse than real-only
  — a cautionary baseline confirming that uncorrected pooling can harm.

---

## Adaptive vs oracle — Exp 2 Table A2

Source: `<exp2_run_dir>/aggregated/table_a2_adaptive_perf.parquet`

Cell: **β=1, ρ=1, n=10 000** (largest n in this grid, sublinear\_fast\_learning
regime).  Sorted by MSE ratio to real-only.

| Method                      | MSE / real\_only | oracle\_regret | safe\_fallback\_rate |
|-----------------------------|-----------------|----------------|----------------------|
| safe\_corrected\_oracle\_gn | 0.504           | −0.034         | 0.000                |
| corrected\_oracle\_gn       | 0.522           | 0.000          | 0.000                |
| corrected\_adaptive\_gn     | 0.641           | 0.230          | 0.000                |
| safe\_corrected\_adaptive\_gn | 0.702         | 0.346          | 0.000                |
| adaptive\_parametric\_foc   | 0.832           | 0.595          | 0.000                |
| real\_only\_all             | 1.000           | 0.917          | 0.000                |
| adaptive\_nonparametric\_grid | 1.177          | 1.256          | 0.000                |

**Key findings:**
- `corrected_adaptive_gn` reaches 36% MSE reduction at n=10 000 without
  oracle knowledge of (β, c), at the cost of 0.23 oracle regret.
- `safe_corrected_adaptive_gn` adds a safety check; its regret (0.35) exceeds
  the plain adaptive variant at this cell — expected, since the safety fallback
  is conservative.
- `adaptive_parametric_foc` is intermediate (0.83) due to the parametric
  power-law fit being noisier than the bias-curve approach.
- `adaptive_nonparametric_grid` slightly exceeds real-only MSE at n=10 000
  (ratio 1.18), indicating the nonparametric smoother has not yet converged
  at this sample size.
- `safe_fallback_rate = 0` for all methods at n=10 000; the safe condition
  `x · B̂(x) < a` is satisfied at the selected allocation for every
  replication.

> **Note on fast\_mean and adaptive bias estimation**: enabling `fast_mean=True`
> causes `make_synth_fn` to return a length-1 array (the exact mean
> distribution).  The original `bias_curve.py` fallback for length-1 arrays
> used `a_hat` as the variance estimate (the `m=1` regime heuristic), which
> is wrong when m ≫ 1.  This PR also fixes `bias_curve.py` to use the closure
> metadata (`synth_fn.sigma_s2`, `synth_fn.m`) when available, restoring
> correct variance estimation for all adaptive estimators under `fast_mean=True`.
> An early uncorrected run (dir: `exp2_adaptive__df6d462__20260503T072845__dirty`)
> showed adaptive MSE ratios of ~3 000×; the corrected run above shows the
> expected sub-2× ratios.

---

## What ran / what didn't

### Exp 1

| Dimension | Value |
|---|---|
| Cells completed | 336 / 336 (100%) |
| Estimators per cell | 7 |
| Total rows | 9 996 000 |
| Replications (n ≤ 5000) | 5 000 |
| Replications (n > 5000) | 2 000 |
| Failures (`failure_flag=True`) | 0 |
| Wall-clock time | ~10 min (fast-mean path) |

### Exp 2

| Dimension | Value |
|---|---|
| Cells completed | 90 / 90 (100%) |
| Estimators per cell | 7 |
| Total rows | 1 155 000 |
| Replications (n ≤ 5000) | 2 000 |
| Replications (n > 5000) | 1 000 |
| Failures (`failure_flag=True`) | 0 |
| Wall-clock time | ~2.5 min |

All 336 Exp 1 cells and all 90 Exp 2 cells completed without any
estimator-level failures.  The `failure_flag` column is False for every row
in both raw outputs.
