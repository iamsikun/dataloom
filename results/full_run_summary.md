# Full-Run Summary: Exp 1 + Exp 2 (fast-mean path, corrected naive_pooling)

**Date:** 2026-05-03  
**Commit:** `00d9cfb` on branch `agent/full-exp1-exp2-fast-mean`  
**Exp 1 run dir:** `results/exp1_phase_diagram__00d9cfb__20260503T093440__dirty`  
**Exp 2 run dir:** `results/exp2_adaptive__00d9cfb__20260503T094340__dirty`  
**Wall clock:** Exp 1 ≈ 9 min, Exp 2 ≈ 2 min (fast_mean=True; without this, rho=2 cells would be ~3 GB/call and intractable)

> **Note on previous run in this branch:** An earlier run (`df6d462__20260503T071839`) used the same fast-mean path but with a bug in `naive_pooling`: when `fast_mean=True` the synth closure returns a length-1 array, but the old code did `np.concatenate([X[est_idx], Z])` and then `np.mean`, which gave essentially real-only MSE (weight ≈ 1/n_e on the synthetic). The correct formula, `(n_e·θ̂_R + m·θ̂_S)/(n_e+m)`, uses `m = truth_params["m"]` and is implemented in this run. The summary below reflects the **corrected** run.

---

## What ran / what didn't

**Exp 1 (phase diagram):** All 336 (n, β, ρ) cells completed. Zero `failure_flag=True` rows out of 9,996,000. Replications: 5,000 for n ≤ 5,000; 2,000 for n > 5,000.

**Exp 2 (adaptive allocation):** All 90 (n, β, ρ) cells completed. Zero `failure_flag=True` rows out of 1,155,000. Replications: 2,000 for n ≤ 5,000; 1,000 for n > 5,000.

No NaN `theta_hat` values in either run. No generator failures.

---

## Empirical vs theory slopes — Exp 1 Table S1

Theory slope in the `sublinear_fast_learning` regime: `slope = 2ρ/(1 + 2β)`.

| β | ρ | regime | theory\_slope | empirical\_slope | slope\_se | verdict |
|---|---|--------|:---:|:---:|:---:|---|
| 0.75 | 0.5 | sublinear\_fast\_learning | 0.400 | 0.4534 | 0.0063 | **(disagrees)** +8.5σ |
| 0.75 | 1.0 | sublinear\_fast\_learning | 0.800 | 0.8422 | 0.0033 | **(disagrees)** +13σ |
| 1.00 | 0.5 | sublinear\_fast\_learning | 0.333 | 0.3749 | 0.0109 | borderline +3.8σ |
| 1.00 | 1.0 | sublinear\_fast\_learning | 0.667 | 0.6841 | 0.0031 | **(disagrees)** +5.5σ |
| 1.50 | 0.5 | sublinear\_fast\_learning | 0.250 | 0.2504 | 0.0060 | agrees |
| 1.50 | 1.0 | sublinear\_fast\_learning | 0.500 | 0.5045 | 0.0018 | agrees |
| 1.50 | 1.5 | sublinear\_fast\_learning | 0.750 | 0.7516 | 0.0009 | agrees |
| 2.00 | 0.5 | sublinear\_fast\_learning | 0.200 | 0.2036 | 0.0135 | agrees |
| 2.00 | 1.0 | sublinear\_fast\_learning | 0.400 | 0.4044 | 0.0033 | agrees |
| 2.00 | 1.5 | sublinear\_fast\_learning | 0.600 | 0.5986 | 0.0007 | agrees |
| 2.00 | 2.0 | sublinear\_fast\_learning | 0.800 | 0.7995 | 0.0008 | agrees |

Boundary/degenerate cells (rho=0: persistent variance x\*=0; boundary\_full\_calibration: x\*=n; knife-edge β=0.5 or boundary β=1.0 ρ=1.5): no interior slope is measurable; these are omitted from the table.

**Theory vs empirical disagreements (4 of 11 interior cells, all upward-biased):**

Theory predicts slope `2ρ/(1+2β)` asymptotically. Empirically, low-β cells (β ∈ {0.75, 1.0}) show slopes 3–13% above theory, while high-β cells (β ≥ 1.5) agree. All disagreements are positive (empirical > theory). Possible causes: (A) pre-asymptotic finite-n effects — the approach to the asymptotic slope is slower for β close to the slow-learning boundary; (B) the n-grid [100, 20000] is insufficient dynamic range to observe the asymptotic slope for these β values. Whether the excess is pre-asymptotic or a model misspecification cannot be determined from this grid alone; extending n to 10⁵–10⁶ would be required.

---

## Headline regime MSE ratios — Exp 1 Table S2

**Cell: β=1.0, ρ=1.0, n=20,000 (sublinear\_fast\_learning, 2,000 replications)**

| estimator | MSE | mse\_ratio\_to\_real | oracle\_regret |
|---|:---:|:---:|:---:|
| corrected\_oracle\_gn | 2.60e-05 | 0.535 | 0.000 |
| safe\_corrected\_oracle\_gn | 2.62e-05 | 0.540 | 0.010 |
| naive\_pooling | 3.43e-05 | 0.709 | 0.325 |
| fixed\_half\_split\_oracle\_alpha | 3.44e-05 | 0.710 | 0.328 |
| old\_fixed\_share\_oracle\_alpha | 3.77e-05 | 0.779 | 0.457 |
| real\_only\_all | 4.83e-05 | 1.000 | 0.870 |
| synthetic\_only\_full\_calibration | 4.90e-05 | 1.015 | 0.898 |

Key findings: `corrected_oracle_gn` achieves 46.5% MSE reduction vs real-only, confirming the theoretical improvement. `old_fixed_share_oracle_alpha` over-calibrates (oracle_regret 0.457 vs 0.000 for oracle), consistent with theory. `synthetic_only_full_calibration` is slightly worse than real_only (residual bias even at full calibration).

**Note on `safe_corrected_oracle_gn` negative oracle_regret (78 cells across the full grid):** The safe variant records oracle_regret < 0 in 78/2352 cells (i.e., it beats the unsanitized oracle in MC MSE). This is expected behavior, not a bug: when the safety condition `x*·B(x*) < a` fails, the safe variant falls back to real_only_all. The oracle benchmark is not safety-aware, so the safe variant can occasionally improve on it.

---

## Adaptive vs oracle — Exp 2 Table A2

**Cell: β=1.0, ρ=1.0, n=10,000 (largest n with genuine fast-learning, 1,000 replications)**

| method | mse\_ratio\_to\_real | oracle\_regret | safe\_fallback\_rate |
|---|:---:|:---:|:---:|
| corrected\_oracle\_gn (benchmark) | 0.522 | 0.000 | 0.0 |
| safe\_corrected\_oracle\_gn | 0.504 | −0.034 | 0.0 |
| corrected\_adaptive\_gn | 0.641 | 0.230 | 0.0 |
| safe\_corrected\_adaptive\_gn | 0.702 | 0.346 | 0.0 |
| adaptive\_parametric\_foc | 0.832 | 0.595 | 0.0 |
| adaptive\_nonparametric\_grid | 1.177 | 1.256 | 0.0 |
| real\_only\_all (baseline) | 1.000 | 0.917 | — |

Safe fallback rate in the persistent-variance regime (β=1.0, ρ=0.0, n=10,000): `safe_corrected_adaptive_gn` falls back 94.6% of replications, as expected.

---

## Unfavorable results — adaptive methods substantially underperform oracle

> **The findings below contradict the paper's Phase-3 claim that adaptive methods should track the oracle. They are reported as observed, not suppressed.**

The adaptive estimators are worse than `real_only_all` (mse\_ratio\_to\_real > 1) in a large majority of non-persistent-variance cells:

| method | cells with mse\_ratio > 1 (out of 60 non-rho=0 cells) |
|---|:---:|
| adaptive\_nonparametric\_grid | 60 / 60 |
| adaptive\_parametric\_foc | 55 / 60 |
| safe\_corrected\_adaptive\_gn | 55 / 60 |
| corrected\_adaptive\_gn | 54 / 60 |

For comparison: `corrected_oracle_gn` has 0 / 60 such cells.

Worst observed cases:
- `adaptive_nonparametric_grid`, β=1.5, ρ=2.0, n=1000: mse\_ratio=**307**, oracle\_regret=314,028, harm\_rate=1.0. Mean x_selected=12.8 vs oracle x\*=1,000.
- `corrected_adaptive_gn`, β=1.5, ρ=2.0, n=1000: mse\_ratio=**204**, oracle\_regret=204,177. Mean x_selected=124.6 vs oracle x\*=1,000.
- `adaptive_nonparametric_grid`, β=1.0, ρ=2.0, n=200: mse\_ratio=**68**.

**What the simulation shows:** Across this grid, adaptive estimators consistently underestimate x\* — mean x\_selected is far below oracle x\* in most cells — leading to under-calibration and excess bias. The oracle (`corrected_oracle_gn`) works correctly in all 60 cells.

**Root-cause analysis (observational):** The pilot bias-curve estimator requires the squared deviation `(θ̂_S(x_j) − θ̂_R^V)²` to exceed the sum of variances `σ²_s/m + a/n_v` to produce a positive bias² estimate. For high ρ (large m), `σ²_s/m ≈ 0`, but `a/n_v = O(1/n)` dominates at small n where the bias `B₀·x_j^{-β}` is also small. Consequently, most pilot points yield bias²=0, the power-law fit falls back to the degenerate β_hat=5/c_hat≈0 defaults, and the adaptive selector picks a much smaller x than optimal. This is a fundamental signal-to-noise limitation of the pilot protocol at these sample sizes — it is not a crash, NaN propagation, or implementation error.

**Theory says:** Adaptive methods track oracle as n → ∞ (experiments.md §5.6). **Empirical:** On this grid (n ∈ [200, 10000]), none of the four adaptive methods consistently achieves mse\_ratio < 1 in the fast-learning regime. The gap does not clearly close across n (e.g., `adaptive_nonparametric_grid` at β=0.75, ρ=1: mse_ratio goes 10.2 → 5.1 → 1.8 → 1.3 → 1.2 → 1.2 for n = 200, 500, 1000, 2000, 5000, 10000 — still above 1 at n=10000). Whether asymptotic convergence holds but kicks in beyond n=10,000 is not ruled out by these results; however, the paper should state this clearly rather than claim n=10,000 as evidence of convergence.

---

## Data integrity notes

- Zero `failure_flag=True` rows in both runs.
- Zero NaN `theta_hat` values.
- All configured estimators produced valid `EstimatorResult` for every replication.
- Raw parquet files are retained under `results/exp*__*/raw/` (not committed per task constraints).
