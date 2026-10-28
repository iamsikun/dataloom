# Cleaned Full-Profile Result Summary

> Historical run record. Original raw outputs for Experiments 1–3 are unavailable.
> The camera-ready release retains R1–R4 outputs and includes full-profile
> reproductions of Experiments 4, A, and B using pinned historical source.
> See [replication instructions and provenance](../docs/replication.md).

**Date:** 2026-05-03  
**Branch:** `main`  
**Current code:** `0c52efa` with dirty working-tree estimator, inference, and generator fixes  
**Notebook:** `notebooks/exp_analysis.ipynb` re-executed after full-profile reruns

## Retained Runs

| experiment | run directory | profile | raw rows | failures |
|---|---|---:|---:|---:|
| Exp 1 phase diagram | `results/exp1_phase_diagram__0c52efa__20260503T192214__dirty` | full | 9,996,000 | 0 |
| Exp 2 adaptive | `results/exp2_adaptive__0c52efa__20260503T190454__dirty` | full | 1,155,000 | 0 |
| Exp 3 multichannel | `results/exp3_multichannel__0c52efa__20260503T200030__dirty` | full | 48,000 | 0 |
| Exp 4 inference | `results/exp4_inference__0c52efa__20260503T200321__dirty` | full | 280,000 | 0 |
| Exp A tabular | `results/expA_tabular__0c52efa__20260503T200908__dirty` | full | 72,000 | 0 |
| Exp B causal | `results/expB_causal__0c52efa__20260503T202616__dirty` | full | 14,400 | 0 |

Removed during cleanup: old smoke-profile Exp 3/4/A/B directories, stale adaptive reruns, and the interrupted Exp A attempt that exposed the Gaussian-copula robustness issue.

## Fixes Found During Full Rerun

- Exp 4 interval estimators still used the old validation-reference bias-curve call with `n_v=0`. The inference adaptive selector now uses the same corrected bias-curve defaults as the adaptive point estimators, and `ci_validation_debiased` explicitly reserves validation data.
- Exp A full uses `gaussian_copula`; tiny or rank-deficient calibration samples could make the empirical correlation matrix non-positive-definite. The generator now eigenvalue-clips and regularizes the correlation matrix.

## Main Takeaway

The full-profile results are consistent with the paper for the corrected oracle and multichannel oracle stories. The adaptive estimator is now conservative and stable, but the finite-sample results still do not support a strong claim that adaptive tracks oracle performance. In tabular and causal full profiles, the corrected adaptive methods mostly fall back or mildly underperform the real-only baseline.

## Exp 1: Phase Diagram

Headline cell: `beta=1.0`, `rho=1.0`, `n=20000`.

| estimator | MSE ratio to real | oracle regret |
|---|---:|---:|
| `corrected_oracle_gn` | 0.535 | 0.000 |
| `safe_corrected_oracle_gn` | 0.540 | 0.010 |
| `naive_pooling` | 0.709 | 0.325 |
| `fixed_half_split_oracle_alpha` | 0.710 | 0.328 |
| `old_fixed_share_oracle_alpha` | 0.779 | 0.457 |
| `real_only_all` | 1.000 | 0.870 |
| `synthetic_only_full_calibration` | 1.015 | 0.898 |

The corrected oracle cuts MSE by roughly 46.5% versus real-only in the headline fast-learning cell. Allocation-scaling has 4 disagreements among interior fast-learning cells, all upward and concentrated near the slow-learning boundary: `(0.75, 0.5)`, `(0.75, 1.0)`, `(1.0, 0.5)`, `(1.0, 1.0)`.

## Exp 2: Adaptive Estimator

Non-`rho=0` harm counts after the adaptive fix:

| method | cells with MSE ratio > 1 | median MSE ratio | max MSE ratio |
|---|---:|---:|---:|
| `corrected_adaptive_gn` | 21 / 60 | 0.986 | 1.460 |
| `safe_corrected_adaptive_gn` | 22 / 60 | 0.983 | 1.453 |
| `adaptive_parametric_foc` | 29 / 60 | 1.000 | 1.432 |
| `adaptive_nonparametric_grid` | 44 / 60 | 1.152 | 1.834 |

Headline adaptive cell: `beta=1.0`, `rho=1.0`, `n=10000`.

| method | MSE ratio to real | oracle regret | mean x selected | oracle x |
|---|---:|---:|---:|---:|
| `safe_corrected_oracle_gn` | 0.504 | -0.034 | 573 | 573 |
| `corrected_oracle_gn` | 0.522 | 0.000 | 573 | 573 |
| `corrected_adaptive_gn` | 0.867 | 0.663 | 563 | 573 |
| `adaptive_parametric_foc` | 0.874 | 0.675 | 560 | 573 |
| `safe_corrected_adaptive_gn` | 0.908 | 0.741 | 577 | 573 |
| `real_only_all` | 1.000 | 0.917 | 0 | 573 |
| `adaptive_nonparametric_grid` | 1.136 | 1.177 | 137 | 573 |

Adaptive is much better than the pre-fix catastrophic run, but it is not oracle-equivalent on the retained grid.

## Exp 3: Multichannel

Full-profile median MSE ratio to corrected multichannel oracle:

| method | median ratio | max ratio |
|---|---:|---:|
| `corrected_multichannel_oracle` | 1.000 | 1.000 |
| `best_single_channel_oracle` | 1.147 | 1.666 |
| `equal_split_two_channels` | 1.493 | 1.745 |
| `old_multichannel_fixed_share` | 2.240 | 3.217 |

No full-profile cell beats the corrected multichannel oracle in Monte Carlo MSE.

## Exp 4: Inference

Full-profile coverage ranges by interval method:

| interval | min | median | max |
|---|---:|---:|---:|
| `ci_gn_naive` | 0.855 | 0.916 | 0.955 |
| `ci_gn_bias_aware` | 0.896 | 0.930 | 0.957 |
| `ci_gn_undersmoothed` | 0.913 | 0.936 | 0.955 |
| `ci_real_only` | 0.938 | 0.949 | 0.956 |
| `ci_validation_debiased` | 0.938 | 0.953 | 0.964 |

The ranking is consistent with the paper's inference warning: naive GN undercovers most, bias-aware/undersmoothed improves, and validation-debiased is closest to nominal in median coverage.

## Exp A: Tabular

Full-profile tabular summary:

| method | cells | median MSE ratio | max MSE ratio | harmful cells | mean fallback |
|---|---:|---:|---:|---:|---:|
| `corrected_adaptive_gn` | 45 | 1.000 | 1.000 | 0 | 1.000 |
| `safe_corrected_adaptive_gn` | 45 | 1.000 | 1.000 | 0 | 1.000 |
| `validation_debiased_gn` | 45 | 1.000 | 1.000 | 0 | 1.000 |
| `old_fixed_share_plugin_alpha` | 36 | 1.294 | 1.501 | 36 | 0.000 |
| `fixed_half_split_plugin_alpha` | 45 | 3.519 | 4.316 | 45 | 0.000 |
| `naive_pooling` | 45 | 5.439 | 117.166 | 45 | 0.000 |
| `synthetic_only_full_calibration` | 45 | 8.874 | 141.827 | 45 | 0.000 |

The corrected adaptive methods avoid harm by falling back to real-only in every full-profile tabular cell. That is stable, but it is not positive evidence that synthetic augmentation helps in this setting.

## Exp B: Causal

Full-profile causal summary, using `real_only_aipw` as the baseline:

| method | cells | median MSE ratio | max MSE ratio | harmful cells | mean fallback |
|---|---:|---:|---:|---:|---:|
| `safe_corrected_adaptive_gn` | 8 | 1.084 | 1.181 | 8 | 0.811 |
| `corrected_adaptive_gn` | 8 | 1.086 | 1.241 | 8 | 0.783 |
| `synthetic_only_full_calibration` | 8 | 1.276 | 1.642 | 8 | 0.000 |
| `validation_debiased_gn` | 8 | 1.459 | 1.889 | 8 | 0.770 |
| `naive_pooling` | 8 | 1.679 | 2.122 | 8 | 0.000 |
| `old_fixed_share_plugin_alpha` | 8 | 1.760 | 2.264 | 8 | 0.000 |
| `fixed_half_split_plugin_alpha` | 8 | 2.277 | 4.075 | 8 | 0.000 |
| `real_only_diff_in_means` | 8 | 5.415 | 15.472 | 8 | 0.000 |

The causal full profile is not favorable for synthetic augmentation. Corrected adaptive is the least harmful synthetic method, but it is still worse than `real_only_aipw` in all 8 cells.

## Rebuttal Experiments (2026-07-27)

All four preregistered rebuttal profiles ran from the dirty `main` worktree at
`c54575d` with seed root `20260727`. There were no failed rows.

| experiment | run directory | rows/cells | failures |
|---|---|---:|---:|
| equality boundary and comparators | `expR1_boundary__c54575d__20260728T042624__dirty` | 480,000 | 0 |
| remainder and non-power-law curves | `expR2_remainder__c54575d__20260728T042624__dirty` | 63 exact cells | 0 |
| hierarchical adaptive diagnostics | `expR3_hierarchical__c54575d__20260728T045921__dirty` | 2,520,000 | 0 |
| Adult auxiliary pseudo-labeling | `expR4_adult_auxiliary__c54575d__20260728T042625__dirty` | 116,000 | 0 |

### R1: equality boundary and the genuine zero-variance oracle

For \(\beta=1,\rho=3/2,a=c=1\), the corrected oracle calibration shares at
\(n=50{,}000\) were:

| \(v_0\) | predicted limit | observed \(x/n\) | absolute error |
|---:|---:|---:|---:|
| 0.75 | 1.000000 | 1.000000 | 0 |
| \(\sqrt 2\) | 1.000000 | 0.997880 | 0.002120 |
| 2.00 | 0.793701 | 0.791820 | 0.001881 |

At equality, \(n-x\) increased from 7 at \(n=200\) to 106 at \(n=50{,}000\);
the fitted gap exponent was 0.4925 while the share converged to one. Across
all 96 method/cell summaries, empirical MSE was within 2.63 Monte Carlo
standard errors of the analytic risk.

The tie-aware \(v_n=0\) profile is not the product-surrogate rule. With
\(a=c=1,n=10{,}000\), the deterministic tie-break selects real-only for
\(\beta<1/2\) (with the exactly equivalent \(x=1\) also minimizing at these
constants), every allocation ties at \(\beta=1/2\), and synthetic-only is the
unique minimizer for \(\beta>1/2\). The product-surrogate shares over the same
\(\beta\) grid range from \(1/3\) to \(3/4\).

### R2: remainder and non-power-law sensitivity

In the diverging \((\beta,\rho)=(1,1)\) regime, allocation slopes over
\(n=10^3,\ldots,10^6\) were 0.6719 (exact), 0.6698 (positive remainder),
0.6747 (negative remainder), and 0.6706 (oscillatory remainder), all within
0.0081 of the predicted \(2/3\). At \(n=10^6\), the four allocations were
between 12,517 and 12,568 and their risks were within \(1.4\times10^{-5}\)
relative to the exact-curve risk.

For the bounded \(\beta=1/4,c=0.1\) case, the limiting integer allocation
changed from 25 under the exact curve to 24 for \(q=-0.05\), 10 for \(q=0.5\),
and real-only for \(q=1\), confirming that the finite-\(x\) remainder controls
the constant optimum. Adding a \(10^{-3}\) plateau to the \(\beta=1\) curve
made the allocation stabilize at 120 by \(n=10^5\), with its share falling to
0.00012 at \(n=10^6\).

### R3: adaptive-score concentration and buffered safety

The hierarchical Gaussian experiment used two regimes, three \(n\)'s, three
external-validation sizes, three counts of independent calibration fits, two
synthetic-redraw counts, and 2,000 replications per cell. In the headline
\((\beta,\rho,n)=(1,1,10{,}000)\) cell with \(n_V=n\) and eight redraws:

| fits | median uniform factor \(C_0\) | direct MSE/real | buffered MSE/real | buffered fallback |
|---:|---:|---:|---:|---:|
| 1 | 13.336 | 3.005 | 1.004 | 1.000 |
| 4 | 3.293 | 0.688 | 0.662 | 0.0005 |
| 16 | 2.333 | 0.684 | 0.665 | 0 |

With \(n_V=5n\), 16 fits, and eight redraws, the median absolute centering and
positive-part errors of the bias score at \(n=10{,}000\) were both 0.98%;
the direct selector's median working-risk ratio was 1.0176 and its empirical
MSE ratio was 0.528, versus 0.511 for the grid oracle. It selected a median
\(x=320\) versus the grid oracle's 640.

The nominal simultaneous buffer is useful but not a proved no-harm rule.
Across all cells the plug-in rule's worst false-safe rate was 99.45%; the
buffer reduced the worst rate to 9.15% and the median to zero. In the no-gain
\(\beta=0.4\) cells its median fallback rate was 99.23% and median false-safe
rate 0.775%, but its median MSE ratio remained 1.051. Shared validation noise
caused undercoverage in some nominal-bound cells, directly supporting the
rebuttal's decision not to claim that repeated fits alone verify uniform
relative consistency.

### R4: positive Adult auxiliary-covariate pipeline

This experiment used the genuine cached UCI Adult table
(SHA-256 `3c8082c730498b67d2501bdf62d5c22ac7e8f445c3027cb88377e3b40bc87ac8`),
1,000 independent external pilot fits, 10,000 auxiliary covariate rows, and
2,000 held-out evaluation replications. It is a ridge-logistic pseudo-label
pipeline, not the earlier Gaussian-copula generator.

| \(n\) | estimand | external-grid \(x\) | \(\alpha\) | MSE/real (paired 95% CI) |
|---:|---|---:|---:|---:|
| 500 | income mean | 500 | 0 | 0.735 [0.699, 0.773] |
| 500 | subgroup gap | 500 | 0 | 0.754 [0.719, 0.789] |
| 1,000 | income mean | 1,000 | 0 | 0.734 [0.699, 0.772] |
| 1,000 | subgroup gap | 1,000 | 0 | 0.771 [0.736, 0.807] |
| 2,000 | income mean | 1,600 | 0.166 | 0.816 [0.779, 0.855] |
| 2,000 | subgroup gap | 2,000 | 0 | 0.800 [0.764, 0.836] |

All six paired upper confidence limits were below one. The buffered policy
made the same selections and did not fall back. Estimated local power-law
slopes ranged from 0.443 to 0.530, \(R^2\) ranged from 0.9972 to 0.9996, and
there were no monotonicity violations. Thus the benchmark supplies a
positive applied setting, but five cells select full calibration and the
sixth selects an 80% share; it does not validate the vanishing-share regime
or the original single-fit adaptive implementation.

## Consistency With Paper Claims

- Supported: corrected oracle allocation improves over real-only and old fixed-share rules in the fast-learning regime.
- Supported: corrected multichannel oracle dominates single-channel, equal-split, and old fixed-share baselines in the full profile.
- Supported with caveat: inference methods follow the expected ordering, with naive GN undercoverage and validation-debiased intervals closest to nominal.
- Partially supported: adaptive allocation is stable and no longer catastrophically under-calibrates.
- Not supported as a finite-sample claim: adaptive methods do not consistently track oracle performance, and full-profile Exp A/B do not show synthetic augmentation improving over the strongest real-only baselines.
