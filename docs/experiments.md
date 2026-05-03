# Experimental Specification for Optimal Real-Data Allocation

This document is the single source of truth for the coding agent. It specifies the synthetic and semi-synthetic experiments for the corrected Generalized Neyman allocation paper.

The experiments should validate four claims:

1. **Exact allocation law:** the optimal calibration size solves the corrected first-order condition, not a fixed-share rule.
2. **Phase diagram:** depending on the generator learning rate $\beta$ and synthetic variance decay rate $\rho$, the oracle allocation is bounded, sublinear, fixed-share only on knife edges, or full calibration.
3. **Adaptive feasibility:** estimating the risk curve and solving the corrected allocation problem tracks the oracle and safely falls back to real-only when synthetic data is harmful.
4. **Inference distinction:** MSE-optimal mixing improves point estimation, but confidence intervals need either negligible bias, validation debiasing, or honest bias adjustment.

---

## 1. Common notation

Use the following notation consistently in all code, tables, and figures.

| Symbol / variable name | Meaning |
|---|---|
| `n` | number of real observations |
| `m` | number of synthetic observations |
| `rho` | synthetic variance decay exponent, with $m_n\asymp n^\rho$ |
| `beta` | synthetic bias learning exponent |
| `a` | real-estimator variance constant $\sigma_R^2$ |
| `sigma_s2` | synthetic observation variance constant $\sigma_S^2$ |
| `v_n` | synthetic estimator variance $\sigma_S^2/m$ |
| `B0` | synthetic bias constant |
| `c` | squared bias constant $B_0^2$ |
| `x` or `n_f` | number of real observations used for calibration |
| `n_e` | number of real observations used for direct estimation |
| `lambda_f` | calibration share $x/n$ |
| `alpha` | weight on real estimator |
| `B_eff` | effective synthetic error $v_n+c x^{-2\beta}$ |
| `R_profile` | profiled risk $V_RB/(V_R+B)$ |

The central working model is

$$
V_R(x)=\frac{a}{n-x},
\qquad
B_n(x)=v_n+c x^{-2\beta},
\qquad
R_n(x)=\frac{V_R(x)B_n(x)}{V_R(x)+B_n(x)}.
$$

The corrected first-order condition is

$$
\left(v_n+c x^{-2\beta}\right)^2
=
2\beta acx^{-(2\beta+1)}.
$$

The safe-improvement condition is

$$
x\left(v_n+c x^{-2\beta}\right)<a.
$$

---

## 2. Estimators to implement

Every experiment should compare the following estimators unless explicitly marked optional.

### 2.1 Real-only estimator

Use all $n$ real observations for direct estimation:

$$
\hat\theta_{R,all}=\hat\theta(X_1,\ldots,X_n).
$$

This is the main baseline. Its theoretical risk in the scalar Gaussian experiment is

$$
R_{real}=a/n.
$$

Implementation name: `real_only_all`.

---

### 2.2 Split real-only estimator

Use only the estimation subset of size $n_e=n-x$:

$$
\hat\theta_{R,split}(x)=\hat\theta(X_i:i\in E_x).
$$

This is not the main baseline, but it is useful diagnostically because it isolates the opportunity cost of calibration.

Implementation name: `real_only_split`.

---

### 2.3 Synthetic-only estimator

Calibrate the generator using $x$ real observations and estimate from synthetic data only:

$$
\hat\theta_S(x)=\hat\theta(Z_1,\ldots,Z_m),
\qquad
Z_j\sim P_x.
$$

Use two versions:

1. `synthetic_only_oracle_x`: uses the oracle calibration size for synthetic-only risk.
2. `synthetic_only_full_calibration`: uses $x=n$ or the largest feasible calibration size.

---

### 2.4 Naive pooling estimator

Pool real estimation data and synthetic data as if both were unbiased draws from the same distribution.

For a sample mean estimand:

$$
\hat\theta_{pool}(x)
=
\frac{n_e\hat\theta_R(x)+m\hat\theta_S(x)}{n_e+m}.
$$

Implementation name: `naive_pooling`.

Expected behavior: this estimator can perform badly when synthetic bias is non-negligible.

---

### 2.5 Fixed 50/50 calibration split

Set

$$
x=\lfloor n/2\rfloor,
\qquad
n_e=n-x.
$$

Use the MSE-optimal mixing weight at this fixed allocation when oracle parameters are known:

$$
\alpha^*(x)=\frac{B_n(x)}{V_R(x)+B_n(x)}.
$$

Implementation name: `fixed_half_split_oracle_alpha`.

Also include a feasible plug-in version when running adaptive experiments:

Implementation name: `fixed_half_split_plugin_alpha`.

---

### 2.6 Old fixed-share rule

This estimator is included only as a stress-test of the old theory. Set

$$
\lambda_{old}=\frac{2\beta}{1+2\beta},
\qquad
x_{old}=\lfloor n\lambda_{old}\rfloor.
$$

Use the oracle mixing weight at this allocation.

Implementation name: `old_fixed_share_oracle_alpha`.

A second version may estimate $\beta$ and plug it into the old formula:

Implementation name: `old_fixed_share_plugin_alpha`.

Expected behavior: in the common $m\asymp n$, $\beta>1/2$ regime, this estimator should over-calibrate and lose efficiency relative to the corrected rule.

---

### 2.7 Corrected oracle Generalized Neyman estimator

Use the true parameters $a,v_n,c,\beta$ to minimize the profiled risk:

$$
x_{oracle}
\in
\arg\min_{x\in\mathcal G_n\cup\{0,n\}}
R_n(x).
$$

Then use

$$
\alpha_{oracle}
=
\frac{B_n(x_{oracle})}{V_R(x_{oracle})+B_n(x_{oracle})}
$$

and

$$
\hat\theta_{oracle}
=
\alpha_{oracle}\hat\theta_R(x_{oracle})
+(1-\alpha_{oracle})\hat\theta_S(x_{oracle}).
$$

Implementation name: `corrected_oracle_gn`.

This is the benchmark for all adaptive methods.

---

### 2.8 Corrected adaptive Generalized Neyman estimator

Estimate $a,v_n,c,\beta$ or estimate the whole risk curve. Select

$$
\hat x
\in
\arg\min_{x\in\mathcal G_n}
\widehat R_n(x),
$$

where

$$
\widehat R_n(x)
=
\frac{\widehat V_R(x)\widehat B_n(x)}{\widehat V_R(x)+\widehat B_n(x)}.
$$

Then use

$$
\hat\alpha
=
\frac{\widehat B_n(\hat x)}{\widehat V_R(\hat x)+\widehat B_n(\hat x)}.
$$

Implementation name: `corrected_adaptive_gn`.

---

### 2.9 Safe adaptive Generalized Neyman estimator

Use the corrected adaptive allocation, but apply the safety check:

$$
\hat x\widehat B_n(\hat x)<\hat a.
$$

If the check fails, return `real_only_all`.

Implementation name: `safe_corrected_adaptive_gn`.

Required diagnostic fields:

- `safe_pass`: boolean;
- `safe_margin`: $\hat a-\hat x\widehat B_n(\hat x)$;
- `fallback_used`: boolean.

---

### 2.10 Validation-debiased estimator

This estimator is used for inference and coverage experiments.

Split real data into calibration, estimation, and validation sets:

$$
n=n_f+n_e+n_v.
$$

Estimate the synthetic bias on validation data:

$$
\hat b_V(x)=\hat\theta_S^V(x)-\hat\theta_R^V.
$$

Define

$$
\tilde\theta_S(x)=\hat\theta_S(x)-\hat b_V(x).
$$

Combine

$$
\hat\theta_{debias}(x)
=
\alpha\hat\theta_R(x)+(1-\alpha)\tilde\theta_S(x).
$$

Implementation name: `validation_debiased_gn`.

---

## 3. Common metrics

All experiments should report the following metrics whenever truth is known.

### 3.1 Point-estimation metrics

For estimator $e$ and Monte Carlo replication $r$, let $\hat\theta_{e,r}$ be the estimate and $\theta^*$ the truth.

Report:

$$
\operatorname{Bias}(e)=\frac1R\sum_{r=1}^R(\hat\theta_{e,r}-\theta^*),
$$

$$
\operatorname{Variance}(e)=\frac1{R-1}\sum_{r=1}^R\left(\hat\theta_{e,r}-\bar\theta_e\right)^2,
$$

$$
\operatorname{MSE}(e)=\frac1R\sum_{r=1}^R(\hat\theta_{e,r}-\theta^*)^2,
$$

$$
\operatorname{RMSE}(e)=\sqrt{\operatorname{MSE}(e)}.
$$

Report MSE ratios:

$$
\operatorname{MSE\ ratio\ to\ real}(e)
=
\frac{\operatorname{MSE}(e)}{\operatorname{MSE}(real\_only\_all)},
$$

and oracle regret:

$$
\operatorname{Oracle\ regret}(e)
=
\frac{\operatorname{MSE}(e)}{\operatorname{MSE}(corrected\_oracle\_gn)}-1.
$$

---

### 3.2 Allocation metrics

For each method that chooses an allocation, report:

- selected calibration size: `x_selected`;
- selected calibration share: `lambda_selected=x_selected/n`;
- selected direct-estimation size: `n_e=n-x_selected`;
- selected mixing weight: `alpha_selected`;
- effective synthetic error: `B_eff_selected`;
- real variance: `V_R_selected`;
- FOC residual:

  $$
  \operatorname{FOCResidual}
  =
  \left(v_n+c x^{-2\beta}\right)^2
  -2\beta acx^{-(2\beta+1)};
  $$

- safety margin:

  $$
  \operatorname{SafetyMargin}=a-xB_n(x).
  $$

---

### 3.3 Scaling metrics

For phase-diagram experiments, estimate the empirical scaling slope by regressing

$$
\log x_n^*
=
\gamma_0+\gamma_1\log n+u_n.
$$

Compare $\hat\gamma_1$ to the theory:

$$
\gamma_{theory}
=
\begin{cases}
0, & \rho=0 \text{ or } \beta<1/2,\\
2\rho/(2\beta+1), & \beta>1/2,\ 0<\rho<\beta+1/2,\\
1, & \beta>1/2,\ \rho>\beta+1/2.
\end{cases}
$$

For the knife-edge cases, report constants separately rather than forcing a universal slope.

---

### 3.4 Inference metrics

For interval-valued estimators, report:

- nominal coverage, usually 95%;
- average interval length;
- median interval length;
- empirical standard deviation of point estimates;
- average estimated standard error;
- bias divided by standard error:

  $$
  \frac{|\operatorname{Bias}|}{\operatorname{Average\ SE}}.
  $$

---

## 4. Synthetic Experiment 1: phase diagram validation

### 4.1 Motivation

This is the most important synthetic experiment. It should show that the corrected first-order condition predicts the actual oracle allocation and that the allocation has the predicted phase transitions.

The key figure is $\log x_n^*$ versus $\log n$ for multiple $(\beta,\rho)$ pairs.

---

### 4.2 Data-generating process

For each Monte Carlo replication:

Real observations:

$$
X_i=\theta^*+\varepsilon_i,
\qquad
\varepsilon_i\sim N(0,a),
\qquad
\theta^*=0.
$$

Synthetic observations after calibration size $x$:

$$
Z_j(x)=\theta^*+B_0x^{-\beta}+u_j,
\qquad
u_j\sim N(0,\sigma_S^2).
$$

The synthetic sample size is

$$
m_n=\lceil\kappa n^\rho\rceil,
$$

so

$$
v_n=\sigma_S^2/m_n.
$$

Default constants:

- $\theta^*=0$;
- $a=1$;
- $B_0=1$;
- $c=B_0^2=1$;
- $\sigma_S^2=1$;
- $\kappa=1$.

Use additional constants in robustness checks:

- $B_0\in\{0.5,1,2\}$;
- $a\in\{0.5,1,2\}$;
- $\sigma_S^2\in\{0.5,1,2\}$;
- $\kappa\in\{0.5,1,5,10\}$.

---

### 4.3 Parameter grid

Use:

$$
n\in\{100,200,500,1000,2000,5000,10000,20000\}.
$$

Use:

$$
\beta\in\{0.25,0.4,0.5,0.75,1.0,1.5,2.0\}.
$$

Use:

$$
\rho\in\{0,0.5,1.0,1.5,2.0,3.0\}.
$$

For each pair $(\beta,\rho)$, classify the theoretical regime before running simulations:

- `persistent_variance` if $\rho=0$;
- `slow_learning` if $\beta<1/2$;
- `parametric_knife_edge` if $\beta=1/2$;
- `sublinear_fast_learning` if $\beta>1/2$ and $0<\rho<\beta+1/2$;
- `boundary_full_calibration` if $\beta>1/2$ and $\rho>\beta+1/2$;
- `boundary_knife_edge` if $\beta>1/2$ and $\rho=\beta+1/2$.

---

### 4.4 Oracle computation

For each $(n,\beta,\rho)$, compute the oracle allocation in two ways:

1. **FOC root:** solve

   $$
   \left(v_n+c x^{-2\beta}\right)^2
   =2\beta acx^{-(2\beta+1)}.
   $$

   Then check whether the solution is feasible and whether it is a local minimum.

2. **Grid minimization:** evaluate

   $$
   R_n(x)=\frac{V_R(x)B_n(x)}{V_R(x)+B_n(x)}
   $$

   on integer grid

   $$
   x\in\{1,2,\ldots,n-1\}
   $$

   plus boundary choices `real_only_all` and `synthetic_only_full_calibration`.

Use grid minimization as the ground-truth oracle for reported results.

---

### 4.5 Estimators tested

For this experiment, include:

1. `real_only_all`;
2. `synthetic_only_full_calibration`;
3. `naive_pooling`;
4. `fixed_half_split_oracle_alpha`;
5. `old_fixed_share_oracle_alpha`;
6. `corrected_oracle_gn`;
7. `safe_corrected_oracle_gn`.

Adaptive methods are tested in Experiment 2.

---

### 4.6 Replications

For risk estimates, use at least:

- `R=5000` replications for $n\le5000$;
- `R=2000` replications for $n>5000$ if runtime is high.

For oracle allocation scaling, no Monte Carlo is necessary because $R_n(x)$ is known analytically in this experiment. Still, Monte Carlo estimates should be generated to verify implementation.

---

### 4.7 Required figures

#### Figure S1: Allocation phase diagram

Create a heatmap with:

- x-axis: $\beta$;
- y-axis: $\rho$;
- color: theoretical regime;
- overlay markers for grid points actually simulated.

#### Figure S2: Calibration size scaling

For representative regimes, plot:

$$
\log x_n^* \quad \text{versus} \quad \log n.
$$

Overlay the theoretical slope.

Recommended panels:

1. fixed $m$: $\rho=0$;
2. slow learner: $\beta=0.4$, $\rho=1$;
3. common fast learner: $\beta=1$, $\rho=1$;
4. boundary regime: $\beta=1$, $\rho=2$.

#### Figure S3: Calibration share

Plot

$$
\lambda_n^*=x_n^*/n
$$

against $n$ on a log scale. The common fast-learning regime should show $x_n^*\uparrow$ but $\lambda_n^*\downarrow$.

#### Figure S4: Risk ratio by estimator

Plot

$$
\operatorname{MSE}(e)/\operatorname{MSE}(real\_only\_all)
$$

for all estimators across $n$.

---

### 4.8 Required tables

#### Table S1: predicted versus empirical allocation scaling

Columns:

- `beta`;
- `rho`;
- `theory_regime`;
- `theory_slope`;
- `empirical_slope`;
- `slope_se`;
- `mean_oracle_lambda_at_largest_n`;
- `boundary_selected_rate`.

#### Table S2: estimator performance

Columns:

- `beta`;
- `rho`;
- `n`;
- `estimator`;
- `bias`;
- `variance`;
- `mse`;
- `mse_ratio_to_real`;
- `oracle_regret`;
- `x_selected`;
- `lambda_selected`;
- `alpha_selected`.

---

## 5. Synthetic Experiment 2: adaptive allocation and learning-curve estimation

### 5.1 Motivation

This experiment tests whether the feasible adaptive method can recover the corrected oracle allocation without knowing $a,v_n,c,\beta$.

It should answer:

1. Can the method estimate $\beta$ accurately enough?
2. Is direct risk-curve minimization more stable than plug-in FOC solving?
3. Does the safe fallback prevent harm when synthetic data is not useful?

---

### 5.2 DGP

Use the same Gaussian DGP as Experiment 1.

Focus on a smaller grid:

$$
\beta\in\{0.4,0.5,0.75,1.0,1.5\},
\qquad
\rho\in\{0,1,2\},
$$

and

$$
n\in\{200,500,1000,2000,5000,10000\}.
$$

---

### 5.3 Pilot grid

Use pilot calibration sizes

$$
\mathcal G_{pilot}
=
\{5,10,20,40,80,160,320,640\}\cap[1,n/2].
$$

For very small $n$, use

$$
\mathcal G_{pilot}
=
\{\lfloor n/20\rfloor,\lfloor n/10\rfloor,\lfloor n/5\rfloor,\lfloor n/3\rfloor\}
$$

after removing duplicates and invalid values.

---

### 5.4 Bias estimation

For each pilot size $x_j$:

1. calibrate the synthetic generator using $x_j$ observations;
2. generate synthetic data;
3. compute $\hat\theta_S(x_j)$;
4. compute independent held-out real estimate $\hat\theta_R^V$;
5. estimate squared bias by

   $$
   \widehat{b^2}(x_j)
   =
   \left[
   \{\hat\theta_S(x_j)-\hat\theta_R^V\}^2
   -
   \widehat{\operatorname{Var}}\{\hat\theta_S(x_j)\}
   -
   \widehat{\operatorname{Var}}\{\hat\theta_R^V\}
   \right]_+.
   $$

Fit

$$
\log \widehat{b^2}(x_j)=\log c-2\beta\log x_j+\epsilon_j.
$$

Use robust regression or weighted least squares if small $x_j$ values are noisy.

---

### 5.5 Adaptive methods to compare

Implement three adaptive versions.

#### A. Parametric FOC plug-in

Estimate $\hat a,\hat v_n,\hat c,\hat\beta$ and solve

$$
\left(\hat v_n+\hat c x^{-2\hat\beta}\right)^2
=
2\hat\beta\hat a\hat c x^{-(2\hat\beta+1)}.
$$

Implementation name: `adaptive_parametric_foc`.

#### B. Parametric grid risk minimization

Estimate $\hat a,\hat v_n,\hat c,\hat\beta$ and minimize

$$
\widehat R_n(x)
=
\frac{\hat a(\hat v_n+
\hat c x^{-2\hat\beta})}
{\hat a+(\hat v_n+
\hat c x^{-2\hat\beta})(n-x)}.
$$

Implementation name: `adaptive_parametric_grid`.

#### C. Nonparametric risk-curve minimization

Estimate $\widehat B_n(x)$ directly using smoothed or interpolated pilot estimates and minimize the empirical profiled risk.

Implementation name: `adaptive_nonparametric_grid`.

The main adaptive method in the paper should be the grid version, because it handles boundary and multiple-root cases more robustly.

---

### 5.6 Required outputs

For each replication and method, save:

- `theta_hat`;
- `x_selected`;
- `lambda_selected`;
- `alpha_selected`;
- `beta_hat`;
- `c_hat`;
- `a_hat`;
- `v_hat`;
- `safe_pass`;
- `fallback_used`;
- `oracle_x`;
- `oracle_alpha`;
- `oracle_risk`;
- `estimated_risk_at_selected_x`;
- `true_risk_at_selected_x`.

---

### 5.7 Required figures and tables

#### Figure A1: estimated learning curves

Plot

$$
\log\widehat{b^2}(x_j)
$$

against

$$
\log x_j.
$$

Overlay the fitted line with slope $-2\hat\beta$.

#### Figure A2: adaptive versus oracle allocation

Scatter plot:

$$
\hat x \quad \text{versus} \quad x_{oracle}.
$$

Also plot

$$
\log \hat x-\log x_{oracle}.
$$

#### Figure A3: adaptive regret

Plot

$$
R_n(\hat x)/R_n(x_{oracle})-1
$$

against $n$.

#### Table A1: learning-curve estimation accuracy

Columns:

- `beta`;
- `rho`;
- `n`;
- `mean_beta_hat`;
- `sd_beta_hat`;
- `bias_beta_hat`;
- `mean_c_hat`;
- `sd_c_hat`.

#### Table A2: adaptive estimator performance

Columns:

- `beta`;
- `rho`;
- `n`;
- `method`;
- `mse_ratio_to_real`;
- `oracle_regret`;
- `allocation_relative_error`;
- `safe_fallback_rate`;
- `harm_rate`, where harm means MSE greater than real-only in that setting.

---

## 6. Synthetic Experiment 3: multichannel allocation

### 6.1 Motivation

The multichannel experiment tests the corrected KKT conditions. It should show that the optimal allocation equalizes marginal MSE reduction across active calibration channels and that the old fixed-share multichannel rule is generally wrong.

---

### 6.2 DGP

Use two calibration channels:

$$
B_n(x_1,x_2)
=
v_n+c_1x_1^{-2\beta_1}+c_2x_2^{-2\beta_2}.
$$

The real variance is

$$
V_R(x_1,x_2)=\frac{a}{n-x_1-x_2}.
$$

Default parameters:

- $a=1$;
- $c_1=c_2=1$;
- $\sigma_S^2=1$;
- $m=n$ so $\rho=1$;
- $(\beta_1,\beta_2)\in\{(0.5,1.0),(0.75,1.5),(0.4,1.0)\}$.

Interpret channel 1 as fine-tuning and channel 2 as in-context learning or retrieval conditioning.

---

### 6.3 Oracle allocation

Grid search over

$$
(x_1,x_2): x_1\ge1,\ x_2\ge1,\ x_1+x_2\le n-1.
$$

Compute

$$
R_n(x_1,x_2)
=
\frac{V_R(x_1,x_2)B_n(x_1,x_2)}
{V_R(x_1,x_2)+B_n(x_1,x_2)}.
$$

The active-channel KKT condition is

$$
B_n(\mathbf x)^2
=
2\beta_kac_kx_k^{-(2\beta_k+1)},
\qquad k=1,2.
$$

At the optimum, report the marginal values:

$$
MV_k=2\beta_kac_kx_k^{-(2\beta_k+1)}.
$$

They should be equal across active channels after scaling by the common denominator.

---

### 6.4 Estimators to compare

1. `best_single_channel_oracle`;
2. `equal_split_two_channels`;
3. `old_multichannel_fixed_share`;
4. `corrected_multichannel_oracle`;
5. `corrected_multichannel_adaptive` if feasible.

The old multichannel fixed-share rule is

$$
\lambda_k^{old}
=
\frac{2\beta_k}{1+2\sum_j\beta_j}.
$$

It is included only as a benchmark to demonstrate over-allocation.

---

### 6.5 Required figures and tables

#### Figure M1: risk surface

For a representative $n$, plot a heatmap of

$$
R_n(x_1,x_2)
$$

over the feasible triangle $x_1+x_2<n$.

Mark:

- corrected oracle allocation;
- equal split;
- old fixed-share allocation;
- best single-channel allocation.

#### Figure M2: marginal value equality

Plot $MV_1$ and $MV_2$ at the selected allocation for each method. Corrected oracle should equalize active marginal values.

#### Table M1: multichannel performance

Columns:

- `beta_1`;
- `beta_2`;
- `n`;
- `method`;
- `x_1`;
- `x_2`;
- `x_total`;
- `lambda_1`;
- `lambda_2`;
- `mse_ratio_to_real`;
- `oracle_regret`;
- `mv_1`;
- `mv_2`.

---

## 7. Synthetic Experiment 4: inference and coverage

### 7.1 Motivation

This experiment separates point-estimation optimality from valid inference. The MSE-optimal estimator may have non-negligible synthetic bias. The paper should show this honestly and then show how validation debiasing or undersmoothing restores coverage.

---

### 7.2 DGP

Use the scalar Gaussian DGP from Experiment 1. Focus on regimes where bias may be non-negligible relative to the standard error:

- $\beta=0.5$, $\rho=1$;
- $\beta=0.75$, $\rho=1$;
- $\beta=1.0$, $\rho=1$;
- $\beta=0.4$, $\rho=1$.

Use

$$
n\in\{200,500,1000,2000,5000\}.
$$

---

### 7.3 Interval methods

Compare the following interval procedures.

#### A. Real-only Wald interval

Use all real data:

$$
\hat\theta_{R,all}\pm1.96\sqrt{\hat a/n}.
$$

Implementation name: `ci_real_only`.

#### B. Naive MSE-optimal Wald interval

Use the corrected MSE-optimal estimator but ignore synthetic bias:

$$
\hat\theta_{GN}\pm1.96\sqrt{\widehat\Omega_n}.
$$

Implementation name: `ci_gn_naive`.

Expected behavior: may undercover when $(1-\alpha)b(x)$ is non-negligible.

#### C. Bias-aware interval

Use

$$
\hat\theta_{GN}\pm
\left(1.96\sqrt{\widehat\Omega_n}+|(1-\hat\alpha)\hat b(\hat x)|\right).
$$

Implementation name: `ci_gn_bias_aware`.

#### D. Undersmoothed interval

Choose additional calibration so that

$$
|(1-\alpha)b(x)|
\le
\epsilon_n\sqrt{\Omega_n(x)}
$$

for a small threshold, such as $\epsilon_n=0.1$.

Implementation name: `ci_gn_undersmoothed`.

#### E. Validation-debiased interval

Use held-out validation data to estimate and subtract synthetic bias. Then construct

$$
\hat\theta_{debias}\pm1.96\sqrt{\widehat\Omega_{debias}}.
$$

Implementation name: `ci_validation_debiased`.

---

### 7.4 Required metrics

For each interval method, report:

- coverage probability;
- average interval length;
- median interval length;
- empirical bias;
- empirical standard deviation;
- average estimated standard error;
- bias divided by standard error.

---

### 7.5 Required figures and tables

#### Figure C1: coverage versus n

Plot coverage for each interval method, with a horizontal line at 95%.

#### Figure C2: interval length versus coverage

Scatter plot of average interval length against coverage. Methods near 95% coverage with shorter length are preferred.

#### Table C1: coverage summary

Columns:

- `beta`;
- `rho`;
- `n`;
- `interval_method`;
- `coverage`;
- `avg_length`;
- `bias`;
- `empirical_sd`;
- `avg_se`;
- `bias_over_se`.

---

## 8. Semi-synthetic Experiment A: tabular synthetic data benchmark

### 8.1 Motivation

The tabular experiment tests whether the corrected allocation logic appears with real covariates, real outcomes, and an actual synthetic-data generator. The purpose is not to prove that a particular generator is best. The purpose is to show that the empirical bias curve can be estimated and that corrected allocation improves over naive synthetic augmentation and fixed-share calibration.

---

### 8.2 Dataset and pseudo-population

Use a public tabular dataset with enough rows to treat the full dataset as an approximate population. Recommended first choice:

- UCI Adult / Census Income.

Alternative datasets if Adult is inconvenient:

- ACS Income from Folktables;
- Bank Marketing;
- California Housing for continuous outcomes;
- MIMIC-style clinical tabular data if access is already available.

Let the full cleaned dataset be

$$
\mathcal D_{pop}=\{W_i\}_{i=1}^N.
$$

Treat the full dataset as the pseudo-population. The ground-truth estimand is computed on $\mathcal D_{pop}$ and is used only for evaluation, not for training or allocation selection.

---

### 8.3 Estimands

Use at least two estimands.

#### Estimand 1: outcome mean

For binary income outcome $Y$:

$$
\theta_1^*=\mathbb E[Y].
$$

Pseudo-population truth:

$$
\theta_1^*=\frac1N\sum_{i=1}^NY_i.
$$

#### Estimand 2: subgroup gap

For a binary group indicator $G$:

$$
\theta_2^*=
\mathbb E[Y\mid G=1]-\mathbb E[Y\mid G=0].
$$

#### Estimand 3: regression coefficient

Fit a pre-specified linear or logistic model on the pseudo-population and define $\theta_3^*$ as one coefficient, such as education, age, or hours worked.

For robustness, report all three if feasible; otherwise use the mean and subgroup gap in the NeurIPS version.

---

### 8.4 Real sample and calibration protocol

For each Monte Carlo replication:

1. Sample $n$ rows without replacement from $\mathcal D_{pop}$ to form the scarce real dataset $\mathcal D_n$.
2. Split $\mathcal D_n$ into calibration, estimation, and validation subsets according to the method being evaluated.
3. Train or calibrate a tabular generator using the calibration subset.
4. Generate $m$ synthetic rows.
5. Compute the synthetic estimator using the same estimand function as for real data.
6. Combine real and synthetic estimators using the relevant allocation and mixing rule.
7. Evaluate against the pseudo-population truth $\theta^*$.

Use

$$
n\in\{200,500,1000,2000,5000\}.
$$

Use

$$
m\in\{n,5n,10n\}
$$

for the main experiment.

---

### 8.5 Synthetic generators

Start with at least one generator that is easy to run reproducibly.

Recommended order:

1. `GaussianCopula` or simple bootstrap-smoothed generator as a debugging baseline;
2. `CTGAN` or `TVAE` for the main tabular generator;
3. `TabDDPM` if compute and implementation bandwidth allow.

The paper should report generator-specific results rather than claiming generator universality.

---

### 8.6 Adaptive bias-curve estimation

For pilot calibration sizes

$$
\mathcal G_{pilot}=\{20,50,100,200,500,1000\}\cap[1,n/2],
$$

estimate

$$
\widehat{b^2}(x)
$$

using validation data. Fit

$$
\log\widehat{b^2}(x)=\log c-2\beta\log x+\epsilon_x.
$$

Because real generators can have nonmonotone learning curves, also fit a monotone smoothed curve. The code should store both.

Implementation outputs:

- `beta_hat_powerlaw`;
- `c_hat_powerlaw`;
- `B_eff_hat_powerlaw`;
- `B_eff_hat_smooth`;
- `selected_model_for_allocation`, with values `powerlaw` or `smooth`.

---

### 8.7 Estimators to compare

Include:

1. `real_only_all`;
2. `synthetic_only_full_calibration`;
3. `naive_pooling`;
4. `fixed_half_split_plugin_alpha`;
5. `old_fixed_share_plugin_alpha`;
6. `corrected_adaptive_gn`;
7. `safe_corrected_adaptive_gn`;
8. `validation_debiased_gn`, especially for the subgroup and regression estimands.

For semi-synthetic data, the oracle allocation can be approximated empirically by repeated Monte Carlo evaluation over the allocation grid. Include:

9. `empirical_oracle_grid`, for benchmarking only.

---

### 8.8 Required metrics

For each estimand, generator, $n$, and $m$:

- bias;
- variance;
- MSE;
- MSE ratio to real-only;
- regret relative to empirical oracle;
- selected calibration size;
- selected calibration share;
- selected mixing weight;
- estimated $\hat\beta$;
- safe fallback rate;
- generator failure rate, if any.

---

### 8.9 Required figures and tables

#### Figure T1: empirical learning curve

Plot

$$
\log\widehat{b^2}(x)
$$

against

$$
\log x
$$

for each estimand and generator. Overlay the fitted power-law slope and the monotone smoother.

#### Figure T2: selected allocation versus n

Plot selected $x$ and $\lambda=x/n$ against $n$. The key visual pattern to look for is increasing $x$ but decreasing $\lambda$.

#### Figure T3: MSE ratio to real-only

Bar or line plot of MSE ratios for all estimators.

#### Table T1: tabular learning-curve summary

Columns:

- `dataset`;
- `generator`;
- `estimand`;
- `n`;
- `m`;
- `beta_hat_mean`;
- `beta_hat_sd`;
- `powerlaw_r2`;
- `monotonicity_violations`.

#### Table T2: tabular performance summary

Columns:

- `dataset`;
- `generator`;
- `estimand`;
- `n`;
- `m`;
- `method`;
- `bias`;
- `variance`;
- `mse`;
- `mse_ratio_to_real`;
- `oracle_regret`;
- `x_selected`;
- `lambda_selected`;
- `alpha_selected`;
- `fallback_rate`.

---

## 9. Semi-synthetic Experiment B: causal digital-twin benchmark

### 9.1 Motivation

This experiment connects the method to econometrics and causal inference. It asks whether a calibrated synthetic outcome model can improve estimation of an average treatment effect, and whether the corrected allocation rule avoids the common failure mode of biased synthetic controls.

---

### 9.2 Dataset and estimand

Use one of the following benchmark families:

1. IHDP-style semi-synthetic treatment-effect data;
2. ACIC-style semi-synthetic treatment-effect data;
3. LaLonde / NSW-style job training data with a semi-synthetic outcome model;
4. A custom semi-synthetic DGP with real covariates and known potential outcomes.

The target estimand is the average treatment effect:

$$
\tau^*=\mathbb E[Y(1)-Y(0)].
$$

The benchmark must provide either true potential outcomes or a known simulation truth. This truth is used only for evaluation.

---

### 9.3 Real estimator

Use a standard real-data estimator of the ATE.

Recommended default:

$$
\hat\tau_{AIPW}
=
\frac1n\sum_{i=1}^n
\left[
\hat\mu_1(X_i)-\hat\mu_0(X_i)
+
\frac{A_i\{Y_i-\hat\mu_1(X_i)\}}{\hat e(X_i)}
-
\frac{(1-A_i)\{Y_i-\hat\mu_0(X_i)\}}{1-\hat e(X_i)}
\right].
$$

Use cross-fitting for nuisance functions.

Implementation name: `real_only_aipw`.

Also include difference-in-means when treatment is randomized:

Implementation name: `real_only_diff_in_means`.

---

### 9.4 Synthetic estimator

Train a conditional generator on the calibration subset. The generator should produce either:

1. synthetic observed triples $(X,A,Y)$; or
2. synthetic potential outcomes $(X,Y(0),Y(1))$; or
3. synthetic counterfactual outcomes conditional on real covariates.

For synthetic potential outcomes, estimate

$$
\hat\tau_S(x)=\frac1m\sum_{j=1}^m\{\tilde Y_j(1)-\tilde Y_j(0)\}.
$$

For synthetic observed triples, estimate ATE using the same AIPW or randomized-trial estimator as in the real data.

---

### 9.5 Allocation and combination

Use the same corrected allocation framework:

$$
\hat\tau_{GN}
=
\hat\alpha\hat\tau_R+(1-\hat\alpha)\hat\tau_S.
$$

The estimated risk curve should be based on held-out validation estimates of ATE bias:

$$
\widehat{b^2}_{ATE}(x)
=
\left[
\{\hat\tau_S(x)-\hat\tau_R^V\}^2
-
\widehat{\operatorname{Var}}\{\hat\tau_S(x)\}
-
\widehat{\operatorname{Var}}\{\hat\tau_R^V\}
\right]_+.
$$

---

### 9.6 Baselines

Compare:

1. `real_only_diff_in_means` when applicable;
2. `real_only_aipw`;
3. `synthetic_only_full_calibration`;
4. `naive_pooling`;
5. `fixed_half_split_plugin_alpha`;
6. `old_fixed_share_plugin_alpha`;
7. `corrected_adaptive_gn`;
8. `safe_corrected_adaptive_gn`;
9. `validation_debiased_gn`;
10. optional PPI-style correction if the implementation naturally supports prediction-powered scores.

---

### 9.7 Required metrics

For each method, report:

- ATE bias;
- ATE variance;
- ATE MSE;
- MSE ratio to real-only AIPW;
- empirical coverage of 95% intervals;
- average interval length;
- selected calibration size;
- selected calibration share;
- selected mixing weight;
- fallback rate;
- estimated $\hat\beta$.

---

### 9.8 Required figures and tables

#### Figure D1: causal learning curve

Plot

$$
\log\widehat{b^2}_{ATE}(x)
$$

against

$$
\log x.
$$

#### Figure D2: ATE MSE by estimator

Plot MSE ratio to real-only AIPW.

#### Figure D3: coverage versus interval length

Compare real-only, naive GN, validation-debiased GN, and bias-aware GN.

#### Table D1: causal performance summary

Columns:

- `benchmark`;
- `n`;
- `m`;
- `generator`;
- `method`;
- `bias`;
- `variance`;
- `mse`;
- `mse_ratio_to_real_aipw`;
- `coverage`;
- `avg_interval_length`;
- `x_selected`;
- `lambda_selected`;
- `alpha_selected`;
- `fallback_rate`.

---

## 10. Optional semi-synthetic Experiment C: demand estimation

This is optional for NeurIPS and more relevant for Management Science.

### 10.1 Motivation

Demand estimation is a natural management-science application because the value of synthetic data can be measured not only by parameter MSE but also by pricing or policy regret.

---

### 10.2 Setup

Use real or semi-synthetic market-level data. Define an estimand such as:

- price elasticity;
- logit demand coefficient;
- average marginal effect of price;
- optimal price under estimated demand.

Let

$$
\theta^*=\text{demand parameter},
\qquad
\pi^*(\theta)=\text{profit-optimal decision}.
$$

Evaluate both parameter error and decision regret:

$$
\operatorname{Regret}_{profit}
=
\Pi(\pi^*(\theta^*);\theta^*)-
\Pi(\pi^*(\hat\theta);\theta^*).
$$

---

### 10.3 Methods and metrics

Use the same estimator set as the tabular experiment, and report:

- coefficient bias;
- coefficient MSE;
- elasticity MSE;
- profit regret;
- selected calibration size;
- fallback rate.

---

## 11. Master output schema

Every experiment should write a long-format results file with one row per replication, method, and estimand.

Required columns:

```text
experiment_id
replication
seed
n
m
rho
beta
a
sigma_s2
B0
c
dataset
generator
estimand
method
theta_true
theta_hat
error
squared_error
x_selected
lambda_selected
n_e
alpha_selected
B_eff_selected
V_R_selected
beta_hat
c_hat
a_hat
v_hat
oracle_x
oracle_lambda
oracle_alpha
oracle_risk
estimated_risk_selected
true_risk_selected
safe_pass
safe_margin
fallback_used
ci_lower
ci_upper
ci_length
covered
runtime_seconds
failure_flag
failure_reason
```

For experiments where some columns are not applicable, fill with `NA` rather than dropping the column.

---

## 12. Master figure list for the NeurIPS paper

The main NeurIPS paper should contain four figures.

### Main Figure 1: theoretical phase diagram

Axes:

- x-axis: $\beta$;
- y-axis: $\rho$.

Color regions:

- bounded/no calibration;
- sublinear calibration;
- parametric knife edge;
- boundary/full calibration.

Overlay the common line $\rho=1$.

---

### Main Figure 2: synthetic scaling validation

Plot $\log x_n^*$ versus $\log n$ for selected regimes, with theoretical slopes.

This figure should make the main correction visually obvious.

---

### Main Figure 3: MSE comparison

Plot MSE ratio to real-only for:

- real-only;
- old fixed-share;
- corrected oracle;
- corrected adaptive;
- safe adaptive.

Use representative regimes.

---

### Main Figure 4: semi-synthetic result

Either:

1. tabular experiment: empirical learning curve plus MSE ratios; or
2. causal experiment: ATE MSE plus coverage/interval length.

Use whichever semi-synthetic experiment is stronger and cleaner by the deadline.

---

## 13. Master table list for the NeurIPS paper

### Main Table 1: phase diagram validation

Columns:

- $\beta$;
- $\rho$;
- predicted regime;
- predicted slope;
- empirical slope;
- corrected oracle final $\lambda$;
- old fixed-share $\lambda$.

### Main Table 2: estimator performance

Columns:

- regime;
- estimator;
- selected $x$;
- selected $\lambda$;
- selected $\alpha$;
- bias;
- variance;
- MSE;
- MSE ratio to real-only;
- oracle regret.

### Main Table 3: semi-synthetic performance

Columns:

- dataset/benchmark;
- generator;
- estimand;
- method;
- selected $x$;
- selected $\lambda$;
- MSE ratio to real-only;
- coverage if applicable;
- fallback rate.

---

## 14. Implementation priorities

The coding agent should implement in this order.

### Priority 1: deterministic oracle calculator

Implement functions:

```python
B_eff(x, v_n, c, beta)
V_real(x, n, a)
R_profile(x, n, a, v_n, c, beta)
foc_residual(x, a, v_n, c, beta)
safe_condition(x, a, v_n, c, beta)
oracle_grid(n, a, v_n, c, beta, grid=None, include_boundaries=True)
```

This step should be tested before any Monte Carlo simulation.

---

### Priority 2: synthetic Gaussian Monte Carlo

Implement Experiment 1. Validate that empirical MSE matches analytic risk.

Sanity checks:

- real-only MSE should be close to $a/n$;
- synthetic-only MSE should be close to $v_n+cx^{-2\beta}$;
- corrected oracle Monte Carlo MSE should match $R_n(x^*)$;
- old fixed-share should over-calibrate in the common fast-learning regime.

---

### Priority 3: adaptive estimator

Implement learning-curve estimation and adaptive grid minimization.

Sanity checks:

- $\hat\beta$ should be approximately unbiased in the clean Gaussian DGP;
- adaptive allocation should approach oracle allocation;
- safe fallback should activate in harmful regimes.

---

### Priority 4: multichannel synthetic experiment

Implement the two-channel risk surface and KKT diagnostics.

---

### Priority 5: semi-synthetic tabular benchmark

Start with a simple generator, then move to CTGAN/TVAE/TabDDPM if feasible.

---

### Priority 6: causal benchmark and coverage

Implement after the point-estimation experiments are stable.

---

## 15. Expected qualitative results

The paper should report whatever the simulations show. However, the theory predicts the following qualitative patterns.

| Regime | Corrected oracle | Corrected adaptive | Old fixed-share rule |
|---|---|---|---|
| fixed $m$ / persistent $v_n$ | bounded calibration or fallback | near oracle | over-calibrates |
| $m\asymp n$, $\beta>1/2$ | sublinear calibration, constant-factor MSE gain | near oracle | over-calibrates |
| $\beta<1/2$ | bounded calibration or fallback | safe if fallback works | often wastes real data |
| nearly noiseless synthetic data | boundary/full calibration | near oracle if boundary detected | may under-calibrate |
| parametric knife edge | constants determine allocation | sensitive to estimated constants | no universal 50/50 rule |

The key visual result should be:

$$
x_n^*\uparrow
\quad\text{but}\quad
x_n^*/n\downarrow
$$

in the common $m\asymp n$, $\beta>1/2$ regime.

---

## 16. Failure modes to log, not hide

The coding agent should explicitly log these cases.

1. **Nonmonotone empirical learning curves.** Real generators may not follow a clean power law at small $x$.
2. **Generator training failures.** Store failures and do not silently drop replications.
3. **Negative debiased squared-bias estimates.** Use the positive part but log the raw value.
4. **Multiple FOC roots.** Prefer grid minimization; log all roots if root solving is used.
5. **Boundary optima.** Do not force interior allocations.
6. **Unsafe synthetic augmentation.** Log safe fallback decisions.
7. **Coverage failure of naive intervals.** This is expected and scientifically useful.

---

## 17. Minimal reproducibility requirements

- Fix all random seeds and store them in output.
- Save configuration files for every run.
- Save raw long-format replication results before aggregation.
- Save aggregated tables separately.
- Save all figures as both `.pdf` and `.png`.
- Use the same estimator names across synthetic and semi-synthetic experiments.
- Do not hard-code theoretical results into simulation outputs; compute them through reusable functions.

---

## 18. One-page summary for the coding agent

Implement the corrected risk model:

$$
R_n(x)=\frac{\frac{a}{n-x}\left(v_n+cx^{-2\beta}\right)}
{\frac{a}{n-x}+v_n+cx^{-2\beta}}.
$$

Never use the old fixed-share rule as the proposed method. Use it only as a baseline.

The proposed method is:

1. estimate or define $B_n(x)$;
2. minimize $R_n(x)$ over a grid including boundaries;
3. apply safety check $xB_n(x)<a$;
4. compute $\alpha=B_n(x)/(V_R(x)+B_n(x))$;
5. combine real and synthetic estimators.

The most important experiment is the phase diagram. The most important plot is $\log x_n^*$ versus $\log n$ with the theoretical slope. The most important semi-synthetic result is an empirical learning curve plus MSE ratios showing that corrected allocation beats fixed-share and naive pooling while safely falling back when synthetic data is harmful.

