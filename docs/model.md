# Optimal Real-Data Allocation for Synthetic-Data-Augmented Inference

**Working title:** *Optimal Real-Data Allocation for Synthetic-Data-Augmented Inference*  
**Alternative title:** *How Much Real Data Should Be Spent Calibrating Synthetic Data?*

This document is a self-contained theory-and-methods draft for the repositioned NeurIPS paper. It replaces the fixed-share allocation claim with an exact marginal-value characterization and a phase diagram. The intended main message is:

> Real data has two uses: direct estimation and calibration of a synthetic-data generator. The optimal allocation is not a universal fixed fraction. It is the solution of a marginal-value equation and exhibits phase transitions. In the common regime where the number of synthetic observations grows proportionally with the number of real observations and the generator learns faster than the parametric threshold, the optimal number of calibration observations grows sublinearly while the optimal calibration share converges to zero.

---

## 1. Conceptual contribution

Foundation models and other generative models make synthetic data cheap to produce, but not automatically trustworthy. A researcher with scarce real data must decide how to use it. Some real observations can be retained for direct estimation; others can be used to improve the synthetic-data generator through fine-tuning, in-context examples, prompt calibration, validation, or debiasing.

This creates an allocation problem that is structurally analogous to Neyman allocation, but with a crucial difference. Classical Neyman allocation distributes observations across strata to reduce variance. Here, observations are distributed across *uses* of data: direct estimation reduces variance, while calibration reduces the bias of a synthetic estimator. The objective is therefore a joint bias-variance risk.

The paper's contribution is to characterize this allocation problem. The central theoretical result is an exact first-order condition:

$$
\left(v_n+c x^{-2\beta}\right)^2
=
2\beta a c x^{-(2\beta+1)},
$$

where $x$ is the number of real observations allocated to calibration, $a$ is the real-estimator variance constant, $v_n$ is the synthetic sampling variance, $c x^{-2\beta}$ is squared synthetic bias after calibration, and $\beta$ is the learning rate of the generator.

This equation implies a phase diagram. In particular, when $v_n\asymp v_0/n$, which corresponds to $m_n\asymp n$ synthetic observations, and $\beta>1/2$,

$$
x_n^*\asymp n^{2/(2\beta+1)},
\qquad
\lambda_n^* = x_n^*/n \to 0.
$$

Thus, the optimal calibration budget grows, but the optimal calibration share vanishes. This is the corrected replacement for any fixed-fraction allocation rule.

---

## 2. Setup

### 2.1 Real data, synthetic data, and estimand

Let $P^*$ be the true data-generating distribution on a measurable space $\mathcal X$. The target estimand is

$$
\theta^* = \theta(P^*) \in \mathbb R^d.
$$

The researcher observes

$$
X_1,\ldots,X_n \stackrel{i.i.d.}{\sim} P^*.
$$

The researcher also has access to a synthetic-data generator, denoted $G$, which can be calibrated using real data. Calibration may include fine-tuning, prompt construction, in-context examples, retrieval-augmented conditioning, or validation-based debiasing. The main theory focuses on a single calibration channel; multi-channel allocation is handled later.

Let

$$
x = n_f
$$

be the number of real observations allocated to generator calibration, and let

$$
n_e=n-x
$$

be the number of real observations retained for direct estimation. The calibration share is

$$
\lambda = x/n.
$$

After calibration on $x$ real observations, the generator induces a synthetic distribution $P_x$. The corresponding pseudo-true parameter is

$$
\theta_x = \theta(P_x),
$$

and the synthetic bias is

$$
b(x)=\theta_x-\theta^*.
$$

The researcher then draws

$$
Z_1,\ldots,Z_{m_n}\sim P_x
$$

synthetic observations and computes a synthetic estimator.

---

### 2.2 Direct real-data estimator

The real-data estimator is computed on the estimation subset of size $n_e=n-x$. In the scalar case, assume

$$
\hat\theta_R(x)-\theta^*
=
\frac{1}{n-x}\sum_{i\in E_x}\psi(X_i)+r_{R,n},
$$

where

$$
\mathbb E[\psi(X_i)]=0,
\qquad
\operatorname{Var}(\psi(X_i))=a,
\qquad
r_{R,n}=o_p((n-x)^{-1/2}).
$$

Thus

$$
V_R(x)=\operatorname{Var}(\hat\theta_R(x))
=
\frac{a}{n-x}+o((n-x)^{-1}).
$$

In the vector case, replace $a$ by a positive definite covariance matrix $\Sigma_R$.

---

### 2.3 Synthetic estimator

Conditional on the calibration data, the synthetic estimator satisfies

$$
\hat\theta_S(x)-\theta_x
=
\frac{1}{m_n}\sum_{j=1}^{m_n}\phi_x(Z_j)+r_{S,n},
$$

with

$$
\mathbb E[\phi_x(Z_j)\mid \mathcal F_x]=0,
\qquad
\operatorname{Var}(\phi_x(Z_j)\mid \mathcal F_x)=\sigma_S^2+o_p(1),
\qquad
r_{S,n}=o_p(m_n^{-1/2}).
$$

Hence the total synthetic error, relative to $\theta^*$, is

$$
B_n(x)
=
\operatorname{Var}(\hat\theta_S(x))+b(x)^2.
$$

The main power-law specification is

$$
b(x)^2 = c x^{-2\beta}\{1+o(1)\},
\qquad
c=B_0^2>0,
\qquad
\beta>0.
$$

Let

$$
v_n = \frac{\sigma_S^2}{m_n}.
$$

The canonical scalar working model is therefore

$$
B_n(x)=v_n+c x^{-2\beta}.
$$

When $m_n\asymp n^\rho$, we write

$$
v_n=v_0 n^{-\rho}\{1+o(1)\},
\qquad
v_0>0,
\qquad
\rho\ge 0.
$$

---

### 2.4 Combined estimator

For a fixed allocation $x$, define the linear combined estimator

$$
\hat\theta(\alpha,x)
=
\alpha\hat\theta_R(x)+(1-\alpha)\hat\theta_S(x),
\qquad
\alpha\in[0,1].
$$

Under sample splitting, the real estimation subset, calibration subset, and synthetic sampling randomness are conditionally independent. The scalar mean-squared error is

$$
\operatorname{MSE}_n(\alpha,x)
=
\alpha^2 V_R(x)+(1-\alpha)^2B_n(x),
$$

or, under the working model,

$$
\operatorname{MSE}_n(\alpha,x)
=
\alpha^2\frac{a}{n-x}
+(1-\alpha)^2\left(v_n+c x^{-2\beta}\right).
$$

This is the central equation of the paper.

---

## 3. Optimal mixing for a fixed allocation

### Proposition 1: Optimal endogenous shrinkage

For any fixed allocation $x\in(0,n)$, define

$$
V_R(x)=\frac{a}{n-x},
\qquad
B_n(x)=v_n+c x^{-2\beta}.
$$

The MSE-optimal weight on the real estimator is

$$
\alpha^*(x)
=
\frac{B_n(x)}{V_R(x)+B_n(x)}.
$$

The corresponding profiled risk is

$$
R_n(x)
=
\min_{\alpha\in[0,1]}\operatorname{MSE}_n(\alpha,x)
=
\frac{V_R(x)B_n(x)}{V_R(x)+B_n(x)}.
$$

Equivalently,

$$
R_n(x)
=
\frac{a B_n(x)}{a+B_n(x)(n-x)}.
$$

#### Key message

The synthetic estimator receives more weight when its effective error

$$
B_n(x)=\text{synthetic sampling variance}+\text{squared synthetic bias}
$$

is small relative to the real-estimator variance. Calibration decreases $B_n(x)$ but increases $V_R(x)$ by reducing the number of observations retained for direct estimation.

#### Proof sketch

For fixed $x$,

$$
\operatorname{MSE}_n(\alpha,x)
=
\alpha^2V_R(x)+(1-\alpha)^2B_n(x).
$$

Differentiate with respect to $\alpha$:

$$
\frac{\partial}{\partial\alpha}\operatorname{MSE}_n(\alpha,x)
=2\alpha V_R(x)-2(1-\alpha)B_n(x).
$$

Setting the derivative equal to zero gives

$$
\alpha^*(x)V_R(x)=(1-\alpha^*(x))B_n(x),
$$

which yields

$$
\alpha^*(x)=\frac{B_n(x)}{V_R(x)+B_n(x)}.
$$

The second derivative is $2\{V_R(x)+B_n(x)\}>0$, so this is the unique minimizer. Substitution gives the harmonic-risk formula.

---

### Vector version

When $\theta^*\in\mathbb R^d$, let

$$
\Sigma_R(x)=\frac{\Sigma_R}{n-x},
\qquad
\Sigma_B(x)=\frac{\Sigma_S}{m_n}+b(x)b(x)'.
$$

For a matrix-weighted estimator

$$
\hat\theta(W,x)=W\hat\theta_R(x)+(I-W)\hat\theta_S(x),
$$

the trace-risk-optimal weight is

$$
W^*(x)=\Sigma_B(x)\{\Sigma_R(x)+\Sigma_B(x)\}^{-1}
$$

when the covariance matrices commute or when the risk is written in the corresponding generalized least-squares norm. For the general noncommuting case, the same expression is the solution to the matrix first-order condition under symmetric positive definite weights; otherwise one can work componentwise, with a scalar loss for each estimand.

---

## 4. Corrected Generalized Neyman allocation

The outer problem chooses the calibration size $x$:

$$
x_n^*\in\arg\min_{0\le x\le n}R_n(x),
$$

where $x=0$ denotes no calibration and the real-data-only fallback is allowed.

Because $x$ is integer in finite samples, the implementation minimizes $R_n(x)$ over an integer grid. The theory treats $x$ as continuous for clarity and then rounds to the nearest admissible integer.

---

### Proposition 2: Exact first-order condition

Assume

$$
B_n(x)=v_n+c x^{-2\beta},
\qquad
V_R(x)=\frac{a}{n-x}.
$$

An interior stationary point $x\in(0,n)$ of the profiled risk satisfies

$$
\boxed{
\left(v_n+c x^{-2\beta}\right)^2
=
2\beta a c x^{-(2\beta+1)}.
}
$$

Equivalently,

$$
\boxed{
v_n^2x^{2\beta+1}
+2cv_nx
+c^2x^{1-2\beta}
=
2\beta ac.
}
$$

#### Key message

The factor $n-x$ cancels. Therefore, the optimal calibration size is not generally a fixed fraction of $n$. It is determined by the synthetic-error curve and the synthetic sampling variance.

#### Proof sketch

Using the profiled-risk expression

$$
R_n(x)=\frac{aB_n(x)}{a+B_n(x)(n-x)},
$$

differentiate with respect to $x$:

$$
R_n'(x)
=
\frac{a\{aB_n'(x)+B_n(x)^2\}}
{\{a+B_n(x)(n-x)\}^2}.
$$

Thus an interior stationary point satisfies

$$
aB_n'(x)+B_n(x)^2=0.
$$

Since

$$
B_n'(x)=-2\beta c x^{-(2\beta+1)},
$$

the first-order condition becomes

$$
B_n(x)^2
=
2\beta ac x^{-(2\beta+1)}.
$$

Substituting $B_n(x)=v_n+c x^{-2\beta}$ yields the displayed equation.

---

### Proposition 3: Exact safe-improvement condition

The optimally mixed estimator at calibration size $x$ improves on the real-data-only estimator using all $n$ real observations if and only if

$$
R_n(x)<\frac{a}{n}.
$$

Under the working model, this is equivalent to

$$
\boxed{
xB_n(x)<a.
}
$$

That is,

$$
\boxed{
x\left(v_n+c x^{-2\beta}\right)<a.
}
$$

#### Key message

A calibration allocation is safe only if the synthetic error times the number of real observations sacrificed for calibration is smaller than the real-estimator variance constant. This gives a simple finite-sample fallback rule:

$$
\text{use synthetic augmentation only if } \hat x\widehat B_n(\hat x)<\hat a.
$$

Otherwise return the all-real estimator.

#### Proof sketch

The real-only risk is $a/n$. The profiled risk is

$$
R_n(x)=\frac{aB_n(x)}{a+B_n(x)(n-x)}.
$$

Then

$$
R_n(x)<\frac{a}{n}
$$

if and only if

$$
\frac{B_n(x)}{a+B_n(x)(n-x)}<\frac1n.
$$

Cross-multiplying gives

$$
nB_n(x)<a+B_n(x)(n-x),
$$

which reduces to

$$
xB_n(x)<a.
$$

---

## 5. Main phase diagram

The first-order condition is exact, but its implications depend on how many synthetic observations are generated. Suppose

$$
v_n=v_0n^{-\rho}\{1+o(1)\},
\qquad v_0>0,
\qquad \rho\ge0.
$$

The exponent $\rho$ summarizes the synthetic sample-size regime. If $m_n\asymp n^\rho$, then $v_n\asymp n^{-\rho}$. The common case $m_n\asymp n$ corresponds to $\rho=1$.

Let $x_n^\circ$ denote an active interior local minimizer satisfying the exact first-order condition, when such a minimizer exists and is feasible. Let $x_n^*$ denote the global oracle decision after comparing the active solution with the boundary choices $x=0$ and $x=n$ and applying the safe-improvement condition.

---

### Theorem 1: Allocation phase diagram

Assume $B_n(x)=v_n+c x^{-2\beta}$, $V_R(x)=a/(n-x)$, $a,c,v_0>0$, and $v_n=v_0n^{-\rho}\{1+o(1)\}$. Then the oracle calibration allocation has the following asymptotic regimes.

#### Regime I: persistent synthetic variance

If $\rho=0$, so $v_n$ converges to a positive constant, every active interior solution is $O(1)$. Consequently,

$$
\lambda_n^*=x_n^*/n\to0,
$$

unless the safe-improvement rule selects no calibration at all.

**Message.** If synthetic sampling variance does not vanish, calibration has finite marginal value. A fixed positive calibration share is asymptotically wasteful.

---

#### Regime II: slow synthetic learning, $\beta<1/2$

If $\beta<1/2$, then every active interior solution is bounded in all safe regimes. In particular,

$$
\lambda_n^*\to0.
$$

Even with negligible synthetic sampling variance, full calibration gives squared bias $c n^{-2\beta}$, which is asymptotically larger than the real-only risk $a/n$. Therefore synthetic augmentation cannot improve the first-order rate over the real-data estimator.

**Message.** Slow-learning generators may provide finite-sample shrinkage gains, but they should not receive a positive asymptotic share of real data.

---

#### Regime III: common sublinear-calibration regime, $\beta>1/2$ and $0<\rho<\beta+1/2$

If

$$
\beta>1/2,
\qquad
0<\rho<\beta+1/2,
$$

then the active interior solution is feasible and satisfies

$$
\boxed{
x_n^\circ
\sim
\left(\frac{2\beta ac}{v_0^2}\right)^{1/(2\beta+1)}
n^{2\rho/(2\beta+1)}.
}
$$

Therefore

$$
\boxed{
\lambda_n^\circ
=\frac{x_n^\circ}{n}
\asymp
n^{2\rho/(2\beta+1)-1}
\to0.
}
$$

In the practically important case $m_n\asymp n$, so $\rho=1$, this becomes

$$
\boxed{
x_n^\circ\asymp n^{2/(2\beta+1)},
\qquad
\lambda_n^\circ\asymp n^{-(2\beta-1)/(2\beta+1)}\to0.
}
$$

**Message.** Calibration grows, but sublinearly. This is the main corrected allocation law.

---

#### Regime IV: boundary/full-calibration regime, $\beta>1/2$ and $\rho>\beta+1/2$

If

$$
\beta>1/2,
\qquad
\rho>\beta+1/2,
$$

then the interior solution predicted by Regime III lies beyond the feasible range $x\le n$. The global oracle moves to the boundary:

$$
\lambda_n^*\to1.
$$

The synthetic-only risk after full calibration is

$$
B_n(n)=v_n+c n^{-2\beta}=o(n^{-1}),
$$

which is asymptotically smaller than the real-only risk $a/n$.

**Message.** Majority or full calibration is possible, but only when the generator learns fast and synthetic sampling variance vanishes sufficiently quickly.

---

#### Regime V: the parametric knife edge, $\beta=1/2$

When $\beta=1/2$, constants matter. The exact first-order condition becomes

$$
\left(v_n+c/x\right)^2=ac/x^2.
$$

Equivalently,

$$
v_nx+c=\sqrt{ac}.
$$

A positive active solution exists only when $a>c$. If $a\le c$, the safe-improvement condition fails asymptotically and the all-real estimator is first-order optimal.

If $a>c$ and $v_n=v_0n^{-\rho}\{1+o(1)\}$, then

$$
x_n^\circ
\sim
\frac{\sqrt{ac}-c}{v_0}n^\rho.
$$

Thus:

- if $0\le\rho<1$, then $x_n^\circ=o(n)$ and $\lambda_n^\circ\to0$;
- if $\rho=1$, the optimal share can converge to a nonzero constant depending on $a,c,v_0$;
- if $\rho>1$, the oracle moves to full calibration when $c<a$.

**Message.** The parametric-rate case is not governed by a universal one-half allocation. It is a constants-driven knife edge.

---

### Proof sketch for Theorem 1

Start from the exact first-order condition in Proposition 2 and multiply by $x^{2\beta+1}$:

$$
v_n^2x^{2\beta+1}+2cv_nx+c^2x^{1-2\beta}=2\beta ac.
$$

This single equation determines the asymptotic regimes.

1. **Persistent synthetic variance.** If $v_n\to v_0>0$, the equation contains no diverging scale in $n$. Active roots remain bounded, so $x_n^*=O(1)$ and $x_n^*/n\to0$.

2. **Slow learning.** If $\beta<1/2$, the term $c^2x^{1-2\beta}$ increases with $x$. Thus an active solution cannot diverge with $n$ without violating the equation; active roots remain bounded. Boundary full calibration is also not optimal because $B_n(n)\ge c n^{-2\beta}\gg n^{-1}$, while the real-only risk is $a/n$.

3. **Fast learning with feasible interior solution.** If $\beta>1/2$ and $x\to\infty$, the terms $2cv_nx$ and $c^2x^{1-2\beta}$ are negligible at the solution when $0<\rho<\beta+1/2$. The equation reduces to

   $$
   v_n^2x^{2\beta+1}\sim2\beta ac.
   $$

   Since $v_n\sim v_0n^{-\rho}$,

   $$
   x_n^\circ
   \sim
   \left(\frac{2\beta ac}{v_0^2}\right)^{1/(2\beta+1)}
   n^{2\rho/(2\beta+1)}.
   $$

   The condition $0<\rho<\beta+1/2$ is exactly the condition under which this root is $o(n)$.

4. **Fast learning with infeasible interior solution.** If $\rho>\beta+1/2$, the same root would be larger than $n$. At $x=n$, both $v_n$ and $c n^{-2\beta}$ are $o(n^{-1})$, so the full-calibration synthetic risk beats the all-real risk. The oracle therefore moves to the boundary.

5. **Parametric knife edge.** Setting $\beta=1/2$ reduces the first-order condition to $(v_n+c/x)^2=ac/x^2$. Taking positive square roots gives $v_nx+c=\sqrt{ac}$, yielding the stated cases.

The finite-sample oracle is obtained by comparing the active root with the boundary risks and applying the exact safe-improvement condition $xB_n(x)<a$.

---

## 6. Risk consequences

The allocation phase diagram also implies a risk phase diagram. In the fast-learning interior regime $\beta>1/2$ and $0<\rho<\beta+1/2$,

$$
B_n(x_n^\circ)\sim v_0n^{-\rho},
\qquad
V_R(x_n^\circ)\sim a/n.
$$

Therefore

$$
R_n(x_n^\circ)
\sim
\frac{(a/n)(v_0n^{-\rho})}{a/n+v_0n^{-\rho}}.
$$

This yields three subcases:

1. If $\rho<1$, then $B_n(x_n^\circ)\gg V_R(x_n^\circ)$ and

   $$
   R_n(x_n^\circ)\sim a/n.
   $$

   Synthetic augmentation gives second-order gains but does not change the first-order rate.

2. If $\rho=1$, then $B_n(x_n^\circ)$ and $V_R(x_n^\circ)$ are of the same order and

   $$
   R_n(x_n^\circ)
   \sim
   \frac{av_0}{a+v_0}\frac1n.
   $$

   Synthetic augmentation gives a first-order constant-factor improvement.

3. If $\rho>1$, then $B_n(x_n^\circ)\ll V_R(x_n^\circ)$ and

   $$
   R_n(x_n^\circ)\sim v_0n^{-\rho}.
   $$

   Synthetic augmentation improves the convergence rate, provided the interior solution remains feasible.

---

## 7. Multi-channel allocation

The single-channel model can be extended to multiple calibration channels. Let $x_k$ be the number of real observations allocated to channel $k$, and let

$$
X=\sum_{k=1}^K x_k,
\qquad
n_e=n-X.
$$

Assume separable synthetic error

$$
B_n(x_1,\ldots,x_K)
=
v_n+
\sum_{k=1}^K c_kx_k^{-2\beta_k}.
$$

The profiled risk is

$$
R_n(\mathbf x)
=
\frac{aB_n(\mathbf x)}{a+B_n(\mathbf x)(n-\sum_kx_k)}.
$$

---

### Proposition 4: Multi-channel KKT conditions

For each active calibration channel $k$ with $x_k>0$, an interior optimum satisfies

$$
\boxed{
B_n(\mathbf x)^2
=
2\beta_kac_kx_k^{-(2\beta_k+1)}.
}
$$

Equivalently, conditional on the common synthetic error level $B$, the active allocation to channel $k$ is

$$
\boxed{
x_k(B)
=
\left(\frac{2\beta_kac_k}{B^2}\right)^{1/(2\beta_k+1)}.
}
$$

The common error level solves the scalar fixed-point equation

$$
\boxed{
B
=
v_n+
\sum_{k\in\mathcal A}
 c_kx_k(B)^{-2\beta_k},
}
$$

where $\mathcal A$ is the active set. Inactive channels satisfy the corresponding KKT inequality: their marginal risk reduction is no larger than the common shadow value at zero or at the lower admissible allocation.

#### Key message

The multi-channel rule is still an equi-marginal principle, but it does not imply fixed allocation shares. Active channels equalize marginal MSE reduction. Faster-learning channels can receive fewer observations if they require fewer examples to reach the common marginal value.

#### Proof sketch

The derivative of the profiled risk with respect to $x_k$ is

$$
\frac{\partial R_n(\mathbf x)}{\partial x_k}
=
\frac{a\{a\partial_kB_n(\mathbf x)+B_n(\mathbf x)^2\}}
{\{a+B_n(\mathbf x)(n-\sum_jx_j)\}^2}.
$$

Since

$$
\partial_kB_n(\mathbf x)
=-2\beta_kc_kx_k^{-(2\beta_k+1)},
$$

an active interior optimum satisfies

$$
B_n(\mathbf x)^2=2\beta_kac_kx_k^{-(2\beta_k+1)}.
$$

Solving this equation for $x_k$ as a function of $B$ gives the displayed fixed-point representation.

---

## 8. Adaptive corrected Generalized Neyman estimator

The oracle allocation depends on unknown quantities: $a$, $v_n$, $c$, and $\beta$. The practical method estimates the risk curve and solves the corrected first-order condition or the discrete risk minimization problem directly.

---

### Algorithm 1: Corrected Generalized Neyman allocation

**Inputs:** real data $X_1,\ldots,X_n$; synthetic generator $G$; candidate calibration grid $\mathcal G_n\subset\{1,\ldots,n-1\}$; synthetic sample size $m_n$; number of folds $K$.

**Step 1: Split data.**  
Partition real data into calibration-pilot, validation, and final-estimation folds. In the final implementation, use $K$-fold cross-fitting so each observation rotates through these roles.

**Step 2: Estimate the real-estimator variance constant.**  
Using real-data influence values or nonparametric bootstrap, estimate

$$
\hat a\approx a.
$$

For a sample mean, $\hat a$ is the sample variance of the real outcome. For a general asymptotically linear estimator, $\hat a$ is the empirical variance of the estimated influence function.

**Step 3: Estimate the synthetic variance.**  
For each calibration size $x_j\in\mathcal G_n$, calibrate the generator, draw synthetic data, and estimate

$$
\hat v_n(x_j)\approx \operatorname{Var}(\hat\theta_S(x_j)).
$$

If the same synthetic sample size $m_n$ is used at all $x_j$, one can pool synthetic variance estimates and use a common $\hat v_n$.

**Step 4: Estimate the synthetic bias curve.**  
For each $x_j$, compute a synthetic estimate $\hat\theta_S(x_j)$ and an independent held-out real estimate $\hat\theta_R^V$. Estimate squared bias by

$$
\widehat{b^2}(x_j)
=
\left[
\{\hat\theta_S(x_j)-\hat\theta_R^V\}^2
-\widehat{\operatorname{Var}}\{\hat\theta_S(x_j)\}
-
\widehat{\operatorname{Var}}\{\hat\theta_R^V\}
\right]_+.
$$

Then either fit the power law

$$
\log\widehat{b^2}(x_j)
=
\log c-2\beta\log x_j+\epsilon_j,
$$

or use a shape-constrained smoothed curve $\widehat B_n(x)$ directly.

**Step 5: Solve the corrected allocation problem.**  
Either solve the estimated first-order equation

$$
\left(\hat v_n+\hat c x^{-2\hat\beta}\right)^2
=
2\hat\beta\hat a\hat c x^{-(2\hat\beta+1)},
$$

or minimize the estimated profiled risk over the grid:

$$
\hat x
\in
\arg\min_{x\in\mathcal G_n}
\widehat R_n(x),
\qquad
\widehat R_n(x)
=
\frac{\widehat V_R(x)\widehat B_n(x)}{\widehat V_R(x)+\widehat B_n(x)}.
$$

The grid-minimization version is preferred in code because it handles boundary cases automatically.

**Step 6: Apply the safe fallback.**  
Use synthetic augmentation only if

$$
\hat x\widehat B_n(\hat x)<\hat a.
$$

If this condition fails, return the real-data-only estimator based on all $n$ real observations.

**Step 7: Compute the mixing weight.**  
For the selected allocation, compute

$$
\hat\alpha
=
\frac{\widehat B_n(\hat x)}{\widehat V_R(\hat x)+\widehat B_n(\hat x)}.
$$

Then return

$$
\hat\theta_{GN}
=
\hat\alpha\hat\theta_R(\hat x)+(1-\hat\alpha)\hat\theta_S(\hat x).
$$

**Step 8: Cross-fit.**  
Repeat the entire procedure over folds and average the resulting fold-specific estimates. Cross-fitting ensures that the observations used to estimate the bias curve and choose the allocation are separated from the observations used for the final real-data estimating equation.

---

### Theorem 2: Adaptive oracle inequality

Let $\mathcal G_n$ be the allocation grid, and let

$$
x_n^\star\in\arg\min_{x\in\mathcal G_n\cup\{0,n\}}R_n(x)
$$

be the oracle grid allocation, including the real-only and synthetic-only boundary choices. Let

$$
\hat x_n\in\arg\min_{x\in\mathcal G_n\cup\{0,n\}}\widehat R_n(x)
$$

be the estimated allocation after applying the same boundary and safety rules.

Assume the estimated risk is uniformly relatively consistent on the grid:

$$
\Delta_n
=
\sup_{x\in\mathcal G_n\cup\{0,n\}}
\left|
\frac{\widehat R_n(x)}{R_n(x)}-1
\right|
\stackrel p\to0.
$$

Then

$$
\boxed{
\frac{R_n(\hat x_n)}{R_n(x_n^\star)}
\stackrel p\to1.
}
$$

If the grid approximates the continuous oracle allocation to relative risk $1+o(1)$, then the adaptive estimator is also continuous-oracle efficient:

$$
\frac{R_n(\hat x_n)}{\inf_{0\le x\le n}R_n(x)}
\stackrel p\to1.
$$

#### Proof sketch

On the event $\Delta_n\le\delta$, for every grid point $x$,

$$
(1-\delta)R_n(x)
\le
\widehat R_n(x)
\le
(1+\delta)R_n(x).
$$

Because $\hat x_n$ minimizes $\widehat R_n$,

$$
\widehat R_n(\hat x_n)
\le
\widehat R_n(x_n^\star).
$$

Therefore

$$
(1-\delta)R_n(\hat x_n)
\le
(1+\delta)R_n(x_n^\star),
$$

so

$$
\frac{R_n(\hat x_n)}{R_n(x_n^\star)}
\le
\frac{1+\delta}{1-\delta}.
$$

The reverse inequality is immediate from the oracle definition. Since $\delta=o_p(1)$, the ratio converges to one.

---

### Corollary 1: Plug-in consistency under power-law estimation

Suppose the adaptive method estimates the power-law parameters and variance constants such that

$$
\hat a/a\to_p1,
\qquad
\hat v_n/v_n\to_p1,
\qquad
\hat c/c\to_p1,
\qquad
\hat\beta-\beta=o_p(1/\log n).
$$

In the fast-learning interior regime $\beta>1/2$ and $0<\rho<\beta+1/2$,

$$
\frac{\hat x_n}{x_n^\circ}\to_p1
$$

and

$$
\frac{R_n(\hat x_n)}{R_n(x_n^\circ)}\to_p1.
$$

#### Proof sketch

In this regime,

$$
x_n^\circ
=
\left(\frac{2\beta ac}{v_0^2}\right)^{1/(2\beta+1)}n^{2\rho/(2\beta+1)}\{1+o(1)\}.
$$

The estimated allocation has the same form with plug-in constants. Consistency of $\hat a,\hat c,\hat v_n$ controls the multiplicative constant. The condition $\hat\beta-\beta=o_p(1/\log n)$ controls the exponent because allocation depends on $n^{2\rho/(2\beta+1)}$. A Taylor expansion of the exponent delivers $\hat x_n/x_n^\circ\to_p1$.

---

## 9. Distribution theory and inference

The MSE-optimal estimator is designed for point estimation. Valid confidence intervals require additional care because synthetic bias may be non-negligible.

For a fixed oracle allocation and mixing weight,

$$
\hat\theta(\alpha^*,x^*)
-
\theta^*
=
\alpha^*\{\hat\theta_R(x^*)-\theta^*\}
+(1-\alpha^*)\{\hat\theta_S(x^*)-\theta_{x^*}\}
+(1-\alpha^*)b(x^*).
$$

The stochastic variance is

$$
\Omega_n(x^*)
=
\alpha^{*2}\frac{a}{n-x^*}
+(1-\alpha^*)^2v_n.
$$

---

### Proposition 5: Centered asymptotic normality

Under the asymptotic linearity and conditional independence assumptions,

$$
\frac{
\hat\theta(\alpha^*,x^*)-
\theta^*-(1-\alpha^*)b(x^*)
}{
\sqrt{\Omega_n(x^*)}
}
\Rightarrow N(0,1).
$$

The same result holds for the adaptive cross-fitted estimator if the estimated allocation and mixing weight are oracle equivalent in the sense of Theorem 2.

#### Proof sketch

Conditional on the calibration data, the centered real and synthetic estimators are independent asymptotically linear statistics. Their weighted sum is therefore asymptotically normal with variance $\Omega_n(x^*)$. Oracle equivalence of the adaptive allocation allows Slutsky's theorem to replace oracle weights by estimated weights.

---

### Corollary 2: Conditions for ordinary Wald inference around $\theta^*$

A Wald interval centered at $\hat\theta(\alpha^*,x^*)$ is valid for $\theta^*$ if

$$
(1-\alpha^*)b(x^*)
=
o\left(\sqrt{\Omega_n(x^*)}\right).
$$

If this condition fails, the MSE-optimal estimator remains a good point estimator, but ordinary confidence intervals under-cover. Inference then requires one of the following:

1. **Undersmoothing:** allocate additional calibration data so that synthetic bias is negligible relative to standard error;
2. **Validation debiasing:** estimate $b(x)$ on held-out real data and subtract it;
3. **Prediction-powered or orthogonal correction:** use real data to correct the synthetic estimating equation;
4. **Honest bias-aware intervals:** report intervals that include an estimated or bounded bias term.

---

## 10. Validation-debiased extension

A validation-corrected version introduces a third use of real data. Let

$$
n=n_e+n_f+n_v.
$$

Use $n_f$ observations to calibrate the generator, $n_v$ observations to estimate its bias, and $n_e$ observations for direct estimation.

For a scalar estimand, define a validation bias estimate

$$
\hat b_V(x)
=
\hat\theta_S^V(x)-\hat\theta_R^V,
$$

where $\hat\theta_S^V(x)$ is the synthetic analogue of the validation estimand and $\hat\theta_R^V$ is the real validation estimate. Then define

$$
\tilde\theta_S(x)
=
\hat\theta_S(x)-\hat b_V(x).
$$

The effective synthetic error becomes approximately

$$
\widetilde B_n(x,n_v)
=
v_n+
\operatorname{Var}\{\hat b_V(x)\}
+
\text{higher-order residual bias}.
$$

The corresponding allocation problem is

$$
\min_{n_e+n_f+n_v=n}
\frac{V_R(n_e)\widetilde B_n(n_f,n_v)}
{V_R(n_e)+\widetilde B_n(n_f,n_v)}.
$$

This extension is useful for the journal version and for the coverage experiments. For the NeurIPS version, it can be presented as an inference-safe variant rather than the main estimator.

---

## 11. Main paper flow

A camera-ready NeurIPS version should follow this structure.

### Section 1: Introduction

- Synthetic data is useful but biased.
- Real data is scarce and has dual value: estimation and calibration.
- The paper asks how to allocate real observations across these uses.
- The answer is a corrected Generalized Neyman allocation with a phase diagram.
- Main surprise: in the common $m_n\asymp n$ and $\beta>1/2$ regime, $x_n^*$ grows but $x_n^*/n\to0$.

### Section 2: Setup

- Define $P^*$, $\theta^*$, real estimator, calibrated generator, synthetic estimator.
- Present the central scalar MSE equation.
- Explain $B_n(x)=v_n+c x^{-2\beta}$ as effective synthetic error.

### Section 3: Optimal mixing and allocation

- Proposition 1: optimal mixing for fixed allocation.
- Proposition 2: exact first-order condition.
- Proposition 3: safe-improvement condition.

### Section 4: Phase diagram

- Theorem 1: allocation regimes.
- Main figure: phase diagram in $(\beta,\rho)$ space.
- Corollary: risk regimes and MSE improvement.

### Section 5: Adaptive estimator

- Algorithm 1: corrected plug-in/grid allocation.
- Theorem 2: adaptive oracle inequality.
- Corollary: plug-in consistency under power-law estimation.

### Section 6: Experiments

- Synthetic phase-diagram validation.
- Adaptive estimator and safety fallback.
- Multi-channel allocation.
- Semi-synthetic tabular and/or causal benchmark.

### Section 7: Discussion

- Point estimation versus inference.
- Relation to prediction-powered inference, double/debiased ML, transfer learning, and optimal design.
- Limitations: learning-curve estimation, generator instability, privacy, multiple estimands, sequential allocation.

---

## 12. Claims to make and claims to avoid

### Claims to make

1. The optimal calibration allocation solves an exact marginal-value equation.
2. The allocation exhibits phase transitions depending on $\beta$ and $\rho$.
3. In the common $m_n\asymp n$, $\beta>1/2$ regime, calibration grows sublinearly.
4. Fixed positive calibration shares are generally not asymptotically optimal in that common regime.
5. Safe fallback is available through the condition $xB_n(x)<a$.
6. MSE-optimal point estimation and valid inference are distinct problems.

### Claims to avoid

1. Do not claim a universal fixed share such as $2\beta/(1+2\beta)$.
2. Do not claim that faster learning always means a larger share of real data goes to calibration.
3. Do not claim synthetic data always improves first-order risk for every $\beta>0$.
4. Do not claim cross-fitting alone removes synthetic bias.
5. Do not overstate the James-Stein analogy; use it as intuition, not as the formal contribution.

---

## 13. Notation summary

| Symbol | Meaning |
|---|---|
| $n$ | number of real observations |
| $x=n_f$ | number of real observations used for generator calibration |
| $n_e=n-x$ | number of real observations used for direct estimation |
| $m_n$ | number of synthetic observations generated |
| $\theta^*$ | target estimand under the real distribution |
| $\theta_x$ | pseudo-true estimand under the calibrated synthetic distribution |
| $b(x)=\theta_x-\theta^*$ | synthetic bias after calibration size $x$ |
| $a$ | real-estimator variance constant |
| $v_n=\sigma_S^2/m_n$ | synthetic sampling variance |
| $c=B_0^2$ | squared bias constant |
| $\beta$ | generator bias-decay exponent |
| $\rho$ | synthetic variance decay exponent, $v_n\asymp v_0n^{-\rho}$ |
| $B_n(x)=v_n+cx^{-2\beta}$ | effective synthetic error |
| $V_R(x)=a/(n-x)$ | real-estimator variance after calibration allocation |
| $\alpha$ | weight on real estimator |
| $R_n(x)$ | profiled MSE after optimizing over $\alpha$ |

---

## 14. Minimal theorem package for the NeurIPS submission

The main paper can contain exactly the following formal results.

1. **Proposition 1: Optimal endogenous shrinkage**

   $$
   \alpha^*(x)=\frac{B_n(x)}{V_R(x)+B_n(x)},
   \qquad
   R_n(x)=\frac{V_R(x)B_n(x)}{V_R(x)+B_n(x)}.
   $$

2. **Proposition 2: Corrected Generalized Neyman condition**

   $$
   \left(v_n+c x^{-2\beta}\right)^2
   =
   2\beta acx^{-(2\beta+1)}.
   $$

3. **Proposition 3: Safe-improvement condition**

   $$
   R_n(x)<a/n
   \quad\Longleftrightarrow\quad
   xB_n(x)<a.
   $$

4. **Theorem 1: Phase diagram**

   If $\beta>1/2$ and $0<\rho<\beta+1/2$,

   $$
   x_n^*
   \asymp n^{2\rho/(2\beta+1)},
   \qquad
   x_n^*/n\to0.
   $$

   Include boundary and knife-edge cases in the theorem statement or appendix.

5. **Theorem 2: Adaptive oracle inequality**

   Uniform relative consistency of the estimated risk curve implies

   $$
   R_n(\hat x_n)/R_n(x_n^*)\to_p1.
   $$

6. **Proposition 5: Centered asymptotic normality**

   The estimator is asymptotically normal around $\theta^*+(1-\alpha^*)b(x^*)$; ordinary inference around $\theta^*$ requires negligible or corrected bias.

---

## 15. References to cite in the full paper

The final paper should cite at least the following literatures.

- Neyman allocation and stratified sampling: Neyman (1934), Cochran (1977).
- Shrinkage and estimator combination: Stein (1956), James and Stein (1961), Bates and Granger (1969), Hansen (2007).
- Cross-fitting and double/debiased ML: Chernozhukov et al. (2018), Newey and Robins (2018).
- Prediction-powered inference and cross-prediction-powered inference: Angelopoulos et al. (2023), Zrnic and Candès (2024).
- Transfer learning and proxy prediction: Pan and Yang (2010), Bastani (2021), Li, Cai, and Li (2022).
- Neural scaling laws and power-law learning curves: Kaplan et al. (2020), Hoffmann et al. (2022).
- Synthetic tabular data generation: Xu et al. (2019), Kotelnikov et al. (2023).
- Semiparametric efficiency and adaptive estimation: Bickel, Klaassen, Ritov, and Wellner (1993), van der Vaart (1998).

