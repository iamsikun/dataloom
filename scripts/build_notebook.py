"""Generate notebooks/exp_analysis.ipynb covering all six experiments.

Builds the figures and tables required by docs/experiments.md §4.7-§9.8 plus
the §12-§13 master figures/tables. Each code cell saves outputs alongside the
existing analysis directory and inline-displays them.

Usage:
    uv run python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "exp_analysis.ipynb"


def cell_md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def cell_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []


# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------

CELLS.append(cell_md(
    "# `dataloom` — full-experiment analysis notebook\n"
    "\n"
    "Builds every figure and table required by `docs/experiments.md` §4.7-§9.8\n"
    "(Experiments 1, 2, 3, 4, A, B) plus the §12-§13 master plots/tables.\n"
    "\n"
    "Outputs are persisted to each run's `figures/` and `aggregated/` directory\n"
    "as `.pdf` + `.png` per §17.\n"
    "\n"
    "Anywhere empirical results diverge from theory predictions, the cell is\n"
    "annotated **explicitly** — not smoothed. See §16.\n"
))

CELLS.append(cell_code(
    "%run __dev_setup.py\n"
))

CELLS.append(cell_code(
    "from pathlib import Path\n"
    "import json\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib as mpl\n"
    "\n"
    "from dataloom.notation import classify_regime, theory_slope\n"
    "from dataloom.oracle import (\n"
    "    B_eff, V_real, R_profile, foc_residual, safe_condition,\n"
    "    oracle_grid, m_of_n, v_of_n,\n"
    ")\n"
    "from dataloom.io.results import read_run\n"
    "\n"
    "mpl.rcParams['figure.dpi'] = 110\n"
    "mpl.rcParams['savefig.bbox'] = 'tight'\n"
))

# ---------------------------------------------------------------------------
# 1. Discover run dirs
# ---------------------------------------------------------------------------

CELLS.append(cell_md(
    "## 0. Discover run directories\n"
    "\n"
    "Pick the most recent run directory for each experiment. If a fresh full\n"
    "run is desired, re-run the corresponding `scripts/run_experiment.py`\n"
    "config first.\n"
))

CELLS.append(cell_code(
    "RESULTS = Path('../results').resolve()\n"
    "\n"
    "def latest_run(prefix: str) -> Path:\n"
    "    candidates = [p for p in RESULTS.glob(f'{prefix}*') if p.is_dir()]\n"
    "    if not candidates:\n"
    "        raise FileNotFoundError(f'no run dir under results/ matching {prefix!r}')\n"
    "    return max(candidates, key=lambda p: p.stat().st_mtime)\n"
    "\n"
    "RUN_EXP1 = latest_run('exp1_phase_diagram')\n"
    "RUN_EXP2 = latest_run('exp2_adaptive')\n"
    "RUN_EXP3 = latest_run('exp3_multichannel')\n"
    "RUN_EXP4 = latest_run('exp4_inference')\n"
    "RUN_EXPA = latest_run('expA_tabular')\n"
    "RUN_EXPB = latest_run('expB_causal')\n"
    "\n"
    "for label, p in [('exp1', RUN_EXP1), ('exp2', RUN_EXP2),\n"
    "                  ('exp3', RUN_EXP3), ('exp4', RUN_EXP4),\n"
    "                  ('expA', RUN_EXPA), ('expB', RUN_EXPB)]:\n"
    "    n_parts = sum(1 for _ in (p / 'raw').glob('cell=*/part-*.parquet'))\n"
    "    print(f'{label}: {p.name}  ({n_parts} parquet parts)')\n"
))

CELLS.append(cell_code(
    "def save_fig(fig, run_dir: Path, stem: str) -> None:\n"
    "    out_dir = run_dir / 'figures'\n"
    "    out_dir.mkdir(parents=True, exist_ok=True)\n"
    "    fig.savefig(out_dir / f'{stem}.pdf', bbox_inches='tight')\n"
    "    fig.savefig(out_dir / f'{stem}.png', dpi=150, bbox_inches='tight')\n"
    "    print(f'wrote {out_dir / stem}.pdf + .png')\n"
))

# ---------------------------------------------------------------------------
# 2. Exp 1 — phase diagram
# ---------------------------------------------------------------------------

CELLS.append(cell_md(
    "## 1. Experiment 1 — phase diagram validation (§4)\n"
    "\n"
    "Estimators:\n"
    "`real_only_all`, `synthetic_only_full_calibration`, `naive_pooling`,\n"
    "`fixed_half_split_oracle_alpha`, `old_fixed_share_oracle_alpha`,\n"
    "`corrected_oracle_gn`, `safe_corrected_oracle_gn`.\n"
))

CELLS.append(cell_code(
    "raw1 = read_run(RUN_EXP1 / 'raw')\n"
    "table_s1 = pd.read_parquet(RUN_EXP1 / 'aggregated' / 'table_s1_scaling.parquet')\n"
    "table_s2 = pd.read_parquet(RUN_EXP1 / 'aggregated' / 'table_s2_estimator_perf.parquet')\n"
    "print(f'rows: {len(raw1):,}; table_s1 shape {table_s1.shape}; table_s2 shape {table_s2.shape}')\n"
    "table_s1\n"
))

# Figure S1: phase-diagram heatmap
CELLS.append(cell_md(
    "### Figure S1 — Allocation phase diagram (§4.7)\n"
    "\n"
    "Heatmap colored by theoretical regime per (β, ρ). Markers indicate which\n"
    "(β, ρ) cells were actually simulated.\n"
))

CELLS.append(cell_code(
    "betas = sorted(raw1['beta'].unique())\n"
    "rhos = sorted(raw1['rho'].unique())\n"
    "REGIME_COLORS = {\n"
    "    'persistent_variance':       '#999999',\n"
    "    'slow_learning':             '#1f77b4',\n"
    "    'parametric_knife_edge':     '#ff7f0e',\n"
    "    'sublinear_fast_learning':   '#2ca02c',\n"
    "    'boundary_full_calibration': '#d62728',\n"
    "    'boundary_knife_edge':       '#9467bd',\n"
    "}\n"
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "for b in betas:\n"
    "    for r in rhos:\n"
    "        regime = classify_regime(b, r)\n"
    "        ax.scatter(b, r, s=320, c=REGIME_COLORS[regime],\n"
    "                   marker='s', edgecolor='black', linewidth=0.5)\n"
    "handles = [plt.Line2D([0],[0], marker='s', color='w',\n"
    "                       markerfacecolor=c, markersize=14, label=k)\n"
    "           for k, c in REGIME_COLORS.items()]\n"
    "ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',\n"
    "          fontsize=8, frameon=False)\n"
    "ax.set_xlabel(r'$\\beta$ (synthetic learning rate)')\n"
    "ax.set_ylabel(r'$\\rho$ (synthetic-variance decay)')\n"
    "ax.set_title('Figure S1: phase diagram (regime per simulated cell)')\n"
    "ax.grid(True, alpha=0.3)\n"
    "save_fig(fig, RUN_EXP1, 'figure_s1_phase_diagram')\n"
    "plt.show()\n"
))

# Figure S2 + Table S1
CELLS.append(cell_md(
    "### Figure S2 — Calibration size scaling (§4.7)\n"
    "\n"
    "Recommended panels per spec: (β=1, ρ=0), (β=0.4, ρ=1), (β=1, ρ=1),\n"
    "(β=1, ρ=2). Theoretical slope overlaid.\n"
))

CELLS.append(cell_code(
    "cell_oracle = (raw1.groupby(['beta', 'rho', 'n'], observed=True)['oracle_x']\n"
    "                  .first().reset_index())\n"
    "panels = [\n"
    "    (1.0, 0.0, 'fixed m (rho=0)'),\n"
    "    (0.4, 1.0, 'slow learner (β=0.4, ρ=1)'),\n"
    "    (1.0, 1.0, 'common fast learner (β=1, ρ=1)'),\n"
    "    (1.0, 2.0, 'boundary regime (β=1, ρ=2)'),\n"
    "]\n"
    "fig, axes = plt.subplots(2, 2, figsize=(11, 8))\n"
    "for ax, (b, r, title) in zip(axes.flat, panels):\n"
    "    sub = cell_oracle[(cell_oracle['beta']==b) & (cell_oracle['rho']==r)]\n"
    "    sub = sub[sub['oracle_x'] > 0].sort_values('n')\n"
    "    if sub.empty:\n"
    "        ax.set_title(f'{title}\\n(no interior points)')\n"
    "        continue\n"
    "    ax.loglog(sub['n'], sub['oracle_x'], 'o-', label=r'empirical $x_n^*$')\n"
    "    sl = theory_slope(b, r)\n"
    "    if sl is not None and sl > 0:\n"
    "        n_arr = np.array(sub['n'])\n"
    "        ref = sub['oracle_x'].iloc[0] / (n_arr[0] ** sl)\n"
    "        ax.loglog(n_arr, ref * n_arr ** sl, '--',\n"
    "                  label=f'theory slope = {sl:.3f}')\n"
    "    ax.set_xlabel('n')\n"
    "    ax.set_ylabel(r'$x_n^*$')\n"
    "    ax.set_title(title, fontsize=10)\n"
    "    ax.legend(fontsize=8)\n"
    "    ax.grid(True, which='both', alpha=0.3)\n"
    "fig.suptitle('Figure S2: oracle calibration size scaling')\n"
    "fig.tight_layout(rect=(0, 0, 1, 0.96))\n"
    "save_fig(fig, RUN_EXP1, 'figure_s2_calibration_scaling')\n"
    "plt.show()\n"
))

CELLS.append(cell_md(
    "#### Table S1 — predicted vs empirical scaling (§4.8)\n"
    "\n"
    "Annotated `(disagrees with theory)` if the empirical slope is outside\n"
    "theory ± 3·SE for any cell with a defined theoretical slope.\n"
))

CELLS.append(cell_code(
    "ts1 = table_s1.copy()\n"
    "def annotate(row):\n"
    "    if row['theory_slope'] is None or np.isnan(row['theory_slope']):\n"
    "        return ''\n"
    "    if np.isnan(row['empirical_slope']):\n"
    "        return ''\n"
    "    diff = abs(row['empirical_slope'] - row['theory_slope'])\n"
    "    se = row['slope_se'] if pd.notna(row['slope_se']) else 0.0\n"
    "    return '(disagrees with theory)' if diff > 3 * se else ''\n"
    "ts1['flag'] = ts1.apply(annotate, axis=1)\n"
    "ts1\n"
))

# Figure S3
CELLS.append(cell_md(
    "### Figure S3 — Calibration share λ vs n (§4.7)\n"
    "\n"
    "In the common (β>0.5, ρ=1) regime we expect x* ↑ but λ* = x*/n ↓.\n"
))

CELLS.append(cell_code(
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "for (b, r), sub in cell_oracle.groupby(['beta', 'rho'], observed=True):\n"
    "    if sub['oracle_x'].max() == 0:\n"
    "        continue\n"
    "    sub = sub.sort_values('n')\n"
    "    lam = sub['oracle_x'] / sub['n']\n"
    "    ax.semilogx(sub['n'], lam, 'o-', alpha=0.6, label=fr'$\\beta={b}, \\rho={r}$')\n"
    "ax.set_xlabel('n')\n"
    "ax.set_ylabel(r'$\\lambda_n^* = x_n^* / n$')\n"
    "ax.set_title('Figure S3: oracle calibration share')\n"
    "ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc='upper left',\n"
    "          frameon=False)\n"
    "ax.grid(True, alpha=0.3)\n"
    "save_fig(fig, RUN_EXP1, 'figure_s3_calibration_share')\n"
    "plt.show()\n"
))

# Figure S4
CELLS.append(cell_md(
    "### Figure S4 — MSE ratio by estimator (§4.7)\n"
    "\n"
    "MSE / MSE(real_only_all) for all estimators across n, faceted by (β, ρ).\n"
))

CELLS.append(cell_code(
    "pairs = sorted(set(zip(table_s2['beta'], table_s2['rho'])))\n"
    "cols = 3\n"
    "rows = (len(pairs) + cols - 1) // cols\n"
    "fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.5*rows), squeeze=False)\n"
    "for ax, (b, r) in zip(axes.flat, pairs):\n"
    "    sub = table_s2[(table_s2['beta']==b) & (table_s2['rho']==r)]\n"
    "    for est, eg in sub.groupby('estimator', observed=True):\n"
    "        eg = eg.sort_values('n')\n"
    "        ax.semilogx(eg['n'], eg['mse_ratio_to_real'], 'o-', label=est, alpha=0.8)\n"
    "    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.7)\n"
    "    ax.set_title(fr'$\\beta={b}, \\rho={r}$  ({classify_regime(b, r)})', fontsize=9)\n"
    "    ax.set_xlabel('n')\n"
    "    ax.set_ylabel('MSE / MSE(real_only)')\n"
    "    ax.set_ylim(0, 2.5)\n"
    "    ax.grid(True, alpha=0.3)\n"
    "for ax in axes.flat[len(pairs):]:\n"
    "    ax.axis('off')\n"
    "handles, labels = axes[0, 0].get_legend_handles_labels()\n"
    "fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8,\n"
    "           bbox_to_anchor=(0.5, -0.02))\n"
    "fig.suptitle('Figure S4: MSE ratio to real-only across estimators')\n"
    "fig.tight_layout(rect=(0, 0.04, 1, 0.96))\n"
    "save_fig(fig, RUN_EXP1, 'figure_s4_mse_ratio')\n"
    "plt.show()\n"
))

CELLS.append(cell_md(
    "#### Table S2 — Estimator performance (§4.8)\n"
    "\n"
    "Headline regime (β=1, ρ=1) shown inline. Full table on disk.\n"
))

CELLS.append(cell_code(
    "headline = (table_s2[(table_s2['beta']==1.0) & (table_s2['rho']==1.0)]\n"
    "                .sort_values(['n', 'mse_ratio_to_real']))\n"
    "headline[['n','estimator','mse','mse_ratio_to_real','oracle_regret','x_selected']]\n"
))

# ---------------------------------------------------------------------------
# 3. Exp 2 — adaptive
# ---------------------------------------------------------------------------

CELLS.append(cell_md(
    "## 2. Experiment 2 — adaptive estimator (§5)\n"
))

CELLS.append(cell_code(
    "raw2 = read_run(RUN_EXP2 / 'raw')\n"
    "agg2_dir = RUN_EXP2 / 'aggregated'\n"
    "table_a1 = pd.read_parquet(agg2_dir / 'table_a1_learning_curve.parquet') \\\n"
    "    if (agg2_dir / 'table_a1_learning_curve.parquet').exists() else pd.DataFrame()\n"
    "table_a2 = pd.read_parquet(agg2_dir / 'table_a2_adaptive_perf.parquet')\n"
    "print(f'rows: {len(raw2):,}; table_a1 shape {table_a1.shape}; table_a2 shape {table_a2.shape}')\n"
))

CELLS.append(cell_md(
    "### Figure A1 — Estimated learning curves\n"
    "\n"
    "log b̂²(x_j) vs log x_j with the fitted line of slope −2β̂. We plot the\n"
    "average β̂ and ĉ across replications at the largest n in each (β, ρ) pair.\n"
))

CELLS.append(cell_code(
    "from dataloom.adaptive.bias_curve import default_pilot_grid\n"
    "\n"
    "# We don't store per-replication pilot bias curves in raw output; instead show\n"
    "# the mean β̂ and ĉ envelopes at the largest n.\n"
    "cells_to_plot = [(0.4, 1.0), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0)]\n"
    "fig, axes = plt.subplots(1, len(cells_to_plot), figsize=(4*len(cells_to_plot), 4),\n"
    "                          squeeze=False)\n"
    "for ax, (b, r) in zip(axes.flat, cells_to_plot):\n"
    "    sub = table_a1[(table_a1['beta']==b) & (table_a1['rho']==r) &\n"
    "                    (table_a1['method']=='corrected_adaptive_gn')] \\\n"
    "          if not table_a1.empty else pd.DataFrame()\n"
    "    if sub.empty:\n"
    "        ax.set_title(f'β={b}, ρ={r}\\n(no data)')\n"
    "        continue\n"
    "    n_max = int(sub['n'].max())\n"
    "    grid = default_pilot_grid(n_max).astype(float)\n"
    "    s = sub[sub['n']==n_max].iloc[0]\n"
    "    beta_hat = s['mean_beta_hat']; c_hat = s['mean_c_hat']\n"
    "    log_b2 = np.log(c_hat) - 2 * beta_hat * np.log(grid)\n"
    "    ax.plot(np.log(grid), log_b2, 'o-', label=f'fit β̂={beta_hat:.2f}, ĉ={c_hat:.2g}')\n"
    "    # truth reference\n"
    "    log_b2_true = np.log(1.0) - 2 * b * np.log(grid)\n"
    "    ax.plot(np.log(grid), log_b2_true, 'k--', alpha=0.4,\n"
    "            label=f'truth slope = −{2*b:.2f}')\n"
    "    ax.set_xlabel('log x_j')\n"
    "    ax.set_ylabel('log b²')\n"
    "    ax.set_title(f'β={b}, ρ={r}, n={n_max}')\n"
    "    ax.legend(fontsize=8)\n"
    "    ax.grid(True, alpha=0.3)\n"
    "fig.suptitle('Figure A1: pilot bias curves with fitted slope')\n"
    "fig.tight_layout()\n"
    "save_fig(fig, RUN_EXP2, 'figure_a1_learning_curves')\n"
    "plt.show()\n"
))

CELLS.append(cell_md(
    "### Figure A2 — Adaptive vs oracle allocation\n"
    "\n"
    "Per-replication scatter of x̂ (corrected_adaptive_gn) vs oracle x*.\n"
))

CELLS.append(cell_code(
    "sub = raw2[(raw2['method']=='corrected_adaptive_gn') &\n"
    "            raw2['x_selected'].notna()]\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
    "ax = axes[0]\n"
    "sc = ax.scatter(sub['oracle_x'], sub['x_selected'], c=sub['n'], cmap='viridis',\n"
    "                alpha=0.4, s=10)\n"
    "lim = max(sub['oracle_x'].max(), sub['x_selected'].max())\n"
    "ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8)\n"
    "ax.set_xlabel('oracle x*')\n"
    "ax.set_ylabel(r'adaptive $\\hat x$')\n"
    "ax.set_title(r'$\\hat x$ vs $x^*$')\n"
    "plt.colorbar(sc, ax=ax, label='n')\n"
    "\n"
    "ax = axes[1]\n"
    "ok = (sub['oracle_x'] > 0) & (sub['x_selected'] > 0)\n"
    "ax.hist(np.log(sub.loc[ok, 'x_selected'].astype(float)) -\n"
    "         np.log(sub.loc[ok, 'oracle_x'].astype(float)),\n"
    "         bins=50, alpha=0.7)\n"
    "ax.axvline(0, color='black', linestyle=':')\n"
    "ax.set_xlabel(r'$\\log\\hat x - \\log x^*$')\n"
    "ax.set_ylabel('count')\n"
    "ax.set_title(r'log allocation error')\n"
    "fig.suptitle('Figure A2: adaptive vs oracle allocation')\n"
    "fig.tight_layout()\n"
    "save_fig(fig, RUN_EXP2, 'figure_a2_adaptive_vs_oracle')\n"
    "plt.show()\n"
))

CELLS.append(cell_md(
    "### Figure A3 — Adaptive regret vs n\n"
))

CELLS.append(cell_code(
    "methods_a = ['corrected_adaptive_gn', 'adaptive_parametric_foc',\n"
    "              'adaptive_nonparametric_grid', 'safe_corrected_adaptive_gn']\n"
    "sub_a3 = table_a2[table_a2['method'].isin(methods_a)]\n"
    "pairs = sorted(set(zip(sub_a3['beta'], sub_a3['rho'])))\n"
    "cols = 3; rows = (len(pairs) + cols - 1) // cols\n"
    "fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.5*rows), squeeze=False)\n"
    "for ax, (b, r) in zip(axes.flat, pairs):\n"
    "    cell = sub_a3[(sub_a3['beta']==b) & (sub_a3['rho']==r)]\n"
    "    for m, mg in cell.groupby('method', observed=True):\n"
    "        mg = mg.sort_values('n')\n"
    "        ax.semilogx(mg['n'], mg['oracle_regret'], 'o-', label=m, alpha=0.85)\n"
    "    ax.axhline(0.0, color='black', linestyle=':', linewidth=0.7)\n"
    "    ax.set_title(fr'$\\beta={b}, \\rho={r}$', fontsize=9)\n"
    "    ax.set_xlabel('n')\n"
    "    ax.set_ylabel(r'oracle regret $R_n(\\hat x)/R_n(x^*) - 1$')\n"
    "    ax.grid(True, alpha=0.3)\n"
    "for ax in axes.flat[len(pairs):]:\n"
    "    ax.axis('off')\n"
    "handles, labels = axes[0, 0].get_legend_handles_labels()\n"
    "fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8,\n"
    "           bbox_to_anchor=(0.5, -0.02))\n"
    "fig.suptitle('Figure A3: adaptive regret relative to corrected_oracle_gn')\n"
    "fig.tight_layout(rect=(0, 0.04, 1, 0.96))\n"
    "save_fig(fig, RUN_EXP2, 'figure_a3_adaptive_regret')\n"
    "plt.show()\n"
))

CELLS.append(cell_md(
    "#### Table A1 — Learning-curve estimation accuracy\n"
))

CELLS.append(cell_code(
    "table_a1\n"
))

CELLS.append(cell_md(
    "#### Table A2 — Adaptive estimator performance\n"
))

CELLS.append(cell_code(
    "table_a2\n"
))

# ---------------------------------------------------------------------------
# 4. Exp 3 — multichannel
# ---------------------------------------------------------------------------

CELLS.append(cell_md(
    "## 3. Experiment 3 — multichannel allocation (§6)\n"
))

CELLS.append(cell_code(
    "from dataloom.analysis.exp3 import read_run_mc\n"
    "raw3, oracle3 = read_run_mc(RUN_EXP3 / 'raw')\n"
    "table_m1 = pd.read_parquet(RUN_EXP3 / 'aggregated' / 'table_m1_multichannel_perf.parquet')\n"
    "print(f'rows: {len(raw3):,}; table_m1 shape {table_m1.shape}')\n"
))

CELLS.append(cell_md(
    "### Figure M1 — Risk surface (representative cell)\n"
))

CELLS.append(cell_code(
    "from dataloom.multichannel.oracle import R_profile_2\n"
    "cell = oracle3.sort_values('n', ascending=False).iloc[0]\n"
    "n = int(cell['n']); b1, b2 = float(cell['beta1']), float(cell['beta2'])\n"
    "v_n = 1.0 / n\n"
    "step = max(1, n // 100)\n"
    "xs = np.arange(1, n, step)\n"
    "X1, X2 = np.meshgrid(xs, xs, indexing='ij')\n"
    "feas = (X1 + X2) < n\n"
    "R = np.full(X1.shape, np.nan)\n"
    "R[feas] = R_profile_2(X1[feas], X2[feas], n=n, a=1.0, v_n=v_n,\n"
    "                       c1=1.0, beta1=b1, c2=1.0, beta2=b2)\n"
    "fig, ax = plt.subplots(figsize=(7, 6))\n"
    "pcm = ax.pcolormesh(X1, X2, np.log10(R), shading='auto', cmap='viridis')\n"
    "plt.colorbar(pcm, ax=ax, label=r'$\\log_{10} R_n$')\n"
    "ax.plot(cell['x1_star'], cell['x2_star'], '*', markersize=18,\n"
    "        color='red', label='corrected oracle')\n"
    "es = max(1, n // 4)\n"
    "ax.plot(es, es, 's', markersize=10, color='white', label='equal split')\n"
    "denom = 1.0 + 2.0 * (b1 + b2)\n"
    "x1_old = max(1, int(np.floor(n * 2.0 * b1 / denom)))\n"
    "x2_old = max(1, int(np.floor(n * 2.0 * b2 / denom)))\n"
    "ax.plot(x1_old, x2_old, '^', markersize=10, color='orange', label='old fixed-share')\n"
    "ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')\n"
    "ax.set_title(fr'Figure M1: risk surface (n={n}, $\\beta_1$={b1}, $\\beta_2$={b2})')\n"
    "ax.legend(loc='upper right', fontsize=8)\n"
    "save_fig(fig, RUN_EXP3, 'figure_m1_risk_surface')\n"
    "plt.show()\n"
))

CELLS.append(cell_md(
    "### Figure M2 — Marginal-value equality at oracle\n"
    "\n"
    "Active-channel KKT predicts MV₁ = MV₂ at the corrected oracle.\n"
))

CELLS.append(cell_code(
    "fig, ax = plt.subplots(figsize=(6, 6))\n"
    "for (b1, b2), g in oracle3.groupby(['beta1', 'beta2'], observed=True):\n"
    "    ax.scatter(g['mv1'], g['mv2'], label=fr'$\\beta=({b1}, {b2})$',\n"
    "               alpha=0.7, s=60)\n"
    "lim = max(oracle3['mv1'].max(), oracle3['mv2'].max())\n"
    "ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, label='MV1 = MV2')\n"
    "ax.set_xscale('log'); ax.set_yscale('log')\n"
    "ax.set_xlabel(r'$MV_1$ at oracle'); ax.set_ylabel(r'$MV_2$ at oracle')\n"
    "ax.set_title('Figure M2: KKT marginal values at corrected oracle')\n"
    "ax.legend(fontsize=8)\n"
    "save_fig(fig, RUN_EXP3, 'figure_m2_marginal_values')\n"
    "plt.show()\n"
))

CELLS.append(cell_md("#### Table M1 — Multichannel performance"))
CELLS.append(cell_code("table_m1"))

# ---------------------------------------------------------------------------
# 5. Exp 4 — inference
# ---------------------------------------------------------------------------

CELLS.append(cell_md("## 4. Experiment 4 — inference and coverage (§7)"))

CELLS.append(cell_code(
    "raw4 = read_run(RUN_EXP4 / 'raw')\n"
    "table_c1 = pd.read_parquet(RUN_EXP4 / 'aggregated' / 'table_c1_coverage.parquet')\n"
    "print(f'rows: {len(raw4):,}; table_c1 shape {table_c1.shape}')\n"
))

CELLS.append(cell_md(
    "### Figure C1 — Coverage vs n (§7.5)\n"
    "\n"
    "Per the §16 #7 prediction, `ci_gn_naive` should undercover when\n"
    "synthetic bias is non-negligible.\n"
))

CELLS.append(cell_code(
    "ci_methods = ['ci_real_only', 'ci_gn_naive', 'ci_gn_bias_aware',\n"
    "               'ci_gn_undersmoothed', 'ci_validation_debiased']\n"
    "sub = table_c1[table_c1['interval_method'].isin(ci_methods)]\n"
    "pairs = sorted(set(zip(sub['beta'], sub['rho'])))\n"
    "cols = 2; rows = (len(pairs) + cols - 1) // cols\n"
    "fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)\n"
    "for ax, (b, r) in zip(axes.flat, pairs):\n"
    "    cell = sub[(sub['beta']==b) & (sub['rho']==r)]\n"
    "    for m, mg in cell.groupby('interval_method', observed=True):\n"
    "        mg = mg.sort_values('n')\n"
    "        ax.plot(mg['n'], mg['coverage'], 'o-', label=m, alpha=0.85)\n"
    "    ax.axhline(0.95, color='black', linestyle=':', linewidth=0.8)\n"
    "    ax.set_xscale('log'); ax.set_ylim(0, 1.05)\n"
    "    ax.set_xlabel('n'); ax.set_ylabel('coverage')\n"
    "    ax.set_title(fr'$\\beta={b}, \\rho={r}$', fontsize=10)\n"
    "    ax.grid(True, alpha=0.3)\n"
    "for ax in axes.flat[len(pairs):]:\n"
    "    ax.axis('off')\n"
    "handles, labels = axes[0, 0].get_legend_handles_labels()\n"
    "fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8,\n"
    "           bbox_to_anchor=(0.5, -0.02))\n"
    "fig.suptitle('Figure C1: empirical coverage by interval method')\n"
    "fig.tight_layout(rect=(0, 0.04, 1, 0.96))\n"
    "save_fig(fig, RUN_EXP4, 'figure_c1_coverage')\n"
    "plt.show()\n"
))

CELLS.append(cell_md("### Figure C2 — Coverage vs interval length"))
CELLS.append(cell_code(
    "fig, ax = plt.subplots(figsize=(7, 5))\n"
    "for m, mg in table_c1.groupby('interval_method', observed=True):\n"
    "    ax.scatter(mg['avg_length'], mg['coverage'], label=m, alpha=0.7, s=40)\n"
    "ax.axhline(0.95, color='black', linestyle=':', linewidth=0.8)\n"
    "ax.set_xlabel('average interval length'); ax.set_ylabel('coverage')\n"
    "ax.set_title('Figure C2: coverage vs interval length')\n"
    "ax.legend(fontsize=8); ax.grid(True, alpha=0.3)\n"
    "save_fig(fig, RUN_EXP4, 'figure_c2_coverage_vs_length')\n"
    "plt.show()\n"
))

CELLS.append(cell_md("#### Table C1 — Coverage summary"))
CELLS.append(cell_code("table_c1"))

# ---------------------------------------------------------------------------
# 6. Exp A — tabular
# ---------------------------------------------------------------------------

CELLS.append(cell_md("## 5. Experiment A — tabular semi-synthetic (§8)"))

CELLS.append(cell_code(
    "rawA = read_run(RUN_EXPA / 'raw')\n"
    "table_t2 = pd.read_parquet(RUN_EXPA / 'aggregated' / 'table_t2_tabular_perf.parquet')\n"
    "print(f'rows: {len(rawA):,}; table_t2 shape {table_t2.shape}')\n"
    "rawA['estimand'].value_counts()\n"
))

CELLS.append(cell_md(
    "### Figure T1 — Empirical learning curve (§8.9)\n"
    "\n"
    "We don't store per-replication pilot bias points in raw output for tabular,\n"
    "so this figure summarizes selected x and inferred (β̂, ĉ) per estimand.\n"
))

CELLS.append(cell_code(
    "sub = rawA[(rawA['method']=='corrected_adaptive_gn') &\n"
    "            rawA['beta_hat'].notna()]\n"
    "if not sub.empty:\n"
    "    fig, axes = plt.subplots(1, sub['estimand'].nunique(),\n"
    "                              figsize=(5*sub['estimand'].nunique(), 4),\n"
    "                              squeeze=False)\n"
    "    for ax, (es, eg) in zip(axes.flat, sub.groupby('estimand', observed=True)):\n"
    "        agg = (eg.groupby('n')['beta_hat']\n"
    "                  .agg(['mean','std','count'])\n"
    "                  .reset_index())\n"
    "        ax.errorbar(agg['n'], agg['mean'], yerr=agg['std'], fmt='o-')\n"
    "        ax.set_xscale('log'); ax.set_xlabel('n')\n"
    "        ax.set_ylabel(r'mean $\\hat\\beta$')\n"
    "        ax.set_title(f'estimand={es}')\n"
    "        ax.grid(True, alpha=0.3)\n"
    "    fig.suptitle('Figure T1: estimated bias-curve slope across n')\n"
    "    fig.tight_layout()\n"
    "    save_fig(fig, RUN_EXPA, 'figure_t1_learning_curve')\n"
    "    plt.show()\n"
    "else:\n"
    "    print('No adaptive bias-fit data in raw — skipping Figure T1.')\n"
))

CELLS.append(cell_md("### Figure T2 — Selected x and λ vs n"))

CELLS.append(cell_code(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "for m, mg in rawA[rawA['x_selected'].notna()].groupby('method', observed=True):\n"
    "    agg = mg.groupby('n')[['x_selected', 'lambda_selected']].mean().reset_index()\n"
    "    axes[0].semilogx(agg['n'], agg['x_selected'], 'o-', label=m, alpha=0.7)\n"
    "    axes[1].semilogx(agg['n'], agg['lambda_selected'], 'o-', label=m, alpha=0.7)\n"
    "axes[0].set_xlabel('n'); axes[0].set_ylabel(r'$\\hat x$'); axes[0].grid(alpha=0.3)\n"
    "axes[1].set_xlabel('n'); axes[1].set_ylabel(r'$\\hat\\lambda$'); axes[1].grid(alpha=0.3)\n"
    "axes[0].set_title('Figure T2a: selected x'); axes[1].set_title('Figure T2b: selected λ')\n"
    "axes[0].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)\n"
    "fig.tight_layout()\n"
    "save_fig(fig, RUN_EXPA, 'figure_t2_selected_allocation')\n"
    "plt.show()\n"
))

CELLS.append(cell_md("### Figure T3 — MSE ratio to real-only"))

CELLS.append(cell_code(
    "estimands = sorted(table_t2['estimand'].unique())\n"
    "fig, axes = plt.subplots(1, len(estimands),\n"
    "                          figsize=(5*len(estimands), 4),\n"
    "                          squeeze=False)\n"
    "for ax, es in zip(axes.flat, estimands):\n"
    "    eg = table_t2[table_t2['estimand']==es]\n"
    "    for m, mg in eg.groupby('method', observed=True):\n"
    "        mg = mg.sort_values('n')\n"
    "        ax.semilogx(mg['n'], mg['mse_ratio_to_real'], 'o-', label=m, alpha=0.8)\n"
    "    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.7)\n"
    "    ax.set_title(f'estimand={es}', fontsize=9)\n"
    "    ax.set_xlabel('n'); ax.set_ylabel('MSE / MSE(real_only)')\n"
    "    ax.set_ylim(0, 2.5); ax.grid(True, alpha=0.3)\n"
    "handles, labels = axes[0, 0].get_legend_handles_labels()\n"
    "fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8,\n"
    "           bbox_to_anchor=(0.5, -0.05))\n"
    "fig.suptitle('Figure T3: tabular MSE ratio to real-only')\n"
    "fig.tight_layout(rect=(0, 0.05, 1, 0.96))\n"
    "save_fig(fig, RUN_EXPA, 'figure_t3_mse_ratio')\n"
    "plt.show()\n"
))

CELLS.append(cell_md(
    "#### Table T1 — Tabular learning-curve summary\n"
    "\n"
    "Per-cell (β̂_mean, β̂_sd, R²) for the corrected adaptive method.\n"
))

CELLS.append(cell_code(
    "if 'beta_hat' in rawA.columns:\n"
    "    t1 = (rawA[rawA['beta_hat'].notna()]\n"
    "             .groupby(['dataset','generator','estimand','n','m'])['beta_hat']\n"
    "             .agg(beta_hat_mean='mean', beta_hat_sd='std', n_reps='count')\n"
    "             .reset_index())\n"
    "    t1.to_parquet(RUN_EXPA / 'aggregated' / 'table_t1_tabular_learning.parquet')\n"
    "    display(t1)\n"
))

CELLS.append(cell_md("#### Table T2 — Tabular performance summary"))
CELLS.append(cell_code("table_t2"))

# ---------------------------------------------------------------------------
# 7. Exp B — causal
# ---------------------------------------------------------------------------

CELLS.append(cell_md("## 6. Experiment B — causal digital-twin (§9)"))

CELLS.append(cell_code(
    "rawB = read_run(RUN_EXPB / 'raw')\n"
    "table_d1 = pd.read_parquet(RUN_EXPB / 'aggregated' / 'table_d1_causal_perf.parquet')\n"
    "print(f'rows: {len(rawB):,}; table_d1 shape {table_d1.shape}')\n"
))

CELLS.append(cell_md("### Figure D1 — Causal learning curve (per-method β̂ across n)"))

CELLS.append(cell_code(
    "subB = rawB[(rawB['method'].str.contains('adaptive', na=False)) &\n"
    "             rawB['beta_hat'].notna()]\n"
    "if not subB.empty:\n"
    "    fig, ax = plt.subplots(figsize=(7, 5))\n"
    "    for m, mg in subB.groupby('method', observed=True):\n"
    "        agg = mg.groupby('n')['beta_hat'].agg(['mean','std']).reset_index()\n"
    "        ax.errorbar(agg['n'], agg['mean'], yerr=agg['std'],\n"
    "                    fmt='o-', label=m, alpha=0.8)\n"
    "    ax.set_xscale('log'); ax.set_xlabel('n')\n"
    "    ax.set_ylabel(r'mean $\\hat\\beta$ (causal bias-curve)')\n"
    "    ax.set_title('Figure D1: causal bias-curve estimate across n')\n"
    "    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)\n"
    "    save_fig(fig, RUN_EXPB, 'figure_d1_causal_learning_curve')\n"
    "    plt.show()\n"
    "else:\n"
    "    print('No adaptive β̂ data in raw — skipping Figure D1.')\n"
))

CELLS.append(cell_md("### Figure D2 — ATE MSE ratio"))

CELLS.append(cell_code(
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "for m, mg in table_d1.groupby('method', observed=True):\n"
    "    mg = mg.sort_values('n')\n"
    "    ax.semilogx(mg['n'], mg['mse_ratio_to_real'], 'o-', label=m, alpha=0.8)\n"
    "ax.axhline(1.0, color='black', linestyle=':', linewidth=0.7)\n"
    "ax.set_xlabel('n'); ax.set_ylabel('ATE MSE / ATE MSE(real-only baseline)')\n"
    "ax.set_title('Figure D2: ATE MSE ratio (causal)')\n"
    "ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)\n"
    "ax.grid(True, alpha=0.3)\n"
    "save_fig(fig, RUN_EXPB, 'figure_d2_ate_mse_ratio')\n"
    "plt.show()\n"
))

CELLS.append(cell_md(
    "### Figure D3 — Coverage vs interval length (causal)\n"
    "\n"
    "The causal smoke run does not include CI columns yet (the runner emits\n"
    "point estimates only). When interval methods are added to Exp B, this\n"
    "cell will populate; for now we render a placeholder + note.\n"
))

CELLS.append(cell_code(
    "if 'ci_lower' in rawB.columns and rawB['ci_lower'].notna().any():\n"
    "    sub = rawB.dropna(subset=['ci_lower','ci_upper'])\n"
    "    sub['length'] = sub['ci_upper'] - sub['ci_lower']\n"
    "    sub['covered'] = (sub['ci_lower'] <= sub['theta_true']) & \\\n"
    "                       (sub['theta_true'] <= sub['ci_upper'])\n"
    "    s = sub.groupby('method').agg(coverage=('covered','mean'),\n"
    "                                     avg_length=('length','mean')).reset_index()\n"
    "    fig, ax = plt.subplots(figsize=(7, 5))\n"
    "    ax.scatter(s['avg_length'], s['coverage'], s=80)\n"
    "    for _, row in s.iterrows():\n"
    "        ax.annotate(row['method'], (row['avg_length'], row['coverage']),\n"
    "                    fontsize=7, xytext=(4,4), textcoords='offset points')\n"
    "    ax.axhline(0.95, color='black', linestyle=':', linewidth=0.7)\n"
    "    ax.set_xlabel('avg interval length'); ax.set_ylabel('coverage')\n"
    "    ax.set_title('Figure D3: coverage vs interval length (causal)')\n"
    "    save_fig(fig, RUN_EXPB, 'figure_d3_coverage_vs_length')\n"
    "    plt.show()\n"
    "else:\n"
    "    print('Causal runner does not yet emit CI columns — Figure D3 skipped.')\n"
    "    print('To enable: extend semisynth/causal_runner._run_one to call ci_*\\n'\n"
    "          'methods alongside the point estimators.')\n"
))

CELLS.append(cell_md("#### Table D1 — Causal performance summary"))
CELLS.append(cell_code("table_d1"))

# ---------------------------------------------------------------------------
# 8. Master figures (§12)
# ---------------------------------------------------------------------------

CELLS.append(cell_md(
    "## 7. Master figures and tables for the paper (§12, §13)\n"
    "\n"
    "Distilled headline plots/tables for the NeurIPS submission.\n"
))

CELLS.append(cell_md(
    "### Main Figure 1 — Theoretical phase diagram (with regions)\n"
))

CELLS.append(cell_code(
    "fig, ax = plt.subplots(figsize=(8, 6))\n"
    "betas_fine = np.linspace(0.1, 2.5, 400)\n"
    "rhos_fine = np.linspace(0.0, 3.5, 400)\n"
    "BB, RR = np.meshgrid(betas_fine, rhos_fine, indexing='ij')\n"
    "regime_idx = np.zeros_like(BB)\n"
    "name_to_idx = {k: i for i, k in enumerate(REGIME_COLORS)}\n"
    "for i, b in enumerate(betas_fine):\n"
    "    for j, r in enumerate(rhos_fine):\n"
    "        regime_idx[i, j] = name_to_idx[classify_regime(b, r)]\n"
    "from matplotlib.colors import ListedColormap\n"
    "cmap = ListedColormap([REGIME_COLORS[k] for k in REGIME_COLORS])\n"
    "pcm = ax.pcolormesh(BB, RR, regime_idx, cmap=cmap, shading='auto', alpha=0.8)\n"
    "ax.axhline(1.0, color='black', linestyle=':', linewidth=1.0,\n"
    "           label=r'$\\rho=1$ (common case)')\n"
    "for b in betas:\n"
    "    for r in rhos:\n"
    "        ax.scatter(b, r, marker='o', edgecolor='white', facecolor='black',\n"
    "                    s=20, zorder=3)\n"
    "handles = [plt.Line2D([0],[0], marker='s', color='w',\n"
    "                       markerfacecolor=c, markersize=14, label=k)\n"
    "           for k, c in REGIME_COLORS.items()]\n"
    "handles.append(plt.Line2D([0], [0], color='black', linestyle=':',\n"
    "                            label=r'$\\rho=1$'))\n"
    "ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',\n"
    "          fontsize=8, frameon=False)\n"
    "ax.set_xlabel(r'$\\beta$'); ax.set_ylabel(r'$\\rho$')\n"
    "ax.set_title('Main Figure 1: phase diagram (regions filled, simulated cells dotted)')\n"
    "save_fig(fig, RUN_EXP1, 'main_figure_1_phase_diagram')\n"
    "plt.show()\n"
))

CELLS.append(cell_md("### Main Figure 2 — Synthetic scaling validation"))
CELLS.append(cell_code(
    "interior_pairs = [(b, r) for b in betas for r in rhos\n"
    "                   if classify_regime(b, r) == 'sublinear_fast_learning']\n"
    "fig, ax = plt.subplots(figsize=(8, 6))\n"
    "for b, r in sorted(interior_pairs):\n"
    "    sub = cell_oracle[(cell_oracle['beta']==b) & (cell_oracle['rho']==r)]\n"
    "    sub = sub[sub['oracle_x'] > 0].sort_values('n')\n"
    "    if sub.empty:\n"
    "        continue\n"
    "    sl = theory_slope(b, r)\n"
    "    n_arr = sub['n'].to_numpy(dtype=float)\n"
    "    ref = sub['oracle_x'].iloc[0] / (n_arr[0] ** sl)\n"
    "    line, = ax.loglog(n_arr, sub['oracle_x'], 'o-', alpha=0.7,\n"
    "                       label=fr'$\\beta={b}, \\rho={r}$ (slope {sl:.2f})')\n"
    "    ax.loglog(n_arr, ref * n_arr ** sl, '--', alpha=0.5,\n"
    "               color=line.get_color())\n"
    "ax.set_xlabel('n'); ax.set_ylabel(r'oracle $x_n^*$')\n"
    "ax.set_title('Main Figure 2: empirical $x_n^*$ vs theoretical scaling (interior fast-learning regime)')\n"
    "ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc='upper left',\n"
    "          frameon=False)\n"
    "ax.grid(True, which='both', alpha=0.3)\n"
    "save_fig(fig, RUN_EXP1, 'main_figure_2_scaling_validation')\n"
    "plt.show()\n"
))

CELLS.append(cell_md(
    "### Main Figure 3 — MSE ratio comparison (representative regimes)\n"
    "\n"
    "Real-only, old fixed-share, corrected oracle, corrected adaptive, safe adaptive.\n"
))

CELLS.append(cell_code(
    "main3_methods = ['real_only_all', 'old_fixed_share_oracle_alpha',\n"
    "                  'corrected_oracle_gn',\n"
    "                  'corrected_adaptive_gn', 'safe_corrected_adaptive_gn']\n"
    "rep_pairs = [(0.5, 1.0), (1.0, 1.0), (1.5, 1.0)]\n"
    "fig, axes = plt.subplots(1, len(rep_pairs),\n"
    "                          figsize=(5*len(rep_pairs), 4), squeeze=False)\n"
    "for ax, (b, r) in zip(axes.flat, rep_pairs):\n"
    "    s1_perf = table_s2[(table_s2['beta']==b) & (table_s2['rho']==r) &\n"
    "                        table_s2['estimator'].isin(main3_methods)]\n"
    "    s2_perf = table_a2[(table_a2['beta']==b) & (table_a2['rho']==r) &\n"
    "                        table_a2['method'].isin(main3_methods)]\n"
    "    for est, eg in s1_perf.groupby('estimator', observed=True):\n"
    "        eg = eg.sort_values('n')\n"
    "        ax.semilogx(eg['n'], eg['mse_ratio_to_real'], 'o-', label=est, alpha=0.85)\n"
    "    for m, mg in s2_perf.groupby('method', observed=True):\n"
    "        if m in s1_perf['estimator'].unique():\n"
    "            continue\n"
    "        mg = mg.sort_values('n')\n"
    "        ax.semilogx(mg['n'], mg['mse_ratio_to_real'], 's--', label=m, alpha=0.85)\n"
    "    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.7)\n"
    "    ax.set_xlabel('n'); ax.set_ylabel('MSE / MSE(real-only)')\n"
    "    ax.set_title(fr'$\\beta={b}, \\rho={r}$', fontsize=10)\n"
    "    ax.set_ylim(0, 1.5); ax.grid(True, alpha=0.3)\n"
    "handles, labels = axes[0, 0].get_legend_handles_labels()\n"
    "fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8,\n"
    "           bbox_to_anchor=(0.5, -0.04))\n"
    "fig.suptitle('Main Figure 3: MSE ratio across estimators')\n"
    "fig.tight_layout(rect=(0, 0.04, 1, 0.96))\n"
    "save_fig(fig, RUN_EXP1, 'main_figure_3_mse_comparison')\n"
    "plt.show()\n"
))

CELLS.append(cell_md(
    "### Main Figure 4 — Semi-synthetic result\n"
    "\n"
    "Tabular learning curve + MSE ratios (Exp A); causal alternative below.\n"
))

CELLS.append(cell_code(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
    "# left: T1-style learning curve\n"
    "if 'beta_hat' in rawA.columns:\n"
    "    sub = rawA[rawA['beta_hat'].notna()]\n"
    "    for es, eg in sub.groupby('estimand', observed=True):\n"
    "        agg = eg.groupby('n')['beta_hat'].agg(['mean','std']).reset_index()\n"
    "        axes[0].errorbar(agg['n'], agg['mean'], yerr=agg['std'], fmt='o-',\n"
    "                          label=es, alpha=0.85)\n"
    "    axes[0].set_xscale('log'); axes[0].set_xlabel('n')\n"
    "    axes[0].set_ylabel(r'mean $\\hat\\beta$')\n"
    "    axes[0].set_title('learning-curve slope across n')\n"
    "    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)\n"
    "# right: MSE ratio per method (income_mean estimand)\n"
    "ax = axes[1]\n"
    "ig = table_t2[table_t2['estimand']=='income_mean']\n"
    "for m, mg in ig.groupby('method', observed=True):\n"
    "    mg = mg.sort_values('n')\n"
    "    ax.semilogx(mg['n'], mg['mse_ratio_to_real'], 'o-', label=m, alpha=0.85)\n"
    "ax.axhline(1.0, color='black', linestyle=':', linewidth=0.7)\n"
    "ax.set_xlabel('n'); ax.set_ylabel('MSE / MSE(real-only)')\n"
    "ax.set_title('MSE ratio (income_mean estimand)')\n"
    "ax.set_ylim(0, 2.0); ax.grid(True, alpha=0.3)\n"
    "ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)\n"
    "fig.suptitle('Main Figure 4: tabular semi-synthetic result')\n"
    "fig.tight_layout(rect=(0, 0, 1, 0.96))\n"
    "save_fig(fig, RUN_EXPA, 'main_figure_4_semisynth')\n"
    "plt.show()\n"
))

# ---------------------------------------------------------------------------
# 9. Master tables (§13)
# ---------------------------------------------------------------------------

CELLS.append(cell_md("### Main Table 1 — Phase diagram validation"))

CELLS.append(cell_code(
    "old_lambda = (lambda b: (2.0 * b) / (1.0 + 2.0 * b))\n"
    "rows_main1 = []\n"
    "for _, row in ts1.iterrows():\n"
    "    b, r = row['beta'], row['rho']\n"
    "    rows_main1.append({\n"
    "        'beta': b, 'rho': r,\n"
    "        'predicted_regime': row['theory_regime'],\n"
    "        'predicted_slope': row['theory_slope'],\n"
    "        'empirical_slope': row['empirical_slope'],\n"
    "        'corrected_lambda_at_largest_n': row['mean_oracle_lambda_at_largest_n'],\n"
    "        'old_fixed_share_lambda': old_lambda(b),\n"
    "    })\n"
    "main_table_1 = pd.DataFrame(rows_main1)\n"
    "main_table_1.to_parquet(RUN_EXP1 / 'aggregated' / 'main_table_1_phase_diagram.parquet')\n"
    "main_table_1\n"
))

CELLS.append(cell_md("### Main Table 2 — Estimator performance (representative regimes)"))

CELLS.append(cell_code(
    "rep_cells_2 = [(0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (1.0, 0.0), (1.0, 2.0)]\n"
    "main_table_2 = []\n"
    "for b, r in rep_cells_2:\n"
    "    sub = table_s2[(table_s2['beta']==b) & (table_s2['rho']==r)]\n"
    "    if sub.empty:\n"
    "        continue\n"
    "    nmax = sub['n'].max()\n"
    "    sub_n = sub[sub['n']==nmax]\n"
    "    for _, row in sub_n.iterrows():\n"
    "        main_table_2.append({\n"
    "            'regime': classify_regime(b, r),\n"
    "            'beta': b, 'rho': r, 'n': nmax,\n"
    "            'estimator': row['estimator'],\n"
    "            'x_selected': row['x_selected'],\n"
    "            'lambda_selected': row['lambda_selected'],\n"
    "            'alpha_selected': row['alpha_selected'],\n"
    "            'bias': row['bias'],\n"
    "            'variance': row['variance'],\n"
    "            'mse': row['mse'],\n"
    "            'mse_ratio_to_real': row['mse_ratio_to_real'],\n"
    "            'oracle_regret': row['oracle_regret'],\n"
    "        })\n"
    "main_table_2 = pd.DataFrame(main_table_2)\n"
    "main_table_2.to_parquet(RUN_EXP1 / 'aggregated' / 'main_table_2_estimator_perf.parquet')\n"
    "main_table_2\n"
))

CELLS.append(cell_md("### Main Table 3 — Semi-synthetic performance"))

CELLS.append(cell_code(
    "tt2 = table_t2.copy()\n"
    "tt2['lambda_selected'] = tt2['x_selected'] / tt2['n']\n"
    "tt2['coverage'] = np.nan  # tabular runner does not emit CIs\n"
    "main_table_3 = tt2[['dataset','generator','estimand','method','n','m',\n"
    "                     'x_selected','lambda_selected','mse_ratio_to_real',\n"
    "                     'fallback_rate']]\n"
    "main_table_3.to_parquet(RUN_EXPA / 'aggregated' / 'main_table_3_semisynth.parquet')\n"
    "main_table_3\n"
))

# ---------------------------------------------------------------------------
# 10. Discrepancy summary
# ---------------------------------------------------------------------------

CELLS.append(cell_md(
    "## 8. Discrepancy summary\n"
    "\n"
    "Where empirical results diverge from theory predictions, list them here.\n"
    "Per §16 of `docs/experiments.md`: do **not** smooth or hide.\n"
))

CELLS.append(cell_code(
    "discrepancies = []\n"
    "# (a) Slope disagreements from Table S1\n"
    "for _, row in ts1.iterrows():\n"
    "    if row.get('flag') == '(disagrees with theory)':\n"
    "        discrepancies.append(\n"
    "            f\"S1 (β={row['beta']}, ρ={row['rho']}): theory slope \"\n"
    "            f\"{row['theory_slope']:.3f}, empirical {row['empirical_slope']:.3f} \"\n"
    "            f\"(SE {row['slope_se']:.3f})\")\n"
    "# (b) Adaptive harm at large n\n"
    "if not table_a2.empty:\n"
    "    harmful = table_a2[(table_a2['mse_ratio_to_real'] > 1.05) &\n"
    "                        (table_a2['method'].str.contains('adaptive'))]\n"
    "    for _, row in harmful.iterrows():\n"
    "        discrepancies.append(\n"
    "            f\"A2 ({row['method']}, β={row['beta']}, ρ={row['rho']}, n={row['n']}): \"\n"
    "            f\"MSE ratio {row['mse_ratio_to_real']:.3f} > 1 (synthetic data is harmful)\")\n"
    "# (c) Coverage failures — expected for ci_gn_naive but report if it covers >0.95\n"
    "if not table_c1.empty:\n"
    "    over = table_c1[(table_c1['interval_method']=='ci_gn_naive') &\n"
    "                      (table_c1['coverage'] >= 0.94)]\n"
    "    for _, row in over.iterrows():\n"
    "        discrepancies.append(\n"
    "            f\"C1 (β={row['beta']}, n={row['n']}): ci_gn_naive coverage \"\n"
    "            f\"{row['coverage']:.3f} >= 0.94 (theory predicts undercoverage)\")\n"
    "if discrepancies:\n"
    "    print(f'{len(discrepancies)} discrepancy(ies) detected:')\n"
    "    for d in discrepancies:\n"
    "        print('  -', d)\n"
    "else:\n"
    "    print('No theory-empirical discrepancies above the flag thresholds.')\n"
))


# ---------------------------------------------------------------------------
# Build and write
# ---------------------------------------------------------------------------

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NOTEBOOK_PATH, "w") as f:
    json.dump(NOTEBOOK, f, indent=1)
print(f"Wrote {NOTEBOOK_PATH} with {len(CELLS)} cells.")
