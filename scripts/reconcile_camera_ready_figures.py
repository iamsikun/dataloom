"""Rebuild legacy camera-ready figures and tables from pinned full-run outputs.

Usage: python scripts/reconcile_camera_ready_figures.py --data-root PATH --output PATH
PATH contains reproduced-runs/ with exactly one full run each for Experiments 4,
A, and B. No newest-run selection, smoke fallback, or experiment execution occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                     "font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9,
                     "axes.labelsize": 10, "lines.linewidth": 1.4})

LABELS = {
    "real_only_all": "All-real", "real_only_aipw": "Real-only AIPW",
    "corrected_adaptive_gn": "Adaptive plug-in",
    "safe_corrected_adaptive_gn": "Safe adaptive",
    "validation_debiased_gn": "Validation-debiased",
    "synthetic_only_full_calibration": "Synthetic-only",
    "naive_pooling": "Naive pooling",
    "fixed_half_split_plugin_alpha": "Half split, plug-in weight",
    "old_fixed_share_plugin_alpha": "Product-surrogate plug-in",
    "real_only_diff_in_means": "Difference in means",
    "ci_gn_naive": "Naive Wald", "ci_gn_bias_aware": "Bias-aware",
    "ci_gn_undersmoothed": "Undersmoothed", "ci_real_only": "All-real",
    "ci_validation_debiased": "Validation-debiased",
}
PERF_ORDER = ["corrected_adaptive_gn", "safe_corrected_adaptive_gn", "validation_debiased_gn",
              "old_fixed_share_plugin_alpha", "fixed_half_split_plugin_alpha", "naive_pooling",
              "synthetic_only_full_calibration", "real_only_diff_in_means"]
CI_ORDER = ["ci_gn_naive", "ci_gn_bias_aware", "ci_gn_undersmoothed", "ci_real_only", "ci_validation_debiased"]
COLORS = {m: plt.get_cmap("tab10")(i) for i, m in enumerate(PERF_ORDER)}
MARKERS = dict(zip(PERF_ORDER, ["o", "s", "^", "D", "v", "P", "X", "*"]))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(x):
    return str(Decimal(str(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def load_run(root, name, expected_rows):
    paths = list((root / "reproduced-runs").glob(name + "__*"))
    assert len(paths) == 1, (name, paths)
    run = paths[0]
    files = sorted((run / "raw").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert len(df) == expected_rows, (name, len(df))
    return run, files, df


def perf_table(df, baseline):
    keys = ["dataset", "generator", "estimand", "n", "m", "method"]
    work = df.dropna(subset=["theta_hat"]).copy()
    work["error"] = work.theta_hat - work.theta_true
    work["squared_error"] = work.error ** 2
    out = work.groupby(keys, observed=True).agg(
        mse=("squared_error", "mean"), bias=("error", "mean"),
        fallback_rate=("fallback_used", "mean"), n_replications=("replication", "count")
    ).reset_index()
    base = out[out.method == baseline].set_index(keys[:-1]).mse.rename("baseline_mse")
    out = out.join(base, on=keys[:-1])
    out["mse_ratio_to_real"] = out.mse / out.baseline_mse
    out["m_factor"] = out.m / out.n
    return out


def perf_summary(table):
    return table.groupby("method", observed=True).agg(
        cells=("mse_ratio_to_real", "count"), median=("mse_ratio_to_real", "median"),
        maximum=("mse_ratio_to_real", "max"),
        harmful=("mse_ratio_to_real", lambda x: int((x > 1).sum())),
        fallback=("fallback_rate", "mean")
    )


def save_page(pdf, fig):
    fig.tight_layout(pad=0.6)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def legend_page(pdf, handles, labels, ncol=2):
    fig = plt.figure(figsize=(6.6, 0.3 + 0.27 * np.ceil(len(labels) / ncol)))
    fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def xaxis(ax, ns):
    ax.set_xscale("log")
    ax.set_xticks(ns, [str(int(n)) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("Real-data budget $n$")
    ax.grid(alpha=.22)


def plot_perf(ax, table, adult=False, title="", ylim=None):
    ns = sorted(table.n.unique())
    if adult:
        baseline = table[table.method == "real_only_all"].sort_values("n")
        ax.plot(baseline.n, baseline.mse_ratio_to_real, "k.-", label="All-real / adaptive / safe / validation")
        methods = PERF_ORDER[3:7]
    else:
        ax.plot(ns, np.ones(len(ns)), "k.-", label="Real-only AIPW")
        methods = PERF_ORDER
    for m in methods:
        t = table[table.method == m].sort_values("n")
        assert not t.n.duplicated().any(), (m, "Synthetic budgets must be separated")
        ax.plot(t.n, t.mse_ratio_to_real, marker=MARKERS[m], color=COLORS[m], markersize=4,
                label=LABELS[m])
    ax.set_yscale("log")
    if ylim:
        assert table.mse_ratio_to_real.min() > ylim[0] and table.mse_ratio_to_real.max() < ylim[1]
        ax.set_ylim(*ylim)
    ax.set_ylabel("MSE / all-real MSE" if adult else "MSE / real-only AIPW MSE")
    ax.set_title(title)
    xaxis(ax, ns)


def learning(df, methods, keys):
    return df[df.method.isin(methods)].dropna(subset=["beta_hat"]).groupby(keys, observed=True).beta_hat.agg(
        mean="mean", sd="std", count="count").reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data_root, args.output
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    audit = {"source": "Full-profile reproductions of the original experiments, historical code 70ac9da",
             "seed_root": 20260502, "runs": {}, "checks": {}}
    data = {}
    for key, name, rows in [("exp4", "exp4_inference", 280000), ("expA", "expA_tabular", 72000),
                            ("expB", "expB_causal", 14400)]:
        run, files, df = load_run(root, name, rows)
        data[key] = df
        info = json.loads((run / "reproduction.json").read_text())
        info["config"] = Path(info["config"]).name
        info["run"] = str(run.relative_to(root))
        audit["runs"][key] = {"path": str(run.relative_to(root)), "rows": len(df),
            "flagged_rows": int(df.failure_flag.fillna(False).sum()),
            "nonfinite_estimates_by_method": df.groupby("method", observed=True).theta_hat.apply(
                lambda x: int(x.isna().sum() + np.isinf(x.dropna().to_numpy(dtype=float)).sum())).to_dict(),
            "raw_sha256": {str(f.relative_to(root)): sha(f) for f in files},
            "reproduction": info}

    ci = data["exp4"].dropna(subset=["theta_hat", "ci_lower", "ci_upper"]).copy()
    ci["covered"] = (ci.ci_lower <= ci.theta_true) & (ci.theta_true <= ci.ci_upper)
    coverage = ci.groupby(["beta", "rho", "n", "method"], observed=True).agg(
        coverage=("covered", "mean"), n_replications=("replication", "count")).reset_index()
    assert len(coverage) == 100 and coverage.n_replications.eq(2000).all()
    cs = coverage.groupby("method", observed=True).coverage.agg(minimum="min", median="median", maximum="max")
    expected = [("0.855", "0.916", "0.955"), ("0.896", "0.930", "0.957"), ("0.913", "0.936", "0.955"),
                ("0.938", "0.949", "0.956"), ("0.938", "0.953", "0.964")]
    assert [tuple(fmt(v) for v in cs.loc[m]) for m in CI_ORDER] == expected
    coverage.to_csv(out / "coverage_per_cell.csv", index=False)
    cs.to_csv(out / "coverage_summary.csv")
    (out / "coverage_rows.tex").write_text("\n".join(
        "    " + LABELS[m] + " & " + " & ".join("$" + fmt(v) + "$" for v in cs.loc[m]) + r" \\" for m in CI_ORDER) + "\n")
    displayed = coverage[coverage.beta.isin([.5, 1.0])]
    displayed.to_csv(out / "figure6_points.csv", index=False)
    with PdfPages(figs / "fig6_coverage.pdf") as pdf:
        for beta in [.5, 1.0]:
            fig, ax = plt.subplots(figsize=(6.6, 2.5))
            for i, m in enumerate(CI_ORDER):
                t = displayed[(displayed.beta == beta) & (displayed.method == m)].sort_values("n")
                assert t.coverage.between(cs.loc[m, "minimum"], cs.loc[m, "maximum"]).all()
                ax.plot(t.n, t.coverage, marker=["o", "s", "^", "D", "v"][i], label=LABELS[m])
            ax.axhline(.95, linestyle="--", color="black", linewidth=.8)
            ax.set_ylim(.84, .975)
            ax.set_ylabel("Empirical coverage")
            ax.set_title(rf"$\beta={beta:g},\ \rho=1$")
            xaxis(ax, sorted(t.n.unique()))
            handles, labels = ax.get_legend_handles_labels()
            save_page(pdf, fig)
        legend_page(pdf, handles, labels, 3)

    at = perf_table(data["expA"], "real_only_all")
    bt = perf_table(data["expB"], "real_only_aipw")
    ast, bst = perf_summary(at), perf_summary(bt)
    assert len(at) == 351 and len(bt) == 72
    for m in PERF_ORDER[:3]:
        assert ast.loc[m, "cells"] == 45 and ast.loc[m, "maximum"] == 1 and ast.loc[m, "fallback"] == 1
        keys = ["estimand", "n", "m", "replication"]
        adata = data["expA"]
        real = adata[adata.method == "real_only_all"].set_index(keys).theta_hat.sort_index()
        other = adata[adata.method == m].set_index(keys).theta_hat.sort_index()
        assert np.array_equal(real.to_numpy(), other.to_numpy())
    assert data["expB"].query("method == 'old_fixed_share_plugin_alpha'").theta_hat.notna().sum() == 1152
    for m, expected_vals in {
        "old_fixed_share_plugin_alpha": (1.294, 1.501), "fixed_half_split_plugin_alpha": (3.519, 4.316),
        "naive_pooling": (5.439, 117.166), "synthetic_only_full_calibration": (8.874, 141.827)
    }.items():
        assert np.allclose(ast.loc[m, ["median", "maximum"]].to_numpy(float), expected_vals, atol=.00050001, rtol=0)
    for m, expected_vals in zip(PERF_ORDER, [(1.086,1.241),(1.084,1.181),(1.459,1.889),(1.760,2.264),
                                            (2.277,4.075),(1.679,2.122),(1.276,1.642),(5.415,15.472)]):
        assert np.allclose(bst.loc[m, ["median", "maximum"]].to_numpy(float), expected_vals, atol=.00050001, rtol=0)
    for name, table, summary in [("adult", at, ast), ("causal", bt, bst)]:
        table.to_csv(out / f"{name}_per_cell.csv", index=False)
        summary.to_csv(out / f"{name}_summary.csv")
        rows = []
        for m in PERF_ORDER:
            if m not in summary.index:
                continue
            row = summary.loc[m]
            rows.append("    " + LABELS[m] + " & " + " & ".join([
                f"${int(row.cells)}$", f"${fmt(row['median'])}$", f"${fmt(row.maximum)}$",
                f"${int(row.harmful)}$", f"${fmt(row.fallback)}$"]) + r" \\")
        (out / f"{name}_rows.tex").write_text("\n".join(rows) + "\n")

    a_plot = at[at.estimand.isin(["income_mean", "subgroup_gap"])]
    a_plot.to_csv(out / "figure10_points.csv", index=False)
    alim = (.88, at.mse_ratio_to_real.max() * 1.2)
    titles = {"income_mean": "Income mean", "subgroup_gap": "Subgroup gap"}
    with PdfPages(figs / "figT3_tabular_mse.pdf") as pdf:
        for mf in [1, 5, 10]:
            fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharey=True)
            for ax, estimand in zip(axes, titles):
                t = a_plot[(a_plot.estimand == estimand) & (a_plot.m_factor == mf)]
                budget = "n" if mf == 1 else f"{mf}n"
                plot_perf(ax, t, adult=True, title=f"{titles[estimand]}, $m={budget}$", ylim=alim)
            handles, labels = axes[0].get_legend_handles_labels()
            axes[1].set_ylabel("")
            save_page(pdf, fig)
        legend_page(pdf, handles, labels, 2)

    al = learning(data["expA"], ["corrected_adaptive_gn"], ["estimand", "n"])
    assert al["mean"].eq(0).all() and al.sd.eq(0).all()
    al.to_csv(out / "adult_learning.csv", index=False)
    with PdfPages(figs / "figT1_tabular_learning.pdf") as pdf:
        for estimand in titles:
            fig, ax = plt.subplots(figsize=(6.6, 2.5))
            t = al[al.estimand == estimand]
            ax.errorbar(t.n, t["mean"], yerr=t.sd, marker="o", capsize=3)
            ax.set_ylabel(r"Estimated exponent $\hat\beta$")
            ax.set_ylim(-.05, .05)
            ax.set_title(titles[estimand])
            xaxis(ax, sorted(t.n.unique()))
            save_page(pdf, fig)
    with PdfPages(figs / "figT_semisynth.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(6.6, 2.5))
        t = al[al.estimand == "income_mean"]
        ax.errorbar(t.n, t["mean"], yerr=t.sd, marker="o", capsize=3)
        ax.set_ylabel(r"Estimated exponent $\hat\beta$")
        ax.set_ylim(-.05, .05)
        ax.set_title("Income mean: estimated calibration exponent")
        xaxis(ax, sorted(t.n.unique()))
        save_page(pdf, fig)
        fig, ax = plt.subplots(figsize=(6.6, 2.65))
        t = at[(at.estimand == "income_mean") & (at.m_factor == 1)]
        t.to_csv(out / "figure11_mse_points.csv", index=False)
        plot_perf(ax, t, adult=True, title="Income mean, $m=n$", ylim=alim)
        handles, labels = ax.get_legend_handles_labels()
        save_page(pdf, fig)
        legend_page(pdf, handles, labels, 2)

    bl = learning(data["expB"], PERF_ORDER[:2], ["method", "n"])
    assert bl["count"].eq(400).all()
    bl.to_csv(out / "causal_learning.csv", index=False)
    with PdfPages(figs / "figD1_causal_learning.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        for m in PERF_ORDER[:2]:
            t = bl[bl.method == m]
            ax.errorbar(t.n, t["mean"], yerr=t.sd, marker=MARKERS[m], color=COLORS[m],
                        capsize=3, label=LABELS[m])
        ax.set_ylabel(r"Estimated exponent $\hat\beta$")
        xaxis(ax, sorted(t.n.unique()))
        ax.legend(frameon=False)
        save_page(pdf, fig)
    bt.to_csv(out / "figure13_points.csv", index=False)
    with PdfPages(figs / "figD2_ate_mse.pdf") as pdf:
        for mf in [1, 5]:
            fig, ax = plt.subplots(figsize=(6.6, 2.65))
            budget = "n" if mf == 1 else f"{mf}n"
            plot_perf(ax, bt[bt.m_factor == mf], title=f"Causal ATE, $m={budget}$",
                      ylim=(.88, bt.mse_ratio_to_real.max()*1.15))
            handles, labels = ax.get_legend_handles_labels()
            save_page(pdf, fig)
        legend_page(pdf, handles, labels, 3)

    # Deterministic grid diagnostic, using the original Experiment 3 grid.
    def risk(x1, x2, n, b1, b2):
        b = 1/n + x1**(-2*b1) + x2**(-2*b2)
        v = 1 / (n-x1-x2)
        return v*b/(v+b)
    mc = []
    for b1, b2 in [(.5, 1.), (.75, 1.5), (.4, 1.)]:
        for n in [500, 1000, 2000, 5000]:
            xs = np.arange(1, n, max(1, n//200), dtype=float)
            x1, x2 = np.meshgrid(xs, xs, indexing="ij")
            feasible = x1+x2 < n
            a, b = x1[feasible], x2[feasible]
            j = np.argmin(risk(a, b, n, b1, b2))
            a, b = a[j], b[j]
            mc.append(dict(beta1=b1, beta2=b2, n=n, step=max(1,n//200), x1=a, x2=b,
                           mv1=2*b1*a**(-2*b1-1), mv2=2*b2*b**(-2*b2-1)))
    mc = pd.DataFrame(mc)
    assert (mc.x1.gt(0) & mc.x2.gt(0) & (mc.x1+mc.x2 < mc.n)).all()
    mc["at_grid_lower_bound"] = mc.x1.eq(1) | mc.x2.eq(1)
    assert mc.at_grid_lower_bound.sum() == 8
    mc.to_csv(out / "multichannel_grid_diagnostic.csv", index=False)
    with PdfPages(figs / "figM2_marginal_values.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for (b1,b2), t in mc.groupby(["beta1","beta2"]):
            boundary = bool(t.at_grid_lower_bound.all())
            label = rf"$(\beta_1,\beta_2)=({b1:g},{b2:g})$"
            label += r": $x_1=x_2=1$" if boundary else ": interior grid"
            ax.scatter(t.mv1, t.mv2, s=50, marker="^" if boundary else "o",
                       facecolors="none" if boundary else "#2ca02c",
                       edgecolors="#1f77b4" if b1 == .4 else "#ff7f0e" if b1 == .5 else "#2ca02c",
                       label=label)
        bounds = [min(mc.mv1.min(),mc.mv2.min())*.6, max(mc.mv1.max(),mc.mv2.max())*1.5]
        ax.plot(bounds, bounds, "k--", linewidth=1, label="Continuous interior condition")
        ax.set(xscale="log", yscale="log", xlim=bounds, ylim=bounds,
               xlabel=r"Channel 1 marginal value $MV_1$", ylabel=r"Channel 2 marginal value $MV_2$")
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=9)
        save_page(pdf, fig)
    with PdfPages(figs / "figM1_risk_surface.pdf") as pdf:
        n, b1, b2 = 2000, .4, 1.
        xs = np.arange(1,n,20,dtype=float)
        x1,x2 = np.meshgrid(xs,xs)
        z = np.full(x1.shape,np.nan); valid=x1+x2<n
        z[valid]=np.log10(risk(x1[valid],x2[valid],n,b1,b2))
        fig, ax=plt.subplots(figsize=(6.4,4.3))
        mesh=ax.pcolormesh(x1,x2,z,shading="auto",cmap="viridis")
        fig.colorbar(mesh,ax=ax,label=r"$\log_{10} R_n(x_1,x_2)$")
        t=mc[(mc.n==n)&(mc.beta1==b1)].iloc[0]
        ax.scatter(t.x1,t.x2,marker="*",s=150,c="red",edgecolor="white",label="Corrected grid oracle")
        ax.scatter(n//4,n//4,marker="s",s=70,c="white",edgecolor="black",label="Equal split")
        ax.scatter(np.floor(n*2*b1/(1+2*(b1+b2))),np.floor(n*2*b2/(1+2*(b1+b2))),
                   marker="^",s=80,c="orange",edgecolor="black",label="Product-surrogate fixed share")
        ax.set(xlabel=r"Channel 1 calibration size $x_1$",ylabel=r"Channel 2 calibration size $x_2$")
        ax.set_xlim(-65, n+35)
        ax.set_ylim(-65, n+35)
        ax.legend(loc="upper right",frameon=True,fontsize=8.5)
        save_page(pdf,fig)

    audit["checks"] = {"coverage_matches_saved_summary": True, "adult_matches_saved_summary": True,
                       "causal_matches_saved_summary": True, "all_adult_adaptive_ratios_equal_one": True,
                       "synthetic_budgets_separated": True, "plotted_ranges_not_clipped": True,
                       "mc_lower_grid_boundary_cases_separately_labeled": 8,
                       "mc_interior_grid_cases": 4}
    audit["files_sha256"]={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob("*"))
                          if f.is_file() and f.name != "audit.json"}
    (out / "audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"checks":audit["checks"], "figures":len(list(figs.glob('*.pdf')))},indent=2))


if __name__ == "__main__":
    main()
