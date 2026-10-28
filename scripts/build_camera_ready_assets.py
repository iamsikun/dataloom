"""Rebuild the rebuttal revision's figures, tables, and numerical audit.

Reads only the four retained full runs. Does not rerun experiments or mutate
historical results. Run with the project environment and --output DIRECTORY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'R1': 'expR1_boundary__c54575d__20260728T042624__dirty',
    'R2': 'expR2_remainder__c54575d__20260728T042624__dirty',
    'R3': 'expR3_hierarchical__c54575d__20260728T045921__dirty',
    'R4': 'expR4_adult_auxiliary__c54575d__20260728T042625__dirty',
}
COLORS = ['#176B87', '#CC7722', '#8D437D', '#487F54']


def table(out, name, caption, label, columns, header, rows, note=''):
    text = '\\begin{table}[!t]\n\\centering\n\\small\n'
    text += '\\caption{' + caption + '}\n\\label{' + label + '}\n'
    text += '\\begin{threeparttable}\n\\setlength{\\tabcolsep}{4pt}\n'
    text += '\\begin{tabular}{' + columns + '}\n\\toprule\n'
    text += ' & '.join(header) + ' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str, row)) + ' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    if note:
        text += '\\begin{tablenotes}\\footnotesize\n\\item ' + note + '\n\\end{tablenotes}\n'
    text += '\\end{threeparttable}\n\\end{table}\n'
    (out / (name + '.tex')).write_text(text)


def read(key, filename):
    return pd.read_parquet(ROOT / 'results' / RUNS[key] / 'aggregated' / (filename + '.parquet'))


def save(fig, out, stem):
    fig.savefig(out / (stem + '.pdf'), bbox_inches='tight')
    fig.savefig(out / (stem + '.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'axes.labelsize': 10, 'legend.fontsize': 9})
    audit = {'runs': {}, 'checks': {}, 'historical_results': 'Original full-run raw directories absent; preserved from manuscript, saved notebook outputs, and results/full_run_summary.md.'}
    expected = {'R1': 480000, 'R2': 63, 'R3': 2520000, 'R4': 116000}
    for key, run in RUNS.items():
        folder = ROOT / 'results' / run
        files = sorted((folder / 'raw').rglob('*.parquet'))
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        failures = sum(int(pq.read_table(f, columns=['failure_flag']).to_pandas().failure_flag.fillna(False).sum()) for f in files)
        assert rows == expected[key] and failures == 0
        audit['runs'][key] = {'path': str(folder), 'rows': rows, 'failed_rows': failures,
            'config_sha256': hashlib.sha256((folder / 'config.yaml').read_bytes()).hexdigest(),
            'aggregates_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((folder / 'aggregated').glob('*.parquet'))}}

    # Continuous six-case figure: boundary cases are lines, not shaded regions.
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout='constrained')
    x = np.linspace(.5, 2.05, 300)
    ax.axvspan(0, .5, facecolor='#F3E8D7')
    ax.fill_between(x, 0, x + .5, color='#DCECF2')
    ax.fill_between(x, x + .5, 3.0, color='#E8E0EF')
    ax.plot([0, 2.05], [0, 0], color='#555555', lw=3, clip_on=False)
    ax.plot([.5, .5], [0, 3], color='#815728', lw=2.4)
    ax.plot(x, x + .5, color='#793A78', lw=2.6)
    ax.axhline(1, color='#176B87', ls='--', lw=1.4)
    ax.text(.22, 1.7, 'II\nBounded\ncalibration', ha='center', va='center')
    ax.text(1.32, .56, 'III: growing size, vanishing share', ha='center', color='#164D61')
    ax.text(1.0, 2.5, 'IV: full calibration', ha='center')
    ax.text(1.36, 1.97, r'VI: $\rho=\beta+1/2$', rotation=28, color='#793A78', ha='center', bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none', 'pad': 2})
    ax.text(.54, 2.0, r'V: $\beta=1/2$', rotation=90, color='#815728', va='center')
    ax.text(2.02, 1.05, r'$\rho=1$', ha='right', color='#176B87')
    ax.text(1.05, -.25, r'I: persistent variance ($\rho=0$)', ha='center', color='#555555')
    ax.set(xlim=(0, 2.05), ylim=(0, 3), xlabel=r'Calibration exponent $\beta$', ylabel=r'Synthetic-variance exponent $\rho$')
    ax.xaxis.labelpad = 25
    ax.set_xticks([0, .5, 1, 1.5, 2]); ax.set_yticks([0, 1, 2, 3])
    save(fig, out, 'fig1_phase_diagram')

    boundary = read('R1', 'table_r1_boundary')
    performance = read('R1', 'table_r1_performance')
    assert np.allclose(boundary.query('n == 50000').sort_values('v0').lambda_selected, [1, .99788, .79182])
    audit['checks']['r1_max_mse_standard_errors'] = float(performance.analytic_standardized_error.abs().max())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout='constrained')
    for i, (v, g) in enumerate(boundary.groupby('v0')):
        axes[0].semilogx(g.n, g.lambda_selected, 'o-', color=COLORS[i], label=rf'$v_0={v:.3g}$')
        axes[0].axhline(g.lambda_limit.iloc[0], color=COLORS[i], ls=':', lw=1)
    axes[0].set(xlabel='Real budget n', ylabel='Oracle calibration share', ylim=(.74, 1.015))
    axes[0].legend(loc='lower right')
    g=boundary[np.isclose(boundary.v0, np.sqrt(2))].sort_values('n')
    axes[1].loglog(g.n, g.n-g.x_selected, 'o-', color=COLORS[1], label='Equality case')
    axes[1].loglog(g.n, (g.n-g.x_selected).iloc[-1]*np.sqrt(g.n/g.n.iloc[-1]), ':', color='gray', label='Slope 1/2 reference')
    axes[1].set(xlabel='Real budget n', ylabel='Real observations left (n - x)')
    axes[1].legend()
    for ax in axes: ax.grid(alpha=.2)
    save(fig, out, 'figR1_boundary')
    table(out, 'table_r1', r'Equality-boundary allocations at $n=50{,}000$ ($\beta=1,\rho=3/2,a=c=1$).', 'tab:r1', 'rrrr', ['$v_0$', 'Limiting share', 'Observed share', 'Absolute error'],
          [[f'{r.v0:.5f}',f'{r.lambda_limit:.5f}',f'{r.lambda_selected:.5f}',f'{r.absolute_share_error:.5f}'] for r in boundary.query('n == 50000').itertuples()])
    z=read('R1','table_r1_zero_variance_oracle')
    table(out,'table_zero_variance',r'Genuine zero-variance working-risk oracle, $a=c=1,n=10{,}000$.','tab:zero-variance','rrrr',[r'$\beta$','Selected $x$','Number of ties','Product-surrogate share'],
          [[f'{r.beta:g}',str(r.x_star_tiebreak),str(r.n_minimizers),f'{r.product_surrogate_share:.3f}'] for r in z.itertuples()],r'Ties select the smallest allocation. For $\beta<1/2$, the minimizers are $0$ and $1$; at $1/2$, all $10{,}001$ integer allocations tie.')

    slopes=read('R2','table_r2_slopes'); div=read('R2','table_r2_diverging'); bounded=read('R2','table_r2_bounded'); plateau=read('R2','table_r2_plateau')
    names={'exact':'Exact power law','positive_remainder':'Positive remainder','negative_remainder':'Negative remainder','oscillatory_remainder':'Oscillatory remainder'}
    table(out,'table_r2_slopes',r'Regime-III sensitivity: $\beta=\rho=a=c=1$, theoretical slope $2/3$.','tab:r2-slopes','lrrr',['Curve','Fitted slope','$x$ at $10^6$','Risk / exact'],
          [[names[r.curve_family],f'{r.empirical_slope:.4f}',str(r.x_at_largest_n),f'{r.risk_ratio_at_largest_n:.6f}'] for r in slopes.itertuples()])
    table(out,'table_r2_bounded',r'Bounded sensitivity for $b_2(x)=0.1x^{-1/2}+q x^{-1}$ at $n=10^6$.','tab:r2-bounded','rrl',['$q$','Selected integer $x$','Original remainder assumption'],
          [[f'{r.remainder_scale:g}',str(r.x_selected),'Satisfied' if abs(r.remainder_scale)<.1 else 'Outside-assumption stress test'] for r in bounded.itertuples()])
    assert list(bounded.sort_values('remainder_scale').x_selected)==[24,25,10,0]
    assert int(plateau.loc[plateau.n.idxmax(),'x_selected'])==120
    fig, axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    for i,(family,g) in enumerate(div.groupby('curve_family')):
        axes[0].loglog(g.n.astype(float),g.x_selected.astype(float),'o-',color=COLORS[i],label=names[family])
    axes[0].set(xlabel='Real budget n',ylabel='Selected calibration size'); axes[0].legend(fontsize=8)
    axes[1].semilogx(plateau.n.astype(float),plateau.x_selected.astype(float),'o-',color=COLORS[0])
    axes[1].axhline(120,ls=':',color='gray');axes[1].set(xlabel='Real budget n',ylabel='Calibration size with bias floor')
    for ax in axes: ax.grid(alpha=.2)
    save(fig,out,'figR2_remainder')

    r3=read('R3','table_r3_performance')
    head=r3.query('beta == 1 and n == 10000 and validation_size == 10000 and redraws_per_fit == 8')
    direct=head[head.method=='direct_grid_plugin'].sort_values('pilot_fits')
    buffered=head[head.method=='uncertainty_buffered_direct_grid'].sort_values('pilot_fits')
    rows=[]
    for d,b in zip(direct.itertuples(),buffered.itertuples()):
        rows.append([str(d.pilot_fits),f'{d.median_uniform_curve_error:.3f}',f'{d.mse_ratio_to_real:.3f}',f'{b.mse_ratio_to_real:.3f}',f'{100*b.fallback_rate:.2f}'])
    table(out,'table_r3',r'R3 headline cell: $\beta=\rho=1,n=n_V=10{,}000$, eight redraws per fitted generator.','tab:r3','rrrrr',['Fits','Median $C_B$','Direct MSE/real','Buffered MSE/real',r'Fallback (\%)'],rows,
          r'$C_B=\max_x\max\{\widehat B_n(x)/B_n(x),B_n(x)/\widehat B_n(x)\}$ measures synthetic-error distortion, not the risk factor $C_0$ in Theorem~\ref{thm:adaptive}. MSE ratios use 2,000 replications; fallback refers to the buffered rule. Independent final draws allow Monte Carlo ratios slightly above one even when fallback is universal.')
    false=r3.groupby('method').false_safe_rate.max()
    assert np.isclose(false['direct_grid_plugin'],.9945) and np.isclose(false['uncertainty_buffered_direct_grid'],.0915)
    audit['checks']['r3_worst_false_safe']=false.to_dict()
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    axes[0].plot(direct.pilot_fits,direct.median_uniform_curve_error,'o-',color=COLORS[0]);axes[0].set(xlabel='Independent calibration fits',ylabel='Median synthetic-error distortion $C_B$',xticks=[1,4,16])
    for i,(d,label) in enumerate([(direct,'Direct plug-in'),(buffered,'Buffered plug-in')]):
        axes[1].plot(d.pilot_fits,d.mse_ratio_to_real,'o-',color=COLORS[i],label=label)
    axes[1].axhline(1,ls=':',color='gray');axes[1].set(xlabel='Independent calibration fits',ylabel='Empirical MSE / all-real',xticks=[1,4,16]);axes[1].legend()
    for ax in axes: ax.grid(alpha=.2)
    save(fig,out,'figR3_adaptive')

    perf=read('R4','table_r4_performance'); curve=read('R4','table_r4_curve')
    direct=perf[perf.method=='externally_piloted_direct_grid'].sort_values(['n','estimand'])
    assert len(direct)==6 and (direct.ratio_ci_high<1).all()
    assert sum(direct.x_selected==direct.n)==5
    label={'income_mean':'Income mean','subgroup_gap':'Subgroup gap'}
    table(out,'table_r4_main',r'Adult auxiliary-data results. A policy learned from external pilots is evaluated on 2,000 fresh empirical-population replications per budget.','tab:r4-main','rlrrl',['$n$','Target','$x/n$',r'$\alpha$',r'MSE / real (95\% CI)'],
          [[f'{r.n:,}',label[r.estimand],f'{r.lambda_selected:.2f}',f'{r.alpha_selected:.3f}',f'{r.mse_ratio_to_real:.3f} [{r.ratio_ci_low:.3f}, {r.ratio_ci_high:.3f}]'] for r in direct.itertuples()],
          r'All intervals are paired percentile bootstrap intervals for the MSE ratio, conditional on the learned policy, not confidence intervals for the estimand. Each budget uses 1,000 external pilot fits per grid point, 5,000 validation rows per pilot replication, and 10,000 auxiliary covariates. Five cells use full calibration; the income mean at $n=2{,}000$ uses an 80\% share. The buffered policy makes the same selections.')
    methods=['synthetic_only_full_calibration','product_surrogate_fixed_share_oracle_alpha','empirical_grid_oracle','externally_piloted_direct_grid','uncertainty_buffered_external_grid']
    rows=[]
    for (n,e),g in perf.groupby(['n','estimand']):
        ratios=g.set_index('method').mse_ratio_to_real
        rows.append([str(n),label[e]]+[f'{ratios[m]:.3f}' for m in methods])
    table(out,'table_r4_all',r'R4 MSE ratios for all augmentation comparators (all-real equals one).','tab:r4-all','rlrrrrr',['$n$','Target','Synth.','Product','Emp. oracle','External','Buffered'],rows,
          r'The empirical grid oracle uses pilot errors relative to the known empirical-population target. The product heuristic uses an externally estimated exponent and mixing weight; its historical identifier ending in oracle\_alpha does not mean that its weight is known. External and buffered policies use the validation-based error curve.')
    table(out,'table_r4_diagnostics',r'Local calibration-curve diagnostics for R4.','tab:r4-diagnostics','rlrr',['$n$','Target',r'$\hat\beta$','$R^2$'],[[str(r.n),label[r.estimand],f'{r.beta_hat:.3f}',f'{r.powerlaw_r2:.4f}'] for r in direct.itertuples()],r'The fitted log--log slope of squared calibration error is $-2\hat\beta$. These are local descriptive fits. All six estimated total-error curves have zero monotonicity violations.')
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    for ax,(e,label_e) in zip(axes,label.items()):
        for i,(n,g) in enumerate(curve[curve.estimand==e].groupby('n')):
            ax.loglog(g.x_selected,np.maximum(g.score_mean,1e-12),'o-',color=COLORS[i],label=f'n={n:,}')
        ax.set(xlabel='Calibration observations',ylabel='Estimated squared calibration error',title=label_e);ax.legend();ax.grid(alpha=.2)
    save(fig,out,'figR4_learning')
    audit['checks']['r4_main']=direct.to_dict('records')
    audit['checks']['allocation_risk_ratio_563_573']=float(((1e-4+563**-2)/(1+(10000-563)*(1e-4+563**-2)))/((1e-4+573**-2)/(1+(10000-573)*(1e-4+573**-2))))
    (out/'evidence.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(f'Created 5 figures and 8 tables in {out}; raw counts and headline assertions passed.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
