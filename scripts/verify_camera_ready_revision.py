"""Reconcile camera-ready claims with retained raw runs; never modify results."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from build_camera_ready_assets import ROOT, RUNS, read


def raw(k):
    return pd.concat([pd.read_parquet(f) for f in sorted((ROOT/'results'/RUNS[k]/'raw').rglob('*.parquet'))], ignore_index=True)


def close(a,b):
    np.testing.assert_allclose(np.asarray(a,float), np.asarray(b,float),rtol=2e-10,atol=1e-13)


def risk(x,n,a,B):
    x=np.asarray(x,float)
    return np.where(x==0,a/n,a*B/(a+(n-x)*B))


def main(out):
    report={}
    r=raw('R1'); p=read('R1','table_r1_performance')
    rows=[]
    for keys,g in r.groupby(['v0','n','method'],observed=True):
        v,n,m=keys; saved=p[(p.v0==v)&(p.n==n)&(p.method==m)].iloc[0]
        e=(g.theta_hat.astype(float)-g.theta_true.astype(float))**2
        close([e.mean(),e.std()/np.sqrt(len(e)),g.x_selected.iloc[0]], [saved.mse,saved.mse_se,saved.x_selected])
        if m=='corrected_oracle_gn':
            x=np.arange(1,n+1); B=v*n**-1.5+x.astype(float)**-2
            risks=np.r_[1/n,risk(x,n,1,B)]
            assert int(np.argmin(risks))==int(saved.x_selected)
        rows.append(abs((e.mean()-saved.analytic_risk)/(e.std()/np.sqrt(len(e)))))
    report['R1']={'method_cells_recomputed':len(rows),'max_standardized_mse_error':max(rows)}
    r=raw('R2'); slopes=read('R2','table_r2_slopes'); checks=[]
    for q in r.itertuples():
        x=np.arange(1,q.n+1,dtype=float)
        if q.curve_family=='bounded_remainder': b=.1*x**-.5+q.remainder_scale/x
        elif q.curve_family=='plateau': b=.001+x**-2
        else:
            b=x**-2
            if q.curve_family=='positive_remainder': b+=.5*x**-2.5
            elif q.curve_family=='negative_remainder': b-=.5*x**-2.5
            elif q.curve_family=='oscillatory_remainder': b+=.5*np.sin(3*np.log(x))*x**-2.5
        risks=np.r_[1/q.n,risk(x,q.n,1,1/q.n+b)]
        assert float(risks[int(q.x_selected)]/risks.min()-1)<1e-8
        if int(np.argmin(risks)) != int(q.x_selected):
            report.setdefault('R2_saved_near_minimizers',[]).append({'curve':q.curve_family,'n':int(q.n),'reported':int(q.x_selected),'strict_argmin':int(np.argmin(risks)),'relative_risk_difference':float(risks[int(q.x_selected)]/risks.min()-1)})
        close(risks.min(),q.true_risk_selected)
        checks.append(int(q.x_selected))
    for q in slopes.itertuples():
        g=r[r.curve_family==q.curve_family]
        close(np.polyfit(np.log(g.n.astype(float)),np.log(g.x_selected.astype(float)),1)[0],q.empirical_slope)
    report['R2']={'saved_integer_selections_checked':len(checks),'slopes':slopes[['curve_family','empirical_slope']].to_dict('records')}
    r=raw('R3'); p=read('R3','table_r3_performance')
    diag=r[r.method=='score_diagnostic']; r=r[r.method!='score_diagnostic'].copy()
    r['se']=(r.theta_hat.astype(float)-r.theta_true.astype(float))**2
    r['fs']=r.safe_pass.fillna(False).astype(bool)&(r.x_selected.astype(float)>0)&(r.x_selected.astype(float)*r.B_true.fillna(0).astype(float)>=r.a.astype(float))
    keys=['beta','rho','n','validation_size','pilot_fits','redraws_per_fit','method']
    rec=r.groupby(keys,observed=True).agg(mse=('se','mean'),false_safe_rate=('fs','mean'),fallback_rate=('fallback_used','mean'),median_uniform_curve_error=('uniform_curve_error','median'),median_working_risk_ratio=('working_risk_ratio','median'))
    p=p.set_index(keys).sort_index(); rec=rec.sort_index()
    for c in rec: close(rec[c],p[c])
    real=rec.xs('real_only_all',level='method').mse
    close(rec.mse/real.reindex(rec.index.droplevel('method')).to_numpy(),p.mse_ratio_to_real)
    saved=read('R3','table_r3_score').set_index(keys[:-1]+['x_selected']).sort_index()
    d=diag.groupby(keys[:-1]+['x_selected'],observed=True).score_untruncated.agg(['mean','std']).sort_index()
    close(d['mean'],saved.score_mean);close(d['std'],saved.score_sd)
    report['R3']={'method_cells_recomputed':len(rec),'score_cells_recomputed':len(d),'method_rows':len(r),'diagnostic_rows':len(diag),'worst_false_safe':rec.groupby('method').false_safe_rate.max().to_dict()}
    r=raw('R4'); p=read('R4','table_r4_performance'); pilots=r[r.method=='external_pilot_curve']; r=r[r.method!='external_pilot_curve'].copy()
    r['se']=(r.theta_hat.astype(float)-r.theta_true.astype(float))**2
    checked=[]
    for (n,e),g in r.groupby(['n','estimand'],observed=True):
        baseline=g[g.method=='real_only_all'].sort_values('replication')
        real=baseline.se.to_numpy(float)
        for m,h in g.groupby('method',observed=True):
            h=h.sort_values('replication'); assert h.replication.tolist()==baseline.replication.tolist()
            errors=h.se.to_numpy(float); q=p[(p.n==n)&(p.estimand==e)&(p.method==m)].iloc[0]
            close([errors.mean(),errors.mean()/real.mean(),h.x_selected.iloc[0],h.alpha_selected.iloc[0]], [q.mse,q.mse_ratio_to_real,q.x_selected,q.alpha_selected])
            if m=='externally_piloted_direct_grid':
                rng=np.random.default_rng(20260727+int(n)+sum(map(ord,m)))
                ix=rng.integers(0,len(real),size=(2000,len(real)))
                ratios=errors[ix].mean(1)/real[ix].mean(1)
                ci=np.quantile(ratios,[.025,.975]);close(ci,[q.ratio_ci_low,q.ratio_ci_high]);assert ci[1]<1
                checked.append({'n':int(n),'estimand':e,'ratio':q.mse_ratio_to_real,'ci':ci.tolist(),'x':int(q.x_selected),'alpha':q.alpha_selected})
    d=pilots.groupby(['n','estimand','x_selected'],observed=True).score_untruncated.mean().sort_index()
    q=read('R4','table_r4_curve').set_index(['n','estimand','x_selected']).sort_index()
    close(d,q.score_mean)
    report['R4']={'method_cells_recomputed':len(p),'evaluation_rows':len(r),'pilot_rows':len(pilots),'main_table_and_bootstrap_recomputed':checked}
    # Deterministic algebra and limiting comparisons independent of the oracle API.
    for beta in [.25,.4,.5,.75,1,1.5]:
        n=10000; x=np.arange(1,n+1,dtype=float); reciprocal=np.r_[n,n-x+x**(2*beta)]
        ties=np.flatnonzero(np.isclose(reciprocal,reciprocal.max(),rtol=0,atol=1e-8)).tolist()
        assert ties==([0,1] if beta<.5 else list(range(n+1)) if beta==.5 else [n])
    for a,c,v in [(1,1,.0001),(4,1,.0001),(4,1,.001),(4,1,.01)]:
        n=1000; x=np.arange(1,n+1,dtype=float); B=v+c/x
        oracle=int(np.argmin(np.r_[a/n,risk(x,n,a,B)]))
        candidates=[0,n]
        if a>c: candidates+=list({int(np.floor((np.sqrt(a*c)-c)/v)),int(np.ceil((np.sqrt(a*c)-c)/v))})
        candidates=[z for z in candidates if 0<=z<=n]
        assert oracle in candidates
    for n in [1000,10000,100000]:
        x=24.; B=.1*x**-.5-.05/x
        scaled=n*n*(float(risk(x,n,1,B))-1/n)
        assert abs(scaled-(x-1/B))<1500/n
    n=10000
    ratio=float(risk(563,n,1,1/n+563**-2)/risk(573,n,1,1/n+573**-2))
    assert f'{ratio:.6f}'=='1.000014'
    # Common-weight quadratic-loss reduction and stationary point.
    W=np.array([[2.,.2],[.2,1.]])
    SR=np.array([[1.,.1],[.1,3.]])/900; MS=np.array([[.002,.001],[.001,.004]])
    A=float(np.trace(W@SR)); B=float(np.trace(W@MS)); alpha=B/(A+B)
    close(np.trace(W@(alpha**2*SR+(1-alpha)**2*MS)),A*B/(A+B))
    assert alpha**2*A+(1-alpha)**2*B <= min(A,B)
    report['mathematics']={'zero_variance_ties':'passed, six exponents','beta_half_constants':'passed, four cases','bounded_second_order':'passed at three budgets','followup_ratio':ratio,'common_weight_vector_identity':'passed'}
    out.write_text(json.dumps(report,indent=2)+'\n');print('All raw-data reconciliations and deterministic mathematical checks passed:',out)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);main(parser.parse_args().output)
