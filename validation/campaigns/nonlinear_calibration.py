"""Opt-in prior-predictive calibration of internally minimized two-parameter CEF.
No resume: existing output directories are refused; partial records remain durable.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,sys,time,platform
from datetime import datetime,timezone
import importlib.metadata
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import logsumexp
from scipy.stats import truncnorm,binomtest,norm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prototype.equilibrium_solver import EquilibriumModel
from prototype.noise_marginal import log_marginal
CENTER=np.array([-40000.,-31000.,-23000.,-12000.,18000.,-9000.,7000.,13000.,22000.,-4.])
FREE=np.array([1,4]);SD=7000.;BOUND=15000.;S0=150.;ALPHA=.01

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def exit_status(status):
    """Release-check contract: success=0, statistical rejection=1, unresolved=2."""
    codes={'no_calibration_failure_detected':0,'calibration_rejected':1,'inconclusive_grid_resolution':2}
    if status not in codes:raise ValueError('Unknown campaign status: '+str(status))
    return codes[status]

def calibration_tests(cdfs,alpha=ALPHA):
    """Ten prespecified exact-binomial tests; Bonferroni familywise alpha."""
    c=np.asarray(cdfs,float)
    if c.ndim!=2 or c.shape[1]!=2 or not len(c) or not np.isfinite(c).all() or np.any((c<0)|(c>1)):
        raise ValueError('Require finite n by2 CDF probabilities')
    rows=[]
    for k in range(2):
        for b in range(4):
            hit=(c[:,k]>=b/4)&((c[:,k]<(b+1)/4) if b<3 else (c[:,k]<=1))
            count=int(hit.sum());p=float(binomtest(count,len(c),.25).pvalue)
            rows.append(dict(parameter=k,test=f'CDF_quartile_{b}',count=count,n=len(c),null_probability=.25,pvalue=p))
        count=int(((c[:,k]>=.025)&(c[:,k]<=.975)).sum())
        rows.append(dict(parameter=k,test='equal_tail_95percent_coverage',count=count,n=len(c),null_probability=.95,pvalue=float(binomtest(count,len(c),.95).pvalue)))
    for row in rows:row['reject']=bool(row['pvalue']<alpha/len(rows))
    return dict(familywise_alpha=alpha,individual_threshold=alpha/len(rows),tests=rows,rejected=any(r['reject'] for r in rows))

def posterior(axis,prediction,observed,truth):
    n=len(axis);spacing=axis[1]-axis[0];trap=np.ones(n);trap[[0,-1]]=.5
    rss=np.sum((prediction-observed)**2,axis=-1)
    aa,bb=np.meshgrid(axis,axis,indexing='ij')
    lp=log_marginal(rss,len(observed),S0)-.5*((aa/SD)**2+(bb/SD)**2)
    lw=lp+np.log(trap[:,None])+np.log(trap[None,:]);w=np.exp(lw-logsumexp(lw))
    cdfs=[];means=[];sds=[];quantiles=[]
    for k in range(2):
        mass=w.sum(axis=1-k);density=mass/(trap*spacing)
        cdf=cumulative_trapezoid(density,axis,initial=0);cdf/=cdf[-1]
        cdfs.append(float(np.interp(truth[k],axis,cdf)));quantiles.append(np.interp([.025,.975],cdf,axis).tolist())
        mean=float(np.sum(mass*axis));means.append(mean);sds.append(float(np.sqrt(np.sum(mass*(axis-mean)**2))))
    edge=(abs(aa)>.9*BOUND)|(abs(bb)>.9*BOUND)
    return dict(cdfs=cdfs,mean=means,sd=sds,quantiles=quantiles,boundary_outer5percent_mass=float(w[edge].sum()))

def converged(coarse,fine):
    sd=np.asarray(fine['sd'])
    metrics=dict(cdf_difference=float(np.max(abs(np.array(coarse['cdfs'])-fine['cdfs']))),mean_difference_sd=float(np.max(abs(np.array(coarse['mean'])-fine['mean'])/sd)),sd_relative_difference=float(np.max(abs(np.array(coarse['sd'])/sd-1))),credible_bound_difference_sd=float(np.max(abs(np.array(coarse['quantiles'])-fine['quantiles'])/sd[:,None])))
    return dict(**metrics,passes=bool(metrics['cdf_difference']<.005 and metrics['mean_difference_sd']<.01 and metrics['sd_relative_difference']<.01 and metrics['credible_bound_difference_sd']<.03))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--replicates',type=int,default=32);ap.add_argument('--seed',type=int,default=202610073);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-grid',type=int,choices=[321,641],default=641);a=ap.parse_args()
    if a.replicates<1:ap.error('replicates must be positive')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    x=np.repeat([.15,.25,.4,.5,.6,.75,.85],3);T=np.tile([2200.,2500.,2800.],7);model=EquilibriumModel(x,T)
    bounds=[]
    for shifts in itertools.product([-BOUND,BOUND],repeat=2):
        th=CENTER.copy();th[FREE]+=shifts;bounds.append(float(model.curvature_bound(th).min()))
    if min(bounds)<=0:raise ValueError('Entire prior rectangle not certified convex')
    prior_edge_mass=1-((norm.cdf(.9*BOUND/SD)-norm.cdf(-.9*BOUND/SD))/(norm.cdf(BOUND/SD)-norm.cdf(-BOUND/SD)))**2
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),environment=dict(machine=platform.machine(),python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['numpy','scipy']}),scope='Repeated prior-predictive coefficient-CDF and equal-tail credible-interval calibration, internally minimized single-phase likelihood; NOT multiphase or real-data calibration',replicates=a.replicates,seed=a.seed,free_indices=FREE.tolist(),center=CENTER.tolist(),coefficient_prior='Independent Gaussian SD7000 truncated EXACTLY at +/-15000 J/mol formula unit; this bounded distribution is the scientific benchmark prior, not a numerical truncation',noise_prior='HalfNormal150 J/mol atoms, untruncated, analytically marginalized; no minimum sigma',grid_levels=[161,321]+([641] if a.max_grid==641 else []),grid_rule='Nested trapezoidal quadrature on full prior support, convergence in CDF<.005, mean<.01 posteriorSD, sd relative<.01, credible bounds<.03 posteriorSD; any unresolved replicate makes campaign inconclusive, none discarded',familywise_alpha=ALPHA,binomial_tests=10,prior_boundary_outer5percent_mass=float(prior_edge_mass),convexity_corner_lower_bounds=bounds,observation_x=x.tolist(),observation_T=T.tolist(),source_hashes={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'prototype/equilibrium_solver.py',Path(__file__).resolve().parents[1]/'prototype/noise_marginal.py']},resume='Unsupported; existing output refused')
    save(out/'specification.json',manifest)
    rng=np.random.default_rng(a.seed);truths=truncnorm.rvs(-BOUND/SD,BOUND/SD,scale=SD,size=(a.replicates,2),random_state=rng);sigmas=abs(rng.normal(0,S0,a.replicates));observations=[]
    for truth,sigma in zip(truths,sigmas):
        th=CENTER.copy();th[FREE]+=truth;observations.append(model.evaluate(th)['energy']+rng.normal(0,sigma,len(x)))
    np.savez_compressed(out/'simulated_datasets.npz',coefficient_shifts=truths,sigmas=sigmas,observations=observations)
    def precompute(n):
        axis=np.linspace(-BOUND,BOUND,n);pred=np.empty((n,n,len(x)))
        for i,u in enumerate(axis):
            for j,v in enumerate(axis):
                th=CENTER.copy();th[FREE]+=[u,v];pred[i,j]=model.evaluate(th)['energy']
            if i%20==0:save(out/'progress.json',dict(stage='grid_predictions',size=n,row=i,total=n,seconds=time.monotonic()-start));print('GRID',n,i,flush=True)
        np.savez_compressed(out/f'predictions_{n}.npz',axis=axis,energy=pred)
        return axis,pred
    axis,pred=precompute(321);records=[]
    for i,(obs,truth) in enumerate(zip(observations,truths)):
        coarse=posterior(axis[::2],pred[::2,::2],obs,truth);fine=posterior(axis,pred,obs,truth)
        records.append(dict(index=i,truth_shift=truth.tolist(),sigma=float(sigmas[i]),levels={'161':coarse,'321':fine},comparison=converged(coarse,fine),selected=fine))
    for r in records:save(out/f"replicate_{r['index']:04d}.json",r)
    if any(not r['comparison']['passes'] for r in records) and a.max_grid==641:
        axis,pred=precompute(641)
        for r,obs,truth in zip(records,observations,truths):
            if r['comparison']['passes']:continue
            fine=posterior(axis,pred,obs,truth);r['levels']['641']=fine;r['comparison']=converged(r['selected'],fine);r['selected']=fine
    unresolved=[r['index'] for r in records if not r['comparison']['passes']]
    for r in records:save(out/f"replicate_{r['index']:04d}.json",r)
    tests=calibration_tests([r['selected']['cdfs'] for r in records]) if not unresolved else None
    result=dict(status='inconclusive_grid_resolution' if unresolved else ('calibration_rejected' if tests['rejected'] else 'no_calibration_failure_detected'),unresolved_replicates=unresolved,calibration=tests,replicates=a.replicates,minimum_simulated_sigma=float(sigmas.min()),maximum_posterior_boundary_mass=max(r['selected']['boundary_outer5percent_mass'] for r in records),seconds=time.monotonic()-start,interpretation='Non-rejection is limited-power evidence under the exact manufactured prior/model only; not proof of calibration. Sigma marginalization is exact but sigma posterior CDF calibration is not tested here. No evidence-normalization claim.')
    save(out/'summary.json',result);save(out/'progress.json',dict(stage='completed',status=result['status'],seconds=time.monotonic()-start));print(json.dumps(result,indent=2),flush=True)
    return exit_status(result['status'])
if __name__=='__main__':sys.exit(main())
