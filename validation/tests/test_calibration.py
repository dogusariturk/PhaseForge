"""Prior-predictive calibration, not physical or conditional frequentist coverage.
Run larger campaigns via PHASEFORGE_CALIBRATION_REPS; default80 is bounded CI.
"""
import json
import os
import unittest
import numpy as np
from scipy.stats import binom
from bayes_linear import Posterior, mm
from cef_fixtures import fixtures


def binomial_band(n,p,alpha):
    return int(binom.ppf(alpha/2,n,p)),int(binom.ppf(1-alpha/2,n,p))


def calibration_campaign(repetitions=80,seed=730120):
    if repetitions<80:raise ValueError('At least80 independent datasets required')
    rng=np.random.default_rng(seed)
    fixture=fixtures()['unequal_substitutional']
    q=rng.uniform(.08,.92,(36,2));t=np.tile([600.,900.,1200.],12)
    X=fixture.design_matrix(q,t);H=fixture.design_matrix([[.37,.68]],1050.)
    tau=np.r_[np.full(9,5000.),5.];noise_scale=100.
    counts={f'{kind}_{level}':0 for kind in ['latent','predictive'] for level in [.8,.95]}
    ranks=[]
    convergence_max={"mean_in_fine_sd":0.,"sd_relative":0.,"interval_endpoint_in_fine_sd":0.}
    for _ in range(repetitions):
        truth=rng.normal(size=10)*tau;sigma=abs(rng.normal())*noise_scale
        y=mm(X,truth)+rng.normal(size=len(X))*sigma
        target=float(mm(H,truth)[0]);future=target+rng.normal()*sigma
        coarse=Posterior(X,y,tau,noise_scale=noise_scale,sigma_bounds=(1e-7,1e5),grid_size=801)
        fit=Posterior(X,y,tau,noise_scale=noise_scale,sigma_bounds=(1e-7,1e5),grid_size=1601)
        fine_sd=np.sqrt(np.diag(fit.covariance))
        mean_error=float(np.max(np.abs(coarse.mean-fit.mean)/fine_sd))
        sd_error=float(np.max(np.abs(np.sqrt(np.diag(coarse.covariance))/fine_sd-1)))
        convergence_max['mean_in_fine_sd']=max(convergence_max['mean_in_fine_sd'],mean_error)
        convergence_max['sd_relative']=max(convergence_max['sd_relative'],sd_error)
        if mean_error>=1e-4 or sd_error>=1e-4:
            raise AssertionError(f'Dataset {_}: noise quadrature moments not converged: {mean_error}, {sd_error}')
        cached={}
        latent_variance=float(np.einsum('ni,ij,nj->n',H,fit.covariance,H,optimize=False)[0])
        for kind in ['latent','predictive']:
            predictive=kind=='predictive'
            target_sd=np.sqrt(latent_variance+(float(np.sum(fit.weights*fit.sigma**2)) if predictive else 0.))
            for level in [.8,.95]:
                interval=fit.intervals(H,predictive=predictive,level=level)[0]
                error=float(np.max(np.abs(interval-coarse.intervals(H,predictive=predictive,level=level)[0]))/target_sd)
                convergence_max['interval_endpoint_in_fine_sd']=max(convergence_max['interval_endpoint_in_fine_sd'],error)
                if error>=1e-3:
                    raise AssertionError(f'Dataset {_}: {kind} {level} interval not converged: {error}')
                cached[(kind,level)]=interval
        for kind,value in [('latent',target),('predictive',future)]:
            for level in [.8,.95]:
                lo,hi=cached[(kind,level)]
                counts[f'{kind}_{level}']+=int(lo<=value<=hi)
        draws,_=fit.sample(99,seed=int(rng.integers(0,2**32-1)))
        ranks.append(int(np.sum(draws[:,0]<truth[0])))
    # Four inclusion statistics + ten rank-bin counts, Bonferroni familywise .01.
    # Each dataset contributes only one independent Bernoulli to each statistic.
    per_test_alpha=.01/14
    bands={key:binomial_band(repetitions,float(key.rsplit('_',1)[1]),per_test_alpha) for key in counts}
    histogram=np.histogram(ranks,bins=np.arange(0,101,10))[0]
    rank_band=binomial_band(repetitions,.1,per_test_alpha)
    return dict(repetitions=repetitions,seed=seed,nominal_familywise_alpha=.01,
                inclusion_counts=counts,acceptance_bands=bands,
                quadrature_nodes=[801,1601],quadrature_max_errors=convergence_max,
                quadrature_thresholds=dict(mean_in_fine_sd=1e-4,sd_relative=1e-4,interval_endpoint_in_fine_sd=1e-3),
                rank_histogram=histogram.tolist(),rank_bin_acceptance_band=rank_band,
                scope='Prior-predictive fixed-site10-coefficient CEF; known ideal offset removed; not nonlinear equilibrium calibration')


class TestCalibration(unittest.TestCase):
    def test_repeated_prior_predictive_calibration(self):
        report=calibration_campaign(int(os.getenv('PHASEFORGE_CALIBRATION_REPS','80')))
        print('\nCALIBRATION '+json.dumps(report),flush=True)
        for key,count in report['inclusion_counts'].items():
            lo,hi=report['acceptance_bands'][key]
            with self.subTest(statistic=key):self.assertTrue(lo<=count<=hi,(key,count,(lo,hi)))
        lo,hi=report['rank_bin_acceptance_band']
        for i,count in enumerate(report['rank_histogram']):
            with self.subTest(rank_bin=i):self.assertTrue(lo<=count<=hi,(i,count,(lo,hi)))

    def test_missing_interaction_detected_on_independent_holdouts(self):
        rng=np.random.default_rng(730121);f=fixtures()['unequal_substitutional']
        q=rng.uniform(.03,.97,(90,2));h=rng.uniform(.03,.97,(100,2))
        t=np.tile([600.,900.,1200.],30);ht=np.tile([750.,1050.],50)
        X=f.design_matrix(q,t);H=f.design_matrix(h,ht)
        # Smooth manufactured term outside the polynomial CEF basis.
        omitted=lambda z:np.sin(6*np.pi*z[:,0])*np.sin(4*np.pi*z[:,1])
        z=omitted(q);hz=omitted(h);amplitude=2000.;sigma=25.
        tau=np.r_[np.full(9,50000.),50.]
        y=mm(X,f.nominal_parameters)+amplitude*z+rng.normal(size=len(q))*sigma
        truth=mm(H,f.nominal_parameters)+amplitude*hz
        fit=lambda A,p:Posterior(A,y,p,noise_scale=2000.,sigma_bounds=(1e-5,1e6),grid_size=801)
        bad=fit(X,tau);good=fit(np.column_stack([X,z]),np.r_[tau,5000.])
        bad_residual=truth-mm(H,bad.mean)
        good_residual=truth-mm(np.column_stack([H,hz]),good.mean)
        rmse_bad=float(np.sqrt(np.mean(bad_residual**2)));rmse_good=float(np.sqrt(np.mean(good_residual**2)))
        corr=float(np.corrcoef(bad_residual,hz)[0,1])
        print('\nMISSPECIFICATION '+json.dumps(dict(omitted_RMSE=rmse_bad,augmented_RMSE=rmse_good,residual_missing_term_correlation=corr)),flush=True)
        self.assertGreater(rmse_bad,5*rmse_good)
        self.assertGreater(corr,.7)
        # This intentionally severe manufactured control is not a universal detector.

if __name__=='__main__':unittest.main()
