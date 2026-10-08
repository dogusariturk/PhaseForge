"""Branch oracles + honest rejection test for the current convex-only backend."""
import unittest
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from fixtures.thermodynamic_oracles import QuarticBranches
from equilibrium_solver import EquilibriumModel,NonConvexError
from cef_fixtures import fixtures

class CompetingBranchOracleTests(unittest.TestCase):
    def test_exact_degenerate_minima_and_unstable_stationary_point(self):
        m=QuarticBranches();stationary=m.stationary(0)
        np.testing.assert_allclose([r['q'] for r in stationary],[-1,0,1],atol=1e-13)
        self.assertEqual([r['kind'] for r in stationary],['minimum','maximum','minimum'])
        minima=m.global_minima(0);self.assertEqual(len(minima),2)
        np.testing.assert_allclose([r['energy'] for r in minima],0,atol=1e-25)
    def test_branch_crossing_and_metastable_minimum(self):
        m=QuarticBranches()
        for h in [-.2,-.02,.02,.2]:
            local=[r for r in m.stationary(h) if r['kind']=='minimum']
            self.assertEqual(len(local),2)
            global_states=m.global_minima(h);self.assertEqual(len(global_states),1)
            best=global_states[0];self.assertGreater(best['q']*h,0)
            self.assertGreater(max(r['energy'] for r in local)-best['energy'],0)
            # Independent bounded minimization of each basin, not one starting guess.
            independent=[minimize_scalar(lambda q:m.energy(q,h),bounds=b,method='bounded',options={'xatol':1e-13}) for b in [(-2,-.4),(.4,2)]]
            self.assertAlmostEqual(best['energy'],min(r.fun for r in independent),places=11)
        eps=1e-6;f0=m.global_minima(0)[0]['energy']
        left=(f0-m.global_minima(-eps)[0]['energy'])/eps
        right=(m.global_minima(eps)[0]['energy']-f0)/eps
        self.assertAlmostEqual(left,1.,places=5);self.assertAlmostEqual(right,-1.,places=5)
    def test_symmetry_and_envelope_derivative_away_from_crossing(self):
        m=QuarticBranches()
        for h in [.03,.17,.5]:
            p=m.global_minima(h)[0];n=m.global_minima(-h)[0]
            self.assertAlmostEqual(p['energy'],n['energy'],places=13)
            self.assertAlmostEqual(p['q'],-n['q'],places=13)
            eps=1e-6;fd=(m.global_minima(h+eps)[0]['energy']-m.global_minima(h-eps)[0]['energy'])/(2*eps)
            self.assertAlmostEqual(fd,-p['q'],places=7)

class CurrentBackendAcceptanceBoundaryTests(unittest.TestCase):
    def test_truly_nonconvex_cef_is_explicitly_rejected(self):
        theta=np.zeros(10);theta[4]=theta[5]=80000.
        temperature=900.;f=fixtures()['unequal_substitutional'];bulk=.5
        def energy(u):return f.independent_energy([[u,(4*bulk-u)/3]],temperature,theta)[0]
        # Exact curvature at u=v=.5: −L/2 +4RT/3 <0.
        analytic=-80000./2+4*8.3145*temperature/3
        self.assertLess(analytic,-20000.)
        step=1e-4;fd=(energy(.5+step)-2*energy(.5)+energy(.5-step))/step**2
        self.assertLess(abs(fd-analytic),.02)
        # Independent analytic derivatives, not the solver's cached polynomial.
        # Bounded energy minimization can stop asymmetrically near endpoints;
        # bracket each stationary point instead and verify positive curvature.
        L=theta[4];RT=8.3145*temperature
        def gradient(u):
            v=(2-u)/3
            return L*(1-2*u)/4+RT*(np.log(u)-np.log1p(-u)-np.log(v)+np.log1p(-v))/4
        def curvature(u):
            v=(2-u)/3
            return -L/2+RT*(1/(u*(1-u))+1/(3*v*(1-v)))/4
        left=brentq(gradient,1e-12,.49,xtol=5e-16,rtol=1e-14)
        right=brentq(gradient,.51,1-1e-12,xtol=5e-16,rtol=1e-14)
        for u in [left,right]:
            self.assertLess(abs(gradient(u)),1e-6)
            self.assertGreater(curvature(u),0)
            self.assertLess(energy(u),energy(.5)-1000)
        self.assertAlmostEqual(left+right,1.,places=12)
        self.assertAlmostEqual(energy(left),energy(right),places=8)
        with self.assertRaises(NonConvexError):EquilibriumModel([bulk],[temperature]).evaluate(theta)

if __name__=='__main__':unittest.main()
