"""Batched solver parity and acceptance-boundary tests; no licensed dependency."""
import itertools
import unittest
import warnings
import numpy as np
from batched_equilibrium import BatchedEquilibriumModel
from equilibrium_solver import EquilibriumModel, NonConvexError
from cef_fixtures import fixtures


class BatchedEquilibriumTests(unittest.TestCase):
    def setUp(self):
        self.theta = fixtures()['unequal_substitutional'].nominal_parameters.copy()
        self.x = np.r_[0., 1., np.repeat([.15,.25,.4,.5,.6,.75,.85], 3)]
        self.T = np.r_[2200.,2800.,np.tile([2200.,2500.,2800.], 7)]

    def test_scalar_parity_corners_posterior_stationarity_and_constraints(self):
        rng = np.random.default_rng(982341)
        theta = np.tile(self.theta, (36, 1))
        theta[:32, [1]] += rng.uniform(-15000,15000,(32,1))
        theta[:32, [4]] += rng.uniform(-15000,15000,(32,1))
        theta[32:, 1] += [p[0] for p in itertools.product([-15000.,15000.],repeat=2)]
        theta[32:, 4] += [p[1] for p in itertools.product([-15000.,15000.],repeat=2)]
        with warnings.catch_warnings(), np.errstate(all='raise'):
            warnings.simplefilter('error', RuntimeWarning)
            result = BatchedEquilibriumModel(self.x,self.T,chunk_size=7).evaluate(theta,return_sites=True)
            scalar = EquilibriumModel(self.x,self.T)
            for i, th in enumerate(theta):
                expected = scalar.evaluate(th)
                for key in ['energy','sites','stationarity','convexity_lower_bound']:
                    np.testing.assert_allclose(result[key][i],expected[key],rtol=0,atol=2e-9)
            sites=result['sites']
            self.assertTrue(np.all((sites>=0)&(sites<=1)))
            np.testing.assert_allclose((sites[:,:,0]+3*sites[:,:,1])/4,np.broadcast_to(self.x,(len(theta),len(self.x))),atol=1e-14,rtol=0)
            tol=1e-8+1e-11*(8.3145*self.T+np.max(abs(theta[:,:9]),axis=1)[:,None]+abs(theta[:,9,None])*self.T)
            self.assertTrue(np.all(abs(result['stationarity'])<=tol))
            other=BatchedEquilibriumModel(self.x,self.T,chunk_size=100).evaluate(theta)
            np.testing.assert_array_equal(other['energy'],result['energy'])

    def test_all_coefficients_and_near_endpoints(self):
        rng=np.random.default_rng(2242)
        theta=np.tile(self.theta,(15,1))+rng.normal(size=(15,10))*np.r_[np.full(9,500.),.1]
        x=[0.,1e-5,.2,.5,.8,1-1e-5,1.]
        with warnings.catch_warnings(), np.errstate(all='raise'):
            warnings.simplefilter('error',RuntimeWarning)
            result=BatchedEquilibriumModel(x,2700).evaluate(theta,return_sites=True)
            scalar=EquilibriumModel(x,2700)
            for i,th in enumerate(theta):
                expected=scalar.evaluate(th)
                np.testing.assert_allclose(result['energy'][i],expected['energy'],atol=2e-9,rtol=0)
                np.testing.assert_allclose(result['sites'][i],expected['sites'],atol=1e-13,rtol=0)

    def test_invalid_empty_and_nonconvex(self):
        model=BatchedEquilibriumModel([0.,.5,1.],900,chunk_size=1)
        for bad in [np.zeros(10),np.zeros((2,9)),np.full((2,10),np.nan),np.full((2,10),np.inf)]:
            with self.assertRaises(ValueError):model.evaluate(bad)
        for chunk in [0,-2,1.2,True]:
            with self.assertRaises(ValueError):BatchedEquilibriumModel([.5],900,chunk_size=chunk)
        empty=model.evaluate(np.empty((0,10)),return_sites=True)
        self.assertEqual(empty['energy'].shape,(0,3));self.assertEqual(empty['sites'].shape,(0,3,2))
        theta=np.zeros((2,10));theta[1,4:6]=80000.
        with self.assertRaisesRegex(NonConvexError,'parameter row 1, state 1'):model.evaluate(theta)
        endpoint=BatchedEquilibriumModel([0.,1.],900).evaluate(theta,return_sites=True)
        np.testing.assert_allclose(endpoint['energy'],0.,atol=1e-10,rtol=0)

if __name__=='__main__':unittest.main()
