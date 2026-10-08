import unittest
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from noise_marginal import log_marginal,draw_sigmas,conditional_moment


def quadrature(r,n,s,power=0.):
    # Independent integration in t=log(sigma), with d_sigma=exp(t)dt.
    def logf(t):return (1-n+power)*t-.5*r*np.exp(-2*t)-.5*np.exp(2*t)/s**2
    center=.5*np.log(r/n)
    fit=minimize_scalar(lambda t:-logf(t),bounds=(center-25,center+25),method='bounded')
    peak=logf(fit.x)
    value,error=quad(lambda t:np.exp(logf(t)-peak),fit.x-35,fit.x+35,epsabs=1e-11,epsrel=1e-11)
    return np.log(value)+peak


class TestNoiseMarginal(unittest.TestCase):
    def test_independent_logsigma_quadrature(self):
        for n in [1,2,7,21,72]:
            for r,s in [(2.,.7),(13125.,50.),(2e5,20.)]:
                with self.subTest(n=n,r=r,s=s):
                    self.assertAlmostEqual(float(log_marginal(r,n,s)),quadrature(r,n,s),places=9)

    def test_normalizing_constants(self):
        r,n,s=37.,7,3.
        expected=quadrature(r,n,s)+.5*np.log(2/np.pi)-np.log(s)-n/2*np.log(2*np.pi)
        self.assertAlmostEqual(float(log_marginal(r,n,s,True)),expected,places=11)

    def test_moments_quadrature(self):
        for r,n,s in [(13125.,21,50.),(2.,1,.7),(40.,7,2.)]:
            for power in [-2.,1.,2.,4.]:
                expected=np.exp(quadrature(r,n,s,power)-quadrature(r,n,s))
                np.testing.assert_allclose(conditional_moment(r,n,s,power),expected,rtol=2e-10)

    def test_conditional_sampling_and_vector_rss(self):
        # Replicated heterogeneous RSS verifies each entry uses its own conditional.
        r=np.tile([4000.,16000.],(20000,1))
        draws=draw_sigmas(r,21,50.,rng=871)
        self.assertEqual(draws.shape,r.shape)
        for j in range(2):
            expected=conditional_moment(r[0,j],21,50.)
            variance=conditional_moment(r[0,j],21,50.,2)-expected**2
            self.assertLess(abs(draws[:,j].mean()-expected),6*np.sqrt(variance/len(draws)))
        np.testing.assert_array_equal(draw_sigmas(4000.,21,rng=41,size=30),draw_sigmas(4000.,21,rng=41,size=30))

    def test_scale_invariance_and_array(self):
        r=np.array([4.,31.,500.]);n=21;s=3.;c=96485.332
        np.testing.assert_allclose(log_marginal(r*c*c,n,s*c)-log_marginal(r,n,s),(1-n)*np.log(c),rtol=1e-13)
        np.testing.assert_allclose(conditional_moment(r*c*c,n,s*c)/c,conditional_moment(r,n,s),rtol=2e-13)
        # Very large argument avoids ordinary K underflow.
        self.assertTrue(np.isfinite(log_marginal(1e8,21,1.)))

    def test_invalid_inputs(self):
        for r in [0.,-1.,np.nan,np.inf,[],[1.,0.]]:
            with self.subTest(r=r),self.assertRaises(ValueError):log_marginal(r,21)
        for n in [0,-2,2.5,True]:
            with self.assertRaises(ValueError):log_marginal(4.,n)
        for s in [0.,-1.,np.nan,np.inf,[1,2]]:
            with self.assertRaises(ValueError):log_marginal(4.,21,s)
        for size in [0,-1,2.2,True]:
            with self.assertRaises(ValueError):draw_sigmas(4.,21,size=size)
        with self.assertRaises(ValueError):conditional_moment(4.,21,power=np.nan)

if __name__=='__main__':unittest.main()
