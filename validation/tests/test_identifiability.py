"""All10 CEF coefficients at known sites; no internally-relaxed likelihood claim."""
import unittest
import numpy as np
from bayes_linear import Posterior,mm
from cef_fixtures import fixtures


class TestIdentifiability(unittest.TestCase):
    def test_temperature_nullspace_and_prior_sensitivity(self):
        rng=np.random.default_rng(730122)
        for name,f in fixtures().items():
            with self.subTest(fixture=name):
                q=rng.uniform(.04,.96,(60,2));T0=900.
                X=f.design_matrix(q,T0);M=f.design_matrix(q,np.tile([600.,900.,1200.],20))
                self.assertEqual(np.linalg.matrix_rank(X),9)
                self.assertEqual(np.linalg.matrix_rank(M),10)
                direction=np.zeros(10);direction[4]=-T0;direction[9]=1.
                np.testing.assert_allclose(mm(X,direction),0.,atol=1e-12)
                y=mm(X,f.nominal_parameters)+rng.normal(size=len(X))*25
                tau=np.r_[np.full(9,50000.),50.]
                fits=[Posterior(X,y,tau*scale,noise_scale=100.,sigma_bounds=(1e-5,1e5),grid_size=801) for scale in [.5,2.]]
                probe=f.design_matrix(np.tile([.4,.6],(2,1)),[T0,1800.])
                variances=[]
                for fit in fits:
                    _,cov=fit.conditional(25.)
                    whitened=cov/fit.tau[:,None]/fit.tau[None,:]
                    _,_,vt=np.linalg.svd(X*fit.tau,full_matrices=True)
                    self.assertAlmostEqual(float(mm(mm(vt[-1],whitened),vt[-1])),1.,places=10)
                    variances.append(np.einsum('ni,ij,nj->n',probe,cov,probe,optimize=False))
                ratios=variances[1]/variances[0]
                self.assertLess(abs(ratios[0]-1),.1)
                self.assertGreater(ratios[1],10.)
                # Finite posterior in a null direction comes from prior, not data.

if __name__=='__main__':unittest.main()
