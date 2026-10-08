"""Independent algebra/numerical checks; no real holdout outcomes tune these tests."""
import unittest
import numpy as np
from scipy.special import logsumexp
from bayes_linear import Posterior,basis

class TestBayesianLinear(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[1.,-.4],[.3,.8],[-.6,.7],[.5,-.2],[.9,.9]])
        self.y=np.array([.18,-.13,.21,.02,.16]);self.tau=np.array([.7,1.2])
    def make(self,X=None,y=None,tau=None,**kwargs):
        return Posterior(self.X if X is None else X,self.y if y is None else y,self.tau if tau is None else tau,noise_scale=kwargs.pop('noise_scale',.3),sigma_bounds=kwargs.pop('sigma_bounds',(1e-8,3.)),**kwargs)
    @staticmethod
    def direct_conditional(X,y,tau,sigma):
        precision=np.diag(1/tau**2)+X.T@X/sigma**2
        cov=np.linalg.solve(precision,np.eye(len(tau)))
        return cov@X.T@y/sigma**2,cov
    def test_fixed_sigma_closed_form(self):
        p=self.make()
        for sigma in [.025,.2,1.3]:
            mean,cov=p.conditional(sigma);m,c=self.direct_conditional(self.X,self.y,self.tau,sigma)
            np.testing.assert_allclose(mean,m,atol=2e-13,rtol=2e-12)
            np.testing.assert_allclose(cov,c,atol=2e-13,rtol=2e-12)
    def test_marginal_weights_direct_observation_covariance_and_jacobian(self):
        p=self.make(grid_size=801,sigma_bounds=(1e-4,3.))
        logs=[];means=[];covs=[]
        for sigma in p.sigma:
            C=(self.X*self.tau**2)@self.X.T+sigma**2*np.eye(len(self.y))
            sign,ld=np.linalg.slogdet(C);self.assertGreater(sign,0)
            logs.append(-.5*(ld+self.y@np.linalg.solve(C,self.y))-.5*(sigma/.3)**2+np.log(sigma))
            m,c=self.direct_conditional(self.X,self.y,self.tau,sigma);means.append(m);covs.append(c)
        logs=np.array(logs);logs[[0,-1]]+=np.log(.5);w=np.exp(logs-logsumexp(logs))
        np.testing.assert_allclose(p.weights,w,atol=2e-12,rtol=2e-8)
        m=w@means;delta=np.array(means)-m;c=np.einsum('i,ijk->jk',w,covs)+(delta.T*w)@delta
        np.testing.assert_allclose(p.mean,m,atol=2e-11,rtol=2e-9)
        np.testing.assert_allclose(p.covariance,c,atol=2e-11,rtol=2e-9)
    def test_rank_deficient_duplicate_columns(self):
        X=np.column_stack([self.X,self.X[:,0]])
        tau=np.array([.7,1.2,.4]);p=self.make(X=X,tau=tau)
        m,c=p.conditional(.17);em,ec=self.direct_conditional(X,self.y,tau,.17)
        np.testing.assert_allclose(m,em,atol=1e-12);np.testing.assert_allclose(c,ec,atol=1e-12)
    def test_more_coefficients_than_observations(self):
        X=np.array([[1.,.2,-.5,0.],[.3,1.,.1,0.]])
        y=np.array([.2,-.1]);tau=np.array([.7,1.2,.4,.9]);p=self.make(X=X,y=y,tau=tau)
        m,c=p.conditional(.17);em,ec=self.direct_conditional(X,y,tau,.17)
        np.testing.assert_allclose(m,em,atol=1e-12);np.testing.assert_allclose(c,ec,atol=1e-12)
        self.assertAlmostEqual(c[-1,-1],.9**2,places=12);self.assertAlmostEqual(m[-1],0.,places=12)
    def test_weak_prior_ols_limit(self):
        p=self.make(tau=np.full(2,1e5));m,c=p.conditional(.2)
        np.testing.assert_allclose(m,np.linalg.lstsq(self.X,self.y,rcond=None)[0],atol=1e-11)
        np.testing.assert_allclose(c,.2**2*np.linalg.inv(self.X.T@self.X),atol=1e-11)
    def test_grid_refinement(self):
        coarse=self.make(grid_size=801);fine=self.make(grid_size=3201)
        np.testing.assert_allclose(coarse.mean,fine.mean,atol=1e-9)
        np.testing.assert_allclose(coarse.covariance,fine.covariance,atol=1e-9)
    def test_joint_draw_moments(self):
        p=self.make();draws,sigma=p.sample(100000,seed=81007)
        np.testing.assert_allclose(draws.mean(0),p.mean,atol=.003)
        np.testing.assert_allclose(np.cov(draws,rowvar=False),p.covariance,rtol=.04,atol=.0003)
        self.assertEqual(len(sigma),100000)
    def test_intervals_and_residual_noise_separation(self):
        p=self.make();rows=np.array([[1.,.5],[0.,0.]])
        latent=p.intervals(rows);pred=p.intervals(rows,predictive=True)
        np.testing.assert_array_equal(latent[1],[0.,0.]);self.assertGreater(pred[1,1]-pred[1,0],0.)
        self.assertGreater(pred[0,1]-pred[0,0],latent[0,1]-latent[0,0])
    def test_basis_has_constant_ternary_convention(self):
        x=np.array([.2,.3,.4,.1]);b=basis(x)[0]
        np.testing.assert_allclose(b[-4:],[.024,.006,.008,.012],atol=1e-15)
        np.testing.assert_array_equal(basis(np.eye(4)),np.zeros((4,16)))
    def test_invalid_data_and_scales(self):
        for kwargs in [dict(y=[1.]),dict(tau=[1.]),dict(tau=[0.,1.]),dict(tau=[np.nan,1.]),dict(y=[1,2,3,4,np.inf]),dict(noise_scale=0.),dict(sigma_bounds=(1.,.1)),dict(grid_size=2)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.make(**kwargs)
        for x in [[.2,.2,.2,.2],[1.1,-.1,0,0],[np.nan,0,0,1]]:
            with self.assertRaises(ValueError):basis(x)
    def test_nonfinite_noise_scale_rejected(self):
        for scale in [np.nan,np.inf]:
            with self.subTest(scale=scale),self.assertRaises(ValueError):self.make(noise_scale=scale)
    def test_nonfinite_conditional_sigma_rejected(self):
        p=self.make()
        for sigma in [np.nan,np.inf]:
            with self.subTest(sigma=sigma),self.assertRaises(ValueError):p.conditional(sigma)

if __name__=='__main__':unittest.main()
