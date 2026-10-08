"""Fast protocol tests; these do not count as calibration replications."""
import unittest
import numpy as np
from campaigns.nonlinear_calibration import calibration_tests,converged,posterior,exit_status
class Contract(unittest.TestCase):
 def test_exit_status_contract(self):
  self.assertEqual(exit_status('no_calibration_failure_detected'),0)
  self.assertEqual(exit_status('calibration_rejected'),1)
  self.assertEqual(exit_status('inconclusive_grid_resolution'),2)
  with self.assertRaises(ValueError):exit_status('unknown')
 def test_bonferroni_and_uniform_pattern(self):
  x=(np.arange(200)+.5)/200
  r=calibration_tests(np.column_stack([x,x[::-1]]))
  self.assertEqual(len(r['tests']),10);self.assertEqual(r['individual_threshold'],.001)
  self.assertFalse(r['rejected'])
 def test_degenerate_ranks_rejected(self):
  self.assertTrue(calibration_tests(np.full((64,2),.5))['rejected'])
 def test_invalid_probabilities(self):
  with self.assertRaises(ValueError):calibration_tests([[np.nan,.5]])
 def test_grid_resolution_fail_closed(self):
  a=dict(cdfs=[.5,.5],mean=[0,0],sd=[1,1],quantiles=[[-2,2],[-2,2]])
  self.assertTrue(converged(a,a)['passes'])
  b=dict(a,quantiles=[[-2,2.1],[-2,2]])
  self.assertFalse(converged(a,b)['passes'])
 def test_symmetric_posterior_cdf(self):
  axis=np.linspace(-15000,15000,33)
  u,v=np.meshgrid(axis,axis,indexing='ij')
  pred=np.stack([u/100,v/100,np.zeros_like(u)],axis=-1)
  r=posterior(axis,pred,np.array([0,0,10]),[0,0])
  np.testing.assert_allclose(r['cdfs'],[.5,.5],atol=1e-13)
if __name__=='__main__':unittest.main()
