"""EQ-01: convex CEF minima, derivatives, conservation and unsupported-state rejection.

Ported from the archived benchmark: removed sibling-directory import and timing
file side effect. Numerical acceptance assertions are unchanged.
"""
import sys,time,unittest,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import sympy as sp
from pycalphad import Database,Model,variables as vv
from cef_fixtures import fixtures
from equilibrium_solver import EquilibriumModel,NonConvexError
f=fixtures()['unequal_substitutional'];theta=f.nominal_parameters
class Tests(unittest.TestCase):
 def test_independent_minimum_and_parsed_energy(self):
  x=np.r_[0,np.linspace(.05,.95,24),1];T=np.full(len(x),900.);m=EquilibriumModel(x,T);r=m.evaluate(theta,True)
  s=f.symbolic(theta);u,v=s['coordinates'];t=s['temperature'];g=s['GM'];der=sp.lambdify((u,v,t),sp.diff(g,u)-sp.diff(g,v)/3,'numpy')
  for j,b in enumerate(x):
   if b in [0,1]:continue
   lo=max(0,4*b-3);hi=min(1,4*b);eps=1e-13
   z=brentq(lambda z:der(z,(4*b-z)/3,900.),lo+eps,hi-eps,xtol=1e-14)
   self.assertLess(abs(z-r['sites'][j,0]),1e-10)
  np.testing.assert_allclose(r['energy'],f.independent_energy(r['sites'],T,theta),atol=1e-9)
  py=Model(Database(f.render_tdb()),f.components,f.name);pg=sp.sympify(py.GM).subs({sp.sympify(vv.T):t,sp.sympify(vv.P):101325,sp.sympify(vv.SiteFraction(f.name,0,'AL')):u,sp.sympify(vv.SiteFraction(f.name,0,'NI')):1-u,sp.sympify(vv.SiteFraction(f.name,1,'AL')):v,sp.sympify(vv.SiteFraction(f.name,1,'NI')):1-v})
  fn=sp.lambdify((u,v,t),pg,'numpy')
  with np.errstate(divide='ignore',invalid='ignore'):np.testing.assert_allclose(r['energy'],fn(r['sites'][:,0],r['sites'][:,1],T),atol=1e-8)
  self.assertAlmostEqual(r['energy'][0],theta[3]/4);self.assertAlmostEqual(r['energy'][-1],theta[0]/4)
 def test_envelope_jacobian_and_nonconvex(self):
  m=EquilibriumModel(np.linspace(.05,.95,24),900.);r=m.evaluate(theta,True)
  for j in range(10):
   step=.01 if j<9 else 1e-4
   d=np.zeros(10);d[j]=step
   fd=(m.evaluate(theta+d)['energy']-m.evaluate(theta-d)['energy'])/(2*step)
   np.testing.assert_allclose(fd,r['jacobian'][:,j],atol=3e-7,rtol=1e-6)
  bad=theta.copy();bad[4]=1e8
  with self.assertRaises(NonConvexError):m.evaluate(bad)
 def test_bounds_and_random_parameters(self):
  rng=np.random.default_rng(8);m=EquilibriumModel(np.linspace(.03,.97,24),np.linspace(900,1300,24))
  for _ in range(20):
   th=theta+rng.normal(size=10)*np.r_[np.full(9,500),.5];r=m.evaluate(th,True)
   self.assertTrue(np.all(r['convexity_lower_bound']>0));np.testing.assert_allclose((r['sites'][:,0]+3*r['sites'][:,1])/4,m.x,atol=1e-14)

if __name__ == "__main__":
    unittest.main()
