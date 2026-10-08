import unittest
import numpy as np
import sympy as sp
from pycalphad import Database,Model,variables as v
from cef_fixtures import fixtures

def pyexpr(f):
 m=Model(Database(f.render_tdb()),f.components,f.name);s=f.symbolic(f.nominal_parameters);x,y=s['coordinates'];T=s['temperature']
 replace={sp.sympify(v.T):T,sp.sympify(v.P):101325}
 for i,(q,ss) in enumerate(zip([x,y],f.species)):
  replace[sp.sympify(v.SiteFraction(f.name,i,ss[0]))]=q;replace[sp.sympify(v.SiteFraction(f.name,i,ss[1]))]=1-q
 return sp.sympify(m.GM).subs(replace),s
class TestCEF(unittest.TestCase):
 def test_energy_derivatives_and_limits(self):
  rng=np.random.default_rng(20261007)
  for name,f in fixtures().items():
   with self.subTest(name=name):
    pg,s=pyexpr(f);x,y=s['coordinates'];T=s['temperature'];ours=s['GM'];fp=sp.lambdify((x,y,T),pg,'numpy');fo=sp.lambdify((x,y,T),ours,'numpy')
    hp=sp.lambdify((x,y,T),sp.hessian(pg,(x,y)),'numpy');ho=sp.lambdify((x,y,T),sp.hessian(ours,(x,y)),'numpy')
    pts=rng.uniform(.015,.985,(31,2));temps=rng.uniform(350,2000,31)
    np.testing.assert_allclose(fp(pts[:,0],pts[:,1],temps),f.independent_energy(pts,temps),rtol=1e-12,atol=1e-8)
    for q,t in zip(pts,temps):
     np.testing.assert_allclose(hp(*q,t),ho(*q,t),rtol=1e-11,atol=1e-7)
     h=3e-5;e=np.eye(2);mixed=(fo(*(q+h*(e[0]+e[1])),t)-fo(*(q+h*(e[0]-e[1])),t)-fo(*(q+h*(-e[0]+e[1])),t)+fo(*(q-h*(e[0]+e[1])),t))/(4*h*h)
     self.assertLess(abs(mixed-ho(*q,t)[0,1])/max(1.,np.max(abs(ho(*q,t)))),1e-6)
    for x0,y0 in [(0,0),(0,1),(1,0),(1,1)]:
     q=np.array([[x0,y0]],float);j={(1,1):0,(1,0):1,(0,1):2,(0,0):3}[(x0,y0)]
     expected=f.nominal_parameters[j]/f.normalization(q)[0]
     self.assertAlmostEqual(f.independent_energy(q,773.15)[0],expected,places=9)
     with np.errstate(divide='ignore',invalid='ignore'):
      self.assertAlmostEqual(float(fp(x0,y0,773.15)),expected,places=8)
     near=np.clip(q,1e-10,1-1e-10);self.assertLess(abs(f.independent_energy(near,773.15)[0]-expected),1e-3)
 def test_temperature_design_identifiability(self):
  rng=np.random.default_rng(81);q=rng.uniform(.05,.95,(70,2))
  for f in fixtures().values():
   self.assertEqual(np.linalg.matrix_rank(f.design_matrix(q,800.)),9)
   D=f.design_matrix(q,np.linspace(350,1800,70));self.assertEqual(np.linalg.matrix_rank(D),10)
   delta=rng.normal(size=10);np.testing.assert_allclose(f.independent_energy(q,800.,f.nominal_parameters+delta)-f.independent_energy(q,800.),f.design_matrix(q,800.)@delta,atol=1e-10)
if __name__=='__main__':unittest.main()
