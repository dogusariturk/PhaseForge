"""Exact regular-solution coexistence topology + actual pycalphad comparisons."""
import unittest
import numpy as np
from fixtures.thermodynamic_oracles import RegularSolution

class CoexistenceOracleTests(unittest.TestCase):
    def test_common_tangent_chemical_potentials_and_spinodal_nesting(self):
        m=RegularSolution()
        for ratio in [.2,.5,.8,.95,.999]:
            T=ratio*m.critical_temperature;a,b=m.coexistence(T);s,t=m.spinodal(T)
            self.assertTrue(0<a<s<.5<t<b<1)
            self.assertAlmostEqual(a+b,1.,places=14)
            self.assertLess(abs(m.derivative(a,T)),1e-7)
            self.assertLess(abs(m.derivative(b,T)),1e-7)
            self.assertAlmostEqual(float(m.energy(a,T)),float(m.energy(b,T)),places=9)
            np.testing.assert_allclose(m.chemical_potentials(a,T),m.chemical_potentials(b,T),atol=1e-7,rtol=0)
            self.assertGreater(m.curvature(a,T),0);self.assertLess(m.curvature(.5,T),0)
            tangent=float(m.energy(a,T));grid=np.linspace(a,b,301)
            self.assertGreaterEqual(float(np.min(m.energy(grid,T)-tangent)),-1e-9)
    def test_lever_rule_mass_balance_and_endpoint_states(self):
        m=RegularSolution();T=.8*m.critical_temperature;a,b=m.coexistence(T)
        for x in [0,.02,a,.2,.5,.8,b,.98,1]:
            eq=m.constrained_equilibrium(x,T);frac=np.array(eq['fractions']);comp=np.array(eq['compositions'])
            self.assertAlmostEqual(frac.sum(),1.,places=14);self.assertTrue(np.all(frac>=0))
            self.assertAlmostEqual(float(frac@comp),x,places=14)
            self.assertLessEqual(eq['energy'],m.energy(x,T)+1e-9)
            self.assertEqual(len(comp),2 if a<x<b else 1)
    def test_critical_topology_and_terminal_behavior(self):
        m=RegularSolution();widths=[]
        for ratio in [.5,.8,.95,.99,.9999]:
            T=ratio*m.critical_temperature;graph=m.topology(T);vertices=graph['vertices']
            self.assertEqual(len(vertices),4);self.assertEqual([e['field'] for e in graph['edges']],['single','two_phase','single'])
            self.assertEqual(vertices[0],0);self.assertEqual(vertices[-1],1)
            for i,e in enumerate(graph['edges']):self.assertEqual((e['left'],e['right']),tuple(vertices[i:i+2]))
            widths.append(vertices[2]-vertices[1])
        self.assertTrue(np.all(np.diff(widths)<0));self.assertLess(widths[-1],.02)
        self.assertIsNone(m.coexistence(m.critical_temperature));self.assertEqual(m.spinodal(m.critical_temperature),(.5,.5))
        self.assertAlmostEqual(m.curvature(.5,m.critical_temperature),0.,places=9)
        critical=m.topology(m.critical_temperature)
        self.assertEqual(critical["vertices"],[0.,.5,1.])
        self.assertTrue(critical["critical"]);self.assertEqual(critical["critical_composition"],.5)
        self.assertEqual([(e["left"],e["right"]) for e in critical["edges"]],[(0.,.5),(.5,1.)])
        for ratio in [1.,1.05,2.]:
            T=ratio*m.critical_temperature;self.assertTrue(all(e['field']=='single' for e in m.topology(T)['edges']))
            self.assertTrue(np.all(m.curvature(np.linspace(.001,.999,401),T)>=-1e-9))

class PycalphadCoexistenceTests(unittest.TestCase):
    def test_actual_equilibrium_below_and_above_critical(self):
        from pycalphad import Database,equilibrium,variables as v
        m=RegularSolution();db=Database(m.render_tdb())
        for ratio in [.8,1.2]:
            T=ratio*m.critical_temperature
            for x in [.05,.2,.5,.8,.95]:
                with self.subTest(ratio=ratio,x=x):
                    eq=equilibrium(db,['AL','NI'],['REG'],{v.T:T,v.P:101325.,v.N:1.,v.X('NI'):x})
                    phases=np.asarray(eq.Phase).ravel();fractions=np.asarray(eq.NP).ravel();xs=np.asarray(eq.X).reshape(-1,2)
                    ni=list(eq.component.values).index('NI');active=sorted((float(xs[i,ni]),float(fractions[i])) for i,p in enumerate(phases) if p and fractions[i]>1e-8)
                    expected=m.constrained_equilibrium(x,T)
                    self.assertEqual(len(active),len(expected['compositions']))
                    np.testing.assert_allclose([a for a,f in active],expected['compositions'],atol=2e-7,rtol=0)
                    np.testing.assert_allclose([f for a,f in active],expected['fractions'],atol=2e-6,rtol=0)
                    self.assertAlmostEqual(sum(a*f for a,f in active),x,places=7)
                    self.assertLess(abs(float(np.asarray(eq.GM).ravel()[0])-expected['energy']),1e-4)

if __name__=='__main__':unittest.main()
