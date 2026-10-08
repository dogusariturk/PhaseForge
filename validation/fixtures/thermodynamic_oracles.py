"""Independent manufactured thermodynamics, not production fitting backends.
No external materials data or diagram interpolation. All energies in J/mol atoms
except the dimensionless quartic oracle, whose arbitrary energy unit is explicit.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

@dataclass(frozen=True)
class QuarticBranches:
    """F(q;h)=(q²−1)²−hq on−2<=q<=2; h is a conjugate field.
    Stationary points are all real roots of a cubic, then compared with endpoints.
    At h=0 exact competing minima q=±1 exchange global stability; q=0 is a maximum.
    """
    bound: float = 2.0
    def energy(self,q,h):return (np.asarray(q)**2-1)**2-h*np.asarray(q)
    def stationary(self,h):
        roots=np.roots([4.,0.,-4.,-h]);out=[]
        for r in roots:
            if abs(r.imag)<1e-10 and abs(r.real)<self.bound:
                q=float(r.real);curv=12*q*q-4
                out.append(dict(q=q,energy=float(self.energy(q,h)),curvature=curv,kind='minimum' if curv>1e-10 else 'maximum' if curv< -1e-10 else 'degenerate'))
        return sorted(out,key=lambda r:r['q'])
    def global_minima(self,h,energy_tolerance=1e-10):
        states=[dict(q=q,energy=float(self.energy(q,h)),kind='endpoint') for q in [-self.bound,self.bound]]+self.stationary(h)
        best=min(r['energy'] for r in states)
        return [r for r in states if abs(r['energy']-best)<=energy_tolerance]

@dataclass(frozen=True)
class RegularSolution:
    """Symmetric stress-free binary g=Ωx(1−x)+RTΣxlnx; no elastic term."""
    omega: float=20000.0
    R: float=8.3145
    @property
    def critical_temperature(self):return self.omega/(2*self.R)
    def energy(self,x,T):
        x=np.asarray(x,float)
        if np.any((x<0)|(x>1)) or T<=0:raise ValueError('x must be within[0,1],T>0')
        def xl(q):
            a=np.zeros_like(q);m=q>0;a[m]=q[m]*np.log(q[m]);return a
        return self.omega*x*(1-x)+self.R*T*(xl(x)+xl(1-x))
    def derivative(self,x,T):return self.omega*(1-2*np.asarray(x))+self.R*T*(np.log(x)-np.log1p(-np.asarray(x)))
    def curvature(self,x,T):return -2*self.omega+self.R*T/(np.asarray(x)*(1-np.asarray(x)))
    def spinodal(self,T):
        if T<self.critical_temperature:
            d=np.sqrt(1-T/self.critical_temperature);return ((1-d)/2,(1+d)/2)
        if T==self.critical_temperature:return (.5,.5)
        return None
    def coexistence(self,T):
        if T<=0:raise ValueError('T must be positive')
        if T>=self.critical_temperature:return None
        left_spin=self.spinodal(T)[0]
        # Finite bracket is explicit: no claimed reliable root below its resolution.
        eps=np.finfo(float).eps
        if self.derivative(eps,T)>=0:raise ValueError('Binodal below floating-point endpoint resolution')
        a=brentq(lambda x:self.derivative(x,T),eps,left_spin,xtol=5e-15,rtol=1e-14)
        return (a,1-a)
    def chemical_potentials(self,x,T):
        g=self.energy(x,T);d=self.derivative(x,T)
        return np.array([g-x*d,g+(1-x)*d]) # components A and B; x means B
    def constrained_equilibrium(self,x,T):
        if not 0<=x<=1:raise ValueError('Invalid bulk composition')
        pair=self.coexistence(T)
        if pair is not None and pair[0]<x<pair[1]:
            a,b=pair;fb=(x-a)/(b-a)
            return dict(compositions=[a,b],fractions=[1-fb,fb],energy=float((1-fb)*self.energy(a,T)+fb*self.energy(b,T)))
        return dict(compositions=[x],fractions=[1.],energy=float(self.energy(x,T)))
    def topology(self,T):
        """Deterministic fixed-T section: vertices, interval edges and field labels."""
        pair=self.coexistence(T)
        if pair is None:
            critical=T==self.critical_temperature
            vertices=[0.,.5,1.] if critical else [0.,1.]
            edges=[dict(left=a,right=b,field='single') for a,b in zip(vertices[:-1],vertices[1:])]
            return dict(vertices=vertices,edges=edges,critical=critical,critical_composition=.5 if critical else None)
        a,b=pair
        return dict(vertices=[0.,a,b,1.],edges=[dict(left=0.,right=a,field='single'),dict(left=a,right=b,field='two_phase'),dict(left=b,right=1.,field='single')],critical=False,critical_composition=None)
    def render_tdb(self):
        return f'''$ Manufactured symmetric regular solution; not an alloy assessment
ELEMENT AL FCC_A1 26.9815 0 0 !
ELEMENT NI FCC_A1 58.6934 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE REG % 1 1 !
CONSTITUENT REG :AL,NI: !
PARAMETER G(REG,AL;0) 1 0; 6000 N !
PARAMETER G(REG,NI;0) 1 0; 6000 N !
PARAMETER L(REG,AL,NI;0) 1 {self.omega:.17g}; 6000 N !
'''
