"""Certified-convex internal CEF minimization; not multiphase equilibrium.
Parameters match sibling cef_fixtures unequal_substitutional, J/mol formula unit.
Energy and envelope Jacobian are per mole atoms. No symbolic work per proposal.
"""
import numpy as np
R=8.3145
class NonConvexError(ValueError):pass
class ConvergenceError(RuntimeError):pass
class EquilibriumModel:
 def __init__(self,x,T):
  self.x,self.T=np.broadcast_arrays(np.asarray(x,float),np.asarray(T,float));self.x=self.x.ravel().copy();self.T=self.T.ravel().copy()
  if np.any(~np.isfinite(self.x)) or np.any((self.x<0)|(self.x>1)) or np.any(~np.isfinite(self.T)) or np.any(self.T<=0):raise ValueError('Require0<=bulkx<=1 and finiteT>0')
  self.lo=np.maximum(0.,4*self.x-3);self.hi=np.minimum(1.,4*self.x);self.interior=(self.x>0)&(self.x<1)
  self.entropy_bound=np.zeros(len(self.x))
  mask=self.interior
  umax=np.clip(.5,self.lo[mask],self.hi[mask]);vmax=np.clip(.5,(4*self.x[mask]-self.hi[mask])/3,(4*self.x[mask]-self.lo[mask])/3)
  self.entropy_bound[mask]=R*self.T[mask]/4*(1/(umax*(1-umax))+1/(3*vmax*(1-vmax)))
  self.basis=np.zeros((len(self.x),10,5))
  mul=np.polynomial.polynomial.polymul
  for n,(bulk,t) in enumerate(zip(self.x,self.T)):
   u=np.array([0.,1.]);v=np.array([4*bulk/3,-1/3]);U=np.array([1.,-1.]);V=np.array([1-4*bulk/3,1/3]);
   bases=[mul(u,v),mul(u,V),mul(U,v),mul(U,V),mul(mul(u,U),v),mul(mul(u,U),V),mul(u,mul(v,V)),mul(U,mul(v,V)),mul(mul(u,U),mul(v,V)),t*mul(mul(u,U),v)]
   for j,b in enumerate(bases):self.basis[n,j,:len(b)]=b/4
 def _poly(self,c,u):return ((c[:,4]*u+c[:,3])*u+c[:,2])*u*u+c[:,1]*u+c[:,0]
 def _grad(self,c,u):return ((4*c[:,4]*u+3*c[:,3])*u+2*c[:,2])*u+c[:,1]
 def _curv(self,c,u):return (12*c[:,4]*u+6*c[:,3])*u+2*c[:,2]
 def curvature_bound(self,theta):
  c=np.einsum('njk,j->nk',self.basis,self._theta(theta),optimize=False)
  low=np.minimum(self._curv(c,self.lo),self._curv(c,self.hi));den=24*c[:,4]
  vertex=np.divide(-6*c[:,3],den,out=np.zeros_like(den),where=den!=0)
  valid=(c[:,4]>0)&(vertex>self.lo)&(vertex<self.hi)
  low=np.where(valid,np.minimum(low,self._curv(c,vertex)),low)
  return low+self.entropy_bound
 @staticmethod
 def _theta(theta):
  theta=np.asarray(theta,float)
  if theta.shape!=(10,) or not np.all(np.isfinite(theta)):raise ValueError('Expected10finitecoefficients')
  return theta
 def evaluate(self,theta,derivatives=False):
  theta=self._theta(theta);c=np.einsum('njk,j->nk',self.basis,theta,optimize=False);bound=self.curvature_bound(theta)
  if np.any(bound[self.interior]<=0):raise NonConvexError('Strict internal convexity not certified by conservative bound')
  ids=np.flatnonzero(self.interior);u=self.x.copy();residual=np.zeros(len(u));iterations=0
  if len(ids):
   cc=c[ids];bulk=self.x[ids];t=self.T[ids];lo=self.lo[ids].copy();hi=self.hi[ids].copy();z=bulk.copy()
   def gradcurv(z):
    v=(4*bulk-z)/3
    g=self._grad(cc,z)+R*t/4*(np.log(z)-np.log1p(-z)-np.log(v)+np.log1p(-v))
    h=self._curv(cc,z)+R*t/4*(1/(z*(1-z))+1/(3*v*(1-v)))
    return g,h
   tol=1e-8+1e-11*(R*t+np.max(np.abs(theta[:9]))+abs(theta[9])*t)
   for iterations in range(100):
    g,h=gradcurv(z);done=np.abs(g)<=tol
    if np.all(done):break
    lo=np.where((g<0)&~done,z,lo);hi=np.where((g>=0)&~done,z,hi)
    proposal=z-g/h;valid=(proposal>lo)&(proposal<hi)&np.isfinite(proposal)
    z=np.where(done,z,np.where(valid,proposal,(lo+hi)/2))
   else:raise ConvergenceError('SafeguardedNewton failed stationarity tolerance')
   u[ids]=z;residual[ids]=g
  v=(4*self.x-u)/3;sites=np.column_stack([u,v])
  if np.any(sites<0) or np.any(sites>1) or np.max(abs((u+3*v)/4-self.x),initial=0)>1e-13:raise ConvergenceError('Siteconstraintfailure')
  def mix(z):
   out=np.zeros_like(z);mask=(z>0)&(z<1);zz=z[mask];out[mask]=zz*np.log(zz)+(1-zz)*np.log1p(-zz);return out
  energy=self._poly(c,u)+R*self.T/4*(mix(u)+3*mix(v))
  if np.any(~np.isfinite(energy)):raise ConvergenceError('Nonfiniteenergy')
  out=dict(energy=energy,sites=sites,stationarity=residual,convexity_lower_bound=bound,iterations=iterations+1)
  if derivatives:out['jacobian']=np.einsum('njk,nk->nj',self.basis,u[:,None]**np.arange(5),optimize=False)
  return out
