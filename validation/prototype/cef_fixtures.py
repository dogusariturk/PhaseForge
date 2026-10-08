"""Manufactured linear-in-parameter CEF fixtures; no physical assessment claims.
Independent CEF construction (not derived from pycalphad expression).
Site coordinate q=(fraction of first species on sublattice0, on sublattice1).
TDB coefficient energies are J/mol formula unit; outputs are J/mol atoms.
"""
from dataclasses import dataclass
import textwrap
import numpy as np
import sympy as sp
R=8.3145 # pycalphad convention; nativeTC may differ by1e-5
@dataclass
class CEFFixture:
 name:str
 species:tuple
 site_ratios:tuple
 nominal_parameters:np.ndarray
 @property
 def phase_name(self):return self.name
 @property
 def components(self):return sorted(set(sum((list(s) for s in self.species),[])))
 @property
 def _constituents(self):
  a,b=self.species[0];c,d=self.species[1]
  return [((a,),(c,)),((a,),(d,)),((b,),(c,)),((b,),(d,)),((a,b),(c,)),((a,b),(d,)),((a,),(c,d)),((b,),(c,d)),((a,b),(c,d))]
 @property
 def parameters(self):
  out=[]
  for i,ca in enumerate(self._constituents):
   typ='G' if i<4 else 'L';identifier=f"{typ}({self.name},{':'.join(','.join(s) for s in ca)};0)"
   out.append(dict(id=identifier+'::constant',phase=self.name,parameter_type=typ,constituent_array=[list(s) for s in ca],parameter_order=0,polynomial_term='constant',units='J/mol_formula_unit',tdb_identifier=identifier))
  temp=dict(out[4]);temp.update(id=temp['tdb_identifier']+'::T',polynomial_term='T',units='J/mol_formula_unit/K');out.append(temp)
  return out
 @property
 def parameter_names(self):return [p['id'] for p in self.parameters]
 @property
 def units(self):return [p['units'] for p in self.parameters]
 def _basis_symbolic(self):
  x,y=sp.symbols('x y',positive=True);T=sp.Symbol('T',positive=True)
  basis=[x*y,x*(1-y),(1-x)*y,(1-x)*(1-y),x*(1-x)*y,x*(1-x)*(1-y),x*y*(1-y),(1-x)*y*(1-y),x*(1-x)*y*(1-y)]
  basis.append(T*basis[4]);return x,y,T,basis
 def symbolic(self,theta=None):
  x,y,T,basis=self._basis_symbolic();a,b=map(sp.sympify,self.site_ratios)
  occupied=[]
  for sf,ss in zip([x,y],self.species):occupied.append((sf if ss[0]!='VA' else 0)+((1-sf) if ss[1]!='VA' else 0))
  norm=a*occupied[0]+b*occupied[1]
  ideal=sp.Float(R)*T*(a*(x*sp.log(x)+(1-x)*sp.log(1-x))+b*(y*sp.log(y)+(1-y)*sp.log(1-y)))/norm
  pars=list(sp.symbols('p:10')) if theta is None else list(map(sp.Float,theta))
  gm=sum(p*f for p,f in zip(pars,basis))/norm+ideal
  return dict(coordinates=(x,y),temperature=T,parameters=tuple(pars),normalization=norm,design=sp.Matrix([f/norm for f in basis]),ideal=ideal,GM=gm)
 def normalization(self,sitefractions):
  q=np.atleast_2d(np.asarray(sitefractions,float));norm=np.zeros(len(q))
  for i,(ratio,ss) in enumerate(zip(self.site_ratios,self.species)):
   if ss[0]!='VA':norm+=ratio*q[:,i]
   if ss[1]!='VA':norm+=ratio*(1-q[:,i])
  return norm
 def design_matrix(self,sitefractions,temperature):
  q=np.atleast_2d(np.asarray(sitefractions,float))
  if q.shape[1]!=2 or np.any(~np.isfinite(q)) or np.any((q<0)|(q>1)):raise ValueError('Expected Nx2 binary-sublattice fractions within[0,1]')
  x,y=q.T;T=np.broadcast_to(np.asarray(temperature,float),(len(q),))
  if np.any(~np.isfinite(T)) or np.any(T<=0):raise ValueError('Positive finite temperature required')
  b=np.column_stack([x*y,x*(1-y),(1-x)*y,(1-x)*(1-y),x*(1-x)*y,x*(1-x)*(1-y),x*y*(1-y),(1-x)*y*(1-y),x*(1-x)*y*(1-y),T*x*(1-x)*y])
  return b/self.normalization(q)[:,None]
 def ideal_offset(self,sitefractions,temperature):
  q=np.atleast_2d(np.asarray(sitefractions,float));self.design_matrix(q,temperature)
  def xl(z):
   out=np.zeros_like(z);mask=z>0;out[mask]=z[mask]*np.log(z[mask]);return out
  entropy=np.einsum('ni,i->n',xl(q)+xl(1-q),np.array(self.site_ratios),optimize=False)
  return R*np.asarray(temperature)*entropy/self.normalization(q)
 def independent_energy(self,sitefractions,temperature,theta=None):
  theta=self.nominal_parameters if theta is None else np.asarray(theta,float)
  return np.einsum('ni,i->n',self.design_matrix(sitefractions,temperature),theta,optimize=False)+self.ideal_offset(sitefractions,temperature)
 def render_tdb(self,theta=None):
  theta=self.nominal_parameters if theta is None else np.asarray(theta,float)
  if theta.shape!=(10,) or not np.all(np.isfinite(theta)):raise ValueError('Expected10 finite coefficients in metadata order')
  refs={'AL':('FCC_A1',26.9815),'NI':('FCC_A1',58.6934),'FE':('BCC_A2',55.845),'C':('GRAPHITE',12.011),'VA':('VACUUM',0.)}
  lines=['$ MANUFACTURED SOFTWARE FIXTURE; NOT A MATERIAL ASSESSMENT','$ Coefficients J/mol formula unit; B coefficients J/mol formula unit/K']
  for e in self.components:
   ref,mass=refs[e];lines.append(f'ELEMENT {e} {ref} {mass} 0 0 !')
  lines+=['TYPE_DEFINITION % SEQ * !',f'PHASE {self.name} % 2 '+ ' '.join(map(str,self.site_ratios))+' !',f"CONSTITUENT {self.name} :{':'.join(','.join(s) for s in self.species)}: !"]
  for i,p in enumerate(self.parameters[:9]):
   expr=f'{theta[i]:.17g}'
   if i==4:expr+=f'{theta[9]:+.17g}*T'
   lines.append(f"PARAMETER {p['tdb_identifier']} 298.15 {expr}; 3000 N !")
  wrapped=[]
  for line in lines:
   if line.startswith('$'):
    wrapped.extend(textwrap.wrap(line,width=78,subsequent_indent='$ ',break_long_words=False,break_on_hyphens=False))
   else:
    wrapped.extend(textwrap.wrap(line,width=78,subsequent_indent=' ',break_long_words=False,break_on_hyphens=False))
  return '\n'.join(wrapped)+'\n'
def fixtures():
 return {
 'unequal_substitutional':CEFFixture('CEF_SUB',(('AL','NI'),('AL','NI')),(1,3),np.array([-40000.,-31000.,-23000.,-12000.,18000.,-9000.,7000.,13000.,22000.,-4.])),
 'interstitial_vacancy':CEFFixture('CEF_INT',(('FE','NI'),('C','VA')),(1,1),np.array([-31000.,-12000.,-19000.,-9000.,12000.,8000.,16000.,-7000.,11000.,3.])),
 }
