import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from uq_bundle import *
from cef_fixtures import fixtures
class BundleTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.fixture=next(iter(fixtures().values()))
  self.meta={'parameters':self.fixture.parameters,'description':'Generic two-sublattice CEF test'}
  ids=[r['id'] for r in self.fixture.parameters];mean=np.asarray(self.fixture.nominal_parameters)
  self.arr={'mean':mean,'joint_draws':np.vstack([mean-1,mean+2,mean+3]),'sigmas':np.array([1.,2.,3.]),'coefficient_ids':np.array(ids)}
  self.text=self.fixture.render_tdb(mean)
 def tearDown(self):self.tmp.cleanup()
 def write(self):return write_bundle(self.text,self.meta,self.arr,self.root)
 def editmeta(self,p,fn):
  mp=p.with_suffix('.uq.json');m=json.loads(mp.read_text());fn(m);mp.write_text(json.dumps(m));lines=p.read_text().splitlines(keepends=True);lines[:3]=[pointer_header(mp.name,hashlib.sha256(mp.read_bytes()).hexdigest())];p.write_text(''.join(lines))
 def editnpz(self,p,fn):
  q=p.with_suffix('.uq.npz')
  with np.load(q,allow_pickle=False) as z:a={k:z[k] for k in z.files}
  fn(a);np.savez_compressed(q,**a)
  self.editmeta(p,lambda m:m.update(npz_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
 def test_roundtrip_order_and_analytic_mean(self):
  p=self.write();text,m,a=load_bundle(p);self.assertEqual(text,strip_tdb_comments(self.text));self.assertEqual(m['parameters'],self.meta['parameters']);np.testing.assert_array_equal(a['joint_draws'],self.arr['joint_draws']);self.assertFalse(np.allclose(a['mean'],a['joint_draws'].mean(axis=0)))
 def test_pycalphad_ignores_pointer_and_roundtrip_loses_it(self):
  from pycalphad import Database,Model
  p=self.write();db=Database(str(p));orig=Database(self.text)
  self.assertEqual(db.symbols,orig.symbols)
  output=self.root/'conventional.tdb';db.to_file(str(output),fmt='tdb',if_incompatible='raise')
  self.assertNotIn('PHASEFORGE_UQ_',output.read_text())
  with self.assertRaises(MissingDirectiveError):load_bundle(output)
 def test_all_sublattice_fixtures_preserve_order(self):
  for name,f in fixtures().items():
   ids=[r['id'] for r in f.parameters];a={**self.arr,'mean':f.nominal_parameters,'joint_draws':np.tile(f.nominal_parameters,(3,1)),'coefficient_ids':np.array(ids)}
   p=write_bundle(f.render_tdb(f.nominal_parameters),{'parameters':f.parameters},a,self.root/name);_,m,_=load_bundle(p);self.assertEqual(m['parameters'],f.parameters)
 def test_duplicate_directive_and_missing_pointer(self):
  p=self.write();original=p.read_text();p.write_text(original.splitlines()[0]+'\n'+original)
  with self.assertRaises(BundleError):load_bundle(p)
  p.write_text(self.text)
  with self.assertRaises(MissingDirectiveError):load_bundle(p)
 def test_changed_payload_and_metadata_checksum(self):
  p=self.write();old=p.read_text();p.write_text(old+'\n')
  with self.assertRaises(BundleError):load_bundle(p)
  p.write_text(old);p.with_suffix('.uq.json').write_text('{}')
  with self.assertRaises(BundleError):load_bundle(p)
 def test_missing_truncated_files(self):
  p=self.write();q=p.with_suffix('.uq.npz');data=q.read_bytes();q.unlink()
  with self.assertRaises(BundleError):load_bundle(p)
  q.write_bytes(data[:40]);self.editmeta(p,lambda m:m.update(npz_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
  with self.assertRaises(BundleError):load_bundle(p)
 def test_paths_absolute_traversal_and_symlinkescape(self):
  p=self.write()
  for bad in ['/tmp/outside.npz','../outside.npz','x/../outside.npz','C:\\outside.npz']:
   self.editmeta(p,lambda m:m.update(npz_path=bad))
   with self.assertRaises(BundleError):load_bundle(p)
  with tempfile.TemporaryDirectory() as tmp:
   target=Path(tmp)/'x.npz';target.write_bytes(b'abc');link=self.root/'escape.npz';link.symlink_to(target)
   self.editmeta(p,lambda m:m.update(npz_path='escape.npz',npz_sha256=hashlib.sha256(b'abc').hexdigest()))
   with self.assertRaises(BundleError):load_bundle(p)
 def test_column_mismatch_and_nonfinite(self):
  p=self.write();self.editnpz(p,lambda a:a.update(coefficient_ids=a['coefficient_ids'][::-1]))
  with self.assertRaises(BundleError):load_bundle(p)
 def test_sigma_and_dimensions(self):
  for bad in [np.zeros(3),np.array([1,np.nan,2]),np.ones(2)]:
   with self.assertRaises(BundleError):write_bundle(self.text,self.meta,{**self.arr,'sigmas':bad},self.root)
  with self.assertRaises(BundleError):write_bundle(self.text,self.meta,{**self.arr,'joint_draws':np.ones((2,2))},self.root)
 def test_duplicate_identifiers_and_wrong_version(self):
  import copy
  m=copy.deepcopy(self.meta);m['parameters'][1]=dict(m['parameters'][0])
  with self.assertRaises(BundleError):write_bundle(self.text,m,self.arr,self.root)
  p=self.write();self.editmeta(p,lambda m:m.update(schema_version=2))
  with self.assertRaises(BundleError):load_bundle(p)
 def test_object_arrays_rejected_no_pickle(self):
  p=self.write();self.editnpz(p,lambda a:a.update(extra=np.array([{'x':1}],dtype=object)))
  with self.assertRaises(BundleError):load_bundle(p)
 def test_positive_temperature_coefficient_units(self):
  import copy
  m=copy.deepcopy(self.meta);k=next(i for i,r in enumerate(m['parameters']) if r['polynomial_term']=='T');m['parameters'][k]['units']='J/mol_formula_unit'
  with self.assertRaises(BundleError):write_bundle(self.text,m,self.arr,self.root)
 def test_wrong_mean_rehashed_rejected(self):
  p=self.write();self.editnpz(p,lambda a:a.update(mean=a['mean']+1))
  with self.assertRaises(BundleError):load_bundle(p)
 def test_wrong_mapping_rehashed_rejected(self):
  p=self.write()
  def change(m):m['parameters'][0]['parameter_type']='L'
  self.editmeta(p,change)
  with self.assertRaises(BundleError):load_bundle(p)
 def test_swapped_sublattice_map_and_unsupported_temperature(self):
  import copy
  f=next(f for f in fixtures().values() if f.name=='CEF_INT');m={'parameters':copy.deepcopy(f.parameters)}
  m['parameters'][0]['constituent_array']=m['parameters'][0]['constituent_array'][::-1]
  arr={**self.arr,'mean':f.nominal_parameters,'joint_draws':np.tile(f.nominal_parameters,(3,1)),'coefficient_ids':np.array([p['id'] for p in m['parameters']])}
  with self.assertRaises(BundleError):write_bundle(f.render_tdb(f.nominal_parameters),m,arr,self.root)
  text=self.text.replace('*T','*T**2')
  with self.assertRaises(BundleError):write_bundle(text,self.meta,self.arr,self.root)
 def test_portable_header_lengths_and_writer_mean_mismatch(self):
  p=self.write();self.assertTrue(all(len(line)<=78 for line in p.read_text().splitlines()[:3]))
  with self.assertRaises(BundleError):write_bundle(self.text,self.meta,{**self.arr,'mean':self.arr['mean']+1},self.root/'new')
 def test_phase_site_ratio_rehashed(self):
  self.meta['phase_model']={'phase':self.fixture.name,'constituents':[list(s) for s in self.fixture.species],'site_ratios':list(self.fixture.site_ratios)}
  p=self.write();self.editmeta(p,lambda m:m['phase_model'].update(site_ratios=[1,1]))
  with self.assertRaises(BundleError):load_bundle(p)
if __name__=='__main__':unittest.main()
