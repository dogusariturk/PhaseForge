"""IO-01: manufactured records -> fit -> bundle -> move -> fresh predictions.

Uses real prototype fitting/export modules, both CEF fixtures and full joint
draws. This is the standalone workflow, not yet PhaseForge CLI integration.
Tolerance is for numerical preservation, not physical assessment accuracy.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

from bayes_linear import Posterior
from cef_fixtures import fixtures
from observation_contract import ObservationBatch
from uq_bundle import load_bundle, write_bundle


class TestWorkflowRoundtrip(unittest.TestCase):
    def test_fit_export_move_reload_and_predict_both_sublattice_models(self):
        rng = np.random.default_rng(2026100721)
        for name, fixture in fixtures().items():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as td:
                work = Path(td)
                sites = rng.uniform(.06, .94, (60, 2))
                temperatures = np.tile([600., 900., 1200.], 20)
                center = fixture.nominal_parameters
                sd = np.r_[np.full(9, 6000.), 5.]
                truth = center + rng.normal(size=10)*sd*.2
                energy = fixture.independent_energy(sites, temperatures, truth)+rng.normal(0, 80., 60)
                records = [dict(kind="fixed_configuration", site_fractions=q.tolist(), temperature_K=float(t), energy_J_mol_atoms=float(e))
                           for q, t, e in zip(sites, temperatures, energy)]
                input_path = work/"observations.json"
                input_path.write_text(json.dumps(records))
                loaded = json.loads(input_path.read_text())
                q = np.array([r["site_fractions"] for r in loaded])
                t = np.array([r["temperature_K"] for r in loaded])
                y = np.array([r["energy_J_mol_atoms"] for r in loaded])
                self.assertTrue(all(r["kind"] == "fixed_configuration" for r in loaded))
                design = fixture.design_matrix(q, t)
                baseline = fixture.independent_energy(q, t, center)
                post = Posterior(design, y-baseline, sd, noise_scale=150., sigma_bounds=(1e-4, 5000.), grid_size=801)
                delta, sigmas = post.sample(256, seed=72)
                mean = center+post.mean
                arrays = dict(joint_draws=delta+center, mean=mean, sigmas=sigmas,
                              coefficient_ids=np.array(fixture.parameter_names))
                metadata = dict(parameters=fixture.parameters,
                                phase_model=dict(phase=fixture.name, constituents=[list(s) for s in fixture.species], site_ratios=list(fixture.site_ratios)),
                                observation_kind="fixed_configuration", residual_units="J/mol_atoms")
                tdb = write_bundle(fixture.render_tdb(mean), metadata, arrays, work/"original")
                shutil.move(str(tdb.parent), work/"relocated")
                _, recovered_meta, recovered = load_bundle(work/"relocated"/tdb.name)
                self.assertEqual(recovered_meta["observation_kind"], "fixed_configuration")
                for key in arrays:
                    np.testing.assert_array_equal(recovered[key], arrays[key])
                query = ObservationBatch("fixed_configuration", np.array([750., 1050.]),
                                         site_fractions=np.array([[.31, .72], [.69, .22]]))
                expected = np.stack([query.predict(fixture, row) for row in arrays["joint_draws"]])
                actual = np.stack([query.predict(fixture, row) for row in recovered["joint_draws"]])
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(np.quantile(actual, [.025, .5, .975], axis=0),
                                              np.quantile(expected, [.025, .5, .975], axis=0))
                repeated, repeated_sigma = post.sample(256, seed=72)
                np.testing.assert_array_equal(repeated+center, recovered["joint_draws"])
                np.testing.assert_array_equal(repeated_sigma, recovered["sigmas"])


if __name__ == "__main__":
    unittest.main()
