"""OBS-01/02: explicit physical meaning of every energy observation.

These test the candidate adapter, not a live PhaseForge entry point. The
negative control deliberately equilibrates fixed configurations and must lose
their distinct energies. Agreement in bulk composition is insufficient.
"""
import unittest
from unittest.mock import patch
import numpy as np
from cef_fixtures import fixtures
from observation_contract import ObservationBatch


class TestObservationContract(unittest.TestCase):
    def test_equal_bulk_compositions_keep_distinct_fixed_configurations(self):
        f = fixtures()["unequal_substitutional"]
        sites = np.array([[.2, .6], [.8, .4]])
        np.testing.assert_allclose((sites[:, 0]+3*sites[:, 1])/4, [.5, .5])
        batch = ObservationBatch("fixed_configuration", np.array([900., 900.]), site_fractions=sites)
        with patch("observation_contract.EquilibriumModel", side_effect=AssertionError("Do not relax fixed data")):
            predicted = batch.predict(f, f.nominal_parameters)
        self.assertGreater(abs(predicted[0]-predicted[1]), 1000.)
        equilibrium = ObservationBatch("internal_equilibrium", np.array([900., 900.]), bulk_composition=np.array([.5, .5]))
        relaxed = equilibrium.predict(f, f.nominal_parameters)
        self.assertAlmostEqual(relaxed[0], relaxed[1], places=10)
        self.assertTrue(np.all(relaxed < predicted))

    def test_internal_state_is_recomputed_when_coefficients_change(self):
        f = fixtures()["unequal_substitutional"]
        batch = ObservationBatch("internal_equilibrium", np.array([2350.]), bulk_composition=np.array([.55]))
        from equilibrium_solver import EquilibriumModel
        theta = f.nominal_parameters.copy()
        original_sites = EquilibriumModel([.55], [2350.]).evaluate(theta)["sites"]
        theta[1] += 6000
        theta[4] -= 6000
        actual = batch.predict(f, theta)
        frozen = f.independent_energy(original_sites, [2350.], theta)
        self.assertGreater(float(frozen[0]-actual[0]), 20.)

    def test_ambiguous_or_unsupported_observations_raise(self):
        f = fixtures()["unequal_substitutional"]
        cases = [
            ObservationBatch("unspecified", np.array([900.])),
            ObservationBatch("fixed_configuration", np.array([900.])),
            ObservationBatch("internal_equilibrium", np.array([900.])),
            ObservationBatch("internal_equilibrium", np.array([900.]), np.array([.5]), np.array([[.5, .5]])),
            ObservationBatch("fixed_configuration", np.array([900.]), site_fractions=np.array([[1.1, .2]])),
            ObservationBatch("fixed_configuration", np.array([0.]), site_fractions=np.array([[.5, .5]])),
        ]
        for batch in cases:
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                batch.predict(f, f.nominal_parameters)
        interstitial = fixtures()["interstitial_vacancy"]
        with self.assertRaises(NotImplementedError):
            ObservationBatch("internal_equilibrium", np.array([900.]), np.array([.5])).predict(interstitial, interstitial.nominal_parameters)


if __name__ == "__main__":
    unittest.main()
