"""Independent analytic density and scalar thermodynamic references for adaptation."""
import itertools
import unittest
import numpy as np
from scipy.stats import norm, truncnorm

from prototype.adaptive_quadrature import focused_axis, integrate, marginal_probabilities
from prototype.batch_equilibrium import BoundedBatchEquilibrium
from prototype.equilibrium_solver import EquilibriumModel
from campaigns.nonlinear_calibration import CENTER, FREE, BOUND, converged


class TestAdaptiveQuadrature(unittest.TestCase):
    def test_focus_preserves_entire_support_and_nests_nodes(self):
        coarse = focused_axis(-15000, 15000, 14100, 73, 33)
        fine = focused_axis(-15000, 15000, 14100, 73, 65)
        np.testing.assert_array_equal(coarse, fine[::2])
        self.assertEqual(coarse[0], -15000)
        self.assertEqual(coarse[-1], 15000)
        self.assertTrue(np.all(np.diff(coarse) > 0))

    def test_triangular_density_has_exact_cell_cdf_and_quantiles(self):
        axis = np.array([0., .03, .2, .73, 1.])
        probability, quantiles = marginal_probabilities(axis, 2*axis, .44, [.025, .5, .975])
        self.assertAlmostEqual(probability, .44**2, places=14)
        np.testing.assert_allclose(quantiles, np.sqrt([.025, .5, .975]), atol=2e-15, rtol=0)
        probability, quantiles = marginal_probabilities(axis, 2*(1-axis), .44, [.025, .5, .975])
        self.assertAlmostEqual(probability, 1-(1-.44)**2, places=14)
        np.testing.assert_allclose(quantiles, 1-np.sqrt(1-np.array([.025, .5, .975])), atol=2e-15, rtol=0)

    def test_narrow_and_boundary_truncated_gaussians_match_exact_distribution(self):
        bounds = np.array([[-1., 1.], [-1., 1.]])
        centers = np.array([.12, .94])
        scales = np.array([.012, .22])
        truths = centers+np.array([.5, -.4])*scales
        density = lambda points: -.5*np.sum(((points-centers)/scales)**2, axis=1)
        coarse = integrate(density, bounds, centers, scales, truths, 129)
        fine = integrate(density, bounds, centers, scales, truths, 257)
        self.assertTrue(converged(coarse, fine)["passes"])
        for index in range(2):
            reference = truncnorm((bounds[index, 0]-centers[index])/scales[index],
                                  (bounds[index, 1]-centers[index])/scales[index],
                                  loc=centers[index], scale=scales[index])
            self.assertLess(abs(fine["mean"][index]-reference.mean())/reference.std(), .001)
            self.assertLess(abs(fine["sd"][index]/reference.std()-1), .001)
            self.assertLess(abs(fine["cdfs"][index]-reference.cdf(truths[index])), .001)
            np.testing.assert_allclose((np.asarray(fine["quantiles"][index])-reference.ppf([.025, .975]))/reference.std(), 0., atol=.002)

    def test_correlated_gaussian_recovers_analytic_marginals(self):
        correlation = .94
        centers = np.array([.4, -.3])
        scales = np.array([.3, 1.1])
        truths = centers+scales*np.array([.8, -.6])
        def density(points):
            standardized = (points-centers)/scales
            return -(standardized[:, 0]**2-2*correlation*np.prod(standardized, axis=1)+standardized[:, 1]**2)/(2*(1-correlation**2))
        result = integrate(density, np.array([[-12., 12.], [-12., 12.]]), centers, scales, truths, 257)
        np.testing.assert_allclose(result["mean"], centers, atol=1e-5)
        np.testing.assert_allclose(result["sd"], scales, rtol=.001)
        np.testing.assert_allclose(result["cdfs"], norm.cdf([.8, -.6]), atol=.001)
        for index in range(2):
            np.testing.assert_allclose((np.array(result["quantiles"][index])-centers[index])/scales[index], norm.ppf([.025, .975]), atol=.003)

    def test_truth_changes_diagnostics_only(self):
        density = lambda points: -.5*np.sum(points**2, axis=1)
        first = integrate(density, [[-5, 5], [-5, 5]], [0, 0], [1, 1], [0, 0], 65)
        second = integrate(density, [[-5, 5], [-5, 5]], [0, 0], [1, 1], [2, -2], 65)
        for key in ["mean", "sd", "quantiles", "log_integral"]:
            self.assertEqual(first[key], second[key])
        self.assertNotEqual(first["cdfs"], second["cdfs"])


class TestBatchEquilibrium(unittest.TestCase):
    def test_matches_scalar_minima_at_random_and_prior_corner_parameters(self):
        compositions = np.repeat([.15, .25, .4, .5, .6, .75, .85], 3)
        temperatures = np.tile([2200., 2500., 2800.], 7)
        scalar = EquilibriumModel(compositions, temperatures)
        batch = BoundedBatchEquilibrium(compositions, temperatures, CENTER, FREE, BOUND)
        shifts = np.vstack([np.random.default_rng(937).uniform(-BOUND, BOUND, (32, 2)), list(itertools.product([-BOUND, BOUND], repeat=2))])
        expected = []
        for shift in shifts:
            coefficients = CENTER.copy()
            coefficients[FREE] += shift
            expected.append(scalar.evaluate(coefficients)["energy"])
        actual = batch.evaluate(shifts, chunk_size=7)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
        np.testing.assert_array_equal(actual, batch.evaluate(shifts, chunk_size=512))
        with self.assertRaises(ValueError):
            batch.evaluate([[BOUND+1, 0]])


if __name__ == "__main__":
    unittest.main()
