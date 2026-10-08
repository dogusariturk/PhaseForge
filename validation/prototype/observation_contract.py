"""Explicit observation semantics for the standalone integration candidate.

This adapter is not yet wired into PhaseForge. Fixed-configuration energies
retain supplied site fractions; internally equilibrated energies minimize only
the supported unequal-ratio substitutional fixture. No phase coexistence or
general nonconvex equilibrium is implied by the latter mode.
"""
from dataclasses import dataclass
import numpy as np

from equilibrium_solver import EquilibriumModel


@dataclass(frozen=True)
class ObservationBatch:
    kind: str
    temperature: np.ndarray
    bulk_composition: np.ndarray | None = None
    site_fractions: np.ndarray | None = None

    def predict(self, fixture, coefficients):
        """Return a vector in J/mol atoms; reject ambiguous or unsupported data."""
        theta = np.asarray(coefficients, dtype=float)
        if theta.shape != (10,) or not np.isfinite(theta).all():
            raise ValueError("Expected ten finite coefficients in fixture order")
        temperature = np.asarray(self.temperature, dtype=float)
        if temperature.ndim != 1 or not len(temperature):
            raise ValueError("Temperature must be a nonempty one-dimensional vector")
        if not np.isfinite(temperature).all() or np.any(temperature <= 0):
            raise ValueError("Temperature must be finite and positive")
        if self.kind == "fixed_configuration":
            if self.site_fractions is None or self.bulk_composition is not None:
                raise ValueError("Fixed configurations require site fractions only")
            sites = np.asarray(self.site_fractions, dtype=float)
            if sites.shape != (len(temperature), 2):
                raise ValueError("One pair of site fractions per temperature required")
            return fixture.independent_energy(sites, temperature, theta)
        if self.kind == "internal_equilibrium":
            if self.bulk_composition is None or self.site_fractions is not None:
                raise ValueError("Internal equilibrium requires bulk composition only")
            # The current fast solver has a specific, documented CEF basis.
            if fixture.species != (("AL", "NI"), ("AL", "NI")) or fixture.site_ratios != (1, 3):
                raise NotImplementedError("Internal equilibrium supports (Al,Ni)1:(Al,Ni)3 only")
            bulk = np.asarray(self.bulk_composition, dtype=float)
            if bulk.shape != temperature.shape:
                raise ValueError("One bulk composition per temperature required")
            return EquilibriumModel(bulk, temperature).evaluate(theta)["energy"]
        raise ValueError("Unknown observation kind; select explicit fixed_configuration or internal_equilibrium")
